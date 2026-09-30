import csv
import itertools
from pathlib import Path

import torch
import torch.nn as nn

from trace_utils import (
    build_study_objects,
    make_example,
)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "toy_translation.csv"
OUTPUT_DIR = HERE / "outputs"

DEVICE = "cpu"
SEED = 42
LEARNING_RATE = 0.05
STEPS = 30
SAMPLE_INDEX = 0


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


(
    pairs,
    src_vocab,
    tgt_vocab,
    model,
) = build_study_objects(
    DATA_PATH,
    seed=SEED,
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_layers=1,
    device=DEVICE,
)

example = make_example(
    pairs[SAMPLE_INDEX],
    src_vocab,
    tgt_vocab,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


def all_indices(shape):
    return itertools.product(*[
        range(size)
        for size in shape
    ])


def index_to_text(index):
    return ",".join(
        str(i)
        for i in index
    )


def snapshot_parameters(model):
    snapshot = {}

    for name, parameter in (
        model.named_parameters()
    ):
        data = (
            parameter
            .detach()
            .cpu()
        )

        for index in all_indices(
            data.shape
        ):
            snapshot[
                (name, index)
            ] = data[index].item()

    return snapshot


def append_tensor_trace(
    rows,
    step,
    name,
    tensor,
):
    data = (
        tensor
        .detach()
        .cpu()
    )

    for index in all_indices(
        data.shape
    ):
        rows.append({
            "step": step,
            "tensor": name,
            "index": index_to_text(
                index
            ),
            "value": data[
                index
            ].item(),
        })


def append_attention_trace(
    rows,
    step,
    attention_type,
    weights,
    query_tokens,
    key_tokens,
):
    # weights:
    # [num_heads, query_len, key_len]

    data = (
        weights
        .detach()
        .cpu()
    )

    for head in range(
        data.shape[0]
    ):
        for q in range(
            data.shape[1]
        ):
            for k in range(
                data.shape[2]
            ):
                rows.append({
                    "step": step,
                    "attention_type":
                        attention_type,
                    "head": head,
                    "query_index": q,
                    "query_token":
                        query_tokens[q],
                    "key_index": k,
                    "key_token":
                        key_tokens[k],
                    "weight":
                        data[
                            head,
                            q,
                            k,
                        ].item(),
                })


training_rows = []
tensor_rows = []
attention_rows = []
parameter_rows = []


print("=" * 70)
print("Training Trace")
print("=" * 70)

print("Source:")
print(example["source_text"])

print("\nTarget:")
print(example["target_text"])

print("\nSteps:", STEPS)
print(
    "Learning rate:",
    LEARNING_RATE,
)


for step in range(STEPS):
    optimizer.zero_grad()

    parameter_before = (
        snapshot_parameters(model)
    )

    trace = model(
        example["src_ids"],
        example[
            "decoder_input_ids"
        ],
    )

    logits = trace["logits"]

    probabilities = trace[
        "probabilities"
    ]

    loss = criterion(
        logits,
        example["gt_ids"],
    )

    predictions = (
        probabilities.argmax(
            dim=-1
        )
    )

    accuracy = (
        predictions
        .eq(example["gt_ids"])
        .float()
        .mean()
        .item()
    )

    positions = torch.arange(
        len(example["gt_ids"]),
        device=DEVICE,
    )

    gt_probabilities = (
        probabilities[
            positions,
            example["gt_ids"],
        ]
    )

    mean_gt_probability = (
        gt_probabilities
        .mean()
        .item()
    )

    # ----------------------------------------------
    # Forward tensors
    # ----------------------------------------------

    forward_tensors = {
        "src_embedding":
            trace["src_embedding"],
        "src_pe":
            trace["src_pe"],
        "encoder_input":
            trace["encoder_input"],
        "encoder_output":
            trace["encoder_output"],
        "tgt_embedding":
            trace["tgt_embedding"],
        "tgt_pe":
            trace["tgt_pe"],
        "decoder_input":
            trace["decoder_input"],
        "decoder_output":
            trace["decoder_output"],
        "logits":
            trace["logits"],
        "probabilities":
            trace["probabilities"],
    }

    for (
        tensor_name,
        tensor,
    ) in forward_tensors.items():
        append_tensor_trace(
            tensor_rows,
            step,
            tensor_name,
            tensor,
        )

    # ----------------------------------------------
    # Attention weights
    # ----------------------------------------------

    append_attention_trace(
        attention_rows,
        step,
        "encoder_self",
        trace[
            "encoder_attention"
        ][0],
        example["src_tokens"],
        example["src_tokens"],
    )

    append_attention_trace(
        attention_rows,
        step,
        "decoder_masked",
        trace[
            "masked_attention"
        ][0],
        example[
            "decoder_input_tokens"
        ],
        example[
            "decoder_input_tokens"
        ],
    )

    append_attention_trace(
        attention_rows,
        step,
        "decoder_cross",
        trace[
            "cross_attention"
        ][0],
        example[
            "decoder_input_tokens"
        ],
        example["src_tokens"],
    )

    # ----------------------------------------------
    # Backward
    # ----------------------------------------------

    loss.backward()

    gradients = {}

    for name, parameter in (
        model.named_parameters()
    ):
        if parameter.grad is None:
            continue

        grad = (
            parameter.grad
            .detach()
            .cpu()
        )

        for index in all_indices(
            grad.shape
        ):
            gradients[
                (name, index)
            ] = grad[index].item()

    # ----------------------------------------------
    # Optimizer update
    # ----------------------------------------------

    optimizer.step()

    parameter_after = (
        snapshot_parameters(model)
    )

    for (
        key,
        value_before,
    ) in parameter_before.items():
        (
            parameter_name,
            index,
        ) = key

        grad = gradients.get(
            key,
            0.0,
        )

        value_after = (
            parameter_after[key]
        )

        parameter_rows.append({
            "step": step,
            "parameter":
                parameter_name,
            "index":
                index_to_text(index),
            "value_before":
                value_before,
            "gradient":
                grad,
            "value_after":
                value_after,
            "delta":
                value_after
                - value_before,
        })

    training_row = {
        "step": step,
        "loss": loss.item(),
        "accuracy": accuracy,
        "mean_gt_probability":
            mean_gt_probability,
    }

    for i, probability in enumerate(
        gt_probabilities
    ):
        training_row[
            f"gt_probability_{i}"
        ] = probability.item()

    training_rows.append(
        training_row
    )

    print(
        f"step={step:02d}  "
        f"loss={loss.item():.6f}  "
        f"accuracy={accuracy:.2f}  "
        f"mean_GT_prob="
        f"{mean_gt_probability:.6f}"
    )


def write_csv(
    path,
    rows,
    fieldnames,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


training_fields = [
    "step",
    "loss",
    "accuracy",
    "mean_gt_probability",
] + [
    f"gt_probability_{i}"
    for i in range(
        len(example["gt_ids"])
    )
]


write_csv(
    OUTPUT_DIR
    / "training_trace.csv",
    training_rows,
    training_fields,
)

write_csv(
    OUTPUT_DIR
    / "tensor_trace.csv",
    tensor_rows,
    [
        "step",
        "tensor",
        "index",
        "value",
    ],
)

write_csv(
    OUTPUT_DIR
    / "attention_trace.csv",
    attention_rows,
    [
        "step",
        "attention_type",
        "head",
        "query_index",
        "query_token",
        "key_index",
        "key_token",
        "weight",
    ],
)

write_csv(
    OUTPUT_DIR
    / "parameter_trace.csv",
    parameter_rows,
    [
        "step",
        "parameter",
        "index",
        "value_before",
        "gradient",
        "value_after",
        "delta",
    ],
)


print("\n" + "=" * 70)
print("Saved")
print("=" * 70)

for filename in [
    "training_trace.csv",
    "tensor_trace.csv",
    "attention_trace.csv",
    "parameter_trace.csv",
]:
    print(
        OUTPUT_DIR / filename
    )

print(
    "\nCSV는 모두 step별 값을 "
    "그대로 저장한다."
)

print(
    "복잡한 소수점 계산은 "
    "손으로 풀지 않고 PyTorch의 "
    "실제 계산값을 관찰한다."
)
