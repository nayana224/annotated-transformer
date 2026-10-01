import sys
from pathlib import Path

import torch
import torch.nn as nn

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from src.data import make_example
from src.factory import build_study_objects
from src.tokenization import invert_vocab


TRAIN_PATH = STUDY_DIR / "data" / "en_ko_train.csv"

DEVICE = "cpu"
SEED = 42
LEARNING_RATE = 0.05
SAMPLE_INDEX = 0


(
    train_pairs,
    src_vocab,
    tgt_vocab,
    model,
) = build_study_objects(
    TRAIN_PATH,
    seed=SEED,
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_layers=1,
    device=DEVICE,
)

example = make_example(
    train_pairs[SAMPLE_INDEX],
    src_vocab,
    tgt_vocab,
    device=DEVICE,
)

id_to_target = invert_vocab(tgt_vocab)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


print("=" * 70)
print("Actual English -> Korean Input / Ground Truth")
print("=" * 70)

print("Source text:")
print(example["source_text"])

print("\nSource tokens:")
print(example["src_tokens"])

print("\nSource IDs:")
print(example["src_ids"])

print("\nTarget text:")
print(example["target_text"])

print("\nDecoder input tokens:")
print(example["decoder_input_tokens"])

print("\nDecoder input IDs:")
print(example["decoder_input_ids"])

print("\nGT tokens:")
print(example["gt_tokens"])

print("\nGT IDs:")
print(example["gt_ids"])


tracked_weight = (
    model
    .decoder_layers[0]
    .masked_self_attention
    .W_Q[0]
    .weight
)

weight_before = tracked_weight[0, 0].detach().item()


# ==================================================
# 1. Forward
# ==================================================

optimizer.zero_grad()

trace = model(
    example["src_ids"],
    example["decoder_input_ids"],
)

logits = trace["logits"]
probabilities = trace["probabilities"]

loss = criterion(
    logits,
    example["gt_ids"],
)

pred_ids = probabilities.argmax(dim=-1)

pred_tokens = [
    id_to_target[token_id.item()]
    for token_id in pred_ids
]

gt_probabilities = probabilities[
    torch.arange(len(example["gt_ids"])),
    example["gt_ids"],
]


print("\n" + "=" * 70)
print("1. Forward")
print("=" * 70)

print("Encoder output:")
print(trace["encoder_output"])
print("shape:", trace["encoder_output"].shape)

print("\nCausal mask:")
print(trace["causal_mask"])

print("\nDecoder output:")
print(trace["decoder_output"])
print("shape:", trace["decoder_output"].shape)

print("\nLogits:")
print(logits)
print("shape:", logits.shape)

print("\nProbabilities:")
print(probabilities)

print("\nGT probabilities:")
for token, prob in zip(
    example["gt_tokens"],
    gt_probabilities,
):
    print(f"P({token:>8}) = {prob.item():.6f}")

print("\nPrediction:")
print(pred_tokens)

print("\nGround Truth:")
print(example["gt_tokens"])

print("\nLoss:")
print(loss.item())


# ==================================================
# 2. Attention
# ==================================================

masked_weights = trace["masked_attention"][0]
cross_weights = trace["cross_attention"][0]


print("\n" + "=" * 70)
print("2. Attention Weights")
print("=" * 70)

print("Masked Self-Attention - Layer 1 / Head 1")
print(masked_weights[0])

print("\nCross-Attention - Layer 1 / Head 1")
print(cross_weights[0])


# ==================================================
# 3. Backward
# ==================================================

loss.backward()

tracked_grad = tracked_weight.grad[0, 0].detach().item()


def grad_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return parameter.grad.norm().item()


print("\n" + "=" * 70)
print("3. Backward")
print("=" * 70)

print(
    "Tracked parameter:\n"
    "decoder Layer 1 / masked self-attention / "
    "Head 1 / W_Q[0,0]"
)

print(f"weight before = {weight_before:.8f}")
print(f"gradient      = {tracked_grad:.8f}")

print("\nGradient norms:")

print(
    "masked W_Q:",
    grad_norm(
        model.decoder_layers[0]
        .masked_self_attention
        .W_Q[0]
        .weight
    ),
)

print(
    "masked W_K:",
    grad_norm(
        model.decoder_layers[0]
        .masked_self_attention
        .W_K[0]
        .weight
    ),
)

print(
    "masked W_V:",
    grad_norm(
        model.decoder_layers[0]
        .masked_self_attention
        .W_V[0]
        .weight
    ),
)

print(
    "cross W_Q:",
    grad_norm(
        model.decoder_layers[0]
        .cross_attention
        .W_Q[0]
        .weight
    ),
)

print(
    "output_linear:",
    grad_norm(model.output_linear.weight),
)


# ==================================================
# 4. Optimizer Step
# ==================================================

optimizer.step()

weight_after = tracked_weight[0, 0].detach().item()
delta = weight_after - weight_before


print("\n" + "=" * 70)
print("4. Optimizer Step")
print("=" * 70)

print(f"weight before = {weight_before:.8f}")
print(f"gradient      = {tracked_grad:.8f}")
print(f"learning rate = {LEARNING_RATE}")
print(f"weight after  = {weight_after:.8f}")
print(f"delta weight  = {delta:.8f}")

print(
    "\nSGD에서는 대략 "
    "weight_after = weight_before - lr * gradient"
)


# ==================================================
# 5. Update 후 Forward를 다시 확인
# ==================================================

with torch.no_grad():
    trace_after = model(
        example["src_ids"],
        example["decoder_input_ids"],
    )

    loss_after = criterion(
        trace_after["logits"],
        example["gt_ids"],
    )


print("\n" + "=" * 70)
print("5. Loss Before / After One Update")
print("=" * 70)

print(f"before: {loss.item():.6f}")
print(f"after : {loss_after.item():.6f}")

print(
    "\n이번 파일의 목적은 실제 EN-KO sentence pair를 사용해 "
    "forward -> loss -> backward -> gradient -> "
    "parameter update를 한 번 끝까지 연결하는 것이다."
)
