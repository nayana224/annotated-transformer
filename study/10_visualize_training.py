import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


training_rows = read_csv(
    OUTPUT_DIR / "training_trace.csv"
)

attention_rows = read_csv(
    OUTPUT_DIR / "attention_trace.csv"
)

parameter_rows = read_csv(
    OUTPUT_DIR / "parameter_trace.csv"
)


steps = [
    int(row["step"])
    for row in training_rows
]


# ==================================================
# 1. Probe / Test Loss vs Step
# ==================================================

probe_losses = [
    float(row["probe_loss"])
    for row in training_rows
]

test_losses = [
    float(row["test_mean_loss"])
    for row in training_rows
]

plt.figure(figsize=(8, 4))

plt.plot(
    steps,
    probe_losses,
    label="Held-out probe",
)

plt.plot(
    steps,
    test_losses,
    label="Test mean",
)

plt.title("Held-out Loss vs Step")
plt.xlabel("Optimizer Step")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.grid()

loss_path = (
    FIGURE_DIR
    / "01_heldout_loss_vs_step.png"
)

plt.tight_layout()
plt.savefig(loss_path, dpi=150)
plt.close()


# ==================================================
# 2. Held-out Accuracy / GT Probability
# ==================================================

probe_gt_probs = [
    float(
        row[
            "probe_mean_gt_probability"
        ]
    )
    for row in training_rows
]

test_accuracies = [
    float(
        row["test_token_accuracy"]
    )
    for row in training_rows
]

plt.figure(figsize=(8, 4))

plt.plot(
    steps,
    probe_gt_probs,
    label="Probe mean GT probability",
)

plt.plot(
    steps,
    test_accuracies,
    label="Test token accuracy",
)

plt.title(
    "Held-out Prediction Quality vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Value")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid()

quality_path = (
    FIGURE_DIR
    / "02_heldout_quality_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    quality_path,
    dpi=150,
)
plt.close()


# ==================================================
# 3. Decoder Cross-Attention
#    Head 1 / probe의 마지막 query가
#    English source token을 보는 weight 변화
# ==================================================

cross_rows = [
    row
    for row in attention_rows
    if (
        row["attention_type"]
        == "decoder_cross"
        and int(row["head"]) == 0
    )
]

last_query_index = max(
    int(row["query_index"])
    for row in cross_rows
)

cross_last_rows = [
    row
    for row in cross_rows
    if int(row["query_index"])
    == last_query_index
]

by_key = defaultdict(list)

for row in cross_last_rows:
    by_key[
        row["key_token"]
    ].append(
        (
            int(row["step"]),
            float(row["weight"]),
        )
    )

plt.figure(figsize=(8, 5))

for key_token, values in (
    by_key.items()
):
    values.sort(
        key=lambda x: x[0]
    )

    x = [
        step
        for step, _ in values
    ]

    y = [
        weight
        for _, weight in values
    ]

    plt.plot(
        x,
        y,
        label=key_token,
    )

plt.title(
    "Probe Cross-Attention "
    "(Head 1, Last Decoder Query)"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Attention Weight")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid()

attention_path = (
    FIGURE_DIR
    / "03_probe_cross_attention_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    attention_path,
    dpi=150,
)
plt.close()


# ==================================================
# 4. 대표 W_Q scalar 변화
# ==================================================

tracked_parameter = (
    "decoder_layers.0."
    "masked_self_attention."
    "W_Q.0.weight"
)

tracked_rows = [
    row
    for row in parameter_rows
    if (
        row["parameter"]
        == tracked_parameter
        and row["index"] == "0,0"
    )
]

parameter_steps = [
    int(row["step"])
    for row in tracked_rows
]

parameter_values = [
    float(row["value_before"])
    for row in tracked_rows
]

plt.figure(figsize=(8, 4))

plt.plot(
    parameter_steps,
    parameter_values,
)

plt.title(
    "Selected W_Q[0,0] vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Parameter Value")
plt.grid()

parameter_path = (
    FIGURE_DIR
    / "04_selected_parameter_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    parameter_path,
    dpi=150,
)
plt.close()


# ==================================================
# 5. 같은 parameter의 gradient 변화
# ==================================================

gradients = [
    float(row["gradient"])
    for row in tracked_rows
]

plt.figure(figsize=(8, 4))

plt.plot(
    parameter_steps,
    gradients,
)

plt.axhline(
    y=0.0,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "Selected W_Q[0,0] Gradient vs Step"
)

plt.xlabel("Optimizer Step")
plt.ylabel("Gradient")
plt.grid()

gradient_path = (
    FIGURE_DIR
    / "05_selected_gradient_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    gradient_path,
    dpi=150,
)
plt.close()


print("=" * 70)
print("Saved Figures")
print("=" * 70)

for path in [
    loss_path,
    quality_path,
    attention_path,
    parameter_path,
    gradient_path,
]:
    print(path)

print(
    "\nCSV 전체 값은 study/outputs/에서 "
    "직접 열어 확인할 수 있다."
)
