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
        return list(
            csv.DictReader(f)
        )


training_rows = read_csv(
    OUTPUT_DIR
    / "training_trace.csv"
)

attention_rows = read_csv(
    OUTPUT_DIR
    / "attention_trace.csv"
)

parameter_rows = read_csv(
    OUTPUT_DIR
    / "parameter_trace.csv"
)


# ==================================================
# 1. Loss vs Step
# ==================================================

steps = [
    int(row["step"])
    for row in training_rows
]

losses = [
    float(row["loss"])
    for row in training_rows
]

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    steps,
    losses,
    marker="o",
)

plt.title(
    "Training Loss vs Step"
)

plt.xlabel("Step")
plt.ylabel("Cross Entropy Loss")
plt.grid()

loss_path = (
    FIGURE_DIR
    / "01_loss_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    loss_path,
    dpi=150,
)
plt.close()


# ==================================================
# 2. Mean GT Probability vs Step
# ==================================================

mean_gt_probs = [
    float(
        row["mean_gt_probability"]
    )
    for row in training_rows
]

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    steps,
    mean_gt_probs,
    marker="o",
)

plt.title(
    "Mean Ground-Truth Probability vs Step"
)

plt.xlabel("Step")
plt.ylabel("Mean GT Probability")
plt.ylim(0.0, 1.0)
plt.grid()

gt_path = (
    FIGURE_DIR
    / "02_gt_probability_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    gt_path,
    dpi=150,
)
plt.close()


# ==================================================
# 3. Decoder Cross-Attention
#    Head 1 / 마지막 query가
#    source token을 보는 weight 변화
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
    if int(
        row["query_index"]
    ) == last_query_index
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

plt.figure(
    figsize=(8, 5)
)

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
        marker="o",
        label=key_token,
    )

plt.title(
    "Decoder Cross-Attention "
    "(Head 1, Last Query)"
)

plt.xlabel("Step")
plt.ylabel("Attention Weight")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid()

attention_path = (
    FIGURE_DIR
    / "03_cross_attention_vs_step.png"
)

plt.tight_layout()
plt.savefig(
    attention_path,
    dpi=150,
)
plt.close()


# ==================================================
# 4. 대표 W_Q scalar의 변화
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

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    parameter_steps,
    parameter_values,
    marker="o",
)

plt.title(
    "Selected W_Q[0,0] vs Step"
)

plt.xlabel("Step")
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

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    parameter_steps,
    gradients,
    marker="o",
)

plt.axhline(
    y=0.0,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "Selected W_Q[0,0] Gradient vs Step"
)

plt.xlabel("Step")
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
    gt_path,
    attention_path,
    parameter_path,
    gradient_path,
]:
    print(path)

print(
    "\nCSV 전체 값은 study/outputs/에서 "
    "직접 열어 확인할 수 있다."
)
