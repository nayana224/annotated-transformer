import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 0. Reproducibility
# ==================================================

torch.manual_seed(42)


# ==================================================
# 1. Configuration
# ==================================================

d_model = 4

vocab = {
    0: "<PAD>",
    1: "<SOS>",
    2: "I",
    3: "love",
    4: "robotics",
    5: "<EOS>",
}

vocab_size = len(vocab)


print("======================================")
print("Configuration")
print("======================================")

print("d_model   :", d_model)
print("vocab_size:", vocab_size)

print("\nVocabulary:")

for token_id, token in vocab.items():
    print(token_id, "=", token)


# ==================================================
# 2. Final Decoder Output
# ==================================================
#
# 05_decoder.py의 최종 output을 그대로 사용
#
# Decoder input:
#
# <SOS> I love
#
# 따라서 각 row는:
#
# row 0 -> I 예측
# row 1 -> love 예측
# row 2 -> robotics 예측
#
#
# shape:
#
# [target_len, d_model]
# = [3, 4]
# ==================================================

decoder_output = torch.tensor([
    [ 0.1769, -1.0267, -0.7050,  1.5548],
    [-0.4914,  0.0975,  1.5519, -1.1579],
    [-1.2938,  0.2286, -0.3912,  1.4563],
])


print("\n======================================")
print("Decoder Output")
print("======================================")

print(decoder_output)
print("shape:", decoder_output.shape)


# ==================================================
# 3. Output Linear Projection
# ==================================================
#
# Decoder output:
#
# [3, 4]
#
# ↓ Linear
#
# logits:
#
# [3, 6]
#
#
# 즉 각 position마다
# vocabulary의 모든 token에 대한 score를 만든다.
# ==================================================

output_linear = nn.Linear(
    d_model,
    vocab_size,
    bias=False
)


print("\n======================================")
print("Output Linear Weight")
print("======================================")

print(output_linear.weight)

print(
    "weight shape:",
    output_linear.weight.shape
)


# ==================================================
# 4. Logits
# ==================================================
#
# logits:
#
# 아직 probability가 아님
#
# 각 vocabulary token에 대한
# raw score
# ==================================================

logits = output_linear(
    decoder_output
)


print("\n======================================")
print("Logits")
print("======================================")

print(logits)

print(
    "shape:",
    logits.shape
)


# ==================================================
# 5. Softmax
# ==================================================
#
# vocab dimension에 대해 softmax
#
# 각 row:
#
# vocabulary 전체에 대한 probability distribution
#
# row sum = 1
# ==================================================

probabilities = F.softmax(
    logits,
    dim=-1
)


print("\n======================================")
print("Probabilities")
print("======================================")

print(probabilities)


print("\nRow sums:")

print(
    probabilities.sum(
        dim=-1
    )
)


# ==================================================
# 6. Prediction
# ==================================================
#
# 각 position에서
# 가장 probability가 큰 token을 선택
# ==================================================

pred_ids = torch.argmax(
    probabilities,
    dim=-1
)


print("\n======================================")
print("Predictions")
print("======================================")

print(
    "Predicted IDs:",
    pred_ids
)


pred_tokens = [
    vocab[token_id.item()]
    for token_id in pred_ids
]


print(
    "Predicted Tokens:",
    pred_tokens
)


# ==================================================
# 7. Ground Truth
# ==================================================
#
# Target:
#
# I love robotics
#
#
# Decoder input:
#
# <SOS> I love
#
#
# GT:
#
# I love robotics
#
# ==================================================

gt_ids = torch.tensor([
    2,   # I
    3,   # love
    4,   # robotics
])


print("\n======================================")
print("Ground Truth")
print("======================================")

print(
    "GT IDs:",
    gt_ids
)


gt_tokens = [
    vocab[token_id.item()]
    for token_id in gt_ids
]


print(
    "GT Tokens:",
    gt_tokens
)


# ==================================================
# 8. GT Probability 확인
# ==================================================
#
# 각 position에서
# 정답 token에 할당된 probability를 확인한다.
#
# position 0 -> P(I)
# position 1 -> P(love)
# position 2 -> P(robotics)
# ==================================================

gt_probabilities = probabilities[
    torch.arange(
        len(gt_ids)
    ),
    gt_ids
]


print("\n======================================")
print("GT Probabilities")
print("======================================")

print(
    "P(I)        :",
    gt_probabilities[0]
)

print(
    "P(love)     :",
    gt_probabilities[1]
)

print(
    "P(robotics) :",
    gt_probabilities[2]
)


# ==================================================
# 9. Manual Cross Entropy
# ==================================================
#
# 한 position의 loss:
#
# -log(P(correct token))
#
#
# 세 position의 평균을 계산한다.
# ==================================================

manual_losses = (
    -torch.log(
        gt_probabilities
    )
)


print("\n======================================")
print("Manual CE Loss per Position")
print("======================================")

print(manual_losses)


manual_loss = (
    manual_losses.mean()
)


print(
    "\nManual Mean Loss:",
    manual_loss
)


# ==================================================
# 10. PyTorch CrossEntropyLoss
# ==================================================
#
# 중요한 점:
#
# CrossEntropyLoss에는
#
# softmax 결과가 아니라
# raw logits를 넣는다.
#
# 내부에서:
#
# LogSoftmax
# +
# NLLLoss
#
# 를 처리한다.
# ==================================================

criterion = nn.CrossEntropyLoss()


loss = criterion(
    logits,
    gt_ids
)


print("\n======================================")
print("PyTorch CrossEntropy Loss")
print("======================================")

print(loss)


# ==================================================
# 11. Manual Loss와 비교
# ==================================================

print("\n======================================")
print("Loss Comparison")
print("======================================")

print(
    "Manual CE :",
    manual_loss
)

print(
    "PyTorch CE:",
    loss
)


# ==================================================
# 12. Label Smoothing
# ==================================================
#
# Attention Is All You Need:
#
# epsilon_ls = 0.1
#
# 정답 token에 probability 1.0을
# 완전히 몰아주지 않고
# 일부를 다른 class에도 분배한다.
# ==================================================

criterion_smoothing = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


loss_smoothing = criterion_smoothing(
    logits,
    gt_ids
)


print("\n======================================")
print("Label Smoothing")
print("======================================")

print(
    "CrossEntropy:",
    loss
)

print(
    "Label Smoothing Loss:",
    loss_smoothing
)