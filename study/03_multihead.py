import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. Transformer input x
# --------------------------------------------------
#
# 이전 embedding + positional encoding 결과라고 가정
#
# token 3개
# d_model = 4
#
# 예:
# token 0 = I
# token 1 = love
# token 2 = robotics

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])

print("x:")
print(x)
print("x shape:", x.shape)


# --------------------------------------------------
# 2. Multi-Head 설정
# --------------------------------------------------

d_model = 4
num_heads = 2

d_k = d_model // num_heads
d_v = d_model // num_heads

print("\nd_model:", d_model)
print("num_heads:", num_heads)
print("d_k:", d_k)
print("d_v:", d_v)


# --------------------------------------------------
# 3. Head 1의 Linear Projection
# --------------------------------------------------
#
# x [3, 4]
#
# ↓ 각각 다른 Linear
#
# Q1, K1, V1 [3, 2]

W_Q1 = nn.Linear(d_model, d_k, bias=False)
W_K1 = nn.Linear(d_model, d_k, bias=False)
W_V1 = nn.Linear(d_model, d_v, bias=False)

Q1 = W_Q1(x)
K1 = W_K1(x)
V1 = W_V1(x)


# --------------------------------------------------
# 4. Head 2의 Linear Projection
# --------------------------------------------------

W_Q2 = nn.Linear(d_model, d_k, bias=False)
W_K2 = nn.Linear(d_model, d_k, bias=False)
W_V2 = nn.Linear(d_model, d_v, bias=False)

Q2 = W_Q2(x)
K2 = W_K2(x)
V2 = W_V2(x)


print("\n=== Head 1 ===")
print("Q1:")
print(Q1)
print("Q1 shape:", Q1.shape)

print("\nK1:")
print(K1)

print("\nV1:")
print(V1)


print("\n=== Head 2 ===")
print("Q2:")
print(Q2)
print("Q2 shape:", Q2.shape)

print("\nK2:")
print(K2)

print("\nV2:")
print(V2)


# --------------------------------------------------
# 5. Scaled Dot-Product Attention 함수
# --------------------------------------------------

def scaled_dot_product_attention(Q, K, V):

    # Q @ K^T
    scores = Q @ K.T

    # scaling
    scaled_scores = scores / math.sqrt(Q.shape[-1])

    # softmax
    attention_weights = F.softmax(
        scaled_scores,
        dim=-1
    )

    # weighted sum of V
    output = attention_weights @ V

    return output, attention_weights


# --------------------------------------------------
# 6. 각 Head에서 Attention 수행
# --------------------------------------------------

head1, weights1 = scaled_dot_product_attention(
    Q1,
    K1,
    V1
)

head2, weights2 = scaled_dot_product_attention(
    Q2,
    K2,
    V2
)


print("\n=== Head 1 Attention Weights ===")
print(weights1)

print("\nHead 1 Output:")
print(head1)
print("shape:", head1.shape)


print("\n=== Head 2 Attention Weights ===")
print(weights2)

print("\nHead 2 Output:")
print(head2)
print("shape:", head2.shape)


# --------------------------------------------------
# 7. 두 Head의 결과를 Concatenate
# --------------------------------------------------
#
# head1 : [3, 2]
# head2 : [3, 2]
#
# concat:
#
# [3, 2] + [3, 2]
#        ↓
#      [3, 4]

concat = torch.cat(
    [head1, head2],
    dim=-1
)

print("\n=== Concatenated Heads ===")
print(concat)
print("concat shape:", concat.shape)


# --------------------------------------------------
# 8. Final Linear Projection
# --------------------------------------------------
#
# 논문의 W^O에 해당
#
# [3, 4]
# ↓
# [3, 4]

W_O = nn.Linear(
    d_model,
    d_model,
    bias=False
)

output = W_O(concat)


print("\n=== Multi-Head Attention Output ===")
print(output)

print("output shape:", output.shape)