import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. 이전 단계의 x
# --------------------------------------------------
#
# Embedding + Positional Encoding 결과라고 가정
# token 3개, d_model = 4

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])

print("x:")
print(x)
print("x shape:", x.shape)


# --------------------------------------------------
# 2. Linear projection으로 Q, K, V 생성
# --------------------------------------------------

d_model = 4
d_k = 4
d_v = 4

W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_v, bias=False)

Q = W_Q(x)
K = W_K(x)
V = W_V(x)

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# --------------------------------------------------
# 3. QK^T
# --------------------------------------------------

scores = Q @ K.T

print("\nQ @ K.T:")
print(scores)
print("scores shape:", scores.shape)


# --------------------------------------------------
# 4. Scaling
# --------------------------------------------------

scaled_scores = scores / math.sqrt(d_k)

print("\nscaled scores:")
print(scaled_scores)


# --------------------------------------------------
# 5. Softmax
# --------------------------------------------------
#
# 각 query가 모든 key에 대해 가지는 weight
# 마지막 dimension 기준으로 softmax

attention_weights = F.softmax(scaled_scores, dim=-1)

print("\nattention weights(softmax):")
print(attention_weights)

print("\nrow sums:")
print(attention_weights.sum(dim=-1))


# --------------------------------------------------
# 6. Weighted sum of V
# --------------------------------------------------

output = attention_weights @ V

print("\nattention output:")
print(output)

print("output shape:", output.shape)