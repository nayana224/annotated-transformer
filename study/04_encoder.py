import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 1. Multi-Head Self-Attention
# ==================================================

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Head별 Linear projection
        self.W_Q = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
            for _ in range(num_heads)
        ])

        self.W_K = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False)
            for _ in range(num_heads)
        ])

        self.W_V = nn.ModuleList([
            nn.Linear(d_model, self.d_v, bias=False)
            for _ in range(num_heads)
        ])

        # Concat 후 최종 projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)


    def scaled_dot_product_attention(self, Q, K, V):

        scores = Q @ K.T

        scaled_scores = scores / math.sqrt(self.d_k)

        attention_weights = F.softmax(
            scaled_scores,
            dim=-1
        )

        output = attention_weights @ V

        return output, attention_weights


    def forward(self, x):

        head_outputs = []
        attention_weights_all = []

        for i in range(self.num_heads):

            Q = self.W_Q[i](x)
            K = self.W_K[i](x)
            V = self.W_V[i](x)

            head_output, attention_weights = (
                self.scaled_dot_product_attention(
                    Q,
                    K,
                    V
                )
            )

            head_outputs.append(head_output)
            attention_weights_all.append(attention_weights)

        # [3,2] + [3,2] → [3,4]
        concat = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.W_O(concat)

        return output, attention_weights_all


# ==================================================
# 2. Encoder Layer 하나
# ==================================================

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):

        # ------------------------------------------
        # Multi-Head Self-Attention
        # ------------------------------------------

        attention_output, attention_weights = (
            self.attention(x)
        )


        # ------------------------------------------
        # Residual + LayerNorm
        # ------------------------------------------

        residual1 = x + attention_output

        x1 = self.norm1(residual1)


        # ------------------------------------------
        # FFN
        # ------------------------------------------

        ffn_output = self.ffn(x1)


        # ------------------------------------------
        # Residual + LayerNorm
        # ------------------------------------------

        residual2 = x1 + ffn_output

        output = self.norm2(residual2)


        return (
            output,
            residual1,
            x1,
            attention_weights
        )


# ==================================================
# 3. Encoder Stack
# ==================================================

class Encoder(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers
    ):
        super().__init__()

        # Encoder Layer를 6개 "각각 따로" 생성
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ])


    def forward(self, x):

        first_residual = None
        first_norm = None

        for i, layer in enumerate(self.layers):

            print(f"\n========== Encoder Layer {i + 1} ==========")

            x, residual1, x1, attention_weights = layer(x)

            print("output:")
            print(x)

            print("shape:", x.shape)

            # 첫 번째 layer의 LayerNorm 결과는
            # 시각화를 위해 저장
            if i == 0:
                first_residual = residual1
                first_norm = x1

        return x, first_residual, first_norm


# ==================================================
# 4. Toy Input
# ==================================================
#
# 실제로는:
#
# token IDs
# → Embedding
# → + Positional Encoding
# → x
#
# 이라고 생각한다.
#
# I / love / robotics
#

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])


print("Initial x:")
print(x)

print("shape:", x.shape)


# ==================================================
# 5. Model 설정
# ==================================================

d_model = 4
num_heads = 2
d_ff = 8

# 논문의 N = 6을 따라감
num_layers = 6


encoder = Encoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)


# ==================================================
# 6. Encoder × 6 Forward
# ==================================================

encoder_output, before_norm, after_norm = encoder(x)


print("\n======================================")
print("Final Encoder Output")
print("======================================")

print(encoder_output)

print("shape:", encoder_output.shape)


# ==================================================
# 7. 첫 번째 Encoder Layer의 LayerNorm 시각화
# ==================================================

before = before_norm.detach().numpy()
after = after_norm.detach().numpy()

tokens = [
    "I",
    "love",
    "robotics"
]


for i, token in enumerate(tokens):

    plt.figure(figsize=(7, 4))

    plt.plot(
        range(d_model),
        before[i],
        marker="o",
        label="Before LayerNorm"
    )

    plt.plot(
        range(d_model),
        after[i],
        marker="o",
        label="After LayerNorm"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        f"Encoder Layer 1 - LayerNorm: {token}"
    )

    plt.xlabel("Feature dimension")
    plt.ylabel("Value")

    plt.xticks(
        range(d_model)
    )

    plt.legend()
    plt.grid()

    plt.show()


# ==================================================
# 8. LayerNorm 통계 확인
# ==================================================

print("\n=== Before LayerNorm ===")

print("mean:")
print(
    before_norm.mean(dim=-1)
)

print("std:")
print(
    before_norm.std(
        dim=-1,
        unbiased=False
    )
)


print("\n=== After LayerNorm ===")

print("mean:")
print(
    after_norm.mean(dim=-1)
)

print("std:")
print(
    after_norm.std(
        dim=-1,
        unbiased=False
    )
)