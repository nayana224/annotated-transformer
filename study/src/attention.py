import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Head별 W_Q를 만든다.
        self.W_Q = nn.ModuleList()

        for head_index in range(num_heads):
            linear_q = nn.Linear(
                d_model,
                self.d_k,
                bias=False,
            )

            self.W_Q.append(linear_q)

        # Head별 W_K를 만든다.
        self.W_K = nn.ModuleList()

        for head_index in range(num_heads):
            linear_k = nn.Linear(
                d_model,
                self.d_k,
                bias=False,
            )

            self.W_K.append(linear_k)

        # Head별 W_V를 만든다.
        self.W_V = nn.ModuleList()

        for head_index in range(num_heads):
            linear_v = nn.Linear(
                d_model,
                self.d_v,
                bias=False,
            )

            self.W_V.append(linear_v)

        # 여러 head를 concat한 뒤 다시 d_model로 projection
        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(
        self,
        query_input,
        key_value_input=None,
        mask=None,
    ):
        if key_value_input is None:
            key_value_input = query_input

        head_outputs = []
        head_weights = []

        for head_index in range(self.num_heads):
            W_Q = self.W_Q[head_index]
            W_K = self.W_K[head_index]
            W_V = self.W_V[head_index]

            Q = W_Q(query_input)
            K = W_K(key_value_input)
            V = W_V(key_value_input)

            scores = Q @ K.transpose(-2, -1)

            scale = math.sqrt(self.d_k)

            scaled_scores = scores / scale

            if mask is not None:
                scaled_scores = scaled_scores.masked_fill(
                    mask,
                    float("-inf"),
                )

            weights = F.softmax(
                scaled_scores,
                dim=-1,
            )

            head_output = weights @ V

            head_outputs.append(head_output)
            head_weights.append(weights)

        concat = torch.cat(
            head_outputs,
            dim=-1,
        )

        output = self.W_O(concat)

        attention_weights = torch.stack(
            head_weights,
            dim=0,
        )

        return output, attention_weights
