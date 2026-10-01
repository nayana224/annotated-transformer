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
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_input, key_value_input=None, mask=None):
        if key_value_input is None:
            key_value_input = query_input

        head_outputs = []
        head_weights = []

        for i in range(self.num_heads):
            Q = self.W_Q[i](query_input)
            K = self.W_K[i](key_value_input)
            V = self.W_V[i](key_value_input)

            scores = (
                Q @ K.transpose(-2, -1)
            ) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            output = weights @ V

            head_outputs.append(output)
            head_weights.append(weights)

        concat = torch.cat(head_outputs, dim=-1)
        output = self.W_O(concat)
        attention_weights = torch.stack(head_weights, dim=0)

        return output, attention_weights
