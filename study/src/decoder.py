import torch.nn as nn

from src.attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, causal_mask):
        masked_output, masked_weights = self.masked_self_attention(
            query_input=x,
            key_value_input=x,
            mask=causal_mask,
        )
        x = self.norm1(x + masked_output)

        cross_output, cross_weights = self.cross_attention(
            query_input=x,
            key_value_input=encoder_output,
            mask=None,
        )
        x = self.norm2(x + cross_output)

        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x, masked_weights, cross_weights
