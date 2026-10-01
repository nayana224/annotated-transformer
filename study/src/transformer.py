import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.decoder import DecoderLayer
from src.encoder import EncoderLayer
from src.positional_encoding import positional_encoding


class TinyTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(
            d_model,
            tgt_vocab_size,
            bias=False,
        )

    def forward(self, src_ids, decoder_input_ids):
        device = src_ids.device

        src_embedding = (
            self.src_embedding(src_ids)
            * math.sqrt(self.d_model)
        )
        src_pe = positional_encoding(
            src_ids.shape[0],
            self.d_model,
            device,
        )
        encoder_input = src_embedding + src_pe

        encoder_output = encoder_input
        encoder_attention = []

        for layer in self.encoder_layers:
            encoder_output, weights = layer(encoder_output)
            encoder_attention.append(weights)

        tgt_embedding = (
            self.tgt_embedding(decoder_input_ids)
            * math.sqrt(self.d_model)
        )
        tgt_pe = positional_encoding(
            decoder_input_ids.shape[0],
            self.d_model,
            device,
        )
        decoder_input = tgt_embedding + tgt_pe

        target_len = decoder_input_ids.shape[0]
        causal_mask = torch.triu(
            torch.ones(
                target_len,
                target_len,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

        decoder_output = decoder_input
        masked_attention = []
        cross_attention = []

        for layer in self.decoder_layers:
            (
                decoder_output,
                masked_weights,
                cross_weights,
            ) = layer(
                decoder_output,
                encoder_output,
                causal_mask,
            )
            masked_attention.append(masked_weights)
            cross_attention.append(cross_weights)

        logits = self.output_linear(decoder_output)
        probabilities = F.softmax(logits, dim=-1)

        return {
            "src_embedding": src_embedding,
            "src_pe": src_pe,
            "encoder_input": encoder_input,
            "encoder_output": encoder_output,
            "tgt_embedding": tgt_embedding,
            "tgt_pe": tgt_pe,
            "decoder_input": decoder_input,
            "decoder_output": decoder_output,
            "logits": logits,
            "probabilities": probabilities,
            "encoder_attention": encoder_attention,
            "masked_attention": masked_attention,
            "cross_attention": cross_attention,
            "causal_mask": causal_mask,
        }
