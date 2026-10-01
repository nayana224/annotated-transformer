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

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
        )

        # Encoder layer들을 하나씩 만든다.
        self.encoder_layers = nn.ModuleList()

        for layer_index in range(num_layers):
            encoder_layer = EncoderLayer(
                d_model,
                num_heads,
                d_ff,
            )

            self.encoder_layers.append(
                encoder_layer
            )

        # Decoder layer들도 하나씩 만든다.
        self.decoder_layers = nn.ModuleList()

        for layer_index in range(num_layers):
            decoder_layer = DecoderLayer(
                d_model,
                num_heads,
                d_ff,
            )

            self.decoder_layers.append(
                decoder_layer
            )

        self.output_linear = nn.Linear(
            d_model,
            tgt_vocab_size,
            bias=False,
        )

    def forward(
        self,
        src_ids,
        decoder_input_ids,
    ):
        device = src_ids.device

        # ==================================================
        # 1. Source Embedding + Positional Encoding
        # ==================================================

        src_embedding = self.src_embedding(
            src_ids
        )

        embedding_scale = math.sqrt(
            self.d_model
        )

        src_embedding = (
            src_embedding
            * embedding_scale
        )

        src_pe = positional_encoding(
            src_ids.shape[0],
            self.d_model,
            device,
        )

        encoder_input = (
            src_embedding
            + src_pe
        )

        # ==================================================
        # 2. Encoder
        # ==================================================

        encoder_output = encoder_input
        encoder_attention = []

        for layer in self.encoder_layers:
            (
                encoder_output,
                weights,
            ) = layer(
                encoder_output
            )

            encoder_attention.append(
                weights
            )

        # ==================================================
        # 3. Target Embedding + Positional Encoding
        # ==================================================

        tgt_embedding = self.tgt_embedding(
            decoder_input_ids
        )

        tgt_embedding = (
            tgt_embedding
            * embedding_scale
        )

        tgt_pe = positional_encoding(
            decoder_input_ids.shape[0],
            self.d_model,
            device,
        )

        decoder_input = (
            tgt_embedding
            + tgt_pe
        )

        # ==================================================
        # 4. Causal Mask
        # ==================================================

        target_len = (
            decoder_input_ids.shape[0]
        )

        mask_matrix = torch.ones(
            target_len,
            target_len,
            dtype=torch.bool,
            device=device,
        )

        causal_mask = torch.triu(
            mask_matrix,
            diagonal=1,
        )

        # ==================================================
        # 5. Decoder
        # ==================================================

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

            masked_attention.append(
                masked_weights
            )

            cross_attention.append(
                cross_weights
            )

        # ==================================================
        # 6. Output Linear
        # ==================================================

        logits = self.output_linear(
            decoder_output
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        result = {
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

        return result
