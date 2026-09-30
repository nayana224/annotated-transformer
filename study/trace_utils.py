import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def load_translation_pairs(csv_path):
    csv_path = Path(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "source": row["source"].strip(),
                "target": row["target"].strip(),
            }
            for row in reader
        ]


def whitespace_tokenize(text):
    return text.strip().split()


def build_vocab(sentences):
    tokens = set()

    for sentence in sentences:
        tokens.update(whitespace_tokenize(sentence))

    vocab = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    for token in sorted(tokens):
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def invert_vocab(vocab):
    return {
        index: token
        for token, index in vocab.items()
    }


def encode_source(text, vocab):
    tokens = whitespace_tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids.append(vocab["<EOS>"])

    return tokens + ["<EOS>"], ids


def encode_target(text, vocab):
    tokens = whitespace_tokenize(text)

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    decoder_input_tokens = ["<SOS>"] + tokens
    gt_tokens = tokens + ["<EOS>"]

    decoder_input_ids = [
        vocab["<SOS>"]
    ] + token_ids

    gt_ids = token_ids + [
        vocab["<EOS>"]
    ]

    return (
        decoder_input_tokens,
        decoder_input_ids,
        gt_tokens,
        gt_ids,
    )


def make_example(pair, src_vocab, tgt_vocab, device="cpu"):
    src_tokens, src_ids = encode_source(
        pair["source"],
        src_vocab,
    )

    (
        decoder_input_tokens,
        decoder_input_ids,
        gt_tokens,
        gt_ids,
    ) = encode_target(
        pair["target"],
        tgt_vocab,
    )

    return {
        "source_text": pair["source"],
        "target_text": pair["target"],
        "src_tokens": src_tokens,
        "src_ids": torch.tensor(
            src_ids,
            dtype=torch.long,
            device=device,
        ),
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_input_ids": torch.tensor(
            decoder_input_ids,
            dtype=torch.long,
            device=device,
        ),
        "gt_tokens": gt_tokens,
        "gt_ids": torch.tensor(
            gt_ids,
            dtype=torch.long,
            device=device,
        ),
    }


def positional_encoding(seq_len, d_model, device):
    position = torch.arange(
        seq_len,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
            device=device,
        )
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(
        seq_len,
        d_model,
        dtype=torch.float32,
        device=device,
    )

    pe[:, 0::2] = torch.sin(
        position * div_term
    )

    pe[:, 1::2] = torch.cos(
        position * div_term
    )

    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_Q = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_k,
                bias=False,
            )
            for _ in range(num_heads)
        ])

        self.W_K = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_k,
                bias=False,
            )
            for _ in range(num_heads)
        ])

        self.W_V = nn.ModuleList([
            nn.Linear(
                d_model,
                self.d_v,
                bias=False,
            )
            for _ in range(num_heads)
        ])

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

        for i in range(self.num_heads):
            Q = self.W_Q[i](query_input)
            K = self.W_K[i](key_value_input)
            V = self.W_V[i](key_value_input)

            scores = (
                Q @ K.transpose(-2, -1)
            ) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(
                    mask,
                    float("-inf"),
                )

            weights = F.softmax(
                scores,
                dim=-1,
            )

            output = weights @ V

            head_outputs.append(output)
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


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d_model,
                d_ff,
            ),
            nn.ReLU(),
            nn.Linear(
                d_ff,
                d_model,
            ),
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

    def forward(self, x):
        attention_output, weights = (
            self.self_attention(x)
        )

        x = self.norm1(
            x + attention_output
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + ffn_output
        )

        return x, weights


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()

        self.masked_self_attention = (
            MultiHeadAttention(
                d_model,
                num_heads,
            )
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.cross_attention = (
            MultiHeadAttention(
                d_model,
                num_heads,
            )
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d_model,
                d_ff,
            ),
            nn.ReLU(),
            nn.Linear(
                d_ff,
                d_model,
            ),
        )

        self.norm3 = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x,
        encoder_output,
        causal_mask,
    ):
        masked_output, masked_weights = (
            self.masked_self_attention(
                query_input=x,
                key_value_input=x,
                mask=causal_mask,
            )
        )

        x = self.norm1(
            x + masked_output
        )

        cross_output, cross_weights = (
            self.cross_attention(
                query_input=x,
                key_value_input=encoder_output,
                mask=None,
            )
        )

        x = self.norm2(
            x + cross_output
        )

        ffn_output = self.ffn(x)

        x = self.norm3(
            x + ffn_output
        )

        return (
            x,
            masked_weights,
            cross_weights,
        )


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

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
            )
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
            )
            for _ in range(num_layers)
        ])

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

        src_embedding = (
            self.src_embedding(src_ids)
            * math.sqrt(self.d_model)
        )

        src_pe = positional_encoding(
            src_ids.shape[0],
            self.d_model,
            device,
        )

        encoder_input = (
            src_embedding + src_pe
        )

        encoder_output = encoder_input
        encoder_attention = []

        for layer in self.encoder_layers:
            (
                encoder_output,
                weights,
            ) = layer(encoder_output)

            encoder_attention.append(weights)

        tgt_embedding = (
            self.tgt_embedding(
                decoder_input_ids
            )
            * math.sqrt(self.d_model)
        )

        tgt_pe = positional_encoding(
            decoder_input_ids.shape[0],
            self.d_model,
            device,
        )

        decoder_input = (
            tgt_embedding + tgt_pe
        )

        target_len = (
            decoder_input_ids.shape[0]
        )

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

            masked_attention.append(
                masked_weights
            )

            cross_attention.append(
                cross_weights
            )

        logits = self.output_linear(
            decoder_output
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

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


def build_study_objects(
    csv_path,
    seed=42,
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_layers=1,
    device="cpu",
):
    torch.manual_seed(seed)

    pairs = load_translation_pairs(
        csv_path
    )

    src_vocab = build_vocab([
        pair["source"]
        for pair in pairs
    ])

    tgt_vocab = build_vocab([
        pair["target"]
        for pair in pairs
    ])

    model = TinyTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    ).to(device)

    return (
        pairs,
        src_vocab,
        tgt_vocab,
        model,
    )


def flatten_tensor_rows(
    step,
    tensor_name,
    tensor,
):
    tensor = tensor.detach().cpu()

    rows = []

    for index in torch.cartesian_prod(*[
        torch.arange(size)
        for size in tensor.shape
    ]):
        if tensor.ndim == 1:
            index_tuple = (
                int(index.item()),
            )
        else:
            index_tuple = tuple(
                int(x)
                for x in index.tolist()
            )

        value = tensor[
            index_tuple
        ].item()

        rows.append({
            "step": step,
            "tensor": tensor_name,
            "index": ",".join(
                str(x)
                for x in index_tuple
            ),
            "value": value,
        })

    return rows
