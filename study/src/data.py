import csv
from pathlib import Path

import torch

from src.tokenization import encode_source, encode_target


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
        "src_ids": torch.tensor(src_ids, dtype=torch.long, device=device),
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_input_ids": torch.tensor(
            decoder_input_ids,
            dtype=torch.long,
            device=device,
        ),
        "gt_tokens": gt_tokens,
        "gt_ids": torch.tensor(gt_ids, dtype=torch.long, device=device),
    }
