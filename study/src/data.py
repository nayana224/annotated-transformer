import csv
from pathlib import Path

import torch

from src.tokenization import encode_source
from src.tokenization import encode_target


def load_translation_pairs(csv_path):
    csv_path = Path(csv_path)

    pairs = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            source_text = row["source"].strip()
            target_text = row["target"].strip()

            pair = {
                "source": source_text,
                "target": target_text,
            }

            pairs.append(pair)

    return pairs


def make_example(
    pair,
    src_vocab,
    tgt_vocab,
    device="cpu",
):
    source_text = pair["source"]
    target_text = pair["target"]

    src_tokens, src_ids = encode_source(
        source_text,
        src_vocab,
    )

    (
        decoder_input_tokens,
        decoder_input_ids,
        gt_tokens,
        gt_ids,
    ) = encode_target(
        target_text,
        tgt_vocab,
    )

    src_ids_tensor = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device,
    )

    decoder_input_ids_tensor = torch.tensor(
        decoder_input_ids,
        dtype=torch.long,
        device=device,
    )

    gt_ids_tensor = torch.tensor(
        gt_ids,
        dtype=torch.long,
        device=device,
    )

    example = {
        "source_text": source_text,
        "target_text": target_text,
        "src_tokens": src_tokens,
        "src_ids": src_ids_tensor,
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_input_ids": decoder_input_ids_tensor,
        "gt_tokens": gt_tokens,
        "gt_ids": gt_ids_tensor,
    }

    return example
