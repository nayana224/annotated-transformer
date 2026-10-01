import torch

from src.data import load_translation_pairs
from src.tokenization import build_vocab
from src.transformer import TinyTransformer


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

    source_sentences = []
    target_sentences = []

    for pair in pairs:
        source_sentences.append(
            pair["source"]
        )

        target_sentences.append(
            pair["target"]
        )

    src_vocab = build_vocab(
        source_sentences
    )

    tgt_vocab = build_vocab(
        target_sentences
    )

    model = TinyTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )

    model = model.to(device)

    return (
        pairs,
        src_vocab,
        tgt_vocab,
        model,
    )
