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

    pairs = load_translation_pairs(csv_path)

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

    return pairs, src_vocab, tgt_vocab, model
