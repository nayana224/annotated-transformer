from pathlib import Path

from trace_utils import (
    build_vocab,
    encode_source,
    encode_target,
    invert_vocab,
    load_translation_pairs,
    whitespace_tokenize,
)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "toy_translation.csv"


def print_vocab(title, vocab):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    id_to_token = invert_vocab(vocab)

    for index in range(len(id_to_token)):
        print(
            f"{index:>2} -> "
            f"{id_to_token[index]}"
        )


pairs = load_translation_pairs(
    DATA_PATH
)

src_vocab = build_vocab([
    pair["source"]
    for pair in pairs
])

tgt_vocab = build_vocab([
    pair["target"]
    for pair in pairs
])


print("=" * 60)
print("Dataset")
print("=" * 60)

for i, pair in enumerate(pairs):
    print(
        f"[{i}] "
        f"{pair['source']} "
        f"-> "
        f"{pair['target']}"
    )


print_vocab(
    "Source Vocabulary",
    src_vocab,
)

print_vocab(
    "Target Vocabulary",
    tgt_vocab,
)


# ==================================================
# 실제 첫 번째 문장을 끝까지 따라간다.
# ==================================================

sample = pairs[0]

source_text = sample["source"]
target_text = sample["target"]


print("\n" + "=" * 60)
print("Raw Source / Target")
print("=" * 60)

print("Source:")
print(source_text)

print("\nTarget:")
print(target_text)


# ==================================================
# 1. Tokenization
# ==================================================

source_tokens = whitespace_tokenize(
    source_text
)

target_tokens = whitespace_tokenize(
    target_text
)


print("\n" + "=" * 60)
print("1. Tokenization")
print("=" * 60)

print("Source tokens:")
print(source_tokens)

print("\nTarget tokens:")
print(target_tokens)


# ==================================================
# 2. Source Encoding
# ==================================================
#
# Encoder input에는 source token 뒤에 <EOS>를 붙인다.
# ==================================================

(
    source_tokens_with_eos,
    source_ids,
) = encode_source(
    source_text,
    src_vocab,
)


print("\n" + "=" * 60)
print("2. Encoder Input")
print("=" * 60)

print("Tokens:")
print(source_tokens_with_eos)

print("\nToken IDs:")
print(source_ids)


# ==================================================
# 3. Target Encoding
# ==================================================
#
# Target:
# I like robots
#
# Decoder input:
# <SOS> I like robots
#
# GT:
# I like robots <EOS>
# ==================================================

(
    decoder_input_tokens,
    decoder_input_ids,
    gt_tokens,
    gt_ids,
) = encode_target(
    target_text,
    tgt_vocab,
)


print("\n" + "=" * 60)
print("3. Decoder Input / Ground Truth")
print("=" * 60)

print("Decoder input tokens:")
print(decoder_input_tokens)

print("\nDecoder input IDs:")
print(decoder_input_ids)

print("\nGT tokens:")
print(gt_tokens)

print("\nGT IDs:")
print(gt_ids)


# ==================================================
# 4. Position-by-position next-token task
# ==================================================

print("\n" + "=" * 60)
print("4. Next-token Prediction Pairs")
print("=" * 60)

for i in range(len(gt_tokens)):
    visible_prefix = (
        decoder_input_tokens[: i + 1]
    )

    print(
        f"position {i}: "
        f"{visible_prefix} "
        f"-> GT = {gt_tokens[i]}"
    )


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(
    "Raw string"
    " -> tokenization"
    " -> token ID"
    " -> embedding"
    " -> Transformer"
)

print(
    "\n이번 파일에서는 embedding 전까지의 "
    "실제 데이터 준비 과정을 확인한다."
)
