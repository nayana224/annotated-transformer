import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from src.data import load_translation_pairs
from src.tokenization import (
    build_vocab,
    encode_source,
    encode_target,
    invert_vocab,
    whitespace_tokenize,
)


TRAIN_PATH = STUDY_DIR / "data" / "en_ko_train.csv"
TEST_PATH = STUDY_DIR / "data" / "en_ko_test.csv"


def print_vocab(title, vocab):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    id_to_token = invert_vocab(vocab)

    for index in range(len(id_to_token)):
        print(f"{index:>2} -> {id_to_token[index]}")


train_pairs = load_translation_pairs(TRAIN_PATH)
test_pairs = load_translation_pairs(TEST_PATH)

src_vocab = build_vocab([
    pair["source"]
    for pair in train_pairs
])

tgt_vocab = build_vocab([
    pair["target"]
    for pair in train_pairs
])


print("=" * 70)
print("English -> Korean Parallel Dataset")
print("=" * 70)

print(f"Train pairs: {len(train_pairs)}")
print(f"Test pairs : {len(test_pairs)}")

print("\nTrain samples:")
for i, pair in enumerate(train_pairs[:5]):
    print(f"[{i}] {pair['source']} -> {pair['target']}")

print("\nTest samples:")
for i, pair in enumerate(test_pairs[:5]):
    print(f"[{i}] {pair['source']} -> {pair['target']}")


print_vocab("Source Vocabulary (English)", src_vocab)
print_vocab("Target Vocabulary (Korean)", tgt_vocab)


# ==================================================
# Train sample 하나를 실제 문자열부터 끝까지 추적한다.
# ==================================================

sample = train_pairs[0]

source_text = sample["source"]
target_text = sample["target"]


print("\n" + "=" * 70)
print("Raw Source / Target")
print("=" * 70)

print("Source:")
print(source_text)

print("\nTarget:")
print(target_text)


# ==================================================
# 1. Tokenization
# ==================================================

source_tokens = whitespace_tokenize(source_text)
target_tokens = whitespace_tokenize(target_text)


print("\n" + "=" * 70)
print("1. Tokenization")
print("=" * 70)

print("Source tokens:")
print(source_tokens)

print("\nTarget tokens:")
print(target_tokens)


# ==================================================
# 2. Source Encoding
# ==================================================

(
    source_tokens_with_eos,
    source_ids,
) = encode_source(
    source_text,
    src_vocab,
)


print("\n" + "=" * 70)
print("2. Encoder Input")
print("=" * 70)

print("Tokens:")
print(source_tokens_with_eos)

print("\nToken IDs:")
print(source_ids)


# ==================================================
# 3. Target Encoding
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


print("\n" + "=" * 70)
print("3. Decoder Input / Ground Truth")
print("=" * 70)

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

print("\n" + "=" * 70)
print("4. Next-token Prediction Pairs")
print("=" * 70)

for i in range(len(gt_tokens)):
    visible_prefix = decoder_input_tokens[: i + 1]

    print(
        f"position {i}: "
        f"{visible_prefix} "
        f"-> GT = {gt_tokens[i]}"
    )


# ==================================================
# 5. Test pair는 Train vocab으로 encode되는지 확인
# ==================================================

test_sample = test_pairs[0]

test_src_tokens, test_src_ids = encode_source(
    test_sample["source"],
    src_vocab,
)

(
    test_decoder_tokens,
    test_decoder_ids,
    test_gt_tokens,
    test_gt_ids,
) = encode_target(
    test_sample["target"],
    tgt_vocab,
)


print("\n" + "=" * 70)
print("5. Unseen Sentence Combination")
print("=" * 70)

print("Test source:")
print(test_sample["source"])

print("\nTest target:")
print(test_sample["target"])

print("\nSource tokens / IDs:")
print(test_src_tokens)
print(test_src_ids)

print("\nDecoder input:")
print(test_decoder_tokens)
print(test_decoder_ids)

print("\nGT:")
print(test_gt_tokens)
print(test_gt_ids)

print(
    "\n이 test 문장은 train에 동일 문장으로 존재하지 않는다. "
    "하지만 구성 token은 train vocabulary 안에 있도록 만들었다."
)


print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    "Raw English / Korean sentence pair"
    " -> whitespace tokenization"
    " -> vocabulary"
    " -> token ID"
    " -> Encoder / Decoder Input + GT"
)

print(
    "\n이번 단계는 WMT식 parallel translation 구조를 "
    "아주 작은 EN-KO 데이터로 재현한다."
)
