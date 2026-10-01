SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def whitespace_tokenize(text):
    clean_text = text.strip()
    tokens = clean_text.split()

    return tokens


def build_vocab(sentences):
    tokens = set()

    # 모든 문장에서 token을 하나씩 모은다.
    for sentence in sentences:
        sentence_tokens = whitespace_tokenize(sentence)

        for token in sentence_tokens:
            tokens.add(token)

    # special token부터 ID를 넣는다.
    vocab = {}

    index = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = index
        index = index + 1

    # 실제 문장 token을 뒤에 추가한다.
    sorted_tokens = sorted(tokens)

    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def invert_vocab(vocab):
    # 원래:
    # token -> ID
    #
    # 뒤집은 뒤:
    # ID -> token

    inverted = {}

    for token, index in vocab.items():
        inverted[index] = token

    return inverted


def encode_source(text, vocab):
    tokens = whitespace_tokenize(text)

    ids = []

    for token in tokens:
        if token in vocab:
            token_id = vocab[token]
        else:
            token_id = vocab["<UNK>"]

        ids.append(token_id)

    # Encoder source 끝에 <EOS> 추가
    tokens_with_eos = tokens.copy()
    tokens_with_eos.append("<EOS>")

    ids.append(vocab["<EOS>"])

    return tokens_with_eos, ids


def encode_target(text, vocab):
    tokens = whitespace_tokenize(text)

    token_ids = []

    for token in tokens:
        if token in vocab:
            token_id = vocab[token]
        else:
            token_id = vocab["<UNK>"]

        token_ids.append(token_id)

    # Decoder Input
    # <SOS> + target
    decoder_input_tokens = ["<SOS>"]

    for token in tokens:
        decoder_input_tokens.append(token)

    decoder_input_ids = [vocab["<SOS>"]]

    for token_id in token_ids:
        decoder_input_ids.append(token_id)

    # Ground Truth
    # target + <EOS>
    gt_tokens = []

    for token in tokens:
        gt_tokens.append(token)

    gt_tokens.append("<EOS>")

    gt_ids = []

    for token_id in token_ids:
        gt_ids.append(token_id)

    gt_ids.append(vocab["<EOS>"])

    return (
        decoder_input_tokens,
        decoder_input_ids,
        gt_tokens,
        gt_ids,
    )
