SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


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

    decoder_input_ids = [vocab["<SOS>"]] + token_ids
    gt_ids = token_ids + [vocab["<EOS>"]]

    return (
        decoder_input_tokens,
        decoder_input_ids,
        gt_tokens,
        gt_ids,
    )
