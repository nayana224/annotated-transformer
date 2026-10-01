import csv
import itertools


def all_indices(shape):
    ranges = []

    for size in shape:
        one_range = range(size)
        ranges.append(one_range)

    # tensor 차원 수가 매번 다르기 때문에
    # itertools.product를 사용해 모든 index 조합을 만든다.
    indices = itertools.product(
        *ranges
    )

    return indices


def index_to_text(index):
    text_parts = []

    for number in index:
        text_parts.append(
            str(number)
        )

    text = ",".join(
        text_parts
    )

    return text


def snapshot_parameters(model):
    snapshot = {}

    for name, parameter in model.named_parameters():
        data = parameter.detach()
        data = data.cpu()

        indices = all_indices(
            data.shape
        )

        for index in indices:
            value = data[index].item()

            key = (
                name,
                index,
            )

            snapshot[key] = value

    return snapshot


def append_tensor_trace(
    rows,
    step,
    name,
    tensor,
):
    data = tensor.detach()
    data = data.cpu()

    indices = all_indices(
        data.shape
    )

    for index in indices:
        row = {
            "step": step,
            "tensor": name,
            "index": index_to_text(index),
            "value": data[index].item(),
        }

        rows.append(row)


def append_attention_trace(
    rows,
    step,
    attention_type,
    weights,
    query_tokens,
    key_tokens,
):
    data = weights.detach()
    data = data.cpu()

    num_heads = data.shape[0]
    query_len = data.shape[1]
    key_len = data.shape[2]

    for head in range(num_heads):
        for q in range(query_len):
            for k in range(key_len):
                row = {
                    "step": step,
                    "attention_type": attention_type,
                    "head": head,
                    "query_index": q,
                    "query_token": query_tokens[q],
                    "key_index": k,
                    "key_token": key_tokens[k],
                    "weight": data[
                        head,
                        q,
                        k,
                    ].item(),
                }

                rows.append(row)


def write_csv(
    path,
    rows,
    fieldnames,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)
