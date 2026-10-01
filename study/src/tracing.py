import csv
import itertools


def all_indices(shape):
    return itertools.product(*[
        range(size)
        for size in shape
    ])


def index_to_text(index):
    return ",".join(str(i) for i in index)


def snapshot_parameters(model):
    snapshot = {}

    for name, parameter in model.named_parameters():
        data = parameter.detach().cpu()

        for index in all_indices(data.shape):
            snapshot[(name, index)] = data[index].item()

    return snapshot


def append_tensor_trace(rows, step, name, tensor):
    data = tensor.detach().cpu()

    for index in all_indices(data.shape):
        rows.append({
            "step": step,
            "tensor": name,
            "index": index_to_text(index),
            "value": data[index].item(),
        })


def append_attention_trace(
    rows,
    step,
    attention_type,
    weights,
    query_tokens,
    key_tokens,
):
    data = weights.detach().cpu()

    for head in range(data.shape[0]):
        for q in range(data.shape[1]):
            for k in range(data.shape[2]):
                rows.append({
                    "step": step,
                    "attention_type": attention_type,
                    "head": head,
                    "query_index": q,
                    "query_token": query_tokens[q],
                    "key_index": k,
                    "key_token": key_tokens[k],
                    "weight": data[head, q, k].item(),
                })


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
