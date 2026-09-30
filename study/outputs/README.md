# Training Trace Outputs

`09_training_trace.py`를 실행하면 이 디렉터리에 학습 과정의 실제 값을 CSV로 저장합니다.

## Files

### `training_trace.csv`

step별 요약값입니다.

- `loss`
- `accuracy`
- `mean_gt_probability`
- position별 GT probability

### `tensor_trace.csv`

forward 과정의 tensor 값을 element 단위로 저장합니다.

예:

- source embedding
- positional encoding
- encoder input/output
- target embedding
- decoder input/output
- logits
- probabilities

형식:

```text
step, tensor, index, value
```

### `attention_trace.csv`

모든 step의 attention weight를 저장합니다.

- Encoder Self-Attention
- Decoder Masked Self-Attention
- Decoder Cross-Attention

형식:

```text
step, attention_type, head, query_index, query_token, key_index, key_token, weight
```

### `parameter_trace.csv`

모델의 모든 learnable parameter scalar에 대해 다음을 저장합니다.

- update 전 값
- gradient
- update 후 값
- delta

형식:

```text
step, parameter, index, value_before, gradient, value_after, delta
```

## Figures

`10_visualize_training.py`를 실행하면 `outputs/figures/` 아래에 다음 그래프를 생성합니다.

- Loss vs Step
- Mean GT Probability vs Step
- Cross-Attention Weight vs Step
- Selected Parameter vs Step
- Selected Gradient vs Step

이 실습에서는 복잡한 소수점 계산을 손으로 전개하기보다, PyTorch가 실제로 계산한 값을 저장하고 비교하는 것을 목표로 합니다.
