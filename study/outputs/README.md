# Training Trace Outputs

`study/training/03_training_trace.py`를 실행하면 이 디렉터리에 EN-KO 학습 과정의 실제 값을 CSV로 저장합니다.

학습은 40개의 train pair를 순회하면서 진행하고,
매 optimizer step 뒤에 train에 없는 고정 test probe와 전체 test set을 다시 평가합니다.

고정 probe:

```text
The teacher likes robots
→ 선생님은 로봇을 좋아한다
```

CSV files are written to `study/outputs/csv/`.

## Files

### `training_trace.csv`

step별 학습/평가 요약값입니다.

- `train_pair_index`
- `train_source`
- `train_target`
- `train_loss`
- `probe_loss`
- `probe_accuracy`
- `probe_mean_gt_probability`
- `test_mean_loss`
- `test_token_accuracy`
- position별 probe GT probability

### `tensor_trace.csv`

고정 test probe의 forward tensor 값을 element 단위로 step마다 저장합니다.

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

고정 test probe에 대한 모든 step의 attention weight를 저장합니다.

- Encoder Self-Attention
- Decoder Masked Self-Attention
- Decoder Cross-Attention

형식:

```text
step, attention_type, head, query_index, query_token, key_index, key_token, weight
```

Attention weight는 곧바로 언어학적 의미라고 해석하기보다,
같은 위치의 weight가 학습 과정에서 어떻게 변하는지 관찰하는 데 사용합니다.

### `parameter_trace.csv`

각 training update에서 모델의 모든 learnable parameter scalar에 대해 다음을 저장합니다.

- update 전 값
- gradient
- update 후 값
- delta

형식:

```text
step, parameter, index, value_before, gradient, value_after, delta
```

## Figures

`study/training/04_visualize_training.py`를 실행하면 `study/outputs/figures/` 아래에 다음 그래프를 생성합니다.

- Held-out Probe / Test Loss vs Step
- Probe GT Probability / Test Token Accuracy vs Step
- Probe Cross-Attention Weight vs Step
- Selected Parameter vs Step
- Selected Gradient vs Step

이 실습에서는 복잡한 소수점 계산을 손으로 전개하기보다,
PyTorch가 실제로 계산한 tensor, gradient, parameter 값을 저장하고 비교하는 것을 목표로 합니다.
