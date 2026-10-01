# Study Guide

이 디렉터리는 **Attention Is All You Need**를 두 단계로 나눠 공부하도록 구성합니다.

```text
basics/
= Transformer 내부 블록을 작은 toy tensor로 분해해서 확인

training/
= 실제 EN-KO 문자열을 넣어 forward → loss → backward → update를 확인

src/
= training 실습에서 재사용하는 구현

data/
= parallel sentence pairs

outputs/
= 학습 trace CSV와 시각화 결과

notebooks/
= 전체 흐름을 합친 통합 notebook
```

## Directory Structure

```text
study/
├── README.md
├── setup_venv.sh
│
├── basics/
│   ├── 01_embedding_pe.py
│   ├── 02_attention.py
│   ├── 03_multihead.py
│   ├── 04_encoder.py
│   ├── 05_decoder.py
│   └── 06_output_loss.py
│
├── training/
│   ├── 01_tokenization_dataset.py
│   ├── 02_single_training_step.py
│   ├── 03_training_trace.py
│   └── 04_visualize_training.py
│
├── src/
│   ├── tokenization.py
│   ├── data.py
│   ├── positional_encoding.py
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── factory.py
│   └── tracing.py
│
├── data/
│   ├── toy_translation.csv
│   ├── en_ko_train.csv
│   └── en_ko_test.csv
│
├── outputs/
│   ├── README.md
│   ├── csv/
│   └── figures/
│
├── notebooks/
│   └── Attention_Is_All_You_Need_Study.ipynb
│
└── images/
```

## Recommended Reading Order

새로운 end-to-end 학습 코드를 분석할 때는 아래 순서를 권장합니다.

### 1. 실제 데이터부터 보기

```text
data/en_ko_train.csv
data/en_ko_test.csv
```

먼저 모델 코드보다 **무엇이 Input이고 무엇이 GT인지** 확인합니다.

### 2. Tokenization / GT 생성

```text
src/tokenization.py
↓
src/data.py
↓
training/01_tokenization_dataset.py
```

여기서 다음 흐름만 확실히 이해하면 됩니다.

```text
Raw string
→ token
→ vocabulary
→ token ID

Target
→ <SOS> + target = Decoder Input
→ target + <EOS> = GT
```

실행:

```bash
python study/training/01_tokenization_dataset.py
```

### 3. Transformer 전체 forward의 큰 흐름

먼저:

```text
src/transformer.py
```

만 읽습니다.

처음에는 내부 Attention 수식을 다시 파고들기보다:

```text
src_ids
→ source embedding + PE
→ Encoder
→ encoder_output

decoder_input_ids
→ target embedding + PE
→ Decoder
→ logits
```

이 큰 흐름을 봅니다.

그다음 필요할 때 아래 파일로 내려갑니다.

```text
src/attention.py
src/encoder.py
src/decoder.py
src/positional_encoding.py
```

### 4. 역전파 한 번 보기

```text
training/02_single_training_step.py
```

실행:

```bash
python study/training/02_single_training_step.py
```

여기서는 코드 전체보다 이 다섯 줄의 연결이 핵심입니다.

```text
model(...)
→ CrossEntropyLoss
→ loss.backward()
→ gradient
→ optimizer.step()
```

### 5. 여러 step의 변화 추적

```text
src/tracing.py
↓
training/03_training_trace.py
```

실행:

```bash
python study/training/03_training_trace.py
```

생성되는 CSV:

```text
outputs/csv/
├── training_trace.csv
├── tensor_trace.csv
├── attention_trace.csv
└── parameter_trace.csv
```

### 6. 값의 변화 시각화

```text
training/04_visualize_training.py
```

실행:

```bash
python study/training/04_visualize_training.py
```

그래프는 `outputs/figures/`에 저장됩니다.

## Why src/ Is Separated

기존 `trace_utils.py`에는 tokenization, dataset, positional encoding,
Attention, Encoder, Decoder, Transformer, tracing 기능이 한 파일에 섞여 있었습니다.

현재는 **파일 하나 = 역할 하나**에 가깝게 분리했습니다.

```text
tokenization.py       문자열 → token / ID
data.py               CSV → 학습 example
positional_encoding.py 위치 정보
attention.py          Multi-Head Attention
encoder.py            Encoder Layer
decoder.py            Decoder Layer
transformer.py        전체 forward data flow
factory.py            실습용 model / vocab 생성
tracing.py            값 기록 및 CSV 저장
```

따라서 처음부터 `src/` 전체를 읽을 필요는 없습니다.
현재 공부 순서에서 필요한 파일만 따라가면 됩니다.
