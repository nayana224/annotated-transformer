# Attention Is All You Need 논문 실습

> **Attention Is All You Need**  
> Ashish Vaswani et al., NeurIPS 2017

이 저장소는 `harvardnlp/annotated-transformer`를 fork하여,
Transformer 논문의 핵심 연산을 직접 코드로 확인하기 위한 개인 학습용 저장소입니다.

원본 구현은 가능한 한 유지하고, 개인 실습 코드는 `study/` 아래에 분리해서 작성합니다.

- 원본 저장소: https://github.com/harvardnlp/annotated-transformer
- 논문: https://arxiv.org/abs/1706.03762
- 논문 정리: https://github.com/nayana224/dl-paper-research

## 학습 목표

Transformer의 전체 데이터 흐름을 작은 tensor와 직접 구현한 코드로 단계별 확인합니다.

```text
Raw Sentence Pair
↓
Tokenization / Vocabulary / Token ID
↓
Embedding + Positional Encoding
↓
Scaled Dot-Product Attention
↓
Multi-Head Attention
↓
Encoder
↓
Decoder
↓
Output Linear / Softmax
↓
Cross Entropy Loss
↓
Backward / Gradient / Optimizer Step
```

각 단계에서 실제 tensor shape, attention weight, residual connection,
loss, gradient, parameter update가 어떻게 이어지는지 직접 확인하는 것이 목표입니다.

## Study

개인 실습 코드는 `study/` 디렉터리에 정리합니다.

```text
study/
├── setup_venv.sh
├── 01_embedding_pe.py
├── 02_attention.py
├── 03_multihead.py
├── 04_encoder.py
├── 05_decoder.py
├── 06_output_loss.py
├── 07_tokenization_dataset.py
├── 08_single_training_step.py
├── 09_training_trace.py
├── 10_visualize_training.py
├── trace_utils.py
├── data/
│   ├── toy_translation.csv
│   ├── en_ko_train.csv
│   └── en_ko_test.csv
├── outputs/
│   └── README.md
├── Attention_Is_All_You_Need_Study.ipynb
└── images/
```

진행 내용:

- [x] 01. Embedding + Positional Encoding
- [x] 02. Scaled Dot-Product Attention
- [x] 03. Multi-Head Attention
- [x] 04. Encoder
- [x] 05. Decoder
- [x] 06. Output / Loss
- [x] 07. 실제 EN-KO 문자열 → Tokenization → Token ID → Decoder Input / GT
- [x] 08. 실제 EN-KO 문장 1개로 Forward → Loss → Backward → Optimizer Step 1회 추적
- [x] 09. 40개 train pair를 순회하며 120 optimizer step 학습 + Tensor / Attention / Gradient / Parameter CSV 저장
- [x] 10. held-out test probe / 전체 test set 변화 시각화

01~06은 `d_model=4`, `num_heads=2`처럼 아주 작은 크기를 사용해
중간값을 직접 출력하고 계산 흐름을 따라갑니다.

07~10의 end-to-end trace에서는 실제 문자열과 더 다양한 vocabulary를 다루기 위해
`d_model=8`, `num_heads=2`의 작은 Transformer를 사용합니다.

## End-to-End Training Trace

기존 01~06 실습은 Transformer의 각 블록을 분리해서 확인하는 단계입니다.

07~10에서는 실제 문자열부터 시작해 학습 과정 전체를 연결합니다.

> 이 데이터는 WMT14 원본 데이터가 아닙니다.  
> WMT의 **parallel sentence pair → tokenization → encoder-decoder translation** 구조를
> 눈으로 추적하기 쉽도록 만든 소규모 English→Korean 학습용 데이터입니다.

### Dataset

```text
study/data/
├── en_ko_train.csv   # 40 sentence pairs
└── en_ko_test.csv    # 10 held-out sentence pairs
```

예시:

```text
Source (English)
I like robots

Target (Korean)
나는 로봇을 좋아한다
```

test set은 train에 동일한 완성 문장으로 넣지 않은 조합으로 구성합니다.
대신 현재 whitespace-tokenizer 실습에서 `<UNK>` 영향보다
**학습된 token 조합이 새로운 문장에 어떻게 적용되는지** 보기 위해,
test 문장을 구성하는 token은 train vocabulary 안에 있도록 구성했습니다.

### 07. Tokenization / Dataset

```bash
python study/07_tokenization_dataset.py
```

다음을 실제 문자열에서 직접 확인합니다.

```text
English Source String
→ whitespace tokenization
→ source vocabulary
→ source token IDs
→ Encoder Input

Korean Target String
→ whitespace tokenization

<SOS> + Target
→ Decoder Input

Target + <EOS>
→ Ground Truth
```

현재는 내부 흐름을 완전히 보이게 하기 위해 직접 만든 whitespace tokenizer를 사용합니다.
추후 BPE / SentencePiece 실습으로 확장할 수 있습니다.

### 08. Single Training Step

```bash
python study/08_single_training_step.py
```

실제 EN-KO train pair 하나에 대해 정확히 1번:

```text
Forward
→ logits / probability
→ Cross Entropy Loss
→ backward()
→ gradient
→ optimizer.step()
→ parameter update
```

를 수행합니다.

대표 `W_Q[0,0]` 값에 대해 다음 값을 직접 출력합니다.

```text
weight before
gradient
learning rate
weight after
delta weight
```

복잡한 소수점 계산을 손으로 전개하는 대신,
PyTorch가 계산한 실제 값을 읽으면서 각 값의 역할과 흐름을 확인합니다.

### 09. Training Trace

```bash
python study/09_training_trace.py
```

40개의 train pair를 순서대로 반복해서 사용하며 기본 120 optimizer step을 수행합니다.

동시에 train에 없는 고정 test 문장 하나를 probe로 정합니다.

```text
The teacher likes robots
→ 선생님은 로봇을 좋아한다
```

각 update 뒤에 이 probe와 전체 10개 test set을 다시 평가합니다.

따라서 단순히 하나의 문장을 반복해서 외우는 과정만 보는 것이 아니라,

```text
training sample update
↓
parameter 변화
↓
held-out probe 변화
↓
전체 test set 변화
```

를 함께 관찰할 수 있습니다.

생성 파일:

```text
study/outputs/
├── training_trace.csv
├── tensor_trace.csv
├── attention_trace.csv
└── parameter_trace.csv
```

저장 내용:

- 현재 train pair와 train loss
- held-out probe loss / accuracy / GT probability
- 전체 test mean loss / token accuracy
- probe의 forward tensor 전체 element
- probe의 Encoder / Decoder attention weight
- 모든 learnable parameter의 update 전 값
- gradient
- update 후 값
- parameter delta

### 10. Visualization

```bash
python study/10_visualize_training.py
```

CSV를 읽어서 다음 변화를 그래프로 저장합니다.

- Held-out Probe / Test Loss vs Step
- Probe GT Probability / Test Token Accuracy vs Step
- Probe Cross-Attention vs Step
- Selected Parameter vs Step
- Selected Gradient vs Step

```text
study/outputs/figures/
```

Attention weight는 사람이 정한 언어학적 의미로 단정하지 않고,
**같은 query/key 위치의 weight가 학습 중 실제로 어떻게 변했는지**를 관찰하는 값으로 사용합니다.

## Study Environment Setup

실습 환경은 `study/.venv/`에 따로 생성합니다.

저장소 루트에서:

```bash
bash study/setup_venv.sh
```

설치되는 주요 패키지:

- CPU PyTorch
- Matplotlib
- Jupyter

이후 다시 활성화하려면:

```bash
source study/.venv/bin/activate
```

통합 notebook은 다음과 같이 실행할 수 있습니다.

```bash
jupyter notebook study/Attention_Is_All_You_Need_Study.ipynb
```

> 이 환경은 논문 구조를 이해하기 위한 개인 실습용입니다.  
> upstream 원본 workflow와는 별도로 관리합니다.

## 실습 원칙

- 원본 `annotated-transformer` 구현은 참고용으로 유지합니다.
- 개인 학습 코드는 가능한 한 `study/` 아래에서만 작성합니다.
- 중간 tensor와 shape을 직접 출력해 연산 흐름을 확인합니다.
- 필요할 때 `torch.manual_seed(42)`를 사용해 결과를 재현합니다.
- 복잡한 소수점 연산 자체보다 tensor의 의미와 data flow를 추적합니다.
- `.py` 파일은 단계별 실습 기록, notebook은 전체 흐름 통합 확인용으로 사용합니다.

---

# Original Repository

Original project:

- Repository: [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)
- Blog post: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)

![image](https://user-images.githubusercontent.com/35882/166251887-9da909a9-660b-45a9-ae72-0aae89fb38d4.png)

## Package Dependencies

Use `requirements.txt` to install library dependencies with pip:

```bash
pip install -r requirements.txt
```

## Notebook Setup

The Annotated Transformer is created using
[jupytext](https://github.com/mwouts/jupytext).

Regular notebooks pose problems for source control because cell outputs end up
in repository history and diffs between commits are difficult to examine.
Using jupytext, a Python script (`.py`) can be kept in sync with the notebook
file by the jupytext plugin.

Prior to using the original repository workflow, install jupytext by following
the [installation instructions](https://github.com/mwouts/jupytext/blob/main/docs/install.md).

To produce the original `.ipynb` notebook:

```bash
make notebook
```

To produce the HTML version:

```bash
make html
```

## Formatting and Linting

The original repository provides Makefile targets for formatting and linting:

```bash
make black
make flake
```

These are useful when working with or contributing back to the upstream
implementation.
