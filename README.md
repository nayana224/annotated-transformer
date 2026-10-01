# Attention Is All You Need 논문 실습

> **Attention Is All You Need**  
> Ashish Vaswani et al., NeurIPS 2017

이 저장소는 `harvardnlp/annotated-transformer`를 fork하여,
Transformer의 핵심 연산과 실제 학습 흐름을 직접 확인하기 위한 개인 학습용 저장소입니다.

원본 구현은 유지하고, 개인 실습 코드는 `study/` 아래에 분리합니다.

- 원본 저장소: https://github.com/harvardnlp/annotated-transformer
- 논문: https://arxiv.org/abs/1706.03762
- 논문 정리: https://github.com/nayana224/dl-paper-research

## Study Structure

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

### Directory roles

`basics/`는 Transformer 내부 구성요소를 작은 toy tensor로 직접 확인한 학습 기록입니다.

`training/`은 실제 English→Korean sentence pair를 사용해
문자열부터 loss, backward, optimizer update까지 연결하는 실습입니다.

`src/`는 training 실습에서 재사용하는 구현을 역할별로 분리한 디렉터리입니다.
이전의 큰 `trace_utils.py`를 tokenization / data / model / tracing 역할로 나눴습니다.

자세한 읽기 순서는:

```text
study/README.md
```

를 참고합니다.

## Recommended Learning Flow

```text
1. basics/
   Transformer 블록별 계산 이해

2. data/en_ko_train.csv
   실제 Input / GT 확인

3. src/tokenization.py
   문자열 → token → token ID

4. training/01_tokenization_dataset.py
   Encoder Input / Decoder Input / GT 확인

5. src/transformer.py
   전체 forward data flow 확인

6. training/02_single_training_step.py
   Forward → Loss → Backward → Optimizer Step

7. training/03_training_trace.py
   여러 step의 tensor / attention / gradient / parameter 저장

8. training/04_visualize_training.py
   저장된 변화 시각화
```

## End-to-End EN-KO Study

이 데이터는 WMT14 원본 데이터가 아닙니다.

WMT의 **parallel sentence pair → tokenization → encoder-decoder translation** 구조를
눈으로 추적하기 쉽도록 만든 소규모 English→Korean 학습 데이터입니다.

```text
study/data/
├── en_ko_train.csv   # 40 sentence pairs
└── en_ko_test.csv    # 10 held-out sentence pairs
```

예:

```text
Source
I like robots

Target
나는 로봇을 좋아한다
```

실행 순서:

```bash
python study/training/01_tokenization_dataset.py
python study/training/02_single_training_step.py
python study/training/03_training_trace.py
python study/training/04_visualize_training.py
```

Training trace CSV는:

```text
study/outputs/csv/
```

시각화 결과는:

```text
study/outputs/figures/
```

에 저장됩니다.

## Study Environment Setup

실습 환경은 `study/.venv/`에 생성합니다.

```bash
bash study/setup_venv.sh
source study/.venv/bin/activate
```

통합 notebook:

```bash
jupyter notebook study/notebooks/Attention_Is_All_You_Need_Study.ipynb
```

## Study Principle

복잡한 소수점 연산을 손으로 끝까지 전개하는 것보다,
PyTorch가 계산한 실제 tensor를 통해 다음 흐름을 읽는 데 집중합니다.

```text
Input
→ representation
→ attention
→ logits
→ GT
→ loss
→ gradient
→ parameter update
```

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
