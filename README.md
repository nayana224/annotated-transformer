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
Token ID
↓
Embedding
↓
Positional Encoding
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
```

각 단계에서 실제 tensor shape, attention weight, residual connection,
그리고 loss 계산이 어떻게 이어지는지 직접 확인하는 것이 목표입니다.

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

실습에서는 `d_model=4`, `num_heads=2`처럼 작은 크기를 사용해
중간값을 직접 출력하고 계산 흐름을 따라갈 수 있도록 구성합니다.

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
