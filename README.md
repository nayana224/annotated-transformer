# Attention Is All You Need — Study Fork

This repository is my personal study fork of the original
[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer).

The goal of this branch is not to rewrite the original implementation, but to
understand the Transformer architecture step by step by tracing actual tensors,
shapes, attention weights, residual paths, and loss computation.

## Study Branch

Current study branch:

```text
study/attention-basics
```

The main study materials are under `study/`.

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
    ├── 01_embedding_pe.png
    ├── 02_attention.png
    ├── 03_multihead.png
    ├── 04_encoder.png
    ├── 05_decoder.png
    └── Transformer_architecture.png
```

## What I Study

I use the following checklist when I consider a paper "read":

1. **Problem** — What limitation did the previous approach have?
2. **Core idea** — What key idea did the authors use to solve it?
3. **Method** — How was that idea implemented in the model architecture?
4. **Input / GT / Output / Loss** — How does the training data flow through the model?
5. **Evidence** — Do the experiments actually support the authors' claims?
6. **My observation** — What did I observe directly from implementation, features, or failure cases?

For this paper, the implementation is intentionally small and traceable:

```text
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
```

The toy setup uses small dimensions such as `d_model=4` and `num_heads=2`
so that intermediate values can be printed and inspected directly.

## Study Environment Setup

The study environment is intentionally isolated inside the `study/` directory.

From the repository root, run:

```bash
bash study/setup_venv.sh
```

This creates:

```text
study/.venv/
```

and installs the packages used by the study code:

- CPU PyTorch
- Matplotlib
- Jupyter

Activate the environment later with:

```bash
source study/.venv/bin/activate
```

Then open the integrated notebook with:

```bash
jupyter notebook study/Attention_Is_All_You_Need_Study.ipynb
```

The repository already ignores `.venv/`, so the local virtual environment is not committed.

## Integrated Notebook

The integrated notebook is:

```text
study/Attention_Is_All_You_Need_Study.ipynb
```

It combines the step-by-step exercises into one continuous flow and includes
the architecture figures stored in `study/images/`.

## Notes

- The `.py` files are kept as step-by-step practice records.
- The `.ipynb` file is used as the integrated paper study notebook.
- Random initialization is fixed with `torch.manual_seed(42)` where appropriate
  so the same values can be reproduced and checked manually.
- This branch is for personal study and may intentionally use simplified toy
  tensors instead of production-style implementations.

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
