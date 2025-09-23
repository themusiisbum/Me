# Simple LLM

This directory contains a **minimal character-level language model** that can be
trained and sampled using only the Python standard library.  The goal is to provide a lightweight
example of the building blocks that power larger language models while keeping
the code accessible and dependency free.

## Quick start

```bash
python -m simple_llm.train --corpus data/tiny_corpus.txt --prompt "Once upon"
```

The command above trains a smoothed bigram model on the provided corpus and
prints a short continuation.

## Files

- `model.py` – implementation of the vocabulary helper and the bigram language
  model.
- `train.py` – small command-line utility for training the model and sampling
  text.
