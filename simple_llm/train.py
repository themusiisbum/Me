"""Command line interface for the NumPy based language model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .model import BigramLanguageModel, CharacterVocabulary


def read_corpus(path: Path) -> Iterable[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file '{path}' does not exist")
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal bigram LLM")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to a training corpus")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to seed generation")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="Number of tokens to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Additive smoothing value")
    args = parser.parse_args()

    vocab = CharacterVocabulary()
    model = BigramLanguageModel(vocab=vocab, smoothing=args.smoothing)
    model.fit(read_corpus(args.corpus))

    continuation = model.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(continuation)


if __name__ == "__main__":
    main()
