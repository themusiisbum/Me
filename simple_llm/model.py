"""A tiny character-level language model implemented with pure Python.

The goal of the module is educational: it exposes a minimal interface for
training and sampling from a statistical language model without depending on
heavy deep-learning frameworks.  The model is a smoothed bigram model trained
from raw text.  While simple, the implementation mirrors the structure of more
complex LLM systems – we build a vocabulary, learn transition probabilities and
provide helpers for text generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Optional

__all__ = ["CharacterVocabulary", "BigramLanguageModel"]


@dataclass
class CharacterVocabulary:
    """Character level vocabulary with optional special tokens."""

    unknown_token: str = "<unk>"
    start_token: str = "<s>"
    end_token: str = "</s>"

    def __post_init__(self) -> None:
        self._itos: List[str] = []
        self._stoi: dict[str, int] = {}

    def build(self, texts: Iterable[str]) -> None:
        characters = {self.unknown_token, self.start_token, self.end_token}
        for text in texts:
            characters.update(text)
        self._itos = sorted(characters)
        self._stoi = {ch: idx for idx, ch in enumerate(self._itos)}

    @property
    def size(self) -> int:
        return len(self._itos)

    def _check_fitted(self) -> None:
        if not self._itos:
            raise RuntimeError("Vocabulary has not been built yet. Call build().")

    def token_to_id(self, token: str) -> int:
        self._check_fitted()
        return self._stoi.get(token, self._stoi[self.unknown_token])

    def id_to_token(self, idx: int) -> str:
        self._check_fitted()
        return self._itos[idx]

    def encode(self, text: str, include_bounds: bool = True) -> List[int]:
        tokens: List[int] = []
        if include_bounds:
            tokens.append(self.token_to_id(self.start_token))
        tokens.extend(self.token_to_id(ch) for ch in text)
        if include_bounds:
            tokens.append(self.token_to_id(self.end_token))
        return tokens

    def decode(self, ids: Iterable[int], strip_bounds: bool = True) -> str:
        characters: List[str] = []
        for idx in ids:
            token = self.id_to_token(idx)
            if strip_bounds and token in {self.start_token, self.end_token}:
                continue
            if token == self.unknown_token:
                characters.append("?")
            else:
                characters.append(token)
        return "".join(characters)

    @property
    def start_id(self) -> int:
        return self.token_to_id(self.start_token)

    @property
    def end_id(self) -> int:
        return self.token_to_id(self.end_token)


@dataclass
class BigramLanguageModel:
    """Simple character-level language model based on bigram counts."""

    vocab: CharacterVocabulary
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self._log_probs: Optional[List[List[float]]] = None

    def fit(self, texts: Iterable[str]) -> None:
        texts = list(texts)
        if not texts:
            raise ValueError("At least one training text is required")

        self.vocab.build(texts)
        vocab_size = self.vocab.size
        counts = [
            [self.smoothing for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

        for text in texts:
            ids = self.vocab.encode(text)
            for current_idx, next_idx in zip(ids[:-1], ids[1:]):
                counts[current_idx][next_idx] += 1.0

        self._log_probs = []
        for row in counts:
            row_sum = sum(row)
            if row_sum == 0:
                self._log_probs.append([float("-inf")] * len(row))
            else:
                self._log_probs.append([
                    math.log(value) - math.log(row_sum) for value in row
                ])

    def _check_trained(self) -> None:
        if self._log_probs is None:
            raise RuntimeError("Model has not been trained yet. Call fit().")

    def next_token_distribution(self, token_id: int) -> List[float]:
        self._check_trained()
        logits = self._log_probs[token_id]
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        total = sum(exp_values)
        if total == 0:
            # If the row_sum was zero (no outgoing transitions), fall back to a uniform
            # distribution to keep the API well-defined.
            return [1.0 / len(exp_values)] * len(exp_values)
        return [value / total for value in exp_values]

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._check_trained()
        if rng is None:
            rng = random.Random()

        generated: List[int] = []
        context_ids = self.vocab.encode(prompt, include_bounds=False)
        current_id = context_ids[-1] if context_ids else self.vocab.start_id

        for _ in range(max_new_tokens):
            logits = self._log_probs[current_id]
            scaled_logits = [value / temperature for value in logits]
            max_logit = max(scaled_logits)
            exp_values = [math.exp(value - max_logit) for value in scaled_logits]
            total = sum(exp_values)
            probs = [value / total for value in exp_values]
            next_id = rng.choices(range(len(probs)), weights=probs, k=1)[0]
            if next_id == self.vocab.end_id:
                break
            generated.append(next_id)
            current_id = next_id

        continuation = self.vocab.decode(generated, strip_bounds=True)
        return prompt + continuation

    def log_probability(self, text: str) -> float:
        self._check_trained()
        ids = self.vocab.encode(text)
        log_prob = 0.0
        for current_idx, next_idx in zip(ids[:-1], ids[1:]):
            log_prob += self._log_probs[current_idx][next_idx]
        return log_prob

    def perplexity(self, text: str) -> float:
        ids = self.vocab.encode(text)
        num_tokens = max(len(ids) - 1, 1)
        return math.exp(-self.log_probability(text) / num_tokens)
