import random
import unittest

from simple_llm import BigramLanguageModel, CharacterVocabulary


class TestSimpleLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            "hello world",
            "hello there",
            "general kenobi",
        ]
        self.vocab = CharacterVocabulary()
        self.model = BigramLanguageModel(self.vocab, smoothing=0.5)
        self.model.fit(self.corpus)

    def test_vocabulary_contains_special_tokens(self) -> None:
        tokens = {self.vocab.start_token, self.vocab.end_token, self.vocab.unknown_token}
        for token in tokens:
            self.assertIn(token, self.vocab._stoi)

    def test_next_token_distribution_normalized(self) -> None:
        probs = self.model.next_token_distribution(self.vocab.start_id)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_generation_uses_prompt(self) -> None:
        rng = random.Random(0)
        prompt = "hello"
        output = self.model.generate(prompt=prompt, max_new_tokens=5, rng=rng)
        self.assertTrue(output.startswith(prompt))

    def test_perplexity_reasonable(self) -> None:
        perplexity = self.model.perplexity("hello world")
        self.assertGreater(perplexity, 0.0)


if __name__ == "__main__":
    unittest.main()
