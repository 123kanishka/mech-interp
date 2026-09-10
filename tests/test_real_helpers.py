import unittest

import yaml

from sae_jlens.real import (
    _completed_sequence_ids,
    _estimate_unigram,
    _feature_token_distributions,
    _shuffle_dataset,
)


class RealHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/sae_jlens.yaml", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)

    def test_only_complete_sequences_resume(self):
        expected = (
            self.config["analysis"]["positions_per_sequence"]
            * len(self.config["sae"]["layers"])
            * 6
        )
        records = [{"sequence_id": 0}] * expected + [{"sequence_id": 1}]
        self.assertEqual(_completed_sequence_ids(records, self.config), {0})

    def test_unigram_is_smoothed_and_normalised(self):
        class Tokenizer:
            def __call__(self, text, **_kwargs):
                return {"input_ids": [0, 1, 1] if text == "a" else [2]}

        probabilities = _estimate_unigram(
            [{"text": "a"}, {"text": "b"}], Tokenizer(), self.config, 4
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue((probabilities > 0).all())
        self.assertGreater(probabilities[1], probabilities[3])

    def test_shuffle_arguments_match_dataset_mode(self):
        class Dataset:
            def __init__(self):
                self.kwargs = None

            def shuffle(self, **kwargs):
                self.kwargs = kwargs
                return self

        in_memory = Dataset()
        streaming = Dataset()
        self.assertIs(_shuffle_dataset(in_memory, seed=42, streaming=False), in_memory)
        self.assertEqual(in_memory.kwargs, {"seed": 42})
        self.assertIs(_shuffle_dataset(streaming, seed=43, streaming=True), streaming)
        self.assertEqual(streaming.kwargs, {"seed": 43, "buffer_size": 10_000})

    def test_feature_token_distribution_uses_top_feature_labels(self):
        import torch

        class Model:
            def unembed(self, value):
                return value

        distributions, metadata = _feature_token_distributions(
            torch.tensor([[3.0, 1.0, 0.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 4.0], [-1.0, 0.0]]),
            Model(),
            k=2,
            vocab_size=2,
        )
        self.assertEqual(metadata[0]["active_feature_ids"], [0, 1])
        self.assertEqual(metadata[0]["feature_token_ids"], [0, 1])
        self.assertAlmostEqual(float(distributions[0][0]), 0.75)
        self.assertAlmostEqual(float(distributions[0][1]), 0.25)
