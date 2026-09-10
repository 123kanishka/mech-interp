import unittest

import yaml

from sae_jlens.real import (
    _completed_sequence_ids,
    _estimate_unigram,
    _independent_readouts,
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

    def test_sae_readout_does_not_use_jlens(self):
        import torch

        class Model:
            def unembed(self, value):
                return value

        class Lens:
            def transport(self, value, _layer):
                return value + 100

        residual = torch.tensor([[1.0, 2.0]])
        reconstruction = torch.tensor([[3.0, 4.0]])
        readouts = _independent_readouts(
            Model(),
            Lens(),
            3,
            residual,
            reconstruction,
            reconstruction + 1,
            reconstruction + 2,
            torch.tensor([[9.0, 9.0]]),
        )
        self.assertTrue(torch.equal(readouts["jlens"], residual + 100))
        self.assertTrue(torch.equal(readouts["sae_reconstruction"], reconstruction))
