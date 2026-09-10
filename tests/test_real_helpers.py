import unittest

import yaml

from sae_jlens.real import _completed_sequence_ids, _estimate_unigram


class RealHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("configs/sae_jlens.yaml", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)

    def test_only_complete_sequences_resume(self):
        expected = (
            self.config["analysis"]["positions_per_sequence"]
            * len(self.config["sae"]["layers"])
            * 5
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
