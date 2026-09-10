import unittest

import numpy as np

from sae_jlens.metrics import (
    compare_distributions,
    frequency_adjust_logits,
    jensen_shannon_similarity,
    rank_biased_overlap,
    reconstruction_quality,
)


class MetricsTest(unittest.TestCase):
    def test_identical_distributions_score_one(self):
        p = np.array([0.1, 0.2, 0.7])
        self.assertAlmostEqual(jensen_shannon_similarity(p, p), 1.0)
        self.assertAlmostEqual(rank_biased_overlap(p, p, 3, 0.9), 1.0)

    def test_comparison_is_finite_and_bounded(self):
        result = compare_distributions(
            np.array([0.7, 0.2, 0.1]),
            np.array([0.1, 0.3, 0.6]),
            target=0,
            top_k=2,
        )
        for name in ("js_similarity", "top_k_overlap", "rbo"):
            self.assertGreaterEqual(result[name], 0.0)
            self.assertLessEqual(result[name], 1.0)

    def test_frequency_adjustment_downweights_common_tokens(self):
        adjusted = frequency_adjust_logits(
            np.zeros(2), np.array([0.9, 0.1]), alpha=1.0
        )
        self.assertLess(adjusted[0], adjusted[1])

    def test_invalid_distribution_rejected(self):
        with self.assertRaises(ValueError):
            jensen_shannon_similarity(np.array([-1.0, 2.0]), np.ones(2))

    def test_reconstruction_quality(self):
        quality = reconstruction_quality(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertAlmostEqual(quality["normalised_mse"], 0.0)
        self.assertAlmostEqual(quality["reconstruction_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
