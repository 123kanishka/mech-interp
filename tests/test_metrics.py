import unittest

import numpy as np

from sae_jlens.metrics import (
    compare_distributions,
    frequency_adjust_logits,
    jensen_shannon_similarity,
    ranked_future_metrics,
    rank_biased_overlap,
    reconstruction_quality,
    weighted_jaccard_similarity,
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

    def test_weighted_jaccard_uses_probability_mass(self):
        p = np.array([0.75, 0.25, 0.0])
        q = np.array([0.25, 0.75, 0.0])
        self.assertAlmostEqual(weighted_jaccard_similarity(p, q), 1.0 / 3.0)

    def test_ranked_future_metrics_deduplicate_feature_labels(self):
        result = ranked_future_metrics([8, 8, 3, 9], [3, 4], k=3)
        self.assertEqual(result["future_hit_at_k"], 1.0)
        self.assertAlmostEqual(result["future_precision_at_k"], 1.0 / 3.0)
        self.assertAlmostEqual(result["future_recall_at_k"], 0.5)
        self.assertAlmostEqual(result["future_mrr_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
