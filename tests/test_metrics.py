"""Unit tests for ranking metrics."""

from __future__ import annotations

import unittest

from src.eval.metrics import evaluate_rankings, mrr, ndcg_at_k, precision_at_k, recall_at_k


class MetricsTests(unittest.TestCase):
    def test_ndcg_is_one_for_perfect_ranking(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1, 0], k=4), 1.0, places=6)

    def test_ndcg_is_zero_when_everything_is_irrelevant(self) -> None:
        self.assertEqual(ndcg_at_k([0, 0, 0], k=3), 0.0)

    def test_mrr_uses_first_relevant_item(self) -> None:
        self.assertAlmostEqual(mrr([[0, 1, 0], [1, 0, 0]]), 0.75, places=6)

    def test_precision_at_k(self) -> None:
        self.assertAlmostEqual(precision_at_k([1, 0, 1, 0], k=2), 0.5, places=6)

    def test_recall_at_k(self) -> None:
        self.assertAlmostEqual(recall_at_k([1, 0, 1, 0], k=3, n_relevant=3), 2 / 3, places=6)

    def test_missing_prediction_job_counts_as_zero(self) -> None:
        predictions = {"J1": ["C1", "C2", "C3"]}
        ground_truth = {
            "J1": {"C1": 3.0, "C2": 1.0},
            "J2": {"C9": 1.0},
        }

        metrics = evaluate_rankings(predictions=predictions, ground_truth=ground_truth, k_values=[1, 3])

        self.assertEqual(int(metrics["n_queries"]), 2)
        self.assertLess(metrics["ndcg@1"], 1.0)
        self.assertLess(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
