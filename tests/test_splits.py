"""Unit tests for job-level split creation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.splits import create_splits, labels_to_ground_truth


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels_df = pd.DataFrame(
            {
                "job_id": ["J1", "J1", "J2", "J2", "J3", "J3", "J4", "J4", "J5", "J5"],
                "candidate_id": ["C1", "C2", "C1", "C3", "C2", "C4", "C5", "C6", "C1", "C7"],
                "relevance": [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            }
        )

    def test_create_splits_keeps_jobs_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_df, val_df, test_df = create_splits(
                labels_df=self.labels_df,
                random_seed=7,
                save_dir=Path(tmp_dir),
            )

        train_jobs = set(train_df["job_id"])
        val_jobs = set(val_df["job_id"])
        test_jobs = set(test_df["job_id"])

        self.assertTrue(train_jobs.isdisjoint(val_jobs))
        self.assertTrue(train_jobs.isdisjoint(test_jobs))
        self.assertTrue(val_jobs.isdisjoint(test_jobs))

    def test_labels_to_ground_truth(self) -> None:
        ground_truth = labels_to_ground_truth(self.labels_df)
        self.assertEqual(ground_truth["J1"]["C1"], 1.0)
        self.assertEqual(ground_truth["J4"]["C6"], 0.0)


if __name__ == "__main__":
    unittest.main()
