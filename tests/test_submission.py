"""Unit tests for submission parsing."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.submission import (
    detect_submission_columns,
    prediction_frame_to_rankings,
    validate_submission,
)


class SubmissionTests(unittest.TestCase):
    def test_detect_submission_columns(self) -> None:
        df = pd.DataFrame(
            {
                "job_id": ["J1", "J1"],
                "candidate_id": ["C2", "C1"],
                "score": [0.3, 0.9],
            }
        )
        columns = detect_submission_columns(df)
        self.assertEqual(columns.job_id, "job_id")
        self.assertEqual(columns.candidate_id, "candidate_id")
        self.assertEqual(columns.ordering, "score")
        self.assertFalse(columns.ordering_ascending)

    def test_prediction_frame_orders_scores_descending(self) -> None:
        df = pd.DataFrame(
            {
                "job_id": ["J1", "J1", "J2"],
                "candidate_id": ["C2", "C1", "C3"],
                "score": [0.2, 0.9, 0.1],
            }
        )

        rankings = prediction_frame_to_rankings(df)
        self.assertEqual(rankings["J1"], ["C1", "C2"])

    def test_validate_submission_finds_duplicates(self) -> None:
        df = pd.DataFrame(
            {
                "job_id": ["J1", "J1"],
                "candidate_id": ["C1", "C1"],
                "rank": [1, 2],
            }
        )

        issues = validate_submission(df)
        self.assertTrue(any("Duplicate" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
