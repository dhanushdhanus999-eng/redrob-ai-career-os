"""Unit tests for submission parsing."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.submission import (
    detect_submission_columns,
    detect_track1_submission_columns,
    prediction_frame_to_rankings,
    validate_submission,
    validate_track1_submission,
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

    def test_detect_submission_columns_without_job_id(self) -> None:
        df = pd.DataFrame(
            {
                "candidate_id": ["C2", "C1"],
                "score": [0.3, 0.9],
            }
        )
        columns = detect_submission_columns(df)
        self.assertIsNone(columns.job_id)
        self.assertEqual(columns.candidate_id, "candidate_id")
        self.assertEqual(columns.ordering, "score")

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

    def test_track1_submission_column_detection(self) -> None:
        df = pd.DataFrame(
            {
                "candidate_id": ["C1"],
                "rank": [1],
                "score": [0.9],
                "reasoning": ["Strong fit"],
            }
        )
        columns = detect_track1_submission_columns(df)
        self.assertEqual(columns.candidate_id, "candidate_id")
        self.assertEqual(columns.rank, "rank")
        self.assertEqual(columns.score, "score")
        self.assertEqual(columns.reasoning, "reasoning")

    def test_track1_submission_validation(self) -> None:
        df = pd.DataFrame(
            {
                "candidate_id": [f"C{i:03d}" for i in range(1, 101)],
                "rank": list(range(1, 101)),
                "score": [1 - i / 1000 for i in range(100)],
                "reasoning": ["fit"] * 100,
            }
        )

        issues = validate_track1_submission(df, valid_candidate_ids=set(df["candidate_id"]))
        self.assertEqual(issues, [])

    def test_track1_submission_rejects_increasing_scores(self) -> None:
        df = pd.DataFrame(
            {
                "candidate_id": [f"C{i:03d}" for i in range(1, 101)],
                "rank": list(range(1, 101)),
                "score": [i / 1000 for i in range(100)],
            }
        )

        issues = validate_track1_submission(df)
        self.assertTrue(any("non-increasing" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
