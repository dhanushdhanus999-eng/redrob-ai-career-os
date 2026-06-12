"""Tests for the internal-consistency / honeypot detector."""

import unittest

from src.utils.consistency import score_consistency


class ConsistencyTests(unittest.TestCase):
    def test_clean_profile_is_consistent(self):
        sd = "Python; expert; endorsements=20; months=60 || FAISS; advanced; endorsements=5; months=24"
        result = score_consistency(sd, total_experience_years=7.0)
        self.assertFalse(result.is_honeypot)
        self.assertEqual(result.consistency_score, 1.0)

    def test_expert_with_zero_months_is_honeypot(self):
        # "expert"/"advanced" proficiency with 0 months used is impossible.
        sd = (
            "TypeScript; expert; endorsements=0; months=0 || "
            "Go; expert; endorsements=1; months=0 || "
            "Docker; advanced; endorsements=2; months=0"
        )
        result = score_consistency(sd, total_experience_years=2.0)
        self.assertTrue(result.is_honeypot)
        self.assertEqual(result.consistency_score, 0.0)
        self.assertGreaterEqual(result.expert_zero_count, 2)

    def test_skill_longer_than_career_flagged(self):
        # Several skills used far longer than the whole 1-year career.
        sd = (
            "A; intermediate; endorsements=1; months=60 || "
            "B; intermediate; endorsements=1; months=72 || "
            "C; intermediate; endorsements=1; months=80 || "
            "D; intermediate; endorsements=1; months=90"
        )
        result = score_consistency(sd, total_experience_years=1.0)
        self.assertTrue(result.is_honeypot)

    def test_empty_input_is_safe(self):
        result = score_consistency("", total_experience_years=0.0)
        self.assertFalse(result.is_honeypot)
        self.assertEqual(result.consistency_score, 1.0)


if __name__ == "__main__":
    unittest.main()
