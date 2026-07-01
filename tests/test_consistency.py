"""Tests for the internal-consistency / honeypot detector."""

import sys
import unittest
from pathlib import Path

import pandas as pd

from src.utils.consistency import score_consistency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PARQUET = PROJECT_ROOT / "data" / "processed" / "challenge_candidates.parquet"
BM25_INDEX = PROJECT_ROOT / "outputs" / "models" / "bm25_demo_index.pkl"

# >10% honeypots in the submitted top-100 = automatic Stage-3 disqualification.
DQ_THRESHOLD = 0.10


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


class HoneypotFalsePositiveTests(unittest.TestCase):
    """The filter must not punish genuinely strong senior candidates."""

    def test_honest_senior_many_expert_skills_not_flagged(self):
        # A real 8-year senior can legitimately be "expert" in many skills, each
        # with a plausible duration. This must NOT read as a honeypot, or we lose
        # exactly the candidates we want at the top of the shortlist.
        sd = " || ".join(
            f"Skill{i}; expert; endorsements=15; months=48" for i in range(8)
        )
        result = score_consistency(sd, total_experience_years=8.0)
        self.assertFalse(result.is_honeypot)
        self.assertGreaterEqual(result.consistency_score, 0.5)

    def test_single_noisy_field_does_not_flag(self):
        # One expert-with-0-months field is noise, not an impossible profile.
        sd = (
            "Python; expert; endorsements=20; months=72 || "
            "Rust; expert; endorsements=0; months=0"
        )
        result = score_consistency(sd, total_experience_years=6.0)
        self.assertFalse(result.is_honeypot)


class HoneypotDQGuardTests(unittest.TestCase):
    """End-to-end proof the ranked top-100 stays under the DQ honeypot rate.

    This is the assertion that turns "we have a honeypot filter" into "we cannot
    be disqualified at Stage 3". It ranks the real candidate pool and recomputes
    the consistency verdict on the emitted top-100. It is skipped automatically
    when the challenge parquet / prebuilt index are absent (e.g. a fresh clone or
    CI), so the fast unit tests above always run.
    """

    def test_top100_honeypot_rate_below_dq_threshold(self):
        if not CANDIDATES_PARQUET.exists() or not BM25_INDEX.exists():
            self.skipTest(
                "challenge_candidates.parquet and/or the prebuilt BM25 index are "
                "not present; run scripts/build_indices.py to enable this check"
            )

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import rank as rank_module

        candidates = rank_module.load_candidates(None)
        submission = rank_module.rank(candidates, recall_k=rank_module.DEFAULT_RECALL_K)
        top = submission.head(100)
        self.assertEqual(len(top), 100, "submission must contain a full top-100")

        lookup = candidates.copy()
        lookup["_cid"] = lookup["candidate_id"].astype(str)
        lookup = lookup.set_index("_cid", drop=False)

        honeypots = 0
        for cid in top["candidate_id"].astype(str):
            row = lookup.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            skills_detailed = row.get("skills_detailed")
            years = row.get("total_experience")
            result = score_consistency(
                "" if pd.isna(skills_detailed) else str(skills_detailed),
                float(years) if pd.notna(years) else 0.0,
            )
            if result.is_honeypot:
                honeypots += 1

        rate = honeypots / len(top)
        self.assertLess(
            rate,
            DQ_THRESHOLD,
            f"top-100 honeypot rate {rate:.1%} meets/exceeds the "
            f"{DQ_THRESHOLD:.0%} Stage-3 disqualification threshold",
        )


if __name__ == "__main__":
    unittest.main()
