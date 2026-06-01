"""Unit tests for Phase 2 parsing helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.schema import detect_candidate_schema, detect_job_schema
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser


class ParsingTests(unittest.TestCase):
    def test_job_parser_extracts_core_fields(self) -> None:
        jobs = pd.DataFrame(
            {
                "job_id": ["J1"],
                "title": ["Senior Data Scientist"],
                "description": [
                    "Remote role based in Bengaluru.\n"
                    "Requires 3 to 5 years of experience.\n"
                    "- Build machine learning models\n"
                    "- Work with stakeholders"
                ],
                "skills": ["Python, Machine Learning, SQL"],
                "location": ["Bengaluru"],
            }
        )
        schema = detect_job_schema(jobs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = JobDescriptionParser(cache_dir=Path(tmp_dir))
            parsed = parser.parse_frame(jobs, schema)

        record = parsed["J1"]
        self.assertEqual(record["seniority"], "senior")
        self.assertEqual(record["location"], "Bengaluru")
        self.assertEqual(record["min_years_experience"], 3.0)
        self.assertEqual(record["max_years_experience"], 5.0)
        self.assertIn("Python", record["must_have_skills"])

    def test_candidate_parser_extracts_skills_and_experience(self) -> None:
        candidates = pd.DataFrame(
            {
                "candidate_id": ["C1"],
                "current_role": ["Senior Backend Engineer"],
                "summary": ["5 years of experience building Python APIs"],
                "skills": ["Python, FastAPI, Docker"],
                "education": ["B.Tech"],
                "location": ["Pune"],
                "updated_at": ["2026-05-30"],
            }
        )
        schema = detect_candidate_schema(candidates)

        with tempfile.TemporaryDirectory() as tmp_dir:
            parser = CandidateProfileParser(cache_dir=Path(tmp_dir))
            parsed = parser.parse_frame(candidates, schema)

        record = parsed["C1"]
        self.assertEqual(record["seniority"], "senior")
        self.assertAlmostEqual(record["total_experience_years"], 5.0, places=6)
        self.assertIn("Python", record["skills"])
        self.assertGreater(record["profile_completeness"], 0.5)


if __name__ == "__main__":
    unittest.main()
