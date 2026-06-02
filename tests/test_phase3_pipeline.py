"""Unit tests for Phase 3 pair building and feature orchestration."""

from __future__ import annotations

import unittest

from src.baselines.common import Phase2DataBundle
from src.data.schema import detect_candidate_schema, detect_job_schema
from src.features.pipeline import (
    Phase3Context,
    build_pair_inputs,
    generate_feature_frames,
    merge_feature_frames,
)
from src.features.semantic_features import SemanticFeatureExtractor
from tests.test_phase3_features import ToyEncoder

import pandas as pd


class Phase3PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = pd.DataFrame(
            {
                "job_id": ["J1"],
                "title": ["Senior Backend Engineer"],
                "description": ["Build Python APIs and data services."],
                "skills": ["Python, FastAPI, SQL"],
                "location": ["Bengaluru"],
            }
        )
        self.candidates = pd.DataFrame(
            {
                "candidate_id": ["C1", "C2"],
                "current_role": ["Backend Engineer", "Frontend Engineer"],
                "summary": [
                    "5 years building Python APIs and data tools.",
                    "3 years building React interfaces.",
                ],
                "skills": ["Python, SQL, Docker", "React, TypeScript"],
                "education": ["B.Tech", "B.Sc"],
                "location": ["Bengaluru", "Pune"],
                "updated_at": ["2026-05-20", "2026-01-02"],
            }
        )
        self.labels = pd.DataFrame(
            {
                "job_id": ["J1", "J1"],
                "candidate_id": ["C1", "C2"],
                "relevance": [3, 0],
            }
        )
        self.bundle = Phase2DataBundle(
            jobs=self.jobs,
            candidates=self.candidates,
            labels=self.labels,
            job_schema=detect_job_schema(self.jobs),
            candidate_schema=detect_candidate_schema(self.candidates),
            label_columns=("job_id", "candidate_id", "relevance"),
        )
        self.context = Phase3Context(
            bundle=self.bundle,
            parsed_jobs={
                "J1": {
                    "title": "Senior Backend Engineer",
                    "must_have_skills": ["Python", "FastAPI"],
                    "nice_to_have_skills": ["Docker"],
                    "seniority": "senior",
                    "min_years_experience": 4,
                    "max_years_experience": 6,
                    "education_required": "bachelor",
                    "domain": "backend",
                }
            },
            parsed_candidates={
                "C1": {
                    "skills": ["Python", "SQL", "Docker"],
                    "current_role": "Backend Engineer",
                    "seniority": "senior",
                    "total_experience_years": 5,
                    "education": "B.Tech",
                    "location": "Bengaluru",
                    "last_active": "2026-05-20",
                    "profile_completeness": 0.9,
                    "summary": "5 years building Python APIs and data tools.",
                },
                "C2": {
                    "skills": ["React", "TypeScript"],
                    "current_role": "Frontend Engineer",
                    "seniority": "mid",
                    "total_experience_years": 3,
                    "education": "B.Sc",
                    "location": "Pune",
                    "last_active": "2026-01-02",
                    "profile_completeness": 0.7,
                    "summary": "3 years building React interfaces.",
                },
            },
        )

    def test_pair_inputs_include_raw_and_parsed_fields(self) -> None:
        pair_inputs = build_pair_inputs(self.labels, self.context)
        self.assertEqual(set(pair_inputs["candidate_id"]), {"C1", "C2"})
        self.assertIn("job_text", pair_inputs.columns)
        self.assertIn("candidate_text", pair_inputs.columns)
        self.assertIn("last_active_date", pair_inputs.columns)

    def test_feature_generation_and_merge(self) -> None:
        pair_inputs = build_pair_inputs(self.labels, self.context)
        semantic_extractor = SemanticFeatureExtractor(
            model_keys=["toy"],
            models={"toy": ToyEncoder()},
        )
        feature_frames = generate_feature_frames(
            pair_inputs,
            semantic_extractor=semantic_extractor,
        )
        merged = merge_feature_frames(self.labels, feature_frames)

        self.assertIn("toy_full_sim", feature_frames["features_semantic"].columns)
        self.assertIn("must_composite", feature_frames["features_skills"].columns)
        self.assertIn("behavioral_composite", feature_frames["features_behavioral"].columns)
        self.assertIn("toy_full_sim", merged.columns)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
