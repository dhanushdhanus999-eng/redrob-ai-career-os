"""Unit tests for Phase 3 feature extractors."""

from __future__ import annotations

from datetime import datetime
import unittest

import numpy as np

from src.features.behavioral_features import BehavioralFeatureExtractor
from src.features.experience_features import ExperienceFeatureExtractor
from src.features.semantic_features import SemanticFeatureExtractor
from src.features.skill_features import SkillFeatureExtractor


class ToyEncoder:
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        del show_progress_bar, convert_to_numpy
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    float("python" in lowered),
                    float("data" in lowered),
                    float("react" in lowered),
                ],
                dtype=np.float32,
            )
            if not vector.any():
                vector += 1e-6
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class Phase3FeatureTests(unittest.TestCase):
    def test_semantic_extractor_supports_custom_models(self) -> None:
        extractor = SemanticFeatureExtractor(
            model_keys=["toy"],
            models={"toy": ToyEncoder()},
        )

        features = extractor.extract_for_pair(
            job_text="Python data platform work",
            candidate_text="Python analytics engineer",
            job_title_text="Senior Python Engineer",
            job_skills_text="Python Data",
            candidate_skills_text="Python SQL",
        )

        self.assertGreater(features["toy_full_sim"], 0.5)
        self.assertGreater(features["toy_title_cand_sim"], 0.5)

    def test_skill_extractor_scores_must_and_nice_skills(self) -> None:
        extractor = SkillFeatureExtractor()
        features = extractor.extract(
            must_have_skills=["Python", "FastAPI"],
            nice_to_have_skills=["Docker"],
            candidate_skills=["Python", "Flask", "Docker"],
        )

        self.assertGreater(features["must_composite"], 0.4)
        self.assertEqual(features["has_all_must_skills"], 0.0)
        self.assertGreater(features["nice_exact_coverage"], 0.9)

    def test_experience_extractor_captures_alignment(self) -> None:
        extractor = ExperienceFeatureExtractor()
        features = extractor.extract(
            job_seniority="senior",
            job_min_years=4,
            job_max_years=6,
            job_education_req="bachelor",
            job_domain="backend",
            job_title="Senior Backend Engineer",
            cand_seniority="senior",
            cand_years_exp=5,
            cand_education="B.Tech",
            cand_current_role="Backend Engineer",
        )

        self.assertEqual(features["seniority_exact_match"], 1.0)
        self.assertEqual(features["exp_in_range"], 1.0)
        self.assertEqual(features["edu_meets_requirement"], 1.0)
        self.assertGreater(features["role_title_token_jaccard"], 0.2)

    def test_behavioral_extractor_derives_recency_and_composite(self) -> None:
        extractor = BehavioralFeatureExtractor(reference_date=datetime(2026, 6, 1))
        features = extractor.extract(
            last_active_date="2026-05-20",
            profile_updated_date="2026-04-15",
            n_applications=3,
            n_profile_views=15,
            profile_completeness=0.8,
            response_rate=0.7,
            avg_response_time_days=2,
            similar_job_apps=4,
            candidate_skill_count=12,
            candidate_summary_length=90,
            candidate_has_location=True,
            candidate_has_education=True,
        )

        self.assertEqual(features["last_active_within_30d"], 1.0)
        self.assertGreater(features["response_speed_score"], 0.5)
        self.assertGreater(features["behavioral_composite"], 0.5)


if __name__ == "__main__":
    unittest.main()
