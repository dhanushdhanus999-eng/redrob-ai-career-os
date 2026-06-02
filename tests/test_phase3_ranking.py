"""Unit tests for Phase 3 ranking and reranking helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ltr_model import prepare_lgb_frame
from src.ranking.cross_encoder import CrossEncoderReranker
from src.ranking.explainer import add_explanations_to_submission
from src.ranking.llm_reranker import LLMReranker


class FakeCrossEncoderModel:
    def predict(
        self,
        pairs: list[list[str]],
        *,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        del show_progress_bar, convert_to_numpy
        return np.asarray([float("python" in candidate.lower()) for _, candidate in pairs])


class FakeMessages:
    def create(self, **kwargs) -> object:
        del kwargs
        return type(
            "FakeResponse",
            (),
            {
                "content": [
                    type("FakeText", (), {"text": '{"ranked_ids":["C2","C1"],"reasoning":"C2 matches more."}'})()
                ]
            },
        )()


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


class Phase3RankingTests(unittest.TestCase):
    def test_prepare_lgb_frame_sorts_and_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "job_id": ["J2", "J1", "J1"],
                "candidate_id": ["C3", "C2", "C1"],
                "relevance": [0, 1, 3],
                "feature_a": [0.1, 0.2, 0.9],
            }
        )

        prepared, feature_cols, group_sizes = prepare_lgb_frame(frame)
        self.assertEqual(prepared["job_id"].tolist(), ["J1", "J1", "J2"])
        self.assertEqual(feature_cols, ["feature_a"])
        self.assertEqual(group_sizes, [2, 1])

    def test_cross_encoder_reranks_with_injected_model(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoderModel())
        ranked = reranker.rerank(
            "Python role",
            [("C1", "React engineer"), ("C2", "Python engineer")],
            top_k=2,
        )
        self.assertEqual(ranked[0][0], "C2")

    def test_llm_reranker_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            candidates = [("C1", "Candidate one"), ("C2", "Candidate two")]

            live = LLMReranker(client=FakeAnthropicClient(), cache_dir=cache_dir)
            first = live.rerank("Job text", candidates, top_k=2)
            cached = LLMReranker(cache_dir=cache_dir)
            second = cached.rerank("Job text", candidates, top_k=2)

        self.assertEqual([candidate_id for candidate_id, _ in first], ["C2", "C1"])
        self.assertEqual(first, second)

    def test_add_explanations_to_submission(self) -> None:
        submission = pd.DataFrame(
            {
                "job_id": ["J1"],
                "candidate_id": ["C1"],
                "rank": [1],
            }
        )
        feature_df = pd.DataFrame(
            {
                "job_id": ["J1"],
                "candidate_id": ["C1"],
                "bge_large_full_sim": [0.8],
                "behavioral_composite": [0.7],
            }
        )
        enriched = add_explanations_to_submission(
            submission,
            parsed_jds={
                "J1": {
                    "title": "Senior Backend Engineer",
                    "must_have_skills": ["Python"],
                    "nice_to_have_skills": ["Docker"],
                    "min_years_experience": 4,
                    "seniority": "senior",
                }
            },
            parsed_candidates={
                "C1": {
                    "skills": ["Python", "Docker"],
                    "total_experience_years": 5,
                    "seniority": "senior",
                }
            },
            feature_df=feature_df,
        )

        self.assertIn("rationale", enriched.columns)
        self.assertIn("Rank #1", enriched.loc[0, "rationale"])


if __name__ == "__main__":
    unittest.main()
