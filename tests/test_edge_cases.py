"""Phase 4 edge-case and robustness tests."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.features.behavioral_features import BehavioralFeatureExtractor
from src.features.experience_features import ExperienceFeatureExtractor
from src.features.graph_features import SkillGraph
from src.features.pipeline import add_confidence_features
from src.retrieval.bm25_retriever import tokenize
from src.utils.skill_ontology import SkillMatcher, normalize_skill


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_none_like() -> None:
    assert tokenize("nan") == []


def test_tokenize_pure_numbers() -> None:
    assert tokenize("12345 67890") == []


def test_tokenize_hinglish_friendly_text() -> None:
    result = tokenize("Looking for Python developer with ML skills")
    assert "python" in result
    assert "developer" in result


def test_normalize_empty() -> None:
    assert normalize_skill("") == ""


def test_normalize_case_insensitive() -> None:
    assert normalize_skill("ReactJS") == normalize_skill("reactjs")


def test_normalize_whitespace() -> None:
    assert normalize_skill("  python  ") == "Python"


def test_skill_match_empty_required() -> None:
    matcher = SkillMatcher()
    result = matcher.match_score([], ["Python", "SQL"])
    assert result["exact_coverage"] == 0.0


def test_skill_match_empty_candidate() -> None:
    matcher = SkillMatcher()
    result = matcher.match_score(["Python", "SQL"], [])
    assert result["exact_coverage"] == 0.0
    assert result["n_missing"] == 2


def test_skill_match_both_empty() -> None:
    matcher = SkillMatcher()
    result = matcher.match_score([], [])
    assert result["composite_score"] == 0.0


def test_experience_all_none() -> None:
    extractor = ExperienceFeatureExtractor()
    result = extractor.extract(
        job_seniority="senior",
        job_min_years=None,
        job_max_years=None,
        job_education_req="bachelor",
        job_domain="tech",
        cand_seniority="unknown",
        cand_years_exp=None,
        cand_education="",
        cand_current_role="",
    )
    assert isinstance(result, dict)
    assert all(isinstance(value, float) for value in result.values())


def test_experience_overqualified() -> None:
    extractor = ExperienceFeatureExtractor()
    result = extractor.extract(
        job_seniority="junior",
        job_min_years=1,
        job_max_years=3,
        job_education_req="bachelor",
        job_domain="tech",
        cand_seniority="senior",
        cand_years_exp=10,
        cand_education="bachelor",
        cand_current_role="tech lead",
    )
    assert result["exp_above_max"] == 1.0
    assert result["seniority_delta"] > 0


def test_behavioral_all_missing() -> None:
    extractor = BehavioralFeatureExtractor(reference_date=datetime(2026, 6, 1))
    result = extractor.extract()
    assert isinstance(result, dict)
    assert all(isinstance(value, float) for value in result.values())


def test_behavioral_invalid_date() -> None:
    extractor = BehavioralFeatureExtractor(reference_date=datetime(2026, 6, 1))
    result = extractor.extract(last_active_date="not-a-date")
    assert result["last_active_days_ago"] == 0.0


def test_behavioral_composite_between_0_and_1() -> None:
    extractor = BehavioralFeatureExtractor(reference_date=datetime(2026, 6, 1))
    result = extractor.extract(
        last_active_date="2026-05-01",
        profile_completeness=0.9,
        n_applications=10,
        response_rate=0.8,
    )
    assert 0.0 <= result["behavioral_composite"] <= 1.0


def test_confidence_features_use_profile_completeness() -> None:
    frame = pd.DataFrame(
        {
            "job_id": ["J1", "J1"],
            "candidate_id": ["C1", "C2"],
            "relevance": [1.0, 0.0],
            "profile_completeness": [0.9, 0.2],
        }
    )
    enriched = add_confidence_features(frame)
    assert enriched.loc[0, "profile_confidence"] == 1.0
    assert enriched.loc[1, "profile_confidence"] == 0.2
    assert enriched.loc[1, "low_profile_confidence"] == 1.0


def test_skill_graph_returns_zero_for_empty_inputs() -> None:
    graph = SkillGraph()
    graph.fit([["Python", "SQL"], ["Python", "Pandas"]])
    assert graph.candidate_graph_score([], ["Python"]) == 0.0
