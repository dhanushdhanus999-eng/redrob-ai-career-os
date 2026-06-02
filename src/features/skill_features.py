"""Skill match features for job-candidate ranking pairs."""

from __future__ import annotations

from src.utils.skill_ontology import SkillMatcher, normalize_skills_list


class SkillFeatureExtractor:
    """Compute exact, fuzzy, and ontology-aware skill overlap features."""

    def __init__(self, matcher: SkillMatcher | None = None) -> None:
        self.matcher = matcher or SkillMatcher()

    def extract(
        self,
        *,
        must_have_skills: list[str] | None,
        nice_to_have_skills: list[str] | None,
        candidate_skills: list[str] | None,
    ) -> dict[str, float]:
        must_have_skills = normalize_skills_list(list(must_have_skills or []))
        nice_to_have_skills = normalize_skills_list(list(nice_to_have_skills or []))
        candidate_skills = normalize_skills_list(list(candidate_skills or []))

        must_match = self.matcher.match_score(must_have_skills, candidate_skills)
        nice_match = self.matcher.match_score(nice_to_have_skills, candidate_skills)

        required_union = normalize_skills_list(must_have_skills + nice_to_have_skills)
        matched_union = {
            *must_match["matched_skills"],
            *nice_match["matched_skills"],
        }

        n_must = len(must_have_skills)
        n_nice = len(nice_to_have_skills)
        n_candidate = len(candidate_skills)

        features = {
            "must_exact_coverage": must_match["exact_coverage"],
            "must_family_coverage": must_match["family_coverage"],
            "must_fuzzy_coverage": must_match["fuzzy_coverage"],
            "must_composite": must_match["composite_score"],
            "n_must_missing": float(must_match["n_missing"]),
            "must_missing_ratio": float(must_match["n_missing"] / n_must) if n_must else 0.0,
            "has_all_must_skills": float(must_match["exact_coverage"] == 1.0) if n_must else 1.0,
            "nice_exact_coverage": nice_match["exact_coverage"],
            "nice_family_coverage": nice_match["family_coverage"],
            "nice_fuzzy_coverage": nice_match["fuzzy_coverage"],
            "nice_composite": nice_match["composite_score"],
            "overall_skill_score": round(
                must_match["composite_score"] * 0.7 + nice_match["composite_score"] * 0.3,
                4,
            ),
            "candidate_skill_overlap_count": float(len(matched_union)),
            "required_skill_coverage": (
                float(len(matched_union) / len(required_union)) if required_union else 0.0
            ),
            "extra_skills_count": float(must_match["extra_skills_count"]),
            "n_must_skills_req": float(n_must),
            "n_nice_skills_req": float(n_nice),
            "n_candidate_skills": float(n_candidate),
        }
        return features
