"""Experience, seniority, education, and title-alignment features."""

from __future__ import annotations

import re


SENIORITY_ORDER = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
    "director": 6,
    "unknown": -1,
}
EDUCATION_ORDER = {
    "any": 0,
    "bachelor": 1,
    "master": 2,
    "phd": 3,
    "unknown": -1,
}
DOMAIN_KEYWORDS = {
    "backend": ("backend", "api", "microservice", "python", "java", "golang"),
    "frontend": ("frontend", "react", "angular", "vue", "ui"),
    "data-science": ("machine learning", "data science", "statistics", "modeling"),
    "data-engineering": ("spark", "airflow", "etl", "data pipeline", "warehouse"),
    "devops": ("docker", "kubernetes", "terraform", "sre", "devops"),
    "product": ("product", "roadmap", "stakeholder", "discovery"),
    "marketing": ("seo", "growth", "campaign", "content"),
}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise_label(value: object) -> str:
    if value is None:
        return "unknown"
    cleaned = str(value).strip().lower()
    return cleaned or "unknown"


def _education_tier(value: object) -> str:
    lowered = _normalise_label(value)
    if lowered in EDUCATION_ORDER:
        return lowered
    if "phd" in lowered or "doctor" in lowered:
        return "phd"
    if any(keyword in lowered for keyword in ("master", "mba", "m.tech", "msc", "ms ")):
        return "master"
    if any(keyword in lowered for keyword in ("bachelor", "b.tech", "btech", "be ", "b.e", "bsc", "ba ")):
        return "bachelor"
    if lowered == "unknown":
        return "unknown"
    return "any"


def _tokenize(text: object) -> set[str]:
    if text is None:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def _role_title_jaccard(job_title: object, candidate_role: object) -> float:
    job_tokens = _tokenize(job_title)
    candidate_tokens = _tokenize(candidate_role)
    if not job_tokens or not candidate_tokens:
        return 0.0
    union = job_tokens | candidate_tokens
    if not union:
        return 0.0
    return round(len(job_tokens & candidate_tokens) / len(union), 4)


def _domain_match_score(job_domain: object, candidate_context: object) -> float:
    job_domain_clean = _normalise_label(job_domain)
    candidate_text = str(candidate_context or "").lower()
    if not job_domain_clean or job_domain_clean == "unknown" or not candidate_text:
        return 0.0
    if job_domain_clean in candidate_text:
        return 1.0
    keywords = DOMAIN_KEYWORDS.get(job_domain_clean, ())
    if keywords and any(keyword in candidate_text for keyword in keywords):
        return 1.0
    return 0.0


class ExperienceFeatureExtractor:
    """Compute structured experience-alignment features."""

    def extract(
        self,
        *,
        job_seniority: str,
        job_min_years: float | None,
        job_max_years: float | None,
        job_education_req: str,
        job_domain: str,
        job_title: str = "",
        cand_seniority: str = "unknown",
        cand_years_exp: float | None = None,
        cand_education: str = "",
        cand_current_role: str = "",
    ) -> dict[str, float]:
        features: dict[str, float] = {}

        job_seniority_clean = _normalise_label(job_seniority)
        cand_seniority_clean = _normalise_label(cand_seniority)
        job_rank = SENIORITY_ORDER.get(job_seniority_clean, -1)
        cand_rank = SENIORITY_ORDER.get(cand_seniority_clean, -1)

        if job_rank >= 0 and cand_rank >= 0:
            features["seniority_exact_match"] = float(job_rank == cand_rank)
            features["seniority_delta"] = float(cand_rank - job_rank)
            features["seniority_within_1"] = float(abs(cand_rank - job_rank) <= 1)
        else:
            features["seniority_exact_match"] = 0.0
            features["seniority_delta"] = 0.0
            features["seniority_within_1"] = 0.0

        min_years = _safe_float(job_min_years)
        max_years = _safe_float(job_max_years)
        candidate_years = _safe_float(cand_years_exp)

        if candidate_years is None:
            candidate_years = 0.0
            features["cand_years_exp_missing"] = 1.0
        else:
            features["cand_years_exp_missing"] = 0.0

        features["cand_years_exp"] = float(candidate_years)
        features["job_years_midpoint"] = float(
            (
                ((min_years or 0.0) + (max_years or min_years or 0.0)) / 2
                if min_years is not None or max_years is not None
                else 0.0
            )
        )

        if min_years is not None and max_years is not None:
            in_range = min_years <= candidate_years <= max_years
            features["exp_in_range"] = float(in_range)
            features["exp_below_min"] = float(candidate_years < min_years)
            features["exp_above_max"] = float(candidate_years > max_years)
            features["exp_deficit"] = max(0.0, min_years - candidate_years)
            features["exp_surplus"] = max(0.0, candidate_years - max_years)
        elif min_years is not None:
            features["exp_in_range"] = float(candidate_years >= min_years)
            features["exp_below_min"] = float(candidate_years < min_years)
            features["exp_above_max"] = 0.0
            features["exp_deficit"] = max(0.0, min_years - candidate_years)
            features["exp_surplus"] = 0.0
        else:
            features["exp_in_range"] = 0.0
            features["exp_below_min"] = 0.0
            features["exp_above_max"] = 0.0
            features["exp_deficit"] = 0.0
            features["exp_surplus"] = 0.0

        job_education = EDUCATION_ORDER.get(_education_tier(job_education_req), -1)
        cand_education_rank = EDUCATION_ORDER.get(_education_tier(cand_education), -1)
        if job_education >= 0 and cand_education_rank >= 0:
            features["edu_meets_requirement"] = float(cand_education_rank >= job_education)
            features["edu_delta"] = float(cand_education_rank - job_education)
        else:
            features["edu_meets_requirement"] = 0.5
            features["edu_delta"] = 0.0

        features["domain_match"] = _domain_match_score(job_domain, cand_current_role)
        features["role_title_token_jaccard"] = _role_title_jaccard(job_title, cand_current_role)
        return features
