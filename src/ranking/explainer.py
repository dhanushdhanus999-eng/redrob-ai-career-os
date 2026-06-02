"""Human-readable ranking rationale helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils.skill_ontology import SkillMatcher


matcher = SkillMatcher()


def _first_available(feature_row: pd.Series | None, prefixes: Iterable[str]) -> float:
    if feature_row is None or feature_row.empty:
        return 0.0
    for prefix in prefixes:
        for column in feature_row.index:
            if column.startswith(prefix):
                return float(feature_row[column])
    return 0.0


def explain_ranking(
    *,
    rank: int,
    job_title: str,
    must_skills: list[str],
    nice_skills: list[str],
    candidate_skills: list[str],
    cand_years_exp: float,
    job_min_years: float,
    seniority_match: bool,
    behavioral_score: float,
    semantic_score: float,
) -> str:
    """Generate a concise rationale for an individual ranked candidate."""
    skill_match = matcher.match_score(must_skills, candidate_skills)
    parts: list[str] = []

    matched_skills = skill_match["matched_skills"]
    missing_skills = skill_match["missing_skills"][:3]
    if matched_skills:
        parts.append(
            "Matches "
            f"{len(matched_skills)} of {len(must_skills)} required skills"
            + (f" including {', '.join(matched_skills[:3])}" if matched_skills[:3] else "")
            + "."
        )
    elif must_skills:
        parts.append(f"Does not directly match the listed must-have skills for {job_title or 'this role'}.")

    if nice_skills and candidate_skills:
        nice_overlap = sorted(set(skill_match["matched_skills"]) & set(nice_skills))
        if nice_overlap:
            parts.append(f"Also covers nice-to-have skills such as {', '.join(nice_overlap[:2])}.")

    if cand_years_exp and job_min_years:
        if cand_years_exp >= job_min_years:
            parts.append(
                f"Meets experience expectations ({cand_years_exp:.0f} years vs {job_min_years:.0f}+ required)."
            )
        else:
            parts.append(
                f"Falls short on experience ({cand_years_exp:.0f} years vs {job_min_years:.0f}+ required)."
            )

    if seniority_match:
        parts.append("Seniority aligns well with the role.")

    if behavioral_score >= 0.65:
        parts.append("Profile activity suggests strong recent engagement.")
    elif behavioral_score <= 0.3:
        parts.append("Recent platform activity appears limited.")

    if semantic_score >= 0.75:
        parts.append("Overall profile language is strongly aligned with the job description.")

    if not parts:
        parts.append("General fit based on the current profile and structured feature signals.")

    return f"Rank #{rank}: {' '.join(parts)}"


def add_explanations_to_submission(
    submission_df: pd.DataFrame,
    parsed_jds: dict[str, dict],
    parsed_candidates: dict[str, dict],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append a rationale column to a ranked output dataframe."""
    working = submission_df.copy()
    working["job_id"] = working["job_id"].astype(str)
    working["candidate_id"] = working["candidate_id"].astype(str)

    feature_lookup = feature_df.copy()
    feature_lookup["job_id"] = feature_lookup["job_id"].astype(str)
    feature_lookup["candidate_id"] = feature_lookup["candidate_id"].astype(str)
    feature_lookup = feature_lookup.set_index(["job_id", "candidate_id"])

    rationales: list[str] = []
    for _, row in working.iterrows():
        job_id = row["job_id"]
        candidate_id = row["candidate_id"]
        feature_row = None
        if (job_id, candidate_id) in feature_lookup.index:
            selected = feature_lookup.loc[(job_id, candidate_id)]
            feature_row = selected.iloc[0] if isinstance(selected, pd.DataFrame) else selected

        parsed_job = parsed_jds.get(job_id, {})
        parsed_candidate = parsed_candidates.get(candidate_id, {})

        rationale = explain_ranking(
            rank=int(row["rank"]),
            job_title=str(parsed_job.get("title", "")),
            must_skills=list(parsed_job.get("must_have_skills", [])),
            nice_skills=list(parsed_job.get("nice_to_have_skills", [])),
            candidate_skills=list(parsed_candidate.get("skills", [])),
            cand_years_exp=float(parsed_candidate.get("total_experience_years") or 0.0),
            job_min_years=float(parsed_job.get("min_years_experience") or 0.0),
            seniority_match=(
                str(parsed_job.get("seniority", "")).lower()
                == str(parsed_candidate.get("seniority", "")).lower()
            ),
            behavioral_score=float(
                feature_row.get("behavioral_composite", 0.5) if feature_row is not None else 0.5
            ),
            semantic_score=_first_available(
                feature_row,
                ("bge_large_full_sim", "e5_large_full_sim", "mpnet_full_sim"),
            ),
        )
        rationales.append(rationale)

    working["rationale"] = rationales
    return working
