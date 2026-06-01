"""Dataset schema inference helpers for jobs and candidate tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from src.data.discovery import find_columns_with_keywords, infer_column_name


def _dedupe(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _non_empty(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class JobTableSchema:
    """Resolved schema for a jobs dataframe."""

    job_id: str
    title: str | None
    description: str | None
    skill_columns: tuple[str, ...]
    location: str | None
    min_experience: str | None
    max_experience: str | None
    text_columns: tuple[str, ...]


@dataclass(frozen=True)
class CandidateTableSchema:
    """Resolved schema for a candidates dataframe."""

    candidate_id: str
    current_role: str | None
    headline: str | None
    summary: str | None
    skill_columns: tuple[str, ...]
    education: str | None
    location: str | None
    total_experience: str | None
    last_active: str | None
    text_columns: tuple[str, ...]


def combine_text_values(row: pd.Series, columns: Sequence[str]) -> str:
    """Concatenate a sequence of columns from a row into a single text field."""
    return " ".join(part for part in (_non_empty(row.get(column)) for column in columns) if part)


def detect_job_schema(df: pd.DataFrame) -> JobTableSchema:
    """Infer the useful columns from a jobs table."""
    job_id = infer_column_name(
        df.columns,
        aliases=("job_id", "jobid", "jd_id", "opening_id"),
        contains=("job", "opening"),
    )
    if job_id is None:
        raise ValueError("Could not infer the job ID column from the jobs dataset.")

    title = infer_column_name(
        df.columns,
        aliases=("title", "job_title", "role", "position", "designation"),
        contains=("title", "role", "position", "designation"),
    )
    description = infer_column_name(
        df.columns,
        aliases=(
            "description",
            "job_description",
            "jd",
            "job_desc",
            "summary",
            "overview",
        ),
        contains=("description", "summary", "overview", "responsibilities"),
    )
    location = infer_column_name(
        df.columns,
        aliases=("location", "city", "state", "country"),
        contains=("location", "city", "state", "country", "remote"),
    )
    min_experience = infer_column_name(
        df.columns,
        aliases=("min_experience", "experience_min", "min_years_experience"),
        contains=("minexperience", "minyears", "experience"),
    )
    max_experience = infer_column_name(
        df.columns,
        aliases=("max_experience", "experience_max", "max_years_experience"),
        contains=("maxexperience", "maxyears"),
    )

    skill_columns = _dedupe(
        find_columns_with_keywords(
            df.columns,
            keywords=("skill", "technology", "requirement", "qualification"),
        )
    )
    text_columns = _dedupe(
        (
            title,
            description,
            location,
            *skill_columns,
            *find_columns_with_keywords(
                df.columns,
                keywords=("responsibility", "overview", "requirement", "qualification"),
            ),
        )
    )

    return JobTableSchema(
        job_id=job_id,
        title=title,
        description=description,
        skill_columns=skill_columns,
        location=location,
        min_experience=min_experience,
        max_experience=max_experience,
        text_columns=text_columns,
    )


def detect_candidate_schema(df: pd.DataFrame) -> CandidateTableSchema:
    """Infer the useful columns from a candidates table."""
    candidate_id = infer_column_name(
        df.columns,
        aliases=("candidate_id", "candidateid", "profile_id", "talent_id"),
        contains=("candidate", "profile", "talent"),
    )
    if candidate_id is None:
        raise ValueError("Could not infer the candidate ID column from the candidates dataset.")

    current_role = infer_column_name(
        df.columns,
        aliases=("current_role", "current_title", "designation", "job_title"),
        contains=("currentrole", "currenttitle", "designation", "jobtitle"),
    )
    headline = infer_column_name(
        df.columns,
        aliases=("headline", "title", "profile_title"),
        contains=("headline", "title"),
    )
    summary = infer_column_name(
        df.columns,
        aliases=("summary", "profile_summary", "bio", "about", "resume_text", "description"),
        contains=("summary", "bio", "about", "resume", "description"),
    )
    education = infer_column_name(
        df.columns,
        aliases=("education", "degree", "qualification"),
        contains=("education", "degree", "qualification"),
    )
    location = infer_column_name(
        df.columns,
        aliases=("location", "city", "state", "country"),
        contains=("location", "city", "state", "country"),
    )
    total_experience = infer_column_name(
        df.columns,
        aliases=("total_experience", "experience_years", "years_experience"),
        contains=("experience", "year"),
    )
    last_active = infer_column_name(
        df.columns,
        aliases=("last_active", "updated_at", "modified_at", "last_seen"),
        contains=("lastactive", "updated", "modified", "lastseen"),
    )

    skill_columns = _dedupe(
        find_columns_with_keywords(
            df.columns,
            keywords=("skill", "technology", "keyword", "competency"),
        )
    )
    text_columns = _dedupe(
        (
            current_role,
            headline,
            summary,
            education,
            location,
            total_experience,
            *skill_columns,
            *find_columns_with_keywords(
                df.columns,
                keywords=("resume", "profile", "project", "about"),
            ),
        )
    )

    return CandidateTableSchema(
        candidate_id=candidate_id,
        current_role=current_role,
        headline=headline,
        summary=summary,
        skill_columns=skill_columns,
        education=education,
        location=location,
        total_experience=total_experience,
        last_active=last_active,
        text_columns=text_columns,
    )
