"""Submission parsing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.data.discovery import infer_column_name


@dataclass(frozen=True)
class SubmissionColumns:
    """Resolved column names for a ranked prediction file."""

    job_id: str
    candidate_id: str
    ordering: str | None
    ordering_ascending: bool


def detect_submission_columns(df: pd.DataFrame) -> SubmissionColumns:
    """Detect job, candidate, and ordering columns from a prediction dataframe."""
    job_col = infer_column_name(
        df.columns,
        aliases=("job_id", "jobid", "jd_id", "opening_id"),
        contains=("job", "opening"),
    )
    cand_col = infer_column_name(
        df.columns,
        aliases=("candidate_id", "candidateid", "profile_id", "talent_id"),
        contains=("candidate", "profile", "talent"),
    )
    ordering_col = infer_column_name(
        df.columns,
        aliases=("rank", "score", "prediction_score", "similarity"),
        contains=("rank", "score", "similarity"),
    )

    if job_col is None or cand_col is None:
        raise ValueError(
            "Could not infer required job/candidate columns from the submission file."
        )

    ordering_ascending = bool(ordering_col and "rank" in ordering_col.lower())
    return SubmissionColumns(
        job_id=job_col,
        candidate_id=cand_col,
        ordering=ordering_col,
        ordering_ascending=ordering_ascending,
    )


def validate_submission(
    df: pd.DataFrame,
    columns: SubmissionColumns | None = None,
) -> list[str]:
    """Return human-readable validation issues for a submission dataframe."""
    columns = columns or detect_submission_columns(df)
    issues: list[str] = []

    if df.empty:
        issues.append("Submission file is empty.")

    for field_name in (columns.job_id, columns.candidate_id):
        if df[field_name].isna().any():
            issues.append(f"Column '{field_name}' contains null values.")

    duplicate_mask = df.duplicated(subset=[columns.job_id, columns.candidate_id], keep=False)
    if duplicate_mask.any():
        issues.append("Duplicate (job_id, candidate_id) rows detected.")

    if columns.ordering and df[columns.ordering].isna().any():
        issues.append(f"Ordering column '{columns.ordering}' contains null values.")

    return issues


def prediction_frame_to_rankings(
    df: pd.DataFrame,
    columns: SubmissionColumns | None = None,
) -> dict[str, list[str]]:
    """Convert a submission dataframe into ordered candidate lists per job."""
    columns = columns or detect_submission_columns(df)
    working_df = df.copy()

    if columns.ordering is None:
        working_df["__row_order__"] = range(len(working_df))
        ordering = "__row_order__"
        ascending = True
    else:
        ordering = columns.ordering
        ascending = columns.ordering_ascending

    working_df = working_df.sort_values(
        by=[columns.job_id, ordering],
        ascending=[True, ascending],
        kind="mergesort",
    )

    predictions: dict[str, list[str]] = {}
    for job_id, group in working_df.groupby(columns.job_id):
        predictions[str(job_id)] = group[columns.candidate_id].astype(str).tolist()
    return predictions


def detect_label_columns(columns: Iterable[str]) -> tuple[str, str, str]:
    """Infer job, candidate, and relevance columns from a label dataset."""
    job_col = infer_column_name(
        columns,
        aliases=("job_id", "jobid", "jd_id", "opening_id"),
        contains=("job", "opening"),
    )
    cand_col = infer_column_name(
        columns,
        aliases=("candidate_id", "candidateid", "profile_id", "talent_id"),
        contains=("candidate", "profile", "talent"),
    )
    rel_col = infer_column_name(
        columns,
        aliases=("relevance", "label", "target", "score", "rel"),
        contains=("relevance", "label", "target", "score", "rel"),
    )

    if job_col is None or cand_col is None or rel_col is None:
        raise ValueError("Could not infer label columns from the labels dataset.")

    return job_col, cand_col, rel_col
