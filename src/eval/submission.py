"""Submission parsing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.data.discovery import infer_column_name


@dataclass(frozen=True)
class SubmissionColumns:
    """Resolved column names for a ranked prediction file."""

    job_id: str | None
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

    if cand_col is None:
        raise ValueError("Could not infer the candidate column from the submission file.")

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

    required_fields = [columns.candidate_id]
    if columns.job_id is not None:
        required_fields.append(columns.job_id)

    for field_name in required_fields:
        if df[field_name].isna().any():
            issues.append(f"Column '{field_name}' contains null values.")

    duplicate_subset = [columns.candidate_id]
    if columns.job_id is not None:
        duplicate_subset.insert(0, columns.job_id)

    duplicate_mask = df.duplicated(subset=duplicate_subset, keep=False)
    if duplicate_mask.any():
        if columns.job_id is None:
            issues.append("Duplicate candidate_id rows detected.")
        else:
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
    grouping_column = columns.job_id or "__job_id__"

    if columns.job_id is None:
        working_df[grouping_column] = "TRACK1_JOB"

    if columns.ordering is None:
        working_df["__row_order__"] = range(len(working_df))
        ordering = "__row_order__"
        ascending = True
    else:
        ordering = columns.ordering
        ascending = columns.ordering_ascending

    working_df = working_df.sort_values(
        by=[grouping_column, ordering],
        ascending=[True, ascending],
        kind="mergesort",
    )

    predictions: dict[str, list[str]] = {}
    for job_id, group in working_df.groupby(grouping_column):
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


@dataclass(frozen=True)
class Track1SubmissionColumns:
    """Column contract for the released single-job submission format."""

    candidate_id: str
    rank: str
    score: str
    reasoning: str | None


def detect_track1_submission_columns(df: pd.DataFrame) -> Track1SubmissionColumns:
    """Infer the candidate/rank/score/reasoning columns for the public challenge format."""
    candidate_id = infer_column_name(
        df.columns,
        aliases=("candidate_id", "candidateid", "profile_id", "talent_id"),
        contains=("candidate", "profile", "talent"),
    )
    rank = infer_column_name(df.columns, aliases=("rank",), contains=("rank",))
    score = infer_column_name(
        df.columns,
        aliases=("score", "prediction_score", "similarity"),
        contains=("score", "similarity"),
    )
    reasoning = infer_column_name(
        df.columns,
        aliases=("reasoning", "rationale", "explanation"),
        contains=("reasoning", "rationale", "explanation"),
    )

    if candidate_id is None or rank is None or score is None:
        raise ValueError(
            "Could not infer the candidate_id, rank, and score columns for the Track 1 submission."
        )

    return Track1SubmissionColumns(
        candidate_id=candidate_id,
        rank=rank,
        score=score,
        reasoning=reasoning,
    )


def validate_track1_submission(
    df: pd.DataFrame,
    *,
    valid_candidate_ids: set[str] | None = None,
) -> list[str]:
    """Validate a single-job submission against the published public spec."""
    columns = detect_track1_submission_columns(df)
    issues: list[str] = []

    if len(df) != 100:
        issues.append("Submission must contain exactly 100 data rows.")

    if df[columns.candidate_id].isna().any():
        issues.append(f"Column '{columns.candidate_id}' contains null values.")

    if df[columns.rank].isna().any():
        issues.append(f"Column '{columns.rank}' contains null values.")

    if df[columns.score].isna().any():
        issues.append(f"Column '{columns.score}' contains null values.")

    duplicate_mask = df.duplicated(subset=[columns.candidate_id], keep=False)
    if duplicate_mask.any():
        issues.append("Duplicate candidate_id rows detected.")

    ranks = pd.to_numeric(df[columns.rank], errors="coerce")
    expected_ranks = list(range(1, len(df) + 1))
    if ranks.isna().any() or sorted(ranks.astype(int).tolist()) != expected_ranks:
        issues.append("Ranks must be the integers 1 through 100, each appearing exactly once.")

    scores = pd.to_numeric(df[columns.score], errors="coerce")
    ordered = pd.DataFrame({"rank": ranks, "score": scores}).sort_values("rank", kind="mergesort")
    if ordered["score"].isna().any():
        issues.append("Score column must be numeric.")
    elif (ordered["score"].diff().fillna(0) > 0).any():
        issues.append("Scores must be non-increasing as rank gets worse.")

    if valid_candidate_ids is not None:
        unknown_ids = sorted(set(df[columns.candidate_id].astype(str)) - valid_candidate_ids)
        if unknown_ids:
            preview = ", ".join(unknown_ids[:5])
            issues.append(
                f"Submission contains candidate IDs that do not exist in the released pool: {preview}"
            )

    return issues
