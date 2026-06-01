"""Shared utilities for the Phase 2 baseline runner scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.discovery import discover_dataset_file
from src.data.io import load_table
from src.data.schema import (
    CandidateTableSchema,
    JobTableSchema,
    combine_text_values,
    detect_candidate_schema,
    detect_job_schema,
)
from src.data.splits import create_splits, labels_to_ground_truth
from src.eval.submission import detect_label_columns
from src.utils.paths import DOCS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


@dataclass(frozen=True)
class Phase2DataBundle:
    """Loaded datasets and inferred schemas needed by the baseline scripts."""

    jobs: pd.DataFrame
    candidates: pd.DataFrame
    labels: pd.DataFrame | None
    job_schema: JobTableSchema
    candidate_schema: CandidateTableSchema
    label_columns: tuple[str, str, str] | None


@dataclass(frozen=True)
class ValidationContext:
    """Validation split and ground truth used for scoring baseline runs."""

    val_df: pd.DataFrame
    ground_truth: dict[str, dict[str, float]]
    job_id_col: str
    candidate_id_col: str
    relevance_col: str

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(self.ground_truth.keys())


def load_phase2_bundle(*, require_labels: bool = False) -> Phase2DataBundle:
    """Load the dataset files needed for Phase 2 and infer their schemas."""
    ensure_project_dirs()

    jobs_path = discover_dataset_file("jobs")
    candidates_path = discover_dataset_file("candidates")

    jobs = load_table(jobs_path)
    candidates = load_table(candidates_path)
    job_schema = detect_job_schema(jobs)
    candidate_schema = detect_candidate_schema(candidates)

    labels: pd.DataFrame | None = None
    label_columns: tuple[str, str, str] | None = None
    try:
        labels_path = discover_dataset_file("labels")
        labels = load_table(labels_path)
        label_columns = detect_label_columns(labels.columns)
    except FileNotFoundError:
        if require_labels:
            raise

    return Phase2DataBundle(
        jobs=jobs,
        candidates=candidates,
        labels=labels,
        job_schema=job_schema,
        candidate_schema=candidate_schema,
        label_columns=label_columns,
    )


def prepare_validation_context(
    bundle: Phase2DataBundle,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    random_seed: int = 42,
) -> ValidationContext | None:
    """Return a validation context if labels exist, otherwise None."""
    if bundle.labels is None or bundle.label_columns is None:
        return None

    job_id_col, candidate_id_col, relevance_col = bundle.label_columns
    val_path = PROCESSED_DATA_DIR / "val.csv"

    if val_path.exists():
        val_df = load_table(val_path)
    else:
        _, val_df, _ = create_splits(
            labels_df=bundle.labels,
            job_id_col=job_id_col,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
            save_dir=PROCESSED_DATA_DIR,
        )

    ground_truth = labels_to_ground_truth(
        labels_df=val_df,
        job_id_col=job_id_col,
        cand_id_col=candidate_id_col,
        rel_col=relevance_col,
    )
    return ValidationContext(
        val_df=val_df,
        ground_truth=ground_truth,
        job_id_col=job_id_col,
        candidate_id_col=candidate_id_col,
        relevance_col=relevance_col,
    )


def build_candidate_documents(bundle: Phase2DataBundle) -> tuple[list[str], list[str]]:
    """Return candidate IDs and concatenated text for retrieval indexing."""
    candidate_ids = bundle.candidates[bundle.candidate_schema.candidate_id].astype(str).tolist()
    documents = [
        combine_text_values(row, bundle.candidate_schema.text_columns)
        for _, row in bundle.candidates.iterrows()
    ]
    return candidate_ids, documents


def iter_job_queries(
    bundle: Phase2DataBundle,
    *,
    only_job_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return job IDs paired with retrieval query text."""
    queries: list[tuple[str, str]] = []
    for _, row in bundle.jobs.iterrows():
        job_id = str(row[bundle.job_schema.job_id])
        if only_job_ids is not None and job_id not in only_job_ids:
            continue
        queries.append((job_id, combine_text_values(row, bundle.job_schema.text_columns)))
    return queries


def save_ranked_predictions(
    predictions: dict[str, list[str]],
    path: str | Path,
    *,
    job_column: str = "job_id",
    candidate_column: str = "candidate_id",
) -> Path:
    """Write a ranked candidate file in the repo's canonical baseline format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for job_id, candidate_ids in predictions.items():
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            rows.append(
                {
                    job_column: job_id,
                    candidate_column: candidate_id,
                    "rank": rank,
                }
            )
    pd.DataFrame(rows, columns=[job_column, candidate_column, "rank"]).to_csv(path, index=False)
    return path


def save_metrics_json(metrics: dict[str, float], path: str | Path) -> Path:
    """Persist evaluation metrics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return path


def append_results_log(
    title: str,
    *,
    metrics: dict[str, float] | None = None,
    note: str | None = None,
) -> None:
    """Append a short baseline section to docs/results_log.md."""
    results_path = DOCS_DIR / "results_log.md"
    lines = [f"\n## {title}\n"]
    if note:
        lines.append(f"{note}\n")
    if metrics is not None:
        lines.append("```json\n")
        lines.append(json.dumps(metrics, indent=2, sort_keys=True))
        lines.append("\n```\n")

    with results_path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
