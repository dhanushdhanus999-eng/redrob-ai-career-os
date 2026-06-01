"""Utilities for deterministic job-level train/validation/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR


def _validate_split_fractions(val_fraction: float, test_fraction: float) -> None:
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be positive.")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")


def create_splits(
    labels_df: pd.DataFrame,
    job_id_col: str = "job_id",
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    random_seed: int = 42,
    save_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split labels into train, validation, and test sets at the job level."""
    _validate_split_fractions(val_fraction=val_fraction, test_fraction=test_fraction)

    if job_id_col not in labels_df.columns:
        raise KeyError(f"Job ID column not found: {job_id_col}")

    save_dir = save_dir or PROCESSED_DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    unique_jobs = labels_df[job_id_col].dropna().astype(str).unique()
    if len(unique_jobs) < 3:
        raise ValueError("At least 3 unique jobs are required to create train/val/test splits.")

    rng = np.random.default_rng(random_seed)
    shuffled_jobs = rng.permutation(unique_jobs)

    n_jobs = len(shuffled_jobs)
    n_test = max(1, int(round(n_jobs * test_fraction)))
    n_val = max(1, int(round(n_jobs * val_fraction)))
    max_reserved = n_jobs - 1

    if n_test + n_val > max_reserved:
        overflow = n_test + n_val - max_reserved
        n_val = max(1, n_val - overflow)

    test_jobs = set(shuffled_jobs[:n_test])
    val_jobs = set(shuffled_jobs[n_test : n_test + n_val])
    train_jobs = set(shuffled_jobs[n_test + n_val :])

    train_df = labels_df[labels_df[job_id_col].astype(str).isin(train_jobs)].copy()
    val_df = labels_df[labels_df[job_id_col].astype(str).isin(val_jobs)].copy()
    test_df = labels_df[labels_df[job_id_col].astype(str).isin(test_jobs)].copy()

    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)

    return train_df, val_df, test_df


def labels_to_ground_truth(
    labels_df: pd.DataFrame,
    job_id_col: str = "job_id",
    cand_id_col: str = "candidate_id",
    rel_col: str = "relevance",
) -> dict[str, dict[str, float]]:
    """Convert a labels dataframe into a nested mapping used by the scorer."""
    required = {job_id_col, cand_id_col, rel_col}
    missing = required.difference(labels_df.columns)
    if missing:
        raise KeyError(f"Missing required label columns: {sorted(missing)}")

    ground_truth: dict[str, dict[str, float]] = {}
    for job_id, group in labels_df.groupby(job_id_col):
        ground_truth[str(job_id)] = {
            str(candidate_id): float(relevance)
            for candidate_id, relevance in zip(group[cand_id_col], group[rel_col], strict=False)
        }
    return ground_truth
