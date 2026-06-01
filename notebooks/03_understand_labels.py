"""Phase 1 Day 3: inspect labels and recreate the submission format."""

from __future__ import annotations

import json

import pandas as pd

from src.data.discovery import discover_dataset_file
from src.data.io import load_table
from src.eval.submission import detect_label_columns
from src.utils.paths import SUBMISSIONS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    try:
        labels_path = discover_dataset_file("labels")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nCopy the labels file into data/raw and rerun.") from exc

    labels = load_table(labels_path)
    print(f"Loaded labels dataset from: {labels_path}")
    print(f"Shape: {labels.shape}")
    print(f"Columns: {labels.columns.tolist()}")
    print("\nHead:")
    print(labels.head(10))

    job_col, cand_col, rel_col = detect_label_columns(labels.columns)
    print("\nDetected schema:")
    print(json.dumps({"job_id": job_col, "candidate_id": cand_col, "relevance": rel_col}, indent=2))

    rel_distribution = labels[rel_col].value_counts(dropna=False).sort_index()
    print("\nRelevance value distribution:")
    print(rel_distribution)

    per_job = labels.groupby(job_col).agg(
        n_candidates=(cand_col, "count"),
        n_relevant=(rel_col, lambda values: (pd.Series(values) > 0).sum()),
        avg_relevance=(rel_col, "mean"),
    )
    print("\nPer-job label summary:")
    print(per_job.describe())

    dummy_submission = pd.DataFrame(
        {
            job_col: ["J001", "J001", "J001", "J002", "J002"],
            cand_col: ["C001", "C002", "C003", "C010", "C011"],
            "rank": [1, 2, 3, 1, 2],
        }
    )
    dummy_path = SUBMISSIONS_DIR / "DUMMY_submission_format.csv"
    dummy_submission.to_csv(dummy_path, index=False)
    print(f"\nSaved dummy submission format to: {dummy_path}")
    print(dummy_submission)
    print("\nUse these findings to finalize docs/dataset_notes.md.")


if __name__ == "__main__":
    main()
