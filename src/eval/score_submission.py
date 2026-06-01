"""CLI for scoring a ranked submission file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.discovery import discover_dataset_file
from src.data.io import load_table
from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings, print_metrics
from src.eval.submission import (
    detect_label_columns,
    detect_submission_columns,
    prediction_frame_to_rankings,
    validate_submission,
)
from src.utils.paths import PROCESSED_DATA_DIR


def resolve_labels_path(explicit_path: Path | None) -> Path:
    """Resolve the label file used for scoring."""
    if explicit_path is not None:
        return explicit_path

    val_path = PROCESSED_DATA_DIR / "val.csv"
    if val_path.exists():
        return val_path

    return discover_dataset_file("labels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a ranked candidate submission file.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to the prediction file.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional label file. Defaults to data/processed/val.csv when present.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args()

    prediction_df = load_table(args.pred)
    submission_columns = detect_submission_columns(prediction_df)
    issues = validate_submission(prediction_df, submission_columns)
    if issues:
        raise ValueError("Submission validation failed:\n- " + "\n- ".join(issues))

    predictions = prediction_frame_to_rankings(prediction_df, submission_columns)

    labels_path = resolve_labels_path(args.labels)
    labels_df = load_table(labels_path)
    job_col, cand_col, rel_col = detect_label_columns(labels_df.columns)
    ground_truth = labels_to_ground_truth(
        labels_df=labels_df,
        job_id_col=job_col,
        cand_id_col=cand_col,
        rel_col=rel_col,
    )

    metrics = evaluate_rankings(predictions=predictions, ground_truth=ground_truth, k_values=args.k)
    print_metrics(metrics)

    log_path = args.pred.parent / f"{args.pred.stem}_metrics.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {log_path}")


if __name__ == "__main__":
    main()
