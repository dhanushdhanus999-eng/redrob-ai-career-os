"""Phase 1 Day 4: create fixed splits and a random baseline."""

from __future__ import annotations

import json
import random

import pandas as pd

from src.data.discovery import discover_dataset_file
from src.data.io import load_table
from src.data.splits import create_splits, labels_to_ground_truth
from src.eval.metrics import evaluate_rankings, print_metrics
from src.eval.submission import detect_label_columns
from src.utils.paths import PROCESSED_DATA_DIR, SUBMISSIONS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    try:
        labels_path = discover_dataset_file("labels")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nCopy the labels file into data/raw and rerun.") from exc

    labels = load_table(labels_path)
    job_col, cand_col, rel_col = detect_label_columns(labels.columns)

    train_df, val_df, test_df = create_splits(
        labels_df=labels,
        job_id_col=job_col,
        val_fraction=0.2,
        test_fraction=0.1,
        random_seed=42,
        save_dir=PROCESSED_DATA_DIR,
    )

    print("Saved train/val/test splits to data/processed.")
    print(
        {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }
    )

    val_ground_truth = labels_to_ground_truth(
        labels_df=val_df,
        job_id_col=job_col,
        cand_id_col=cand_col,
        rel_col=rel_col,
    )

    all_candidates = labels[cand_col].astype(str).unique().tolist()
    random.seed(42)
    random_predictions: dict[str, list[str]] = {}
    for job_id in val_df[job_col].astype(str).unique():
        shuffled = all_candidates.copy()
        random.shuffle(shuffled)
        random_predictions[str(job_id)] = shuffled[:100]

    baseline_metrics = evaluate_rankings(
        predictions=random_predictions,
        ground_truth=val_ground_truth,
        k_values=[1, 5, 10, 20],
    )
    print("\nRandom baseline:")
    print_metrics(baseline_metrics)

    baseline_rows = []
    for job_id, candidates in random_predictions.items():
        for rank, candidate_id in enumerate(candidates, start=1):
            baseline_rows.append({job_col: job_id, cand_col: candidate_id, "rank": rank})

    baseline_path = SUBMISSIONS_DIR / "random_baseline.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)

    metrics_path = SUBMISSIONS_DIR / "random_baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(baseline_metrics, handle, indent=2)

    print(f"Saved baseline submission to: {baseline_path}")
    print(f"Saved baseline metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
