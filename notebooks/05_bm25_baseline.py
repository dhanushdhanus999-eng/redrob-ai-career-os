"""Phase 2 Day 5: BM25 retrieval baseline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import (
    append_results_log,
    build_candidate_documents,
    iter_job_queries,
    load_phase2_bundle,
    prepare_validation_context,
    save_metrics_json,
    save_ranked_predictions,
)
from src.eval.metrics import evaluate_rankings, print_metrics
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.paths import MODELS_DIR, SUBMISSIONS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    try:
        bundle = load_phase2_bundle(require_labels=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs/candidates files into data/raw and rerun."
        ) from exc

    candidate_ids, documents = build_candidate_documents(bundle)
    retriever = BM25Retriever()
    retriever.build_index(documents=documents, candidate_ids=candidate_ids)
    model_path = MODELS_DIR / "bm25_index.pkl"
    retriever.save(model_path)

    validation = prepare_validation_context(bundle)
    target_job_ids = set(validation.job_ids) if validation is not None else None
    predictions: dict[str, list[str]] = {}
    for job_id, query in iter_job_queries(bundle, only_job_ids=target_job_ids):
        predictions[job_id] = [candidate_id for candidate_id, _ in retriever.retrieve(query, top_k=100)]

    submission_path = save_ranked_predictions(
        predictions,
        SUBMISSIONS_DIR / "bm25_baseline.csv",
    )
    print(f"Saved BM25 index to: {model_path}")
    print(f"Saved BM25 baseline submission to: {submission_path}")

    if validation is None:
        note = "Predictions generated, but validation scoring is blocked until labels are available."
        append_results_log("BM25 Baseline", note=note)
        print(note)
        return

    metrics = evaluate_rankings(
        predictions=predictions,
        ground_truth=validation.ground_truth,
        k_values=[1, 5, 10, 20],
    )
    metrics_path = save_metrics_json(metrics, SUBMISSIONS_DIR / "bm25_baseline_metrics.json")
    append_results_log("BM25 Baseline", metrics=metrics)
    print_metrics(metrics)
    print(f"Saved BM25 metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
