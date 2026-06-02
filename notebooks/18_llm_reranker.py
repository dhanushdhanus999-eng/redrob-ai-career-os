"""Phase 3 Day 18: apply the cached LLM reranker to a shortlist submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import save_ranked_predictions
from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings, print_metrics
from src.eval.submission import prediction_frame_to_rankings
from src.features.pipeline import (
    build_candidate_text_lookup,
    build_job_text_lookup,
    load_phase3_context,
    load_split_frame,
)
from src.ranking.llm_reranker import LLMReranker
from src.utils.paths import SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default=str(SUBMISSIONS_DIR / "cross_encoder_val.csv"))
    parser.add_argument("--output-name", default="llm_rerank_val")
    parser.add_argument("--model-name", default="claude-haiku-4-5-20251001")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--sample-jobs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True)
        val_df = load_split_frame("val", context)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"{exc}\nValidation labels are required for this script.") from exc

    submission_path = Path(args.submission)
    if not submission_path.exists():
        raise SystemExit(f"Submission file not found: {submission_path}")

    predictions = prediction_frame_to_rankings(pd.read_csv(submission_path))
    if args.sample_jobs > 0:
        selected_job_ids = list(predictions.keys())[: args.sample_jobs]
        predictions = {job_id: predictions[job_id] for job_id in selected_job_ids}

    reranker = LLMReranker(model=args.model_name)
    job_texts = build_job_text_lookup(context)
    candidate_texts = build_candidate_text_lookup(context)

    reranked_predictions: dict[str, list[str]] = {}
    for job_id, candidate_ids in predictions.items():
        candidates = [
            (candidate_id, candidate_texts.get(candidate_id, ""))
            for candidate_id in candidate_ids[: args.top_k]
        ]
        reranked = reranker.rerank(job_texts.get(job_id, ""), candidates, top_k=args.top_k)
        reranked_predictions[job_id] = [candidate_id for candidate_id, _ in reranked]

    output_path = save_ranked_predictions(
        reranked_predictions,
        SUBMISSIONS_DIR / f"{args.output_name}.csv",
    )
    print(f"Saved LLM reranked submission to: {output_path}")
    if not reranker.enabled:
        print("No Anthropic client available; output preserves the incoming ranking order.")

    if reranked_predictions:
        ground_truth = labels_to_ground_truth(val_df)
        selected_gt = {
            job_id: ground_truth.get(job_id, {})
            for job_id in reranked_predictions
        }
        metrics = evaluate_rankings(reranked_predictions, selected_gt, k_values=[5, 10, 20])
        print_metrics(metrics)


if __name__ == "__main__":
    main()
