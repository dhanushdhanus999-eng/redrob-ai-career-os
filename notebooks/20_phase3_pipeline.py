"""Phase 3 Day 20: run the core ranking pipeline end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import append_results_log, save_metrics_json, save_ranked_predictions
from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings, print_metrics
from src.features.pipeline import (
    build_candidate_text_lookup,
    build_job_text_lookup,
    build_pair_inputs,
    generate_feature_frames,
    load_phase3_context,
    load_split_frame,
    merge_feature_frames,
    save_feature_frames,
)
from src.features.semantic_features import SemanticFeatureExtractor
from src.models.ltr_model import LTRModel
from src.ranking.cross_encoder import CrossEncoderReranker
from src.ranking.explainer import add_explanations_to_submission
from src.ranking.llm_reranker import LLMReranker
from src.utils.paths import MODELS_DIR, SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["bge_large", "mpnet"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name", default="phase3_pipeline")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--run-cross-encoder", action="store_true")
    parser.add_argument("--cross-encoder-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--cross-encoder-top-n", type=int, default=100)
    parser.add_argument("--cross-encoder-top-k", type=int, default=50)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-model", default=None,
                        help="Ollama model tag; defaults to OLLAMA_MODEL or qwen2.5:7b")
    parser.add_argument("--llm-top-k", type=int, default=30)
    parser.add_argument("--refresh-parsed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True, refresh_parsed=args.refresh_parsed)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs, candidates, and labels files into data/raw and rerun."
        ) from exc

    semantic_extractor = SemanticFeatureExtractor(model_keys=args.models, device=args.device)
    merged_splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        labels_df = load_split_frame(split, context)
        pair_inputs = build_pair_inputs(labels_df, context)
        feature_frames = generate_feature_frames(pair_inputs, semantic_extractor=semantic_extractor)
        save_feature_frames(feature_frames, split)
        merged_splits[split] = merge_feature_frames(labels_df, feature_frames)
        print(f"[{split}] feature blocks generated and saved.")

    train_data = merged_splits["train"]
    val_data = merged_splits["val"]

    model = LTRModel()
    model.train(train_data, val_data, num_boost_round=args.num_boost_round)
    model_path = MODELS_DIR / args.output_name
    model.save(model_path)

    predictions = model.rank_frame(val_data, top_k=args.top_k)
    stage_name = "ltr"

    if args.run_cross_encoder:
        cross_encoder = CrossEncoderReranker(
            model_name=args.cross_encoder_model,
            device=args.device,
        )
        job_texts = build_job_text_lookup(context)
        candidate_texts = build_candidate_text_lookup(context)
        reranked_predictions: dict[str, list[str]] = {}
        for job_id, candidate_ids in predictions.items():
            candidates = [
                (candidate_id, candidate_texts.get(candidate_id, ""))
                for candidate_id in candidate_ids[: args.cross_encoder_top_n]
            ]
            reranked = cross_encoder.rerank(
                job_texts.get(job_id, ""),
                candidates,
                top_k=args.cross_encoder_top_k,
            )
            reranked_predictions[job_id] = [candidate_id for candidate_id, _ in reranked]
        predictions = reranked_predictions
        stage_name = "cross_encoder"

    if args.run_llm:
        llm_reranker = LLMReranker(model=args.llm_model)
        job_texts = build_job_text_lookup(context)
        candidate_texts = build_candidate_text_lookup(context)
        reranked_predictions = {}
        for job_id, candidate_ids in predictions.items():
            candidates = [
                (candidate_id, candidate_texts.get(candidate_id, ""))
                for candidate_id in candidate_ids[: args.llm_top_k]
            ]
            reranked = llm_reranker.rerank(
                job_texts.get(job_id, ""),
                candidates,
                top_k=args.llm_top_k,
            )
            reranked_predictions[job_id] = [candidate_id for candidate_id, _ in reranked]
        predictions = reranked_predictions
        stage_name = "llm"

    submission_path = save_ranked_predictions(
        predictions,
        SUBMISSIONS_DIR / f"{args.output_name}_{stage_name}_val.csv",
    )
    metrics = evaluate_rankings(predictions, labels_to_ground_truth(val_data), k_values=[1, 5, 10, 20])
    metrics_path = save_metrics_json(
        metrics,
        SUBMISSIONS_DIR / f"{args.output_name}_{stage_name}_metrics.json",
    )

    submission_df = pd.read_csv(submission_path)
    enriched = add_explanations_to_submission(
        submission_df,
        context.parsed_jobs,
        context.parsed_candidates,
        val_data,
    )
    rationale_path = submission_path.with_name(f"{submission_path.stem}_with_rationale.csv")
    enriched.to_csv(rationale_path, index=False)

    append_results_log(f"Phase 3 Pipeline ({stage_name})", metrics=metrics)
    print_metrics(metrics)
    print(f"Saved model to: {model_path}")
    print(f"Saved submission to: {submission_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved rationale-enriched submission to: {rationale_path}")


if __name__ == "__main__":
    main()
