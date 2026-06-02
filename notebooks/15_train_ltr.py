"""Phase 3 Day 15: train and score the LambdaRank model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import append_results_log, save_metrics_json, save_ranked_predictions
from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings, print_metrics
from src.features.pipeline import load_merged_feature_split, load_phase3_context
from src.models.ltr_model import LTRModel
from src.utils.paths import FIGURES_DIR, MODELS_DIR, SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="ltr_phase3")
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True)
        train_data = load_merged_feature_split("train", context)
        val_data = load_merged_feature_split("val", context)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nRun notebooks/11 through notebooks/14 first, with the organizer dataset in data/raw."
        ) from exc

    model = LTRModel()
    train_params = None
    if args.optuna_trials > 0:
        train_params = model.tune(
            train_data,
            val_data,
            n_trials=args.optuna_trials,
            num_boost_round=max(100, args.num_boost_round // 2),
        )
    model.train(
        train_data,
        val_data,
        params=train_params,
        num_boost_round=args.num_boost_round,
    )

    model_path = MODELS_DIR / args.output_name
    model.save(model_path)

    predictions = model.rank_frame(val_data, top_k=args.top_k)
    submission_path = save_ranked_predictions(
        predictions,
        SUBMISSIONS_DIR / f"{args.output_name}_val.csv",
    )

    ground_truth = labels_to_ground_truth(val_data)
    metrics = evaluate_rankings(predictions, ground_truth, k_values=[1, 5, 10, 20])
    metrics_path = save_metrics_json(metrics, SUBMISSIONS_DIR / f"{args.output_name}_metrics.json")
    append_results_log("LTR Model (LightGBM)", metrics=metrics)

    figure_path = FIGURES_DIR / f"{args.output_name}_feature_importance.png"
    importance = model.feature_importance_df().head(25)
    plt.figure(figsize=(10, 8))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1], color="#4F8EF7")
    plt.title("Top 25 Feature Importances (Gain)")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    print_metrics(metrics)
    print(json.dumps(model.best_params, indent=2, default=str))
    print(f"Saved model to: {model_path}")
    print(f"Saved validation submission to: {submission_path}")
    print(f"Saved validation metrics to: {metrics_path}")
    print(f"Saved feature importance plot to: {figure_path}")


if __name__ == "__main__":
    main()
