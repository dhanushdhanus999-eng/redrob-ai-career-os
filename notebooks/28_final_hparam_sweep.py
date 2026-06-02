"""Phase 4 Day 28: run a final Optuna sweep and save the tuned LTR model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings
from src.features.pipeline import PHASE3_FEATURE_BLOCKS, load_merged_feature_split, load_phase3_context
from src.models.ltr_model import LTRModel
from src.utils.paths import MODELS_DIR, SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--output-name", default="ltr_final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True)
        train_data = load_merged_feature_split("train", context, feature_names=PHASE3_FEATURE_BLOCKS)
        val_data = load_merged_feature_split("val", context, feature_names=PHASE3_FEATURE_BLOCKS)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nFinal hyperparameter tuning is blocked until organizer labels and Phase 3 feature artifacts are available."
        ) from exc

    model = LTRModel()
    best_params = model.tune(train_data, val_data, n_trials=args.n_trials, num_boost_round=300)
    print("Best params:")
    print(json.dumps(best_params, indent=2))

    model.train(
        train_data,
        val_data,
        params=best_params,
        num_boost_round=args.num_boost_round,
    )
    model_path = MODELS_DIR / args.output_name
    model.save(model_path)

    predictions = model.rank_frame(val_data, top_k=100)
    metrics = evaluate_rankings(predictions, labels_to_ground_truth(val_data), k_values=[10])
    metrics_path = SUBMISSIONS_DIR / f"{args.output_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    config_path = PROJECT_ROOT / "configs" / "best_hparams.json"
    config_path.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved best params to: {config_path}")
    print(f"Saved final model to: {model_path}")
    print(f"Saved validation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
