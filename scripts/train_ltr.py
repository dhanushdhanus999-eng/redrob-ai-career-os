"""Train LightGBM LambdaRank on the LTR feature matrix.

Reads:  data/processed/features_ltr.parquet
Writes: outputs/models/ltr_model.pkl            {"model": lgb_model, "feature_cols": [...]}
        outputs/models/ltr_feature_importance.csv

Usage:
    python scripts/train_ltr.py
    python scripts/train_ltr.py --trials 50   # Optuna HPO then final train
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

FEATURE_COLS: list[str] = [
    "must_composite", "nice_composite", "must_exact_cov", "nice_exact_cov",
    "n_must_matched", "n_must_missing", "exp_score", "cand_years",
    "role_score", "career_score", "beh_score", "completeness",
    "response_rate", "recency", "github_score", "saved", "search_app",
    "assessment", "open_to_work", "location_score",
]

MAX_GROUP = 10_000  # LightGBM LambdaRank hard limit per query group


def _make_groups(n: int) -> list[int]:
    full, rem = divmod(n, MAX_GROUP)
    return [MAX_GROUP] * full + ([rem] if rem else [])


DEFAULT_PARAMS: dict = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[5, 10, 20],
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)


def _train(X_train, y_train, group_train, X_val, y_val, group_val, params: dict):
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data   = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=0,
                        help="Number of Optuna HPO trials (0 = skip)")
    args = parser.parse_args()

    ensure_project_dirs()

    features_path = PROCESSED_DATA_DIR / "features_ltr.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {features_path}.\n"
            "Run: python scripts/generate_features.py"
        )

    print(f"Loading feature matrix from {features_path}…")
    df = pd.read_parquet(features_path)
    print(f"  {len(df):,} rows, {len(FEATURE_COLS)} features")

    # 80/20 deterministic train/val split (tail 20% = val)
    n_val   = max(1, int(len(df) * 0.20))
    n_train = len(df) - n_val
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:]

    X_train = train_df[FEATURE_COLS].fillna(0.0).values
    y_train = train_df["relevance"].values.astype(np.float32)
    X_val   = val_df[FEATURE_COLS].fillna(0.0).values
    y_val   = val_df["relevance"].values.astype(np.float32)

    # LightGBM LambdaRank limits each query group to ≤ 10,000 rows.
    # Split the single JD into equal-sized sub-groups for training.
    group_train = _make_groups(n_train)
    group_val   = _make_groups(n_val)

    print(f"  Train: {n_train:,} rows ({len(group_train)} groups)  |  Val: {n_val:,} rows ({len(group_val)} groups)")

    params = dict(DEFAULT_PARAMS)

    if args.trials > 0:
        print(f"\nRunning Optuna HPO ({args.trials} trials)…")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                import lightgbm as lgb
                trial_params = dict(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[10],
                    num_leaves=trial.suggest_int("num_leaves", 31, 127),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    verbose=-1,
                )
                train_data = lgb.Dataset(X_train, label=y_train, group=_make_groups(n_train))
                val_data   = lgb.Dataset(X_val, label=y_val, group=_make_groups(n_val), reference=train_data)
                model = lgb.train(
                    trial_params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
                # Proxy objective: avg LTR score of grade-2 val candidates
                val_scores = model.predict(X_val)
                grade2_mask = y_val == 2
                if grade2_mask.sum() == 0:
                    return 0.0
                return float(val_scores[grade2_mask].mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials=args.trials, show_progress_bar=True)
            best = study.best_params
            print(f"  Best params: {best}")
            params.update(best)
        except ImportError:
            print("  Optuna not installed — skipping HPO")

    print("\nTraining LightGBM LambdaRank…")
    model = _train(X_train, y_train, group_train, X_val, y_val, group_val, params)

    import lightgbm as lgb
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print("\nTop-10 feature importances (gain):")
    for _, feat_row in importance_df.head(10).iterrows():
        print(f"  {feat_row['feature']:25s}  {feat_row['importance']:.1f}")

    model_path = MODELS_DIR / "ltr_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    print(f"\nModel saved to {model_path}")

    fi_path = MODELS_DIR / "ltr_feature_importance.csv"
    importance_df.to_csv(fi_path, index=False)
    print(f"Feature importance saved to {fi_path}")


if __name__ == "__main__":
    main()
