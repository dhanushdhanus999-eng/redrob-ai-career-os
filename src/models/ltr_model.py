"""Learning-to-rank model helpers for Phase 3."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    lgb = None


BASE_COLUMNS = {"job_id", "candidate_id", "relevance"}


def _require_lightgbm() -> None:
    if lgb is None:
        raise ModuleNotFoundError(
            "lightgbm is required for Phase 3 LTR training. Install the project dependencies "
            "from pyproject.toml and rerun the training pipeline."
        )


def prepare_lgb_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Sort a feature frame and return feature columns plus group sizes."""
    required = BASE_COLUMNS.difference(df.columns)
    if required:
        raise KeyError(f"Missing required ranking columns: {sorted(required)}")

    working = df.copy()
    working["job_id"] = working["job_id"].astype(str)
    working["candidate_id"] = working["candidate_id"].astype(str)
    working["relevance"] = pd.to_numeric(working["relevance"], errors="coerce").fillna(0.0)
    working = working.sort_values(["job_id", "candidate_id"], kind="mergesort").reset_index(drop=True)

    feature_cols = [column for column in working.columns if column not in BASE_COLUMNS]
    group_sizes = (
        working.groupby("job_id", sort=False)
        .size()
        .astype(int)
        .tolist()
    )
    return working, feature_cols, group_sizes


def prepare_lgb_dataset(df: pd.DataFrame) -> tuple[lgb.Dataset, list[str]]:
    """Convert a feature frame into a LightGBM ranking dataset."""
    _require_lightgbm()
    working, feature_cols, group_sizes = prepare_lgb_frame(df)
    features = working[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    labels = working["relevance"].to_numpy(dtype=np.float32)
    dataset = lgb.Dataset(
        features,
        label=labels,
        group=group_sizes,
        feature_name=feature_cols,
    )
    return dataset, feature_cols


class LTRModel:
    """Thin wrapper around a LightGBM LambdaRank model."""

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.best_params: dict[str, Any] = {}

    @staticmethod
    def default_params() -> dict[str, Any]:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10, 20],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 500,
    ) -> None:
        """Train the LambdaRank model."""
        _require_lightgbm()
        train_dataset, self.feature_cols = prepare_lgb_dataset(train_df)
        val_dataset, _ = prepare_lgb_dataset(val_df)

        train_params = self.default_params()
        if params is not None:
            train_params.update(params)

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(50),
        ]
        self.model = lgb.train(
            train_params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[val_dataset],
            valid_names=["val"],
            callbacks=callbacks,
        )
        self.best_params = train_params

    def tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        n_trials: int = 20,
        num_boost_round: int = 300,
    ) -> dict[str, Any]:
        """Run a lightweight Optuna search and return the best parameter set."""
        _require_lightgbm()
        import optuna

        train_dataset, _ = prepare_lgb_dataset(train_df)
        val_dataset, _ = prepare_lgb_dataset(val_df)

        def objective(trial: optuna.Trial) -> float:
            params = self.default_params()
            params.update(
                {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                }
            )
            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=num_boost_round,
                valid_sets=[val_dataset],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            return float(model.best_score["val"]["ndcg@10"])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = self.default_params()
        best_params.update(study.best_params)
        return best_params

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The LTR model has not been trained or loaded yet.")
        return self.model.predict(features)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the input frame with model scores appended."""
        if self.model is None:
            raise RuntimeError("The LTR model has not been trained or loaded yet.")
        working = df.copy()
        for column in self.feature_cols:
            if column not in working.columns:
                working[column] = 0.0
        matrix = working[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        working["score"] = self.predict(matrix)
        return working

    def rank_frame(self, df: pd.DataFrame, *, top_k: int = 100) -> dict[str, list[str]]:
        """Rank candidates per job and return ordered candidate IDs."""
        scored = self.score_frame(df)
        predictions: dict[str, list[str]] = {}
        for job_id, group in scored.groupby("job_id"):
            ordered = group.sort_values("score", ascending=False, kind="mergesort")
            predictions[str(job_id)] = ordered["candidate_id"].astype(str).head(top_k).tolist()
        return predictions

    def feature_importance_df(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("The LTR model has not been trained or loaded yet.")
        importance = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_cols,
                    "importance": importance,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("The LTR model has not been trained or loaded yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path) + ".lgb")
        with open(str(path) + ".meta.pkl", "wb") as handle:
            pickle.dump(
                {
                    "feature_cols": self.feature_cols,
                    "best_params": self.best_params,
                },
                handle,
            )

    def load(self, path: str | Path) -> None:
        _require_lightgbm()
        path = Path(path)
        self.model = lgb.Booster(model_file=str(path) + ".lgb")
        with open(str(path) + ".meta.pkl", "rb") as handle:
            payload = pickle.load(handle)
        self.feature_cols = list(payload.get("feature_cols", []))
        self.best_params = dict(payload.get("best_params", {}))
