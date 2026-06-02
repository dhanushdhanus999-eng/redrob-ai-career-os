"""Phase 4 Day 23: run an ablation study for feature-group contribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.splits import labels_to_ground_truth
from src.eval.metrics import evaluate_rankings
from src.features.pipeline import PHASE3_FEATURE_BLOCKS, load_merged_feature_split, load_phase3_context
from src.models.ltr_model import LTRModel
from src.utils.paths import DOCS_DIR


OUTPUT_PATH = DOCS_DIR / "ablation.md"
EXCLUDE_PATTERNS = {
    "semantic": lambda column: any(marker in column for marker in ("_full_sim", "_title_", "_skills_sim")),
    "skills": lambda column: any(
        marker in column
        for marker in (
            "coverage",
            "composite",
            "n_must",
            "n_nice",
            "n_candidate_skills",
            "extra_skills_count",
            "required_skill_coverage",
        )
    ),
    "experience": lambda column: any(
        marker in column
        for marker in ("seniority", "exp_", "cand_years", "edu_", "domain_", "role_title")
    ),
    "behavioral": lambda column: any(
        marker in column
        for marker in (
            "active_",
            "profile_complete",
            "profile_confidence",
            "n_applications",
            "n_profile",
            "response",
            "similar_job",
            "behavioral",
            "confidence",
        )
    ),
    "graph": lambda column: "graph" in column or "centrality" in column,
}


def write_blocked(note: str) -> None:
    OUTPUT_PATH.write_text(f"# Ablation Study\n\n{note}\n", encoding="utf-8")


def score_feature_subset(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    val_gt: dict[str, dict[str, float]],
    *,
    excluded_groups: list[str],
) -> float:
    drop_cols = {
        column
        for group in excluded_groups
        for column in train_data.columns
        if group in EXCLUDE_PATTERNS and EXCLUDE_PATTERNS[group](column)
    }
    train_subset = train_data.drop(columns=sorted(drop_cols), errors="ignore")
    val_subset = val_data.drop(columns=sorted(drop_cols), errors="ignore")

    model = LTRModel()
    model.train(train_subset, val_subset, num_boost_round=300)
    predictions = model.rank_frame(val_subset, top_k=100)
    metrics = evaluate_rankings(predictions, val_gt, k_values=[10])
    return float(metrics["ndcg@10"])


def main() -> None:
    try:
        context = load_phase3_context(require_labels=True)
        train_data = load_merged_feature_split("train", context, feature_names=PHASE3_FEATURE_BLOCKS)
        val_data = load_merged_feature_split("val", context, feature_names=PHASE3_FEATURE_BLOCKS)
        val_gt = labels_to_ground_truth(val_data[["job_id", "candidate_id", "relevance"]])
    except (FileNotFoundError, ValueError) as exc:
        write_blocked(
            f"Blocked locally: {exc}. This analysis will run once organizer labels and Phase 3 feature artifacts are available."
        )
        print(f"Saved blocked-note ablation report to: {OUTPUT_PATH}")
        return

    ablation_results = {
        "Full model": score_feature_subset(train_data, val_data, val_gt, excluded_groups=[]),
        "No behavioral": score_feature_subset(train_data, val_data, val_gt, excluded_groups=["behavioral"]),
        "No semantic": score_feature_subset(train_data, val_data, val_gt, excluded_groups=["semantic"]),
        "No skills": score_feature_subset(train_data, val_data, val_gt, excluded_groups=["skills"]),
        "No experience": score_feature_subset(train_data, val_data, val_gt, excluded_groups=["experience"]),
        "Skills only": score_feature_subset(
            train_data,
            val_data,
            val_gt,
            excluded_groups=["semantic", "experience", "behavioral"],
        ),
        "Semantic only": score_feature_subset(
            train_data,
            val_data,
            val_gt,
            excluded_groups=["skills", "experience", "behavioral"],
        ),
    }

    full_score = ablation_results["Full model"]
    lines = [
        "# Ablation Study\n\n",
        "| Configuration | NDCG@10 | Delta |\n",
        "|---|---:|---:|\n",
    ]
    for label, score in sorted(ablation_results.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {label} | {score:.4f} | {score - full_score:+.4f} |\n")
    lines.append("\n## Raw Results\n\n```json\n")
    lines.append(json.dumps(ablation_results, indent=2, sort_keys=True))
    lines.append("\n```\n")
    OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Saved ablation report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
