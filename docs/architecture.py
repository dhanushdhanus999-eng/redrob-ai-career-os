"""Generate Phase 5 README visuals.

Run from the repository root:
    python docs/architecture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "outputs" / "submissions"


def _save(fig: plt.Figure, filename: str) -> Path:
    path = DOCS_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_architecture() -> Path:
    """Draw the multi-stage ranking architecture."""
    fig, axis = plt.subplots(figsize=(13, 9))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 12)
    axis.axis("off")

    stages = [
        ("Job Description Input", "Released JD or pasted recruiter JD", "#17324D"),
        ("Stage 0: Structured Parsing", "skills, seniority, domain, years, location", "#2E86AB"),
        ("Stage 1: Hybrid Recall", "BM25 + dense retrieval + RRF fusion", "#7B2CBF"),
        ("Stage 2: Feature Engineering", "semantic, skill, experience, behavioral", "#F18F01"),
        ("Stage 3: LightGBM LambdaRank", "trained when labels/artifacts are available", "#C73E1D"),
        ("Stage 4: Cross-Encoder Rerank", "deeper interaction on the short list", "#2D6A4F"),
        ("Stage 5: LLM Listwise Rerank", "qualitative review and rationale support", "#44BBA4"),
        ("Ranked Shortlist", "top candidates with explanations", "#17324D"),
    ]

    y_positions = [11, 9.6, 8.2, 6.8, 5.4, 4.0, 2.6, 1.2]
    for (title, subtitle, color), y_position in zip(stages, y_positions, strict=False):
        box = FancyBboxPatch(
            (2.1, y_position - 0.45),
            9.8,
            0.9,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#F8FAFC",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            7,
            y_position + 0.13,
            title,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            7,
            y_position - 0.17,
            subtitle,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    for start_y, end_y in zip(y_positions[:-1], y_positions[1:], strict=False):
        arrow = FancyArrowPatch(
            (7, start_y - 0.48),
            (7, end_y + 0.48),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color="#475569",
        )
        axis.add_patch(arrow)

    axis.text(
        7,
        11.85,
        "Intelligent Candidate Discovery - System Architecture",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
    )
    axis.text(
        7,
        0.25,
        "Hidden-eval honest: supervised metrics and LTR importance appear only when labels/artifacts exist.",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    return _save(fig, "architecture_diagram.png")


def _load_ltr_importance() -> pd.DataFrame | None:
    model_path = MODELS_DIR / "ltr_final"
    if not Path(str(model_path) + ".lgb").exists():
        return None
    try:
        from src.models.ltr_model import LTRModel

        model = LTRModel()
        model.load(model_path)
        importance = model.feature_importance_df().head(20)
        if importance.empty:
            return None
        importance["group"] = importance["feature"].map(_feature_group)
        return importance
    except Exception:
        return None


def _feature_group(feature_name: str) -> str:
    lowered = feature_name.lower()
    if any(token in lowered for token in ("active", "response", "profile", "saved", "behavior")):
        return "Behavioral"
    if any(token in lowered for token in ("skill", "coverage", "composite", "exact")):
        return "Skill"
    if any(token in lowered for token in ("sim", "semantic", "embedding")):
        return "Semantic"
    if any(token in lowered for token in ("experience", "seniority", "education", "location")):
        return "Experience"
    return "Other"


def draw_feature_importance() -> Path:
    """Draw real LTR importance when possible, otherwise an honest signal map."""
    colors = {
        "Behavioral": "#F18F01",
        "Skill": "#2E86AB",
        "Semantic": "#7B2CBF",
        "Experience": "#44BBA4",
        "Other": "#64748B",
    }
    importance = _load_ltr_importance()

    if importance is None:
        importance = pd.DataFrame(
            [
                ("behavioral_recency", 8, "Behavioral"),
                ("behavioral_availability", 7, "Behavioral"),
                ("behavioral_recruiter_engagement", 7, "Behavioral"),
                ("skill_exact_family_fuzzy", 6, "Skill"),
                ("semantic_profile_alignment", 5, "Semantic"),
                ("experience_year_range", 4, "Experience"),
                ("seniority_alignment", 3, "Experience"),
                ("profile_confidence", 3, "Behavioral"),
            ],
            columns=["feature", "importance", "group"],
        )
        title = "Feature Signal Map - LTR Importance Pending Labels"
        xlabel = "Designed signal coverage, not model gain"
    else:
        title = "Feature Importance - Top 20 by LightGBM Gain"
        xlabel = "Gain"

    working = importance.sort_values("importance", ascending=True)
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    axis.barh(
        working["feature"],
        working["importance"],
        color=[colors.get(group, colors["Other"]) for group in working["group"]],
    )
    axis.set_title(title, fontsize=14, fontweight="bold", pad=14)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=9)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in colors.items()
        if label in set(working["group"])
    ]
    axis.legend(handles=legend_handles, loc="lower right", frameon=False)
    return _save(fig, "feature_importance_colored.png")


def _read_metric_scores() -> list[tuple[str, float]]:
    candidates = sorted(SUBMISSIONS_DIR.glob("*metrics.json"))
    scores: list[tuple[str, float]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = payload.get("ndcg@10") or payload.get("ndcg_10") or payload.get("NDCG@10")
        if value is None:
            continue
        label = path.stem.replace("_metrics", "").replace("_", " ").title()
        scores.append((label, float(value)))
    return scores


def draw_ndcg_progression() -> Path:
    """Draw NDCG progression when metrics exist; otherwise show blocked status."""
    metric_scores = _read_metric_scores()
    fig, axis = plt.subplots(figsize=(10, 5.5))

    if metric_scores:
        labels = [label for label, _ in metric_scores]
        scores = [score for _, score in metric_scores]
        bars = axis.bar(labels, scores, color=["#2E86AB", "#7B2CBF", "#F18F01", "#44BBA4"])
        axis.set_ylim(0, max(scores) * 1.2)
        axis.set_ylabel("NDCG@10")
        axis.set_title("Model Progression - NDCG@10", fontsize=14, fontweight="bold", pad=14)
        for bar, score in zip(bars, scores, strict=False):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        stages = ["Random", "BM25", "Dense", "Hybrid", "LTR", "Cross-Enc", "LLM"]
        axis.bar(stages, [0] * len(stages), color="#CBD5E1")
        axis.set_ylim(0, 1)
        axis.set_ylabel("NDCG@10")
        axis.set_title("Model Progression - Hidden Labels Pending", fontsize=14, fontweight="bold", pad=14)
        axis.text(
            0.5,
            0.62,
            "No public relevance labels are available in the released bundle.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
        )
        axis.text(
            0.5,
            0.48,
            "The repo implements scoring utilities and ranking stages.\n"
            "NDCG is reported only after labels or official feedback exist.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
        )

    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", rotation=20)
    return _save(fig, "ndcg_progression.png")


def main() -> None:
    generated = [
        draw_architecture(),
        draw_feature_importance(),
        draw_ndcg_progression(),
    ]
    for path in generated:
        print(f"Generated {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
