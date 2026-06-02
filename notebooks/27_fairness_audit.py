"""Phase 4 Day 27: audit ranking outputs for coarse bias signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.splits import labels_to_ground_truth
from src.features.pipeline import load_phase3_context, load_split_frame
from src.utils.paths import DOCS_DIR, SUBMISSIONS_DIR


OUTPUT_PATH = DOCS_DIR / "fairness_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-path", default="")
    return parser.parse_args()


def find_default_prediction_path() -> Path | None:
    candidates = sorted(
        (
            path
            for path in SUBMISSIONS_DIR.glob("*.csv")
            if "rationale" not in path.stem.lower()
            and "metrics" not in path.stem.lower()
            and "template" not in path.stem.lower()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def write_blocked(note: str) -> None:
    OUTPUT_PATH.write_text(f"# Fairness Audit Report\n\n{note}\n", encoding="utf-8")


def completeness_bucket(value: float) -> str:
    if value < 0.3:
        return "thin"
    if value < 0.8:
        return "medium"
    return "rich"


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_path) if args.pred_path else find_default_prediction_path()
    if pred_path is None or not pred_path.exists():
        write_blocked(
            "Blocked locally: no validation prediction CSV was found under `outputs/submissions/`."
        )
        print(f"Saved blocked-note fairness report to: {OUTPUT_PATH}")
        return

    try:
        context = load_phase3_context(require_labels=True)
        val_df = load_split_frame("val", context)
    except (FileNotFoundError, ValueError) as exc:
        write_blocked(
            f"Blocked locally: {exc}. Fairness auditing will run once organizer labels are available."
        )
        print(f"Saved blocked-note fairness report to: {OUTPUT_PATH}")
        return

    val_gt = labels_to_ground_truth(val_df)
    preds_df = pd.read_csv(pred_path)
    pred_dict = {
        str(job_id): group.sort_values("rank")["candidate_id"].astype(str).tolist()
        for job_id, group in preds_df.groupby("job_id")
    }

    location_rank_positions: dict[str, list[int]] = {}
    completeness_rank_positions: dict[str, list[int]] = {}

    for job_id, ranked_candidates in pred_dict.items():
        relevant = {
            str(candidate_id)
            for candidate_id, relevance in val_gt.get(job_id, {}).items()
            if relevance > 0
        }
        for rank, candidate_id in enumerate(ranked_candidates[:100], start=1):
            if candidate_id not in relevant:
                continue
            parsed_candidate = context.parsed_candidates.get(str(candidate_id), {})
            location = str(parsed_candidate.get("location") or "unknown")
            location_rank_positions.setdefault(location, []).append(rank)

            completeness = float(parsed_candidate.get("profile_completeness", 0.5))
            completeness_rank_positions.setdefault(completeness_bucket(completeness), []).append(rank)

    lines = [
        "# Fairness Audit Report\n\n",
        f"- Prediction file: `{pred_path.name}`\n\n",
        "## Location Bias\n\n",
    ]
    if location_rank_positions:
        lines.append("| Location | Avg relevant rank | Count |\n")
        lines.append("|---|---:|---:|\n")
        for location, ranks in sorted(location_rank_positions.items(), key=lambda item: np.mean(item[1])):
            if len(ranks) < 3:
                continue
            lines.append(f"| {location} | {np.mean(ranks):.1f} | {len(ranks)} |\n")
    else:
        lines.append("Location data was not sufficiently available for analysis.\n")

    lines.extend(
        [
            "\n## Profile Completeness Bias\n\n",
            "| Bucket | Avg relevant rank | Count |\n",
            "|---|---:|---:|\n",
        ]
    )
    for bucket in ("thin", "medium", "rich"):
        ranks = completeness_rank_positions.get(bucket, [])
        if ranks:
            lines.append(f"| {bucket} | {np.mean(ranks):.1f} | {len(ranks)} |\n")
        else:
            lines.append(f"| {bucket} | N/A | 0 |\n")

    lines.extend(
        [
            "\n## Mitigations In This Repo\n\n",
            "1. Added `profile_confidence` features so thin profiles are treated as low-confidence rather than automatically irrelevant.\n",
            "2. Behavioral features rely on recency and engagement rather than raw activity counts alone.\n",
            "3. Skill matching uses synonym and family-level normalization to reduce equivalent-skill penalties.\n",
        ]
    )
    OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Saved fairness report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
