"""Phase 4 Day 24: analyze failure modes for a validation prediction file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.splits import labels_to_ground_truth
from src.eval.metrics import ndcg_at_k
from src.features.pipeline import load_phase3_context, load_split_frame
from src.utils.paths import DOCS_DIR, SUBMISSIONS_DIR


OUTPUT_PATH = DOCS_DIR / "error_analysis.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-path", default="")
    parser.add_argument("--top-jobs", type=int, default=10)
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
    OUTPUT_PATH.write_text(f"# Error Analysis\n\n{note}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_path) if args.pred_path else find_default_prediction_path()
    if pred_path is None or not pred_path.exists():
        write_blocked(
            "Blocked locally: no validation prediction CSV was found under `outputs/submissions/`."
        )
        print(f"Saved blocked-note error analysis report to: {OUTPUT_PATH}")
        return

    try:
        context = load_phase3_context(require_labels=True)
        val_df = load_split_frame("val", context)
    except (FileNotFoundError, ValueError) as exc:
        write_blocked(
            f"Blocked locally: {exc}. Error analysis will run once organizer labels are available."
        )
        print(f"Saved blocked-note error analysis report to: {OUTPUT_PATH}")
        return

    val_gt = labels_to_ground_truth(val_df)
    preds_df = pd.read_csv(pred_path)
    pred_dict = {
        str(job_id): group.sort_values("rank")["candidate_id"].astype(str).tolist()
        for job_id, group in preds_df.groupby("job_id")
    }

    per_job_scores = {}
    for job_id, ranked_candidates in pred_dict.items():
        truth = val_gt.get(job_id, {})
        relevances = [truth.get(candidate_id, 0.0) for candidate_id in ranked_candidates]
        per_job_scores[job_id] = ndcg_at_k(relevances, k=10)

    scores = pd.Series(per_job_scores).sort_values()
    worst_jobs = scores.head(args.top_jobs)
    lines = [
        "# Error Analysis\n\n",
        f"- Prediction file: `{pred_path.name}`\n",
        f"- Jobs analysed: `{len(per_job_scores)}`\n",
        f"- Mean NDCG@10: `{scores.mean():.4f}`\n\n",
        "## Lowest-Scoring Jobs\n\n",
    ]

    thin_profile_misses = 0
    total_missed_relevant = 0
    for job_id, score in worst_jobs.items():
        truth = val_gt.get(job_id, {})
        ranked_candidates = pred_dict.get(job_id, [])
        top50_pred = set(ranked_candidates[:50])
        top5_gt = [candidate_id for candidate_id, _ in sorted(truth.items(), key=lambda item: item[1], reverse=True)[:5]]
        missed = [candidate_id for candidate_id in top5_gt if candidate_id not in top50_pred]
        total_missed_relevant += len(missed)
        for candidate_id in missed:
            completeness = context.parsed_candidates.get(str(candidate_id), {}).get("profile_completeness", 1.0)
            if completeness < 0.5:
                thin_profile_misses += 1

        parsed_job = context.parsed_jobs.get(str(job_id), {})
        lines.extend(
            [
                f"### Job {job_id}\n",
                f"- NDCG@10: {score:.4f}\n",
                f"- Title: {parsed_job.get('title', '')}\n",
                f"- Must-have skills: {parsed_job.get('must_have_skills', [])[:5]}\n",
                f"- Missed relevant candidates: {missed}\n\n",
            ]
        )

    lines.extend(
        [
            "## Failure Patterns\n\n",
            f"- Thin-profile misses among reviewed relevant candidates: `{thin_profile_misses}/{max(total_missed_relevant, 1)}`\n",
            "- These misses are a strong cue to keep `profile_confidence` and profile-richness signals in the final model.\n",
            "- Any repeatedly missed niche skills should be added to `src/utils/failure_fixes.py` and the skill ontology.\n",
        ]
    )
    OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Saved error analysis report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
