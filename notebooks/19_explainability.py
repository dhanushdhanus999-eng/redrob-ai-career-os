"""Phase 3 Day 19: attach ranking rationales to a ranked output file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import load_merged_feature_split, load_phase3_context
from src.ranking.explainer import add_explanations_to_submission
from src.utils.paths import SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default=str(SUBMISSIONS_DIR / "ltr_phase3_val.csv"))
    parser.add_argument("--feature-split", default="val")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    if not submission_path.exists():
        raise SystemExit(f"Submission file not found: {submission_path}")

    try:
        context = load_phase3_context(require_labels=False)
        feature_df = load_merged_feature_split(args.feature_split, context)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nGenerate the relevant Phase 3 feature blocks before adding explanations."
        ) from exc

    submission_df = pd.read_csv(submission_path)
    enriched = add_explanations_to_submission(
        submission_df,
        context.parsed_jobs,
        context.parsed_candidates,
        feature_df,
    )
    output_path = (
        Path(args.output)
        if args.output
        else submission_path.with_name(f"{submission_path.stem}_with_rationale.csv")
    )
    enriched.to_csv(output_path, index=False)
    print(f"Saved rationale-enriched submission to: {output_path}")


if __name__ == "__main__":
    main()
