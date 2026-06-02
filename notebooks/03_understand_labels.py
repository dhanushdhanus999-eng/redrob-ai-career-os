"""Phase 1 Day 3: inspect the hidden-evaluation submission contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.challenge_bundle import discover_challenge_bundle, read_docx_text
from src.utils.paths import PROCESSED_DATA_DIR, SUBMISSIONS_DIR, ensure_project_dirs


COMPOSITE_PATTERN = re.compile(
    r"Final composite\s*=\s*0\.50\s*×\s*NDCG@10\s*\+\s*0\.30\s*×\s*NDCG@50\s*\+\s*0\.15\s*×\s*MAP\s*\+\s*0\.05\s*×\s*P@10",
    re.IGNORECASE,
)


def main() -> None:
    ensure_project_dirs()
    bundle = discover_challenge_bundle()
    submission_spec = read_docx_text(bundle.submission_spec)
    sample_submission = pd.read_csv(bundle.sample_submission)

    print(f"Loaded submission spec from: {bundle.submission_spec}")
    print(f"Loaded sample submission from: {bundle.sample_submission}")
    print(f"Sample submission shape: {sample_submission.shape}")
    print(f"Columns: {sample_submission.columns.tolist()}")
    print("\nHead:")
    print(sample_submission.head(10))

    summary = {
        "sample_submission_rows": int(len(sample_submission)),
        "sample_submission_columns": sample_submission.columns.tolist(),
        "required_columns_in_order": ["candidate_id", "rank", "score", "reasoning"],
        "exact_top_k_required": 100,
        "score_must_be_non_increasing": True,
        "hidden_metrics": {
            "ndcg@10_weight": 0.50,
            "ndcg@50_weight": 0.30,
            "map_weight": 0.15,
            "p@10_weight": 0.05,
        },
        "composite_formula_confirmed": bool(COMPOSITE_PATTERN.search(submission_spec)),
        "job_is_hidden_eval_single_query": True,
    }
    summary_path = PROCESSED_DATA_DIR / "phase1_submission_spec_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    template_path = SUBMISSIONS_DIR / "submission_template.csv"
    pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"]).to_csv(
        template_path,
        index=False,
    )

    copied_sample_path = SUBMISSIONS_DIR / "sample_submission_reference.csv"
    sample_submission.to_csv(copied_sample_path, index=False)

    print("\nSubmission-spec analysis complete.")
    print(f"Saved structured summary to: {summary_path}")
    print(f"Saved blank submission template to: {template_path}")
    print(f"Copied sample submission reference to: {copied_sample_path}")


if __name__ == "__main__":
    main()
