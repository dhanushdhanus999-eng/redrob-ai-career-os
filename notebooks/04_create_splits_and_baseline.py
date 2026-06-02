"""Phase 1 Day 4: validate the public bundle and sample submission end-to-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.challenge_bundle import (
    discover_challenge_bundle,
    load_candidate_id_set,
    read_docx_text,
)
from src.eval.submission import validate_track1_submission
from src.utils.paths import PROCESSED_DATA_DIR, SUBMISSIONS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    bundle = discover_challenge_bundle()

    sample_submission = pd.read_csv(bundle.sample_submission)
    candidate_ids = load_candidate_id_set(bundle)
    validation_issues = validate_track1_submission(
        sample_submission,
        valid_candidate_ids=candidate_ids,
    )

    bundle_summary = {
        "bundle_root": str(bundle.root),
        "candidates_path": str(bundle.candidates),
        "candidate_schema_path": str(bundle.candidate_schema),
        "job_description_path": str(bundle.job_description),
        "sample_submission_path": str(bundle.sample_submission),
        "submission_spec_path": str(bundle.submission_spec),
        "candidate_pool_size": len(candidate_ids),
        "sample_submission_rows": int(len(sample_submission)),
        "sample_submission_valid": not validation_issues,
        "sample_submission_issues": validation_issues,
        "job_description_word_count": len(read_docx_text(bundle.job_description).split()),
    }

    summary_path = PROCESSED_DATA_DIR / "phase1_bundle_validation.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle_summary, handle, indent=2)

    template_rows = sample_submission[["candidate_id", "rank", "score"]].copy()
    template_rows["reasoning"] = ""
    template_output_path = SUBMISSIONS_DIR / "submission_working_template.csv"
    template_rows.to_csv(template_output_path, index=False)

    print("Bundle validation complete.")
    print(json.dumps(bundle_summary, indent=2))
    print(f"\nSaved validation summary to: {summary_path}")
    print(f"Saved working submission template to: {template_output_path}")


if __name__ == "__main__":
    main()
