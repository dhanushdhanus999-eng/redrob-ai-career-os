"""Phase 2 Day 8: structured job-description parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import load_phase2_bundle
from src.parsing.jd_parser import JobDescriptionParser
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    try:
        bundle = load_phase2_bundle(require_labels=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs file into data/raw and rerun."
        ) from exc

    parser = JobDescriptionParser()
    parsed_jobs = parser.parse_frame(bundle.jobs, bundle.job_schema)

    output_path = PROCESSED_DATA_DIR / "parsed_jds.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed_jobs, handle, indent=2, default=str)

    print(f"Parsed {len(parsed_jobs)} jobs -> {output_path}")
    if parsed_jobs:
        first_job = next(iter(parsed_jobs.values()))
        print(json.dumps(first_job, indent=2, default=str))


if __name__ == "__main__":
    main()
