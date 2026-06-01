"""Phase 2 Day 9: structured candidate profile parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import load_phase2_bundle
from src.parsing.candidate_parser import CandidateProfileParser
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    try:
        bundle = load_phase2_bundle(require_labels=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the released candidates file into data/raw and rerun."
        ) from exc

    parser = CandidateProfileParser()
    parsed_candidates = parser.parse_frame(bundle.candidates, bundle.candidate_schema)

    output_path = PROCESSED_DATA_DIR / "parsed_candidates.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed_candidates, handle, indent=2, default=str)

    print(f"Parsed {len(parsed_candidates)} candidates -> {output_path}")
    if parsed_candidates:
        first_candidate = next(iter(parsed_candidates.values()))
        print(json.dumps(first_candidate, indent=2, default=str))

    total = len(parsed_candidates)
    if total == 0:
        return

    has_skills = sum(1 for record in parsed_candidates.values() if record.get("skills"))
    has_experience = sum(
        1 for record in parsed_candidates.values() if record.get("total_experience_years") is not None
    )
    has_seniority = sum(
        1 for record in parsed_candidates.values() if record.get("seniority") != "unknown"
    )
    print(
        {
            "has_skills": round(has_skills / total, 4),
            "has_experience": round(has_experience / total, 4),
            "has_seniority": round(has_seniority / total, 4),
        }
    )


if __name__ == "__main__":
    main()
