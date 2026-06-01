"""Phase 2 Day 10: skill ontology smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.skill_ontology import SkillMatcher, normalize_skill


def _load_real_example() -> tuple[list[str], list[str]] | None:
    parsed_jds_path = PROCESSED_DATA_DIR / "parsed_jds.json"
    parsed_candidates_path = PROCESSED_DATA_DIR / "parsed_candidates.json"
    if not parsed_jds_path.exists() or not parsed_candidates_path.exists():
        return None

    with parsed_jds_path.open("r", encoding="utf-8") as handle:
        parsed_jds = json.load(handle)
    with parsed_candidates_path.open("r", encoding="utf-8") as handle:
        parsed_candidates = json.load(handle)

    for job_record in parsed_jds.values():
        required = job_record.get("must_have_skills") or job_record.get("nice_to_have_skills") or []
        if not required:
            continue
        for candidate_record in parsed_candidates.values():
            candidate_skills = candidate_record.get("skills") or []
            if candidate_skills:
                return required, candidate_skills
    return None


def main() -> None:
    matcher = SkillMatcher()
    real_example = _load_real_example()

    if real_example is None:
        required_skills = ["Python", "Machine Learning", "AWS", "PostgreSQL", "Docker"]
        candidate_skills = ["py", "pytorch", "tensorflow", "gcp", "postgres", "k8s"]
    else:
        required_skills, candidate_skills = real_example

    result = matcher.match_score(required_skills=required_skills, candidate_skills=candidate_skills)
    print(json.dumps(result, indent=2))

    samples = ["reactjs", "ml", "nodejs", "k8s", "py", "js"]
    print({sample: normalize_skill(sample) for sample in samples})


if __name__ == "__main__":
    main()
