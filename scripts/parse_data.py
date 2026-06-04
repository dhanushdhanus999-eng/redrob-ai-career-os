"""Parse the released job description and candidate profiles into JSON.

Writes structured representations used by feature engineering and the
submission generator. Results are cached on disk so repeated runs are fast.

Usage:
    python scripts/parse_data.py            # parse JD + first 5 000 candidates
    python scripts/parse_data.py --all      # parse all 100 000 candidates (slow)
    python scripts/parse_data.py --limit 0  # parse JD only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import load_phase2_bundle
from src.data.schema import combine_text_values
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs


def parse_jd(bundle) -> dict:
    out = PROCESSED_DATA_DIR / "parsed_jd.json"
    parser = JobDescriptionParser()
    job_row  = bundle.jobs.iloc[0]
    job_text = combine_text_values(job_row, bundle.job_schema.text_columns)
    seed = {
        "job_id":   str(job_row[bundle.job_schema.job_id]),
        "title":    str(job_row.get(bundle.job_schema.title or "job_title", "")),
        "location": str(job_row.get(bundle.job_schema.location or "location", "")),
        "min_years_experience": str(job_row.get(bundle.job_schema.min_experience or "min_experience", "")),
        "max_years_experience": str(job_row.get(bundle.job_schema.max_experience or "max_experience", "")),
    }
    parsed = parser.parse_text(job_text, seed=seed)
    with out.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"JD saved  → {out}")
    print(f"  title: {parsed.get('title')}")
    print(f"  must skills: {parsed.get('must_have_skills', [])[:6]}")
    return parsed


def parse_candidates(bundle, limit: int | None) -> dict[str, dict]:
    out = PROCESSED_DATA_DIR / "parsed_candidates.json"
    parser = CandidateProfileParser()
    df = bundle.candidates if limit is None else bundle.candidates.head(limit)

    print(f"Parsing {len(df):,} candidate profiles…")
    t0 = time.perf_counter()
    parsed: dict[str, dict] = {}
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        p = parser.parse_row(row, bundle.candidate_schema)
        parsed[p["candidate_id"]] = p
        if i % 5000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i:,} / {len(df):,}  ({elapsed:.0f}s)")

    with out.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False)
    print(f"Candidates saved ({len(parsed):,}) → {out}  ({time.perf_counter()-t0:.1f}s)")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all", action="store_true",
        help="Parse all candidates (100 000 — slow; results are cached after first run)",
    )
    group.add_argument(
        "--limit", type=int, default=5000,
        help="Number of candidates to parse (default: 5 000). Use 0 to skip candidate parsing.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    bundle = load_phase2_bundle(require_labels=False)
    print(f"Bundle loaded — {len(bundle.candidates):,} candidates, {len(bundle.jobs)} job(s)\n")

    parse_jd(bundle)

    limit = None if args.all else (args.limit if args.limit > 0 else 0)
    if limit != 0:
        parse_candidates(bundle, limit)
    else:
        print("Skipping candidate parsing (--limit 0).")

    print("\nParsing complete.")


if __name__ == "__main__":
    main()
