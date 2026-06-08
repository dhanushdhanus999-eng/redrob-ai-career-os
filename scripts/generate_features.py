"""Generate 20-feature LTR training matrix from pseudo-labels and candidate pool.

Reads:  data/processed/pseudo_labels.csv
        candidate pool via load_phase2_bundle()
Writes: data/processed/features_ltr.parquet

Usage:
    python scripts/generate_features.py
    python scripts/generate_features.py --limit 1000   # fast test
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.baselines.common import load_phase2_bundle
from src.parsing.candidate_parser import CandidateProfileParser
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs
from src.utils.role_relevance import score_career_trajectory, score_role_relevance
from src.utils.skill_ontology import SkillMatcher

# Import shared constants and helpers from the submission generator (do NOT redefine)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_submission import (  # noqa: E402
    TRACK1_MIN_YEARS,
    TRACK1_MAX_YEARS,
    TRACK1_MUST_HAVE_SKILLS,
    TRACK1_NICE_TO_HAVE_SKILLS,
    _safe_float,
    _safe_text,
    score_behavior,
    score_experience,
    score_location,
)

FEATURE_COLS: list[str] = [
    "must_composite", "nice_composite", "must_exact_cov", "nice_exact_cov",
    "n_must_matched", "n_must_missing", "exp_score", "cand_years",
    "role_score", "career_score", "beh_score", "completeness",
    "response_rate", "recency", "github_score", "saved", "search_app",
    "assessment", "open_to_work", "location_score",
]


def _compute_recency(row: pd.Series) -> float:
    recency = 0.4
    raw_last = _safe_text(row.get("last_active"))
    if raw_last:
        try:
            active_date = datetime.fromisoformat(raw_last[:10]).date()
            days_since = max((date.today() - active_date).days, 0)
            recency = max(0.0, 1.0 - days_since / 180.0)
        except ValueError:
            pass
    return recency


def compute_row_features(
    row: pd.Series,
    cand_parser: CandidateProfileParser,
    skill_matcher: SkillMatcher,
    candidate_schema,
) -> dict:
    """Compute all 20 LTR features for a single candidate row."""
    parsed_cand = cand_parser.parse_row(row, candidate_schema)
    cand_skills = list(parsed_cand.get("skills") or [])
    cand_years = _safe_float(
        parsed_cand.get("total_experience_years"),
        default=_safe_float(row.get("total_experience")),
    )
    cand_location = _safe_text(row.get("location")) or _safe_text(row.get("country"))

    must_match = skill_matcher.match_score(TRACK1_MUST_HAVE_SKILLS, cand_skills)
    nice_match = skill_matcher.match_score(TRACK1_NICE_TO_HAVE_SKILLS, cand_skills)

    # Behavioral sub-signals (mirrors score_behavior internals for per-feature granularity)
    completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
    if completeness == 0.0:
        completeness = _safe_float(row.get("profile_completeness"))
    response_rate = _safe_float(row.get("recruiter_response_rate"))
    github_score  = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
    saved         = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
    search_app    = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
    assessment    = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
    open_to_work  = 1.0 if bool(row.get("open_to_work_flag")) else 0.0
    recency       = _compute_recency(row)

    return {
        "must_composite": must_match["composite_score"],
        "nice_composite": nice_match["composite_score"],
        "must_exact_cov": must_match["exact_coverage"],
        "nice_exact_cov": nice_match["exact_coverage"],
        "n_must_matched": float(must_match["n_exact_matched"]),
        "n_must_missing": float(must_match["n_missing"]),
        "exp_score":      score_experience(cand_years, TRACK1_MIN_YEARS, TRACK1_MAX_YEARS),
        "cand_years":     cand_years,
        "role_score":     score_role_relevance(
                              _safe_text(row.get("current_role")),
                              _safe_text(row.get("headline")),
                          ),
        "career_score":   score_career_trajectory(_safe_text(row.get("career_history_text"))),
        "beh_score":      score_behavior(row),
        "completeness":   completeness,
        "response_rate":  response_rate,
        "recency":        recency,
        "github_score":   github_score,
        "saved":          saved,
        "search_app":     search_app,
        "assessment":     assessment,
        "open_to_work":   open_to_work,
        "location_score": score_location(cand_location),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N candidates (0 = all)")
    args = parser.parse_args()

    ensure_project_dirs()

    labels_path = PROCESSED_DATA_DIR / "pseudo_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Pseudo-labels not found at {labels_path}.\n"
            "Run: python scripts/create_pseudo_labels.py"
        )

    print("Loading pseudo-labels…")
    labels_df = pd.read_csv(labels_path)
    print(f"  {len(labels_df):,} pseudo-labels loaded")

    print("Loading candidate bundle…")
    bundle = load_phase2_bundle(require_labels=False)
    candidates = bundle.candidates.copy()
    cand_id_col = bundle.candidate_schema.candidate_id
    candidates["_cid"] = candidates[cand_id_col].astype(str)
    labels_df["candidate_id"] = labels_df["candidate_id"].astype(str)

    # suffixes=("", "_lbl") prevents pandas from renaming the original candidate_id
    # column to candidate_id_x when cand_id_col == "candidate_id"
    merged = candidates.merge(
        labels_df[["candidate_id", "relevance"]],
        left_on="_cid",
        right_on="candidate_id",
        how="inner",
        suffixes=("", "_lbl"),
    )
    if "candidate_id_lbl" in merged.columns:
        merged = merged.drop(columns=["candidate_id_lbl"])
    print(f"  {len(merged):,} candidates with labels")

    if args.limit > 0:
        merged = merged.head(args.limit)
        print(f"  Limiting to {len(merged):,} candidates (--limit {args.limit})")

    cand_parser   = CandidateProfileParser()
    skill_matcher = SkillMatcher()

    print(f"\nGenerating features for {len(merged):,} candidates…")
    rows: list[dict] = []
    for i, (_, row) in enumerate(merged.iterrows()):
        if i > 0 and i % 10_000 == 0:
            print(f"  … {i:,}/{len(merged):,} processed")
        feats = compute_row_features(row, cand_parser, skill_matcher, bundle.candidate_schema)
        feats["candidate_id"] = str(row["_cid"])
        feats["relevance"]    = int(row["relevance"])
        rows.append(feats)

    print(f"  Done — {len(rows):,} feature rows generated")

    out_df = pd.DataFrame(rows, columns=["candidate_id", "relevance"] + FEATURE_COLS)

    dist = out_df["relevance"].value_counts().sort_index()
    print("\nGrade distribution:")
    for grade, count in dist.items():
        pct = count / len(out_df) * 100
        label = {0: "irrelevant", 1: "weak positive", 2: "strong positive"}.get(int(grade), "?")
        print(f"  Grade {grade} ({label}): {count:,}  ({pct:.1f}%)")

    out_path = PROCESSED_DATA_DIR / "features_ltr.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\nFeatures saved to {out_path}")
    print(f"Shape: {out_df.shape}  ({len(FEATURE_COLS)} feature columns + candidate_id + relevance)")


if __name__ == "__main__":
    main()
