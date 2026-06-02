"""Phase 1 Day 3: inspect and flatten the released candidate pool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.challenge_bundle import (
    discover_challenge_bundle,
    flatten_candidate_record,
    save_canonical_candidate_dataset,
    stream_candidate_records,
)
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def _safe_days_since(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return (pd.Timestamp.utcnow().tz_localize(None) - parsed).dt.days


def main() -> None:
    ensure_project_dirs()
    bundle = discover_challenge_bundle()

    output_path = save_canonical_candidate_dataset(bundle)
    candidates = pd.read_parquet(output_path)

    print(f"Challenge bundle root: {bundle.root}")
    print(f"Saved canonical candidate dataset to: {output_path}")
    print(f"Shape: {candidates.shape}")
    print(f"Columns: {candidates.columns.tolist()}")

    if candidates.empty:
        raise SystemExit("Candidate dataset is empty after flattening.")

    sample_candidate = flatten_candidate_record(next(stream_candidate_records(bundle.candidates, limit=1)))
    print("\nFlattened sample candidate:")
    print(json.dumps(sample_candidate, indent=2, default=str)[:5000])

    plt.figure(figsize=(10, 4))
    candidates["total_experience"].dropna().astype(float).clip(upper=25).hist(
        bins=30,
        color="#4C956C",
    )
    plt.title("Candidate experience distribution (clipped at 25 years)")
    plt.xlabel("Years of experience")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_experience_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    candidates["skill_count"].clip(upper=25).hist(bins=25, color="#577590")
    plt.title("Skills per candidate (clipped at 25)")
    plt.xlabel("Skill count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_skill_count_distribution.png")
    plt.close()

    top_countries = candidates["country"].fillna("Unknown").value_counts().head(10)
    plt.figure(figsize=(10, 4))
    top_countries.sort_values().plot(kind="barh", color="#F4A261")
    plt.title("Top 10 candidate countries")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_country_distribution.png")
    plt.close()

    activity_days = _safe_days_since(candidates["last_active"])
    plt.figure(figsize=(10, 4))
    activity_days.dropna().clip(lower=0, upper=365).hist(bins=30, color="#BC4749")
    plt.title("Days since last activity (clipped at 365)")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_last_active_days.png")
    plt.close()

    summary = {
        "candidate_dataset_path": str(output_path),
        "n_candidates": int(len(candidates)),
        "n_columns": int(candidates.shape[1]),
        "avg_total_experience": round(float(candidates["total_experience"].dropna().mean()), 3),
        "median_total_experience": round(float(candidates["total_experience"].dropna().median()), 3),
        "avg_skill_count": round(float(candidates["skill_count"].mean()), 3),
        "avg_profile_completeness_score": round(
            float(candidates["profile_completeness_score"].dropna().mean()),
            3,
        ),
        "avg_notice_period_days": round(float(candidates["notice_period_days"].dropna().mean()), 3),
        "open_to_work_rate": round(float(candidates["open_to_work_flag"].fillna(False).mean()), 3),
        "verified_email_rate": round(float(candidates["verified_email"].fillna(False).mean()), 3),
        "linkedin_connected_rate": round(float(candidates["linkedin_connected"].fillna(False).mean()), 3),
        "median_last_active_days": round(float(activity_days.dropna().median()), 3),
        "top_countries": top_countries.to_dict(),
        "top_titles": candidates["current_role"].fillna("Unknown").value_counts().head(10).to_dict(),
    }
    summary_path = PROCESSED_DATA_DIR / "phase1_candidate_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nCandidate EDA complete.")
    print(f"Saved summary to: {summary_path}")
    print("Saved figures:")
    print(f"  - {FIGURES_DIR / 'candidate_experience_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'candidate_skill_count_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'candidate_country_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'candidate_last_active_days.png'}")


if __name__ == "__main__":
    main()
