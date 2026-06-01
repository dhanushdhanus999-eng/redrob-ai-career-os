"""Phase 1 Day 3: exploratory analysis for candidate data."""

from __future__ import annotations

import ast
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from src.data.discovery import discover_dataset_file, find_columns_with_keywords
from src.data.io import load_table
from src.utils.paths import FIGURES_DIR, ensure_project_dirs


def parse_skills(value: object) -> list[str]:
    """Parse a skill payload stored as a list-like string or comma list."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    ensure_project_dirs()
    try:
        candidates_path = discover_dataset_file("candidates")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nCopy the official candidate file into data/raw and rerun."
        ) from exc

    candidates = load_table(candidates_path)
    print(f"Loaded candidates dataset from: {candidates_path}")
    print(f"Shape: {candidates.shape}")
    print(f"Columns: {candidates.columns.tolist()}")

    if candidates.empty:
        raise SystemExit("Candidates dataset is empty.")

    completeness_score = candidates.notna().mean(axis=1)
    print("\nCompleteness score summary:")
    print(completeness_score.describe())

    plt.figure(figsize=(10, 4))
    completeness_score.hist(bins=25, color="#55A630")
    plt.title("Candidate profile completeness")
    plt.xlabel("Fraction of non-null fields")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_completeness.png")
    plt.close()

    skills_columns = find_columns_with_keywords(candidates.columns, keywords=("skill", "skills"))
    if skills_columns:
        skills_col = skills_columns[0]
        parsed_skills = candidates[skills_col].apply(parse_skills)
        skill_counter = Counter(skill for skills in parsed_skills for skill in skills)
        print(f"\nSkills column: {skills_col}")
        print(f"Average skills per candidate: {parsed_skills.map(len).mean():.2f}")
        print("Top candidate skills:")
        print(pd.Series(dict(skill_counter.most_common(20))))

    experience_columns = find_columns_with_keywords(
        candidates.columns,
        keywords=("experience", "years", "exp"),
    )
    for column in experience_columns[:4]:
        series = pd.to_numeric(candidates[column], errors="coerce").dropna()
        if series.empty:
            continue
        print(f"\nExperience summary for {column}:")
        print(series.describe())

        plt.figure(figsize=(10, 4))
        series.clip(upper=series.quantile(0.98)).hist(bins=30, color="#80B918")
        plt.title(f"Distribution for {column}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"candidate_{column}_distribution.png")
        plt.close()

    activity_columns = find_columns_with_keywords(
        candidates.columns,
        keywords=("date", "time", "last", "active", "updated", "applied", "login", "click"),
    )
    print("\nPotential activity or behavioral columns:")
    print(activity_columns if activity_columns else "None detected automatically.")

    for column in activity_columns[:5]:
        parsed = pd.to_datetime(candidates[column], errors="coerce")
        if parsed.notna().sum() == 0:
            continue
        days_ago = (pd.Timestamp.utcnow().tz_localize(None) - parsed).dt.days
        print(f"\nRecency summary for {column}:")
        print(days_ago.describe())

    print("\nCandidate EDA complete. Review figures in outputs/figures.")
    print("Capture confirmed behavioral signals in docs/dataset_notes.md and Memory.")


if __name__ == "__main__":
    main()
