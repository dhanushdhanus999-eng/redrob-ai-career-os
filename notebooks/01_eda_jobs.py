"""Phase 1 Day 2: exploratory analysis for job data."""

from __future__ import annotations

import ast
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.discovery import discover_dataset_file, find_columns_with_keywords
from src.data.io import load_table
from src.utils.paths import FIGURES_DIR, ensure_project_dirs


def parse_skills(value: object) -> list[str]:
    """Parse a skill payload stored as a list, JSON-ish string, or comma list."""
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


def choose_text_column(df: pd.DataFrame) -> str | None:
    """Pick the most likely descriptive text column."""
    text_candidates = find_columns_with_keywords(
        df.columns,
        keywords=("description", "job_description", "jd", "summary", "details"),
    )
    return text_candidates[0] if text_candidates else None


def choose_skills_column(df: pd.DataFrame) -> str | None:
    """Pick the most likely skills column."""
    skill_candidates = find_columns_with_keywords(
        df.columns,
        keywords=("skill", "skills", "required_skill", "must_have"),
    )
    return skill_candidates[0] if skill_candidates else None


def main() -> None:
    ensure_project_dirs()
    try:
        jobs_path = discover_dataset_file("jobs")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nCopy the official jobs file into data/raw and rerun.") from exc

    jobs = load_table(jobs_path)
    print(f"Loaded jobs dataset from: {jobs_path}")
    print(f"Shape: {jobs.shape}")
    print(f"Columns: {jobs.columns.tolist()}")
    print("\nDtypes:")
    print(jobs.dtypes)

    if jobs.empty:
        raise SystemExit("Jobs dataset is empty.")

    plt.figure(figsize=(12, 4))
    sns.heatmap(jobs.isna(), yticklabels=False, cbar=False, cmap="YlOrRd")
    plt.title("Jobs missingness heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "jobs_missingness.png")
    plt.close()

    text_col = choose_text_column(jobs)
    if text_col:
        word_counts = jobs[text_col].fillna("").astype(str).str.split().map(len)
        print(f"\nPrimary text column: {text_col}")
        print(word_counts.describe())

        plt.figure(figsize=(10, 4))
        word_counts.hist(bins=40, color="#33658A")
        plt.title(f"Word count distribution for {text_col}")
        plt.xlabel("Words")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "jobs_text_length.png")
        plt.close()

    categorical_columns = [
        column
        for column in jobs.select_dtypes(include="object").columns
        if jobs[column].nunique(dropna=True) <= 20
    ]
    for column in categorical_columns[:4]:
        print(f"\nTop values for {column}:")
        print(jobs[column].value_counts(dropna=False).head(10))

    skills_col = choose_skills_column(jobs)
    if skills_col:
        parsed_skills = jobs[skills_col].apply(parse_skills)
        jobs["__skill_count__"] = parsed_skills.map(len)
        skill_counter = Counter(skill for skills in parsed_skills for skill in skills)
        print(f"\nSkills column: {skills_col}")
        print(f"Average skills per job: {jobs['__skill_count__'].mean():.2f}")
        print(f"Unique skills: {len(skill_counter)}")
        print("Top skills:")
        print(pd.Series(dict(skill_counter.most_common(20))))

        top_skills = pd.Series(dict(skill_counter.most_common(20)))
        plt.figure(figsize=(12, 5))
        top_skills.plot(kind="bar", color="#2F4858")
        plt.title("Top 20 job skills")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "jobs_top_skills.png")
        plt.close()

    numeric_columns = find_columns_with_keywords(
        jobs.columns,
        keywords=("experience", "years", "salary", "ctc"),
    )
    for column in numeric_columns[:4]:
        series = pd.to_numeric(jobs[column], errors="coerce").dropna()
        if series.empty:
            continue
        print(f"\nNumeric summary for {column}:")
        print(series.describe())

    print("\nJobs EDA complete. Review the saved figures in outputs/figures.")
    print("Update docs/dataset_notes.md with the confirmed findings after review.")


if __name__ == "__main__":
    main()
