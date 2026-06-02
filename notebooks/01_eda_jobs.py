"""Phase 1 Day 2: inspect the released single-job challenge brief."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.challenge_bundle import (
    discover_challenge_bundle,
    parse_job_description,
    read_docx_text,
    save_canonical_job_dataset,
)
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def _section_word_counts(text: str) -> pd.Series:
    markers = [
        "Let's be honest about this role",
        "What you'd actually be doing",
        'What we mean by "5-9 years"',
        "Things you absolutely need",
        "Things we'd like you to have but won't reject you for",
        "Things we explicitly do NOT want",
        "On location, comp, and logistics",
        "The vibe check",
        "How to read between the lines",
        "Final note for the participants of the Redrob hackathon",
    ]
    sections: dict[str, int] = {}
    current = "Preamble"
    words: list[str] = []

    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        if line in markers:
            sections[current] = len(words)
            current = line
            words = []
            continue
        words.extend(line.split())

    sections[current] = len(words)
    return pd.Series(sections).sort_values(ascending=False)


def _extract_skill_terms(skills_text: str) -> list[str]:
    skill_counter = Counter()
    for raw_line in skills_text.splitlines():
        line = raw_line.strip(" -•\t")
        if not line or ":" in line:
            continue
        for token in line.split(","):
            cleaned = token.strip()
            if cleaned:
                skill_counter[cleaned] += 1
    return [skill for skill, _ in skill_counter.most_common()]


def main() -> None:
    ensure_project_dirs()
    bundle = discover_challenge_bundle()
    job_text = read_docx_text(bundle.job_description)
    job_row = parse_job_description(job_text)

    output_path = save_canonical_job_dataset(bundle)

    print(f"Challenge bundle root: {bundle.root}")
    print(f"Saved canonical jobs dataset to: {output_path}")
    print("\nReleased job row:")
    print(json.dumps(job_row, indent=2, default=str)[:5000])

    word_counts = _section_word_counts(job_text)
    plt.figure(figsize=(12, 5))
    word_counts.sort_values().plot(kind="barh", color="#2F4858")
    plt.title("Job-description section lengths")
    plt.xlabel("Words")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "job_description_section_lengths.png")
    plt.close()

    summary = {
        "job_dataset_path": str(output_path),
        "job_id": job_row["job_id"],
        "job_title": job_row["job_title"],
        "company": job_row["company"],
        "location": job_row["location"],
        "employment_type": job_row["employment_type"],
        "min_experience": job_row["min_experience"],
        "max_experience": job_row["max_experience"],
        "word_count": len(job_text.split()),
        "top_sections_by_length": word_counts.head(5).to_dict(),
        "must_have_keywords": _extract_skill_terms(str(job_row["required_skills"]))[:10],
        "nice_to_have_keywords": _extract_skill_terms(str(job_row["preferred_skills"]))[:10],
    }
    summary_path = PROCESSED_DATA_DIR / "phase1_job_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nJob EDA complete.")
    print(f"Saved section-length figure to: {FIGURES_DIR / 'job_description_section_lengths.png'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
