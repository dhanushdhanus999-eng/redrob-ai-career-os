"""Create pseudo-relevance labels for LTR training without organiser ground truth.

Strategy:
  Grade 2 (strong positive): AI/ML engineering title + 4-12 years experience
  Grade 1 (weak positive):   adjacent engineering title + 3-12 years experience
  Grade 0 (irrelevant):      explicitly excluded title OR < 2 years experience

These pseudo-labels let notebooks/15_train_ltr.py train LightGBM LambdaRank
on all 51+ engineered features rather than relying on hardcoded weight vectors.
The pseudo-labels are imperfect but directionally correct and vastly better than
a uniform score.

Usage:
    python scripts/create_pseudo_labels.py
    python scripts/create_pseudo_labels.py --preview   # show distribution only

After running this:
    python notebooks/04_create_splits_and_baseline.py  # create train/val splits
    python notebooks/11_semantic_features.py            # generate feature frames
    python notebooks/12_skill_features.py
    python notebooks/13_experience_features.py
    python notebooks/14_behavioral_features.py
    python notebooks/15_train_ltr.py                   # train LightGBM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import load_phase2_bundle
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs
from src.utils.role_relevance import NEGATIVE_ROLE_TOKENS

TRACK1_JOB_ID = "REDROB_TRACK1_MAIN_JD"

# Strong AI/ML titles → grade 2
STRONG_AI_TOKENS: frozenset[str] = frozenset({
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "data scientist",
    "research scientist",
    "applied scientist",
    "deep learning engineer",
    "nlp engineer",
    "computer vision engineer",
    "search engineer",
    "ranking engineer",
    "recommendation engineer",
    "mlops engineer",
    "ml platform engineer",
    "ai researcher",
    "ml researcher",
    "applied ml",
    "applied ai",
})

# Adjacent engineering titles → grade 1
ADJACENT_TOKENS: frozenset[str] = frozenset({
    "software engineer",
    "software developer",
    "backend engineer",
    "backend developer",
    "data engineer",
    "analytics engineer",
    "platform engineer",
    "infrastructure engineer",
    "full stack engineer",
    "research engineer",
    "scientist",
    "developer",
})


def _normalise(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(text).lower().strip())


def assign_relevance(row: pd.Series) -> int:
    role  = _normalise(row.get("current_role") or "")
    years = float(row.get("total_experience") or 0)

    # Immediately irrelevant — too little experience
    if years < 2:
        return 0

    # Hard negative roles
    if any(t in role for t in NEGATIVE_ROLE_TOKENS):
        return 0

    # Strong positive: AI/ML title in good experience range
    if any(t in role for t in STRONG_AI_TOKENS):
        return 2 if 4 <= years <= 12 else 1

    # Adjacent engineering title
    if any(t in role for t in ADJACENT_TOKENS):
        return 1 if 3 <= years <= 12 else 0

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true",
                        help="Show label distribution without writing the file")
    args = parser.parse_args()

    ensure_project_dirs()

    print("Loading candidate bundle…")
    bundle = load_phase2_bundle(require_labels=False)
    candidates = bundle.candidates.copy()
    cand_id_col = bundle.candidate_schema.candidate_id

    print(f"Assigning pseudo-labels to {len(candidates):,} candidates…")
    candidates["relevance"] = candidates.apply(assign_relevance, axis=1)
    candidates["job_id"]    = TRACK1_JOB_ID

    label_cols = ["job_id", cand_id_col, "relevance"]
    labels_df  = candidates[label_cols].rename(columns={cand_id_col: "candidate_id"})

    dist = labels_df["relevance"].value_counts().sort_index()
    print("\nLabel distribution:")
    for grade, count in dist.items():
        pct = count / len(labels_df) * 100
        label = {0: "irrelevant", 1: "weak positive", 2: "strong positive"}.get(int(grade), "?")
        print(f"  Grade {grade} ({label}): {count:,}  ({pct:.1f}%)")

    if args.preview:
        print("\n--preview: file not written.")
        return

    out = PROCESSED_DATA_DIR / "pseudo_labels.csv"
    labels_df.to_csv(out, index=False)
    print(f"\nPseudo labels saved to {out}")
    print("Next: run notebooks/04 through 15 to generate features and train LTR.")


if __name__ == "__main__":
    main()
