"""Validate a public Track 1 submission file against the published contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.challenge_bundle import discover_challenge_bundle, load_candidate_id_set
from src.eval.submission import validate_track1_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Path to the submission CSV.")
    args = parser.parse_args()

    bundle = discover_challenge_bundle()
    submission = pd.read_csv(args.pred)
    issues = validate_track1_submission(
        submission,
        valid_candidate_ids=load_candidate_id_set(bundle),
    )

    if issues:
        print("Submission is invalid:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print(f"Submission is valid: {args.pred}")


if __name__ == "__main__":
    main()
