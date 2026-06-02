"""Phase 3 Day 14: behavioral feature generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import (
    build_pair_inputs,
    generate_feature_frames,
    load_phase3_context,
    load_split_frame,
    save_feature_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--refresh-parsed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True, refresh_parsed=args.refresh_parsed)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs, candidates, and labels files into data/raw and rerun."
        ) from exc

    for split in args.splits:
        labels_df = load_split_frame(split, context)
        pair_inputs = build_pair_inputs(labels_df, context)
        feature_df = generate_feature_frames(
            pair_inputs,
            feature_names=["features_behavioral"],
        )["features_behavioral"]
        output_path = save_feature_frame(feature_df, "features_behavioral", split)
        print(f"[{split}] saved {len(feature_df)} behavioral rows to {output_path}")


if __name__ == "__main__":
    main()
