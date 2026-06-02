"""Phase 3 Day 11: semantic feature generation."""

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
from src.features.semantic_features import SemanticFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--models", nargs="+", default=["bge_large", "mpnet"])
    parser.add_argument("--device", default="cpu")
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

    extractor = SemanticFeatureExtractor(model_keys=args.models, device=args.device)
    for split in args.splits:
        labels_df = load_split_frame(split, context)
        pair_inputs = build_pair_inputs(labels_df, context)
        feature_frames = generate_feature_frames(
            pair_inputs,
            feature_names=["features_semantic"],
            semantic_extractor=extractor,
        )
        feature_df = feature_frames["features_semantic"]
        output_path = save_feature_frame(feature_df, "features_semantic", split)
        print(f"[{split}] saved {len(feature_df)} semantic rows to {output_path}")


if __name__ == "__main__":
    main()
