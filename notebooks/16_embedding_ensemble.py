"""Phase 3 Day 16: extend semantic features with another embedding model."""

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
    load_feature_frame,
    load_phase3_context,
    load_split_frame,
    save_feature_frame,
)
from src.features.semantic_features import SemanticFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--model", default="e5_large")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nCopy the released jobs, candidates, and labels files into data/raw and rerun."
        ) from exc

    extractor = SemanticFeatureExtractor(model_keys=[args.model], device=args.device)
    for split in args.splits:
        try:
            existing = load_feature_frame("features_semantic", split)
        except FileNotFoundError:
            existing = None

        labels_df = load_split_frame(split, context)
        pair_inputs = build_pair_inputs(labels_df, context)
        new_features = generate_feature_frames(
            pair_inputs,
            feature_names=["features_semantic"],
            semantic_extractor=extractor,
        )["features_semantic"]

        merged = new_features if existing is None else existing.merge(
            new_features,
            on=["job_id", "candidate_id"],
            how="left",
            suffixes=("", "_new"),
        )
        duplicate_new_columns = [column for column in merged.columns if column.endswith("_new")]
        for column in duplicate_new_columns:
            base_column = column[:-4]
            merged[base_column] = merged[column].fillna(merged.get(base_column))
        merged = merged.drop(columns=duplicate_new_columns)

        output_path = save_feature_frame(merged, "features_semantic", split)
        sim_columns = [column for column in merged.columns if column.endswith("_sim")]
        print(f"[{split}] saved {len(merged)} semantic rows to {output_path}")
        if sim_columns:
            print(merged[sim_columns].corr().round(3))


if __name__ == "__main__":
    main()
