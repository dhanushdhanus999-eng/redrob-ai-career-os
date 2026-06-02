"""Phase 4 Day 26: fine-tune embeddings on labeled job-candidate pairs when available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import load_phase3_context, load_split_frame
from src.models.finetune_embeddings import create_training_pairs, finetune
from src.utils.paths import MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-name", default="finetuned_embeddings")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-triplets", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        context = load_phase3_context(require_labels=True)
        train_df = load_split_frame("train", context)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nEmbedding fine-tuning is blocked until organizer labels are available locally."
        ) from exc

    pairs = create_training_pairs(
        train_df=train_df,
        jobs_df=context.bundle.jobs,
        candidates_df=context.bundle.candidates,
        parsed_jds=context.parsed_jobs,
        parsed_cands=context.parsed_candidates,
    )
    print(f"Created {len(pairs)} training triplets")
    if len(pairs) < args.min_triplets:
        print("Not enough pairs for meaningful fine-tuning; keeping the base embedding model.")
        return

    output_path = MODELS_DIR / args.output_name
    finetune(
        pairs,
        base_model=args.base_model,
        output_path=str(output_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Saved fine-tuned embedding model to: {output_path}")


if __name__ == "__main__":
    main()
