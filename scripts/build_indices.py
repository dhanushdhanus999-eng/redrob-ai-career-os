"""Build retrieval indices from the canonical candidate dataset.

Builds the BM25 index used by the submission generator and the demo.
Run this once after processing the raw data, and again whenever the
candidate dataset changes.

Usage:
    python scripts/build_indices.py          # BM25 only (fast)
    python scripts/build_indices.py --dense  # BM25 + dense BGE-large (slow, GPU recommended)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_candidate_documents, load_phase2_bundle
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.paths import MODELS_DIR, ensure_project_dirs


def build_bm25(bundle) -> Path:
    out = MODELS_DIR / "bm25_demo_index.pkl"
    if out.exists():
        print(f"BM25 index already exists at {out}")
        print("  Delete it and rerun to rebuild.")
        return out

    print(f"Building BM25 index for {len(bundle.candidates):,} candidates…")
    t0 = time.perf_counter()
    ids, docs = build_candidate_documents(bundle)
    retriever = BM25Retriever()
    retriever.build_index(documents=docs, candidate_ids=ids)
    retriever.save(out)
    print(f"BM25 index saved to {out}  ({time.perf_counter()-t0:.1f}s)")
    return out


def build_dense(bundle) -> Path:
    out = MODELS_DIR / "dense_demo_index"
    meta = MODELS_DIR / "dense_demo_index.meta.pkl"
    if meta.exists():
        print(f"Dense index already exists at {out}")
        print("  Delete dense_demo_index.* files and rerun to rebuild.")
        return out

    try:
        from src.retrieval.dense_retriever import DenseRetriever
    except ImportError as exc:
        print(f"Dense retriever unavailable: {exc}")
        return out

    from src.data.schema import combine_text_values

    print(f"Building dense BGE-large index for {len(bundle.candidates):,} candidates…")
    print("This takes 10-30 minutes on CPU; much faster with a GPU.")
    t0 = time.perf_counter()

    cand_id_col = bundle.candidate_schema.candidate_id
    ids = bundle.candidates[cand_id_col].astype(str).tolist()
    docs = [
        combine_text_values(row, bundle.candidate_schema.text_columns)
        for _, row in bundle.candidates.iterrows()
    ]

    retriever = DenseRetriever(model_name="BAAI/bge-large-en-v1.5")
    retriever.build_index(documents=docs, candidate_ids=ids)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(out)
    print(f"Dense index saved to {out}.*  ({time.perf_counter()-t0:.1f}s)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dense", action="store_true",
        help="Also build the dense BGE-large embedding index (slow — GPU recommended)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing indices and rebuild from scratch",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    if args.force:
        bm25_path = MODELS_DIR / "bm25_demo_index.pkl"
        dense_path = MODELS_DIR / "dense_demo_index.meta.pkl"
        for p in (bm25_path, dense_path):
            if p.exists():
                import shutil
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                print(f"Deleted {p}")

    print("Loading candidate bundle…")
    bundle = load_phase2_bundle(require_labels=False)
    print(f"  {len(bundle.candidates):,} candidates, {len(bundle.jobs)} job(s)")

    build_bm25(bundle)

    if args.dense:
        build_dense(bundle)

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()
