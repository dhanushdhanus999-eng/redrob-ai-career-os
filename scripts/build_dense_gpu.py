"""
Build Dense BGE-large Index — Standalone GPU Script
====================================================
Run this on any machine with a GPU (RTX 5090, Colab T4, etc.)

REQUIREMENTS (install once):
    pip install sentence-transformers faiss-cpu pyarrow pandas torch tqdm

INPUT  (place in same folder as this script):
    challenge_candidates.parquet

OUTPUT (copy both files to  outputs/models/  on your main machine):
    dense_demo_index.faiss
    dense_demo_index.meta.pkl

USAGE:
    python build_dense_gpu.py
    python build_dense_gpu.py --batch-size 256
    python build_dense_gpu.py --parquet path/to/challenge_candidates.parquet
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path,
                        default=Path(__file__).parent / "challenge_candidates.parquet")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size. 256 = safe for any GPU. Try 512 if fast.")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    # ── 0. Validate ──────────────────────────────────────────────────────────
    if not args.parquet.exists():
        raise FileNotFoundError(
            f"\nParquet not found: {args.parquet}\n"
            "Place challenge_candidates.parquet next to this script."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    faiss_out = args.out_dir / "dense_demo_index.faiss"
    meta_out  = args.out_dir / "dense_demo_index.meta.pkl"

    # ── 1. Load candidates ───────────────────────────────────────────────────
    log(f"\n[1/4] Loading {args.parquet.name} ...")
    import pandas as pd
    df = pd.read_parquet(args.parquet)
    log(f"      {len(df):,} rows  |  {len(df.columns)} columns")

    id_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("candidate_id", "cand_id", "candidateid"))),
        df.columns[0],
    )
    log(f"      ID column: {id_col}")

    TEXT_COLS_PRIORITY = [
        "current_role", "headline", "summary", "profile_text",
        "education", "skills", "experience_summary",
        "career_history_text", "bio", "about",
    ]
    text_cols = [c for c in TEXT_COLS_PRIORITY if c in df.columns]
    extra = [c for c in df.select_dtypes(include="object").columns
             if c not in text_cols and c != id_col]
    text_cols = text_cols + extra[:max(0, 12 - len(text_cols))]
    log(f"      Text columns ({len(text_cols)}): {text_cols}")

    # Vectorized document building — avoids slow iterrows
    log("      Building documents (vectorized) ...")
    candidate_ids = df[id_col].astype(str).tolist()
    parts_df = df[text_cols].fillna("").astype(str)
    # mask out "nan" / "none" / empty strings
    mask = parts_df.apply(lambda col: ~col.str.lower().isin({"nan", "none", ""}))
    documents = parts_df.where(mask, "").apply(
        lambda row: " ".join(v for v in row if v), axis=1
    ).tolist()
    log(f"      Sample: {documents[0][:120]}")
    log(f"      Avg doc length: {sum(len(d) for d in documents) // len(documents)} chars")

    # ── 2. Load model ────────────────────────────────────────────────────────
    log(f"\n[2/4] Loading model: {args.model} ...")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"      Device: cuda  ({torch.cuda.get_device_name(0)})")
        log(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log("      Device: cpu  (no GPU — will be slow)")

    model = SentenceTransformer(args.model, device=device)
    log("      Model loaded.")

    # ── 3. Encode — manual batch loop with explicit progress ─────────────────
    n       = len(documents)
    bs      = args.batch_size
    n_batch = math.ceil(n / bs)
    log(f"\n[3/4] Encoding {n:,} candidates  |  batch_size={bs}  |  {n_batch} batches")
    log("      Progress printed every batch.\n")

    import numpy as np
    all_vecs: list[np.ndarray] = []
    t0 = time.perf_counter()
    t_last = t0

    for i in range(n_batch):
        start = i * bs
        end   = min(start + bs, n)
        batch = documents[start:end]

        vecs = model.encode(
            batch,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,   # we handle progress ourselves
            convert_to_numpy=True,
        )
        all_vecs.append(np.asarray(vecs, dtype="float32"))

        # Print a clear progress line after every batch
        elapsed      = time.perf_counter() - t0
        batch_time   = time.perf_counter() - t_last
        pct          = (i + 1) / n_batch * 100
        done_cands   = end
        remaining    = elapsed / (i + 1) * (n_batch - i - 1)
        print(
            f"  Batch {i+1:3d}/{n_batch}  [{pct:5.1f}%]  "
            f"cands {done_cands:>7,}/{n:,}  "
            f"batch={batch_time:.1f}s  elapsed={elapsed:.0f}s  eta={remaining:.0f}s",
            flush=True,
        )
        t_last = time.perf_counter()

    embeddings = np.vstack(all_vecs).astype("float32")
    total_time = time.perf_counter() - t0
    log(f"\n      Encoding done in {total_time:.1f}s — shape: {embeddings.shape}")

    # ── 4. Build FAISS index and save ────────────────────────────────────────
    log(f"\n[4/4] Building FAISS IndexFlatIP and saving ...")
    import faiss

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log(f"      Vectors in index: {index.ntotal:,}  dim={dim}")

    faiss.write_index(index, str(faiss_out))
    log(f"      Saved: {faiss_out}  ({os.path.getsize(faiss_out) / 1e6:.1f} MB)")

    with open(meta_out, "wb") as fh:
        pickle.dump({"candidate_ids": candidate_ids, "index_type": "flat"}, fh)
    log(f"      Saved: {meta_out}  ({os.path.getsize(meta_out) / 1e6:.2f} MB)")

    # ── Done ─────────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("SUCCESS — copy these two files to your laptop:")
    log(f"  {faiss_out.name}")
    log(f"  {meta_out.name}")
    log("")
    log("Destination on laptop:")
    log(r"  c:\Projects\India Runs\outputs\models\dense_demo_index.faiss")
    log(r"  c:\Projects\India Runs\outputs\models\dense_demo_index.meta.pkl")
    log("")
    log("Then run:")
    log("  python scripts/generate_submission.py --recall-k 2000 --llm-rerank --validate")
    log("=" * 60)


if __name__ == "__main__":
    main()
