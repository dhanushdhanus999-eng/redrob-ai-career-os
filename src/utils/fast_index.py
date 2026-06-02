"""Helpers for building faster FAISS indexes for dense retrieval."""

from __future__ import annotations

import math

import numpy as np

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    faiss = None  # type: ignore[assignment]


def _require_faiss() -> None:
    if faiss is None:
        raise ModuleNotFoundError(
            "faiss-cpu is required for fast dense indexes. Install project dependencies first."
        )


def _ensure_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D float32 array.")
    return array


def build_flat_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """Build an exact inner-product index."""
    _require_faiss()
    vectors = _ensure_embeddings(embeddings)
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    return index


def build_hnsw_index(embeddings: np.ndarray, M: int = 32) -> "faiss.IndexHNSWFlat":
    """Build an approximate HNSW index for fast cosine / inner-product search."""
    _require_faiss()
    vectors = _ensure_embeddings(embeddings)
    index = faiss.IndexHNSWFlat(int(vectors.shape[1]), int(M), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(vectors)
    return index


def build_ivf_index(
    embeddings: np.ndarray,
    *,
    n_lists: int | None = None,
    nprobe: int = 32,
) -> "faiss.IndexIVFFlat":
    """Build an IVF flat index tuned for medium-to-large candidate pools."""
    _require_faiss()
    vectors = _ensure_embeddings(embeddings)
    n_vectors, dimension = vectors.shape
    if n_lists is None:
        n_lists = max(1, int(math.sqrt(n_vectors)))
    n_lists = max(1, min(int(n_lists), n_vectors))

    quantizer = faiss.IndexFlatIP(int(dimension))
    index = faiss.IndexIVFFlat(quantizer, int(dimension), n_lists, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(n_lists, int(nprobe))
    return index


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    index_type: str = "flat",
    hnsw_m: int = 32,
    ivf_n_lists: int | None = None,
    ivf_nprobe: int = 32,
) -> "faiss.Index":
    """Build one of the supported FAISS index types."""
    normalized_type = str(index_type).strip().lower()
    if normalized_type == "flat":
        return build_flat_index(embeddings)
    if normalized_type in {"hnsw", "hnsw_flat"}:
        return build_hnsw_index(embeddings, M=hnsw_m)
    if normalized_type == "ivf":
        return build_ivf_index(embeddings, n_lists=ivf_n_lists, nprobe=ivf_nprobe)
    raise ValueError(f"Unsupported FAISS index type: {index_type}")
