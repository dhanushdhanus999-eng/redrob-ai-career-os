"""Retrieval building blocks for Phase 2 baselines.

``BM25Retriever`` and the hybrid helpers are light; ``DenseRetriever`` pulls in
torch / sentence-transformers / faiss, so it is loaded lazily (PEP 562) to keep
``import src.retrieval`` cheap for CPU-only / network-free contexts (e.g. the
hosted demo and the submission path).
"""

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]


def __getattr__(name: str):  # PEP 562 lazy import
    if name == "DenseRetriever":
        from src.retrieval.dense_retriever import DenseRetriever

        return DenseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
