"""Hybrid retrieval by fusing lexical and dense rankings."""

from __future__ import annotations

from typing import Protocol, Sequence


class RetrieverProtocol(Protocol):
    """Protocol shared by the baseline retrievers."""

    def retrieve(self, query: str, top_k: int = 500) -> list[tuple[str, float]]: ...


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with standard Reciprocal Rank Fusion."""
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists.")

    weights = list(weights or [1.0] * len(ranked_lists))
    fused_scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights, strict=False):
        for rank, (candidate_id, _) in enumerate(ranked_list, start=1):
            fused_scores[candidate_id] = fused_scores.get(candidate_id, 0.0) + weight / (k + rank)

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Fuse BM25 and dense retrieval outputs with weighted RRF."""

    def __init__(
        self,
        bm25_retriever: RetrieverProtocol,
        dense_retriever: RetrieverProtocol,
        *,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.weights = weights

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 500,
        recall_k: int = 1000,
    ) -> list[tuple[str, float]]:
        """Retrieve from both systems, fuse them, and return the final top-k list."""
        recall_depth = max(int(top_k), int(recall_k))
        bm25_results = self.bm25_retriever.retrieve(query, top_k=recall_depth)
        dense_results = self.dense_retriever.retrieve(query, top_k=recall_depth)
        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=self.rrf_k,
            weights=self.weights,
        )
        return fused[:top_k]
