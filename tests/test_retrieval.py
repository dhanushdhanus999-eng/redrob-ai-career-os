"""Unit tests for Phase 2 retrieval modules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


class ToyEncoder:
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    float("python" in lowered),
                    float("java" in lowered),
                    float("data" in lowered),
                ],
                dtype=np.float32,
            )
            if not vector.any():
                vector += 1e-6
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class StaticRetriever:
    def __init__(self, mapping: dict[str, list[tuple[str, float]]]) -> None:
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int = 500) -> list[tuple[str, float]]:
        return self.mapping[query][:top_k]


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            "Python data pipelines and analytics",
            "Java spring microservices",
            "Excel operations and reporting",
        ]
        self.candidate_ids = ["C1", "C2", "C3"]

    def test_bm25_prefers_keyword_overlap(self) -> None:
        retriever = BM25Retriever()
        retriever.build_index(documents=self.documents, candidate_ids=self.candidate_ids)

        results = retriever.retrieve("python data", top_k=2)

        self.assertEqual(results[0][0], "C1")

    def test_dense_retriever_supports_custom_encoder(self) -> None:
        retriever = DenseRetriever(encoder=ToyEncoder())
        retriever.build_index(documents=self.documents, candidate_ids=self.candidate_ids)

        results = retriever.retrieve("python data", top_k=2)

        self.assertEqual(results[0][0], "C1")

    def test_dense_retriever_can_roundtrip_index(self) -> None:
        retriever = DenseRetriever(encoder=ToyEncoder())
        retriever.build_index(documents=self.documents, candidate_ids=self.candidate_ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dense_test"
            retriever.save(path)

            loaded = DenseRetriever(encoder=ToyEncoder())
            loaded.load(path)
            results = loaded.retrieve("java", top_k=1)

        self.assertEqual(results[0][0], "C2")

    def test_rrf_promotes_shared_candidates(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                [("C1", 0.9), ("C2", 0.8), ("C3", 0.7)],
                [("C2", 0.95), ("C4", 0.5), ("C1", 0.4)],
            ],
            k=10,
        )

        self.assertEqual(fused[0][0], "C2")

    def test_hybrid_retriever_uses_fused_order(self) -> None:
        hybrid = HybridRetriever(
            StaticRetriever({"query": [("C1", 1.0), ("C2", 0.5)]}),
            StaticRetriever({"query": [("C2", 0.9), ("C3", 0.4)]}),
            rrf_k=10,
        )

        results = hybrid.retrieve("query", top_k=3, recall_k=3)

        self.assertEqual(results[0][0], "C2")


if __name__ == "__main__":
    unittest.main()
