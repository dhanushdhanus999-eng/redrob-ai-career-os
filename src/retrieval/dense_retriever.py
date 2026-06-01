"""Dense retrieval using sentence-transformers and FAISS."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

try:
    import faiss
except ModuleNotFoundError:
    faiss = None  # type: ignore[assignment]


class _NumpyInnerProductIndex:
    """Minimal inner-product index used when FAISS is unavailable."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Vector matrix has an unexpected shape.")
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
            indices = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return scores, indices

        similarities = queries @ self.vectors.T
        order = np.argsort(-similarities, axis=1, kind="mergesort")
        top_order = order[:, :k]
        top_scores = np.take_along_axis(similarities, top_order, axis=1)

        if top_order.shape[1] < k:
            pad = k - top_order.shape[1]
            top_scores = np.pad(top_scores, ((0, 0), (0, pad)), constant_values=-np.inf)
            top_order = np.pad(top_order, ((0, 0), (0, pad)), constant_values=-1)
        return top_scores.astype(np.float32), top_order.astype(np.int64)


class EncoderProtocol(Protocol):
    """Minimal interface required by the dense retriever."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class DenseRetriever:
    """Dense retriever backed by an arbitrary text encoder and a FAISS index."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        encoder: EncoderProtocol | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.encoder = encoder or SentenceTransformerEncoder(model_name=model_name, device=device)
        self.index: object | None = None
        self.candidate_ids: list[str] = []

    def build_index(
        self,
        documents: Sequence[str],
        candidate_ids: Sequence[str],
        batch_size: int = 64,
    ) -> None:
        """Encode the corpus and build an inner-product FAISS index."""
        if len(documents) != len(candidate_ids):
            raise ValueError("documents and candidate_ids must have the same length.")
        if not documents:
            raise ValueError("At least one candidate document is required.")

        embeddings = self.encoder.encode(
            list(documents),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if embeddings.ndim != 2:
            raise ValueError("The encoder must return a 2D embedding matrix.")

        self.candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
        self.index = self._new_index(int(embeddings.shape[1]))
        self.index.add(np.asarray(embeddings, dtype=np.float32))

    def build_from_frame(
        self,
        candidates_df: pd.DataFrame,
        text_cols: Sequence[str],
        id_col: str = "candidate_id",
        batch_size: int = 64,
    ) -> None:
        """Build the index directly from a candidate dataframe."""
        documents: list[str] = []
        candidate_ids = candidates_df[id_col].astype(str).tolist()
        for _, row in candidates_df.iterrows():
            parts = [str(row[column]).strip() for column in text_cols if pd.notna(row.get(column))]
            documents.append(" ".join(part for part in parts if part))
        self.build_index(documents=documents, candidate_ids=candidate_ids, batch_size=batch_size)

    def retrieve(self, query: str, top_k: int = 500) -> list[tuple[str, float]]:
        """Return the top-k dense retrieval results for a query."""
        if self.index is None:
            raise RuntimeError("Dense index has not been built yet.")
        if not isinstance(query, str) or not query.strip():
            return []

        query_embedding = self.encoder.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        limit = min(max(int(top_k), 1), self.index.ntotal)
        scores, indices = self.index.search(np.asarray(query_embedding, dtype=np.float32), limit)

        results: list[tuple[str, float]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            results.append((self.candidate_ids[int(index)], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index and candidate metadata to disk."""
        if self.index is None:
            raise RuntimeError("Dense index has not been built yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if faiss is not None and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, str(path) + ".faiss")
        else:
            with Path(str(path) + ".index.pkl").open("wb") as handle:
                pickle.dump({"vectors": self.index.vectors}, handle)
        with (Path(str(path) + ".meta.pkl")).open("wb") as handle:
            pickle.dump({"candidate_ids": self.candidate_ids}, handle)

    def load(self, path: str | Path) -> None:
        """Load a persisted FAISS index and candidate metadata."""
        path = Path(path)
        faiss_path = Path(str(path) + ".faiss")
        numpy_path = Path(str(path) + ".index.pkl")
        if faiss is not None and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        else:
            with numpy_path.open("rb") as handle:
                payload = pickle.load(handle)
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            self.index = self._new_index(int(vectors.shape[1]))
            self.index.add(vectors)
        with (Path(str(path) + ".meta.pkl")).open("rb") as handle:
            payload = pickle.load(handle)
        self.candidate_ids = [str(candidate_id) for candidate_id in payload["candidate_ids"]]

    @staticmethod
    def _new_index(dimension: int) -> object:
        if faiss is not None:
            return faiss.IndexFlatIP(dimension)
        return _NumpyInnerProductIndex(dimension)
