"""BM25 lexical retrieval for candidate ranking baselines."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils.text_utils import clean_text

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    BM25Okapi = None  # type: ignore[assignment]


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./-]*")


class _FallbackBM25:
    """Small BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = [list(tokens) for tokens in corpus_tokens]
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.asarray([len(tokens) for tokens in self.corpus_tokens], dtype=float)
        self.avgdl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.term_frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}

        for tokens in self.corpus_tokens:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequencies.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.idf = {
            token: np.log(1 + (len(self.corpus_tokens) - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus_tokens), dtype=float)
        if self.avgdl == 0:
            return scores

        for token in query_tokens:
            token_idf = self.idf.get(token, 0.0)
            if token_idf == 0.0:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[index] / self.avgdl
                )
                scores[index] += token_idf * (term_frequency * (self.k1 + 1)) / denominator
        return scores


def tokenize(text: str) -> list[str]:
    """Tokenise free text into lightly normalised BM25 terms."""
    normalized = clean_text(text)
    if not normalized:
        return []

    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(normalized.lower()):
        token = raw_token.strip("._-")
        if re.fullmatch(r"\d+(?:[./-]\d+)*", token):
            continue
        if token and token not in ENGLISH_STOP_WORDS:
            tokens.append(token)
    return tokens


class BM25Retriever:
    """Simple BM25 retriever over candidate documents."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.candidate_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []

    def build_index(self, documents: Sequence[str], candidate_ids: Sequence[str]) -> None:
        """Build a BM25 index from documents and their candidate IDs."""
        if len(documents) != len(candidate_ids):
            raise ValueError("documents and candidate_ids must have the same length.")
        if not documents:
            raise ValueError("At least one candidate document is required.")

        self.candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
        self.corpus_tokens = [tokenize(str(document)) for document in documents]
        self.bm25 = self._build_bm25(self.corpus_tokens)

    def build_from_frame(
        self,
        candidates_df: pd.DataFrame,
        text_cols: Sequence[str],
        id_col: str = "candidate_id",
    ) -> None:
        """Build the index directly from a candidate dataframe."""
        documents: list[str] = []
        candidate_ids = candidates_df[id_col].astype(str).tolist()
        for _, row in candidates_df.iterrows():
            parts = [clean_text(row.get(column)) for column in text_cols if pd.notna(row.get(column))]
            documents.append(" ".join(part for part in parts if part))
        self.build_index(documents=documents, candidate_ids=candidate_ids)

    def retrieve(self, query: str, top_k: int = 500) -> list[tuple[str, float]]:
        """Return the top-k candidate IDs with BM25 scores for a query."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built yet.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        if scores.size == 0:
            return []

        limit = min(max(int(top_k), 1), len(scores))
        top_indices = np.argsort(-scores, kind="mergesort")[:limit]
        return [(self.candidate_ids[index], float(scores[index])) for index in top_indices]

    def save(self, path: str | Path) -> None:
        """Persist the tokenised corpus so the BM25 index can be rebuilt later."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "candidate_ids": self.candidate_ids,
                    "corpus_tokens": self.corpus_tokens,
                },
                handle,
            )

    def load(self, path: str | Path) -> None:
        """Load a persisted BM25 corpus and rebuild the index."""
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        self.candidate_ids = [str(candidate_id) for candidate_id in payload["candidate_ids"]]
        self.corpus_tokens = [list(tokens) for tokens in payload["corpus_tokens"]]
        self.bm25 = self._build_bm25(self.corpus_tokens)

    @staticmethod
    def _build_bm25(corpus_tokens: Sequence[Sequence[str]]) -> BM25Okapi | _FallbackBM25:
        if BM25Okapi is not None:
            return BM25Okapi(corpus_tokens)
        return _FallbackBM25(corpus_tokens)
