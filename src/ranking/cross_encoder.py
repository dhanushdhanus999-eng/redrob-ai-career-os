"""Cross-encoder reranking helpers for the final shortlist."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class CrossEncoderReranker:
    """Apply a cross-encoder to rerank a small candidate shortlist."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        *,
        device: str = "cpu",
        model: object | None = None,
    ) -> None:
        if model is not None:
            self.model = model
            self.model_name = model_name
            return

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        job_text: str,
        candidates: list[tuple[str, str]] | Iterable[tuple[str, str]],
        *,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Return candidates sorted by cross-encoder score, best first."""
        candidate_list = list(candidates)
        if not candidate_list:
            return []

        pairs = [[job_text, candidate_text] for _, candidate_text in candidate_list]
        try:
            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(
                [candidate_id for candidate_id, _ in candidate_list],
                np.asarray(scores, dtype=float).tolist(),
                strict=False,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
