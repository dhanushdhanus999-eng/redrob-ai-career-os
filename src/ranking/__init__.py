"""Phase 3 ranking and reranking helpers.

``explainer`` and ``llm_reranker`` are light; ``CrossEncoderReranker`` pulls in
sentence-transformers / torch, so it is loaded lazily (PEP 562) to keep
``import src.ranking`` cheap for CPU-only contexts (e.g. the hosted demo).
"""

from src.ranking.explainer import add_explanations_to_submission, explain_ranking
from src.ranking.llm_reranker import LLMReranker

__all__ = [
    "CrossEncoderReranker",
    "LLMReranker",
    "add_explanations_to_submission",
    "explain_ranking",
]


def __getattr__(name: str):  # PEP 562 lazy import
    if name == "CrossEncoderReranker":
        from src.ranking.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
