"""Targeted fixes discovered during Phase 4 error analysis."""

from __future__ import annotations


ADDITIONAL_SKILL_SYNONYMS = {
    "generative ai": "Generative AI",
    "genai": "Generative AI",
    "llms": "Large Language Models",
    "rag": "Retrieval Augmented Generation",
}


def profile_confidence(completeness: float | int | str | None) -> float:
    """Return confidence in a candidate score based on profile completeness."""
    try:
        value = float(completeness or 0.0)
    except (TypeError, ValueError):
        value = 0.0

    if value >= 0.8:
        return 1.0
    if value >= 0.5:
        return 0.7
    if value >= 0.3:
        return 0.4
    return 0.2
