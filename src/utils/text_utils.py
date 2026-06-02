"""Text cleaning and normalisation helpers for Indian-context datasets."""

from __future__ import annotations

import re
import unicodedata


_NULLISH_VALUES = {"", "nan", "none", "null", "n/a", "na"}


def clean_text(text: object) -> str:
    """Normalize raw text while preserving multilingual content."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.lower() in _NULLISH_VALUES:
        return ""
    return normalized


def is_meaningful(text: object, min_words: int = 3) -> bool:
    """Return True when text has enough non-trivial tokens to be useful."""
    normalized = clean_text(text)
    if not normalized:
        return False
    words = [word for word in normalized.split() if len(word) > 1]
    return len(words) >= min_words


def truncate_to_tokens(text: object, max_tokens: int = 512) -> str:
    """Approximate token-aware truncation using a rough words-per-token ratio."""
    normalized = clean_text(text)
    if not normalized:
        return ""
    max_words = max(1, int(max_tokens * 0.75))
    return " ".join(normalized.split()[:max_words])
