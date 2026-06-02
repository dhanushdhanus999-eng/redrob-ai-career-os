"""Disk-based caching helpers for expensive computations."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.utils.paths import CACHE_DIR


def cache_key(*parts: object) -> str:
    """Create a deterministic cache key for arbitrary JSON-safe inputs."""
    payload = json.dumps(parts, default=str, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def cached(
    namespace: str,
    key: str,
    fn: Callable[[], Any],
    *,
    force_recompute: bool = False,
) -> Any:
    """Load a cached pickle result or compute and persist it."""
    path = CACHE_DIR / namespace / f"{key}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    if not force_recompute and path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    result = fn()
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return result


class EmbeddingCache:
    """Small helper around per-text embedding arrays stored as `.npy` files."""

    def __init__(self, namespace: str = "embeddings") -> None:
        self.dir = CACHE_DIR / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str, model_key: str) -> Path:
        return self.dir / f"{cache_key(model_key, text)}.npy"

    def get(self, text: str, model_key: str) -> np.ndarray | None:
        path = self._path(text, model_key)
        if not path.exists():
            return None
        return np.load(path)

    def set(self, text: str, model_key: str, embedding: np.ndarray) -> np.ndarray:
        path = self._path(text, model_key)
        array = np.asarray(embedding, dtype=np.float32)
        np.save(path, array)
        return array

    def get_or_compute(
        self,
        text: str,
        model_key: str,
        encode_fn: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        cached_embedding = self.get(text, model_key)
        if cached_embedding is not None:
            return cached_embedding
        return self.set(text, model_key, encode_fn(text))
