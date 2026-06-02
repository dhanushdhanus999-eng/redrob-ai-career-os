"""Utilities for efficient batched model inference."""

from __future__ import annotations

from typing import Iterator, Sequence, TypeVar


T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield successive fixed-size chunks from a sequence."""
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def batch_encode(
    model: object,
    texts: Sequence[str],
    *,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> "np.ndarray":
    """Encode texts in batches and return one stacked float32 matrix."""
    import numpy as np

    ordered = list(texts)
    if not ordered:
        return np.empty((0, 0), dtype=np.float32)

    all_embeddings = []
    for batch in chunked(ordered, batch_size):
        kwargs = {
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        }
        try:
            encoded = model.encode(batch, convert_to_numpy=True, **kwargs)
        except TypeError:
            encoded = model.encode(batch, **kwargs)
        all_embeddings.append(np.asarray(encoded, dtype=np.float32))
    return np.vstack(all_embeddings).astype(np.float32)
