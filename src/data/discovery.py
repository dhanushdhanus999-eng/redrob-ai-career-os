"""Helpers for locating dataset files and likely schema columns."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.utils.paths import EXTERNAL_DATA_DIR, RAW_DATA_DIR

DATASET_PATTERNS = {
    "jobs": (
        "*job*.csv",
        "*job*.parquet",
        "*job*.json",
        "*jobs*.xlsx",
    ),
    "candidates": (
        "*candidate*.csv",
        "*candidate*.parquet",
        "*candidate*.json",
        "*profile*.csv",
        "*resume*.csv",
    ),
    "labels": (
        "*label*.csv",
        "*ground*truth*.csv",
        "*relevance*.csv",
        "*qrels*.csv",
        "*train*.csv",
    ),
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def discover_dataset_files(
    dataset_kind: str,
    search_dirs: Iterable[Path] | None = None,
) -> list[Path]:
    """Return all matching dataset files for a given kind."""
    patterns = DATASET_PATTERNS.get(dataset_kind)
    if patterns is None:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    roots = tuple(search_dirs or (RAW_DATA_DIR, EXTERNAL_DATA_DIR))
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
    return list(dict.fromkeys(matches))


def discover_dataset_file(
    dataset_kind: str,
    search_dirs: Iterable[Path] | None = None,
) -> Path:
    """Return the best matching file for a given dataset kind."""
    matches = discover_dataset_files(dataset_kind=dataset_kind, search_dirs=search_dirs)
    if not matches:
        raise FileNotFoundError(
            f"No {dataset_kind} file found under data/raw or data/external yet."
        )
    return matches[0]


def find_columns_with_keywords(columns: Iterable[str], keywords: Iterable[str]) -> list[str]:
    """Return every column whose normalised name contains a keyword."""
    normalised = {column: _normalise_name(column) for column in columns}
    keyset = [_normalise_name(keyword) for keyword in keywords]
    return [
        column
        for column, clean_name in normalised.items()
        if any(keyword in clean_name for keyword in keyset)
    ]


def infer_column_name(
    columns: Iterable[str],
    aliases: Iterable[str],
    contains: Iterable[str] | None = None,
) -> str | None:
    """Infer a likely schema column from aliases and fallback keywords."""
    columns = list(columns)
    normalised_map = {_normalise_name(column): column for column in columns}

    for alias in aliases:
        found = normalised_map.get(_normalise_name(alias))
        if found is not None:
            return found

    if contains is None:
        contains = aliases

    matches = find_columns_with_keywords(columns, contains)
    return matches[0] if matches else None
