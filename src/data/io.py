"""Tabular dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a supported tabular file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")
