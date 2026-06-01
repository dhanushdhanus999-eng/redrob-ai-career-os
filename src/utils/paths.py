"""Project path helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
CACHE_DIR = OUTPUTS_DIR / "cache"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_project_dirs() -> None:
    """Create the main project directories if they do not exist yet."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        SUBMISSIONS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        FIGURES_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
