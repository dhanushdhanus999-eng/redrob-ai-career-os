"""Semantic similarity features between jobs and candidate profiles."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.utils.batching import batch_encode
from src.utils.cache import EmbeddingCache
from src.utils.text_utils import clean_text


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return clean_text(value)


def _cosine_from_normalized(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    score = float(np.dot(left, right))
    return float(np.clip(score, -1.0, 1.0))


class SemanticFeatureExtractor:
    """Compute semantic similarity features using one or more embedding models."""

    MODEL_CONFIGS: Mapping[str, str] = {
        "bge_large": "BAAI/bge-large-en-v1.5",
        "e5_large": "intfloat/e5-large-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
    }
    FIELD_NAMES = (
        "job_text",
        "candidate_text",
        "job_title_text",
        "job_skills_text",
        "candidate_skills_text",
    )

    def __init__(
        self,
        model_keys: list[str] | tuple[str, ...] | None = None,
        *,
        device: str = "cpu",
        models: Mapping[str, object] | None = None,
        batch_size: int = 128,
        cache_embeddings: bool = True,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        if model_keys is None:
            model_keys = tuple(models.keys()) if models is not None else ("bge_large", "mpnet")

        self.model_keys = tuple(model_keys)
        self.device = device
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.models: dict[str, object] = {}

        if models is not None:
            for key in self.model_keys:
                if key not in models:
                    raise KeyError(f"Missing custom semantic model for key: {key}")
                self.models[key] = models[key]
            return

        from sentence_transformers import SentenceTransformer

        for key in self.model_keys:
            if key not in self.MODEL_CONFIGS:
                raise KeyError(f"Unknown semantic model key: {key}")
            self.models[key] = SentenceTransformer(self.MODEL_CONFIGS[key], device=device)

    def _encode(self, model_key: str, texts: list[str]) -> np.ndarray:
        model = self.models[model_key]
        return batch_encode(
            model,
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _embedding_lookup(self, model_key: str, texts: list[str]) -> dict[str, np.ndarray]:
        ordered_unique = list(dict.fromkeys(_clean_text(text) for text in texts))
        if not ordered_unique:
            ordered_unique = [""]
        lookup: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in ordered_unique:
            if not self.cache_embeddings:
                missing.append(text)
                continue
            cached_embedding = self.embedding_cache.get(text, model_key)
            if cached_embedding is None:
                missing.append(text)
            else:
                lookup[text] = np.asarray(cached_embedding, dtype=np.float32)

        if missing:
            encoded = self._encode(model_key, missing)
            for text, embedding in zip(missing, encoded, strict=False):
                array = np.asarray(embedding, dtype=np.float32)
                if self.cache_embeddings:
                    self.embedding_cache.set(text, model_key, array)
                lookup[text] = array
        return lookup

    def extract_for_pair(
        self,
        *,
        job_text: str,
        candidate_text: str,
        job_title_text: str = "",
        job_skills_text: str = "",
        candidate_skills_text: str = "",
    ) -> dict[str, float]:
        """Extract semantic features for a single job-candidate pair."""
        frame = self.extract_batch(
            [
                {
                    "job_text": job_text,
                    "candidate_text": candidate_text,
                    "job_title_text": job_title_text,
                    "job_skills_text": job_skills_text,
                    "candidate_skills_text": candidate_skills_text,
                }
            ]
        )
        return frame.iloc[0].to_dict() if not frame.empty else {}

    def extract_batch(self, pairs: list[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
        """Extract semantic features for a batch of pairs."""
        df = pd.DataFrame(pairs).copy()
        if df.empty:
            return pd.DataFrame()

        for field in self.FIELD_NAMES:
            if field not in df.columns:
                df[field] = ""
            df[field] = df[field].map(_clean_text)

        id_columns = [column for column in ("job_id", "candidate_id") if column in df.columns]
        feature_df = df[id_columns].copy()

        for model_key in self.model_keys:
            lookups = {
                field_name: self._embedding_lookup(model_key, df[field_name].tolist())
                for field_name in self.FIELD_NAMES
            }
            feature_df[f"{model_key}_full_sim"] = [
                _cosine_from_normalized(
                    lookups["job_text"][job_text],
                    lookups["candidate_text"][candidate_text],
                )
                for job_text, candidate_text in zip(
                    df["job_text"],
                    df["candidate_text"],
                    strict=False,
                )
            ]
            feature_df[f"{model_key}_title_cand_sim"] = [
                _cosine_from_normalized(
                    lookups["job_title_text"][job_title_text],
                    lookups["candidate_text"][candidate_text],
                )
                for job_title_text, candidate_text in zip(
                    df["job_title_text"],
                    df["candidate_text"],
                    strict=False,
                )
            ]
            feature_df[f"{model_key}_skills_sim"] = [
                _cosine_from_normalized(
                    lookups["job_skills_text"][job_skills_text],
                    lookups["candidate_skills_text"][candidate_skills_text],
                )
                for job_skills_text, candidate_skills_text in zip(
                    df["job_skills_text"],
                    df["candidate_skills_text"],
                    strict=False,
                )
            ]

        return feature_df
