"""Structured parsing helpers for candidate profiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.data.schema import CandidateTableSchema, combine_text_values
from src.utils.paths import CACHE_DIR


_CACHE_DIR = CACHE_DIR / "candidate_parsed"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _split_skills(value: str) -> list[str]:
    parts = re.split(r"[,\n;|]+", value)
    seen: set[str] = set()
    skills: list[str] = []
    for part in parts:
        cleaned = part.strip(" .:-")
        if len(cleaned) < 2:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        skills.append(cleaned)
    return skills


def extract_years_experience(text: str) -> float | None:
    """Extract a single years-of-experience estimate from free text."""
    patterns = (
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*yrs?\s+exp",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def normalize_seniority(title: str) -> str:
    """Map a role title to a coarse seniority bucket."""
    lowered = title.lower()
    if any(keyword in lowered for keyword in ("intern", "trainee", "fresher", "graduate")):
        return "intern"
    if any(keyword in lowered for keyword in ("junior", "associate", "jr")):
        return "junior"
    if any(keyword in lowered for keyword in ("director", "vp", "head", "chief", "cto", "ceo")):
        return "director"
    if any(keyword in lowered for keyword in ("principal", "staff")):
        return "principal"
    if any(keyword in lowered for keyword in ("lead", "manager")):
        return "lead"
    if any(keyword in lowered for keyword in ("senior", "sr")):
        return "senior"
    if lowered:
        return "mid"
    return "unknown"


def _merge_non_empty(base: dict, incoming: Mapping[str, object]) -> dict:
    merged = dict(base)
    for key, value in incoming.items():
        if value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


class CandidateProfileParser:
    """Rule-based candidate parser with optional LLM fallback and caching."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, identifier: str) -> Path:
        digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def parse_row(
        self,
        row: pd.Series,
        schema: CandidateTableSchema,
        *,
        llm_parser: Callable[[str], Mapping[str, object]] | None = None,
    ) -> dict:
        """Parse a candidate row into a structured dictionary."""
        candidate_id = str(row[schema.candidate_id])
        profile_text = combine_text_values(row, schema.text_columns)
        cache_path = self._cache_path(f"{candidate_id}::{profile_text}")
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        skill_values = []
        for column in schema.skill_columns:
            skill_values.extend(_split_skills(_clean_text(row.get(column))))

        current_role = _clean_text(row.get(schema.current_role)) if schema.current_role else ""
        headline = _clean_text(row.get(schema.headline)) if schema.headline else ""
        summary = _clean_text(row.get(schema.summary)) if schema.summary else ""

        total_years = None
        if schema.total_experience:
            raw_experience = _clean_text(row.get(schema.total_experience))
            if raw_experience:
                try:
                    total_years = float(raw_experience)
                except ValueError:
                    total_years = extract_years_experience(raw_experience)
        if total_years is None:
            total_years = extract_years_experience(f"{headline} {summary}")

        completeness_fields = [
            current_role,
            headline,
            summary,
            ", ".join(skill_values),
            _clean_text(row.get(schema.education)) if schema.education else "",
            _clean_text(row.get(schema.location)) if schema.location else "",
            _clean_text(row.get(schema.last_active)) if schema.last_active else "",
        ]
        non_empty_count = sum(bool(value) for value in completeness_fields)
        profile_completeness = non_empty_count / len(completeness_fields) if completeness_fields else 0.0

        parsed = {
            "candidate_id": candidate_id,
            "skills": skill_values,
            "current_role": current_role or headline,
            "headline": headline,
            "seniority": normalize_seniority(f"{current_role} {headline}".strip()),
            "total_experience_years": total_years,
            "location": _clean_text(row.get(schema.location)) if schema.location else "",
            "education": _clean_text(row.get(schema.education)) if schema.education else "",
            "last_active": _clean_text(row.get(schema.last_active)) if schema.last_active else "",
            "profile_completeness": round(profile_completeness, 4),
            "summary": summary,
        }

        if llm_parser is not None and (not parsed["skills"] or parsed["total_experience_years"] is None):
            parsed = _merge_non_empty(parsed, llm_parser(profile_text))

        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(parsed, handle, indent=2, default=str)
        return parsed

    def parse_frame(
        self,
        candidates_df: pd.DataFrame,
        schema: CandidateTableSchema,
        *,
        llm_parser: Callable[[str], Mapping[str, object]] | None = None,
    ) -> dict[str, dict]:
        """Parse an entire candidate dataframe into a mapping keyed by candidate_id."""
        parsed_candidates: dict[str, dict] = {}
        for _, row in candidates_df.iterrows():
            parsed = self.parse_row(row, schema, llm_parser=llm_parser)
            parsed_candidates[parsed["candidate_id"]] = parsed
        return parsed_candidates
