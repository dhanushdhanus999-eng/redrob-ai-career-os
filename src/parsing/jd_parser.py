"""Structured parsing helpers for job descriptions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.data.schema import JobTableSchema, combine_text_values
from src.utils.paths import CACHE_DIR


_SENIORITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intern", ("intern", "trainee", "fresher", "graduate")),
    ("junior", ("junior", "associate", "entry level", "jr")),
    ("mid", ("mid", "mid-level")),
    ("senior", ("senior", "sr", "staff")),
    ("lead", ("lead", "manager")),
    ("principal", ("principal", "architect")),
    ("director", ("director", "vp", "head", "chief")),
)
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "backend": ("backend", "api", "microservice", "distributed systems"),
    "frontend": ("frontend", "react", "angular", "vue", "ui"),
    "data-science": ("machine learning", "data science", "statistics", "modeling"),
    "data-engineering": ("spark", "airflow", "etl", "data pipeline"),
    "devops": ("docker", "kubernetes", "devops", "terraform", "sre"),
    "product": ("product management", "roadmap", "stakeholder"),
    "marketing": ("seo", "campaign", "growth", "content"),
}
_INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fintech": ("fintech", "banking", "payments", "lending"),
    "healthtech": ("healthcare", "clinical", "medical", "healthtech"),
    "e-commerce": ("e-commerce", "retail", "marketplace", "shopping"),
    "saas": ("saas", "b2b software", "subscription"),
}
_CACHE_DIR = CACHE_DIR / "jd_parsed"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _split_list_values(text: str) -> list[str]:
    parts = re.split(r"[,\n;|]+", text)
    seen: set[str] = set()
    values: list[str] = []
    for part in parts:
        cleaned = part.strip(" .:-")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(cleaned)
    return values


def _extract_years_range(text: str) -> tuple[float | None, float | None]:
    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    plus_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, flags=re.IGNORECASE)
    if plus_match:
        minimum = float(plus_match.group(1))
        return minimum, None

    return None, None


def _safe_float(value: object) -> float | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        extracted_min, _ = _extract_years_range(cleaned)
        return extracted_min


def _infer_seniority(text: str) -> str:
    lowered = text.lower()
    for seniority, keywords in _SENIORITY_PATTERNS:
        if any(keyword in lowered for keyword in keywords):
            return seniority
    return "unknown"


def _infer_domain(text: str) -> str | None:
    lowered = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return domain
    return None


def _infer_industry(text: str) -> str | None:
    lowered = text.lower()
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return industry
    return None


def _infer_education(text: str) -> str:
    lowered = text.lower()
    if "phd" in lowered or "doctorate" in lowered:
        return "phd"
    if any(keyword in lowered for keyword in ("master", "m.tech", "mba", "ms ")):
        return "master"
    if any(keyword in lowered for keyword in ("bachelor", "b.tech", "b.e", "bs ")):
        return "bachelor"
    return "unknown"


def _extract_location(text: str) -> str | None:
    match = re.search(
        r"(?:location|based in|job location)\s*[:\-]\s*([^\n|;]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .")
    return None


def _extract_remote_flag(text: str) -> bool | None:
    lowered = text.lower()
    if "remote" in lowered or "work from home" in lowered:
        return True
    if "on-site" in lowered or "onsite" in lowered:
        return False
    if "hybrid" in lowered:
        return None
    return None


def _extract_responsibilities(text: str, limit: int = 5) -> list[str]:
    bullets = re.findall(r"^[\-\*\u2022]\s*(.+)$", text, flags=re.MULTILINE)
    if bullets:
        return [bullet.strip() for bullet in bullets[:limit]]

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and len(line.strip().split()) >= 3
    ]
    return lines[:limit]


def _merge_non_empty(base: dict, incoming: Mapping[str, object]) -> dict:
    merged = dict(base)
    for key, value in incoming.items():
        if value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


class JobDescriptionParser:
    """Rule-based job parser with optional LLM enrichment and disk caching."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, identifier: str) -> Path:
        digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def parse_text(
        self,
        job_text: str,
        *,
        seed: Mapping[str, object] | None = None,
        llm_parser: Callable[[str], Mapping[str, object]] | None = None,
    ) -> dict:
        """Parse free-text job content into a structured JSON-safe dict."""
        seed = dict(seed or {})
        cache_key = f"{seed.get('job_id', '')}::{job_text}"
        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        explicit_skills = []
        for column_name in ("skills", "required_skills", "preferred_skills"):
            explicit_skills.extend(_split_list_values(_clean_text(seed.get(column_name))))

        min_years, max_years = _extract_years_range(job_text)
        if min_years is None:
            min_years = _safe_float(seed.get("min_years_experience"))
        if max_years is None:
            max_years = _safe_float(seed.get("max_years_experience"))

        parsed = {
            "title": _clean_text(seed.get("title")) or job_text.splitlines()[0].strip()[:120],
            "seniority": _infer_seniority(
                f"{_clean_text(seed.get('title'))} {job_text}"
            ),
            "domain": _infer_domain(f"{_clean_text(seed.get('title'))} {job_text}"),
            "must_have_skills": explicit_skills,
            "nice_to_have_skills": _split_list_values(_clean_text(seed.get("preferred_skills"))),
            "min_years_experience": min_years,
            "max_years_experience": max_years,
            "education_required": _infer_education(job_text),
            "location": _clean_text(seed.get("location")) or _extract_location(job_text),
            "remote_ok": _extract_remote_flag(job_text),
            "key_responsibilities": _extract_responsibilities(job_text),
            "industry": _infer_industry(job_text),
        }

        if llm_parser is not None:
            parsed = _merge_non_empty(parsed, llm_parser(job_text))

        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(parsed, handle, indent=2, default=str)
        return parsed

    def parse_frame(
        self,
        jobs_df: pd.DataFrame,
        schema: JobTableSchema,
        *,
        llm_parser: Callable[[str], Mapping[str, object]] | None = None,
    ) -> dict[str, dict]:
        """Parse an entire jobs dataframe into a mapping keyed by job_id."""
        parsed_jobs: dict[str, dict] = {}
        for _, row in jobs_df.iterrows():
            job_id = str(row[schema.job_id])
            seed = {
                "job_id": job_id,
                "title": _clean_text(row.get(schema.title)) if schema.title else "",
                "location": _clean_text(row.get(schema.location)) if schema.location else "",
                "min_years_experience": (
                    _clean_text(row.get(schema.min_experience)) if schema.min_experience else ""
                ),
                "max_years_experience": (
                    _clean_text(row.get(schema.max_experience)) if schema.max_experience else ""
                ),
            }
            if schema.skill_columns:
                seed["skills"] = ", ".join(
                    _clean_text(row.get(column)) for column in schema.skill_columns if _clean_text(row.get(column))
                )

            job_text = combine_text_values(row, schema.text_columns)
            parsed_jobs[job_id] = self.parse_text(job_text, seed=seed, llm_parser=llm_parser)
        return parsed_jobs
