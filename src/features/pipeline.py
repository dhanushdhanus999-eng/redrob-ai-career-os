"""Shared data preparation and feature orchestration for Phase 3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.baselines.common import Phase2DataBundle, load_phase2_bundle
from src.data.io import load_table
from src.data.schema import combine_text_values
from src.data.splits import create_splits
from src.eval.submission import detect_label_columns
from src.features.behavioral_features import BehavioralFeatureExtractor
from src.features.experience_features import ExperienceFeatureExtractor
from src.features.semantic_features import SemanticFeatureExtractor
from src.features.skill_features import SkillFeatureExtractor
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser
from src.utils.paths import PROCESSED_DATA_DIR, ensure_project_dirs


PARSED_JOBS_PATH = PROCESSED_DATA_DIR / "parsed_jds.json"
PARSED_CANDIDATES_PATH = PROCESSED_DATA_DIR / "parsed_candidates.json"
PHASE3_FEATURE_BLOCKS = (
    "features_semantic",
    "features_skills",
    "features_experience",
    "features_behavioral",
)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_row_signal(row: pd.Series, aliases: tuple[str, ...]) -> object | None:
    if row is None or row.empty:
        return None
    alias_set = {_normalise_name(alias) for alias in aliases}
    for column in row.index:
        if _normalise_name(str(column)) in alias_set:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return value

    for column in row.index:
        normalised = _normalise_name(str(column))
        if any(alias in normalised for alias in alias_set):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return value
    return None


def _save_json(path: Path, payload: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@dataclass(frozen=True)
class Phase3Context:
    """Loaded data and parsed artifacts required by the Phase 3 pipeline."""

    bundle: Phase2DataBundle
    parsed_jobs: dict[str, dict]
    parsed_candidates: dict[str, dict]


def canonicalize_labels_frame(
    labels_df: pd.DataFrame,
    label_columns: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    """Rename a labels-like dataframe to canonical job/candidate/relevance columns."""
    if label_columns is None:
        if {"job_id", "candidate_id", "relevance"}.issubset(labels_df.columns):
            label_columns = ("job_id", "candidate_id", "relevance")
        else:
            label_columns = detect_label_columns(labels_df.columns)

    job_column, candidate_column, relevance_column = label_columns
    frame = labels_df.rename(
        columns={
            job_column: "job_id",
            candidate_column: "candidate_id",
            relevance_column: "relevance",
        }
    ).copy()
    frame["job_id"] = frame["job_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["relevance"] = pd.to_numeric(frame["relevance"], errors="coerce").fillna(0.0)
    return frame


def load_or_create_parsed_artifacts(
    bundle: Phase2DataBundle,
    *,
    refresh: bool = False,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load cached parsed JSON artifacts or create them from the Phase 2 parsers."""
    ensure_project_dirs()
    if not refresh and PARSED_JOBS_PATH.exists() and PARSED_CANDIDATES_PATH.exists():
        with PARSED_JOBS_PATH.open("r", encoding="utf-8") as handle:
            parsed_jobs = json.load(handle)
        with PARSED_CANDIDATES_PATH.open("r", encoding="utf-8") as handle:
            parsed_candidates = json.load(handle)
        return parsed_jobs, parsed_candidates

    job_parser = JobDescriptionParser()
    candidate_parser = CandidateProfileParser()

    parsed_jobs = job_parser.parse_frame(bundle.jobs, bundle.job_schema)
    parsed_candidates = candidate_parser.parse_frame(bundle.candidates, bundle.candidate_schema)

    _save_json(PARSED_JOBS_PATH, parsed_jobs)
    _save_json(PARSED_CANDIDATES_PATH, parsed_candidates)
    return parsed_jobs, parsed_candidates


def load_phase3_context(
    *,
    require_labels: bool = False,
    refresh_parsed: bool = False,
) -> Phase3Context:
    """Load the Phase 2 datasets plus parsed artifacts needed for Phase 3."""
    bundle = load_phase2_bundle(require_labels=require_labels)
    parsed_jobs, parsed_candidates = load_or_create_parsed_artifacts(bundle, refresh=refresh_parsed)
    return Phase3Context(
        bundle=bundle,
        parsed_jobs=parsed_jobs,
        parsed_candidates=parsed_candidates,
    )


def ensure_label_splits(
    context: Phase3Context,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Materialize train/val/test split files if labels are available."""
    bundle = context.bundle
    if bundle.labels is None or bundle.label_columns is None:
        raise FileNotFoundError(
            "Phase 3 training requires organizer labels, but no labels file was found."
        )

    split_paths = {
        split_name: PROCESSED_DATA_DIR / f"{split_name}.csv"
        for split_name in ("train", "val", "test")
    }
    if all(path.exists() for path in split_paths.values()):
        return {
            split_name: canonicalize_labels_frame(
                load_table(path),
                bundle.label_columns,
            )
            for split_name, path in split_paths.items()
        }

    job_column, _, _ = bundle.label_columns
    train_df, val_df, test_df = create_splits(
        labels_df=bundle.labels,
        job_id_col=job_column,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
        save_dir=PROCESSED_DATA_DIR,
    )
    return {
        "train": canonicalize_labels_frame(train_df, bundle.label_columns),
        "val": canonicalize_labels_frame(val_df, bundle.label_columns),
        "test": canonicalize_labels_frame(test_df, bundle.label_columns),
    }


def load_split_frame(split: str, context: Phase3Context) -> pd.DataFrame:
    """Load a canonicalized train/val/test split."""
    split_path = PROCESSED_DATA_DIR / f"{split}.csv"
    if split_path.exists():
        return canonicalize_labels_frame(load_table(split_path), context.bundle.label_columns)
    return ensure_label_splits(context)[split]


def build_job_text_lookup(context: Phase3Context) -> dict[str, str]:
    """Return canonical job text used throughout retrieval and reranking."""
    bundle = context.bundle
    return {
        str(row[bundle.job_schema.job_id]): combine_text_values(row, bundle.job_schema.text_columns)
        for _, row in bundle.jobs.iterrows()
    }


def build_candidate_text_lookup(context: Phase3Context) -> dict[str, str]:
    """Return canonical candidate text used throughout retrieval and reranking."""
    bundle = context.bundle
    return {
        str(row[bundle.candidate_schema.candidate_id]): combine_text_values(
            row,
            bundle.candidate_schema.text_columns,
        )
        for _, row in bundle.candidates.iterrows()
    }


def _build_indexed_frame(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    indexed = df.copy()
    indexed["__canonical_id__"] = indexed[id_column].astype(str)
    return indexed.set_index("__canonical_id__", drop=False)


def _candidate_behavioral_inputs(row: pd.Series, parsed_candidate: dict) -> dict[str, object]:
    profile_updated = _find_row_signal(
        row,
        ("profile_updated", "updated_at", "modified_at", "last_updated"),
    )
    return {
        "last_active_date": parsed_candidate.get("last_active") or _find_row_signal(
            row,
            ("last_active", "last_seen", "active_at"),
        ),
        "profile_updated_date": profile_updated or parsed_candidate.get("last_active"),
        "n_applications": _find_row_signal(
            row,
            ("n_applications", "applications", "application_count"),
        ),
        "n_profile_views": _find_row_signal(
            row,
            ("n_profile_views", "profile_views", "view_count"),
        ),
        "n_skills_added_recent": _find_row_signal(
            row,
            ("n_skills_added_recent", "skills_added_recent", "recent_skills_added"),
        ),
        "profile_completeness": parsed_candidate.get("profile_completeness", 0.0),
        "response_rate": _find_row_signal(
            row,
            ("response_rate", "reply_rate"),
        ),
        "avg_response_time_days": _find_row_signal(
            row,
            ("avg_response_time_days", "response_time_days", "reply_time_days"),
        ),
        "has_applied_before": _find_row_signal(
            row,
            ("has_applied_before", "applied_before", "previous_applicant"),
        ),
        "similar_job_apps": _find_row_signal(
            row,
            ("similar_job_apps", "related_job_applications"),
        ),
        "candidate_skill_count": len(parsed_candidate.get("skills", [])),
        "candidate_summary_length": len(str(parsed_candidate.get("summary", "")).split()),
        "candidate_has_location": bool(parsed_candidate.get("location")),
        "candidate_has_education": bool(parsed_candidate.get("education")),
    }


def build_pair_inputs(labels_df: pd.DataFrame, context: Phase3Context) -> pd.DataFrame:
    """Join labels with raw and parsed job/candidate context needed for features."""
    labels = canonicalize_labels_frame(labels_df, context.bundle.label_columns)
    bundle = context.bundle

    jobs = _build_indexed_frame(bundle.jobs, bundle.job_schema.job_id)
    candidates = _build_indexed_frame(bundle.candidates, bundle.candidate_schema.candidate_id)

    rows: list[dict[str, object]] = []
    for _, label_row in labels.iterrows():
        job_id = str(label_row["job_id"])
        candidate_id = str(label_row["candidate_id"])

        job_row = jobs.loc[job_id] if job_id in jobs.index else pd.Series(dtype=object)
        candidate_row = (
            candidates.loc[candidate_id] if candidate_id in candidates.index else pd.Series(dtype=object)
        )

        parsed_job = context.parsed_jobs.get(job_id, {})
        parsed_candidate = context.parsed_candidates.get(candidate_id, {})

        candidate_skills = list(parsed_candidate.get("skills", []))
        must_skills = list(parsed_job.get("must_have_skills", []))
        nice_skills = list(parsed_job.get("nice_to_have_skills", []))
        raw_job_title = (
            _clean_text(job_row.get(bundle.job_schema.title))
            if bundle.job_schema.title and not job_row.empty
            else ""
        )
        parsed_job_title = str(parsed_job.get("title", "")).strip()

        row = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "relevance": float(label_row["relevance"]),
            "job_text": (
                combine_text_values(job_row, bundle.job_schema.text_columns)
                if not job_row.empty
                else ""
            ),
            "candidate_text": (
                combine_text_values(candidate_row, bundle.candidate_schema.text_columns)
                if not candidate_row.empty
                else ""
            ),
            "job_title_text": parsed_job_title or raw_job_title,
            "job_skills_text": " ".join(must_skills + nice_skills),
            "candidate_skills_text": " ".join(candidate_skills),
            "must_have_skills": must_skills,
            "nice_to_have_skills": nice_skills,
            "candidate_skills": candidate_skills,
            "job_seniority": parsed_job.get("seniority", "unknown"),
            "job_min_years": parsed_job.get("min_years_experience"),
            "job_max_years": parsed_job.get("max_years_experience"),
            "job_education_req": parsed_job.get("education_required", "unknown"),
            "job_domain": parsed_job.get("domain", ""),
            "job_title": parsed_job_title or raw_job_title,
            "cand_seniority": parsed_candidate.get("seniority", "unknown"),
            "cand_years_exp": parsed_candidate.get("total_experience_years"),
            "cand_education": parsed_candidate.get("education", ""),
            "cand_current_role": parsed_candidate.get("current_role", ""),
        }
        row.update(_candidate_behavioral_inputs(candidate_row, parsed_candidate))
        rows.append(row)

    return pd.DataFrame(rows)


def _skill_feature_frame(pair_inputs: pd.DataFrame) -> pd.DataFrame:
    extractor = SkillFeatureExtractor()
    rows = []
    for _, row in pair_inputs.iterrows():
        features = extractor.extract(
            must_have_skills=row["must_have_skills"],
            nice_to_have_skills=row["nice_to_have_skills"],
            candidate_skills=row["candidate_skills"],
        )
        features["job_id"] = row["job_id"]
        features["candidate_id"] = row["candidate_id"]
        rows.append(features)
    return pd.DataFrame(rows)


def _experience_feature_frame(pair_inputs: pd.DataFrame) -> pd.DataFrame:
    extractor = ExperienceFeatureExtractor()
    rows = []
    for _, row in pair_inputs.iterrows():
        features = extractor.extract(
            job_seniority=row["job_seniority"],
            job_min_years=row["job_min_years"],
            job_max_years=row["job_max_years"],
            job_education_req=row["job_education_req"],
            job_domain=row["job_domain"],
            job_title=row["job_title"],
            cand_seniority=row["cand_seniority"],
            cand_years_exp=row["cand_years_exp"],
            cand_education=row["cand_education"],
            cand_current_role=row["cand_current_role"],
        )
        features["job_id"] = row["job_id"]
        features["candidate_id"] = row["candidate_id"]
        rows.append(features)
    return pd.DataFrame(rows)


def _behavioral_feature_frame(pair_inputs: pd.DataFrame) -> pd.DataFrame:
    extractor = BehavioralFeatureExtractor()
    rows = []
    for _, row in pair_inputs.iterrows():
        features = extractor.extract(
            last_active_date=row["last_active_date"],
            profile_updated_date=row["profile_updated_date"],
            n_applications=row["n_applications"],
            n_profile_views=row["n_profile_views"],
            n_skills_added_recent=row["n_skills_added_recent"],
            profile_completeness=row["profile_completeness"],
            response_rate=row["response_rate"],
            avg_response_time_days=row["avg_response_time_days"],
            has_applied_before=row["has_applied_before"],
            similar_job_apps=row["similar_job_apps"],
            candidate_skill_count=row["candidate_skill_count"],
            candidate_summary_length=row["candidate_summary_length"],
            candidate_has_location=row["candidate_has_location"],
            candidate_has_education=row["candidate_has_education"],
        )
        features["job_id"] = row["job_id"]
        features["candidate_id"] = row["candidate_id"]
        rows.append(features)
    return pd.DataFrame(rows)


def generate_feature_frames(
    pair_inputs: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
    semantic_extractor: SemanticFeatureExtractor | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate one or more feature blocks for the given pair inputs."""
    feature_names = tuple(feature_names or PHASE3_FEATURE_BLOCKS)
    frames: dict[str, pd.DataFrame] = {}

    if "features_semantic" in feature_names:
        semantic_extractor = semantic_extractor or SemanticFeatureExtractor()
        semantic_columns = [
            "job_id",
            "candidate_id",
            "job_text",
            "candidate_text",
            "job_title_text",
            "job_skills_text",
            "candidate_skills_text",
        ]
        frames["features_semantic"] = semantic_extractor.extract_batch(pair_inputs[semantic_columns])

    if "features_skills" in feature_names:
        frames["features_skills"] = _skill_feature_frame(pair_inputs)

    if "features_experience" in feature_names:
        frames["features_experience"] = _experience_feature_frame(pair_inputs)

    if "features_behavioral" in feature_names:
        frames["features_behavioral"] = _behavioral_feature_frame(pair_inputs)

    return frames


def merge_feature_frames(
    base_df: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge one or more Phase 3 feature blocks into a canonical base dataframe."""
    merged = canonicalize_labels_frame(base_df).copy()
    for frame in feature_frames.values():
        if frame.empty:
            continue
        working = frame.copy()
        working["job_id"] = working["job_id"].astype(str)
        working["candidate_id"] = working["candidate_id"].astype(str)
        merged = merged.merge(working, on=["job_id", "candidate_id"], how="left")
    return merged


def feature_block_path(feature_name: str, split: str) -> Path:
    """Return the canonical split-specific output path for a feature block."""
    return PROCESSED_DATA_DIR / f"{feature_name}_{split}.parquet"


def save_feature_frame(frame: pd.DataFrame, feature_name: str, split: str) -> Path:
    """Persist a feature block using the split-specific naming convention."""
    path = feature_block_path(feature_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def save_feature_frames(feature_frames: dict[str, pd.DataFrame], split: str) -> list[Path]:
    """Persist multiple feature blocks for the same split."""
    return [
        save_feature_frame(frame, feature_name, split)
        for feature_name, frame in feature_frames.items()
    ]


def load_feature_frame(feature_name: str, split: str) -> pd.DataFrame:
    """Load a split-specific feature block, falling back to the legacy shared path."""
    split_path = feature_block_path(feature_name, split)
    if split_path.exists():
        return pd.read_parquet(split_path)

    legacy_path = PROCESSED_DATA_DIR / f"{feature_name}.parquet"
    if legacy_path.exists():
        return pd.read_parquet(legacy_path)

    raise FileNotFoundError(f"Feature block not found for {feature_name} ({split}).")


def load_merged_feature_split(
    split: str,
    context: Phase3Context,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Load a canonical split and merge the saved Phase 3 feature blocks."""
    feature_names = tuple(feature_names or PHASE3_FEATURE_BLOCKS)
    base_df = load_split_frame(split, context)
    frames = {
        feature_name: load_feature_frame(feature_name, split)
        for feature_name in feature_names
    }
    return merge_feature_frames(base_df, frames)
