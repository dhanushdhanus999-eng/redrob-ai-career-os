"""Behavioral and activity-derived candidate features."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        timestamp = pd.to_datetime(cleaned, utc=False)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if hasattr(timestamp, "to_pydatetime"):
        parsed = timestamp.to_pydatetime()
    else:
        parsed = timestamp
    return parsed.replace(tzinfo=None)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


class BehavioralFeatureExtractor:
    """Compute recency, engagement, and profile-quality features."""

    def __init__(self, *, reference_date: datetime | None = None) -> None:
        self.reference_date = (reference_date or datetime.now()).replace(tzinfo=None)

    def _recency_features(self, date_value: object, prefix: str) -> dict[str, float]:
        parsed = _parse_datetime(date_value)
        if parsed is None:
            return {
                f"{prefix}_days_ago": 0.0,
                f"{prefix}_within_30d": 0.0,
                f"{prefix}_within_90d": 0.0,
                f"{prefix}_within_180d": 0.0,
                f"{prefix}_recency_score": 0.0,
            }

        days_ago = max(0, (self.reference_date - parsed).days)
        return {
            f"{prefix}_days_ago": float(days_ago),
            f"{prefix}_within_30d": float(days_ago <= 30),
            f"{prefix}_within_90d": float(days_ago <= 90),
            f"{prefix}_within_180d": float(days_ago <= 180),
            f"{prefix}_recency_score": float(1.0 / (1.0 + days_ago / 30.0)),
        }

    def extract(
        self,
        *,
        last_active_date: str | None = None,
        profile_updated_date: str | None = None,
        n_applications: int | float | None = None,
        n_profile_views: int | float | None = None,
        n_skills_added_recent: int | float | None = None,
        profile_completeness: float = 0.0,
        response_rate: float | None = None,
        avg_response_time_days: float | None = None,
        has_applied_before: bool = False,
        similar_job_apps: int | float | None = None,
        candidate_skill_count: int | float | None = None,
        candidate_summary_length: int | float | None = None,
        candidate_has_location: bool = False,
        candidate_has_education: bool = False,
    ) -> dict[str, float]:
        features = {}
        features.update(self._recency_features(last_active_date, "last_active"))
        features.update(self._recency_features(profile_updated_date, "profile_updated"))

        n_applications_value = float(_safe_float(n_applications) or 0.0)
        n_profile_views_value = float(_safe_float(n_profile_views) or 0.0)
        n_skills_added_value = float(_safe_float(n_skills_added_recent) or 0.0)
        similar_job_apps_value = float(_safe_float(similar_job_apps) or 0.0)
        candidate_skill_count_value = float(_safe_float(candidate_skill_count) or 0.0)
        candidate_summary_length_value = float(_safe_float(candidate_summary_length) or 0.0)

        features["n_applications"] = n_applications_value
        features["n_profile_views"] = n_profile_views_value
        features["n_skills_added_recent"] = n_skills_added_value
        features["similar_job_apps"] = similar_job_apps_value
        features["candidate_skill_count"] = candidate_skill_count_value
        features["candidate_summary_length"] = candidate_summary_length_value

        for key in ("n_applications", "n_profile_views", "similar_job_apps"):
            features[f"{key}_log"] = float(np.log1p(features[key]))

        response_rate_value = _safe_float(response_rate)
        avg_response_time_value = _safe_float(avg_response_time_days)
        features["response_rate"] = 0.5 if response_rate_value is None else _clamp01(response_rate_value)
        features["avg_response_time_days"] = (
            0.0 if avg_response_time_value is None else float(max(0.0, avg_response_time_value))
        )
        features["response_speed_score"] = (
            0.5
            if avg_response_time_value is None
            else float(1.0 / (1.0 + max(0.0, avg_response_time_value) / 7.0))
        )

        features["profile_completeness"] = _clamp01(float(profile_completeness or 0.0))
        features["has_applied_before"] = float(_safe_bool(has_applied_before))
        features["candidate_has_location"] = float(_safe_bool(candidate_has_location))
        features["candidate_has_education"] = float(_safe_bool(candidate_has_education))

        features["is_active_job_seeker"] = float(
            n_applications_value > 0
            or similar_job_apps_value > 0
            or features["last_active_within_30d"] > 0
        )
        features["skill_freshness_score"] = _clamp01(n_skills_added_value / 10.0)
        features["profile_richness_score"] = float(
            np.mean(
                [
                    features["profile_completeness"],
                    _clamp01(candidate_skill_count_value / 20.0),
                    _clamp01(candidate_summary_length_value / 150.0),
                    features["candidate_has_location"],
                    features["candidate_has_education"],
                ]
            )
        )

        recency_component = max(
            features["last_active_recency_score"],
            features["profile_updated_recency_score"],
        )
        engagement_component = float(
            np.mean(
                [
                    features["is_active_job_seeker"],
                    features["response_rate"],
                    features["response_speed_score"],
                    _clamp01(features["similar_job_apps_log"] / np.log1p(10.0)),
                ]
            )
        )
        quality_component = float(
            np.mean(
                [
                    features["profile_richness_score"],
                    features["skill_freshness_score"],
                ]
            )
        )
        features["behavioral_composite"] = float(
            np.mean([recency_component, engagement_component, quality_component])
        )
        return features
