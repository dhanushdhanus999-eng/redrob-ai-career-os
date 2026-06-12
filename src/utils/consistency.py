"""Internal-consistency / honeypot detection for Track 1.

The organiser's submission spec states the candidate pool contains ~80 honeypot
profiles with *subtly impossible* internal data (e.g. "8 years of experience at a
company founded 3 years ago", or "'expert' proficiency in 10 skills with 0 years
used"). These are forced to relevance tier 0 in the hidden ground truth, and a
submission with a honeypot rate > 10% in its top 100 is **disqualified at Stage 3**.

A pure keyword/embedding ranker cannot see these contradictions — a honeypot that
lists every must-have skill at "expert" level scores *higher* than an honest
candidate. This module reads the structured profile and scores how internally
plausible it is, so impossible profiles can be pushed out of the shortlist.

The primary, high-precision signal available in the flattened data is the
per-skill ``proficiency`` + ``duration_months`` carried in ``skills_detailed``
(format: ``name; proficiency; endorsements=N; months=M || ...``):

* "expert"/"advanced" proficiency claimed with ``months == 0`` is impossible.
* A skill used for far more months than the candidate's total career length is
  impossible.

Thresholds were calibrated against the released 100K pool: the combined rule
below flags ~96 candidates (the documented honeypot count is ~80), while the
individual noisy signals are kept at a low weight so genuine candidates are not
harmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_MONTHS_PATTERN = re.compile(r"months\s*=\s*(\d+)", re.IGNORECASE)
_HIGH_PROFICIENCY = ("expert", "advanced")


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of a profile-plausibility check."""

    consistency_score: float          # 1.0 = fully plausible, 0.0 = impossible
    is_honeypot: bool                 # True => exclude from the shortlist
    expert_zero_count: int            # high proficiency claimed with 0 months used
    impossible_duration_count: int    # skill used longer than the whole career
    reasons: list[str] = field(default_factory=list)


def _parse_skills_detailed(skills_detailed: str) -> list[tuple[str, int | None]]:
    """Return (proficiency, months) tuples parsed from the skills_detailed string."""
    parsed: list[tuple[str, int | None]] = []
    if not skills_detailed:
        return parsed
    for part in str(skills_detailed).split("||"):
        segments = [seg.strip() for seg in part.split(";")]
        if len(segments) < 2:
            continue
        proficiency = segments[1].lower()
        match = _MONTHS_PATTERN.search(part)
        months = int(match.group(1)) if match else None
        parsed.append((proficiency, months))
    return parsed


def score_consistency(
    skills_detailed: str,
    total_experience_years: float,
) -> ConsistencyResult:
    """Score how internally consistent a candidate profile is.

    Args:
        skills_detailed: the ``skills_detailed`` flattened string for the candidate.
        total_experience_years: the candidate's claimed total years of experience.

    Returns:
        A :class:`ConsistencyResult`. Clear honeypots get ``is_honeypot=True`` and
        ``consistency_score=0.0``; mild contradictions reduce the score smoothly.
    """
    skills = _parse_skills_detailed(skills_detailed)
    exp_months = max(float(total_experience_years or 0.0), 0.0) * 12.0

    expert_zero = sum(
        1
        for prof, months in skills
        if prof in _HIGH_PROFICIENCY and months is not None and months == 0
    )
    impossible_duration = sum(
        1
        for _prof, months in skills
        if months is not None and exp_months > 0 and months > exp_months * 1.5 + 12
    )

    reasons: list[str] = []
    if expert_zero:
        reasons.append(
            f"{expert_zero} skill(s) claim expert/advanced proficiency with 0 months of use"
        )
    if impossible_duration:
        reasons.append(
            f"{impossible_duration} skill(s) used longer than the candidate's total career"
        )

    # Combined high-precision honeypot rule (calibrated on the released pool).
    is_honeypot = (
        expert_zero >= 2
        or impossible_duration >= 4
        or (expert_zero >= 1 and impossible_duration >= 2)
    )

    if is_honeypot:
        return ConsistencyResult(
            consistency_score=0.0,
            is_honeypot=True,
            expert_zero_count=expert_zero,
            impossible_duration_count=impossible_duration,
            reasons=reasons or ["profile contains internally impossible claims"],
        )

    # Graded penalty for milder contradictions — never below 0.5 for non-honeypots
    # so an honest candidate with one noisy field is not unduly punished.
    penalty = 0.15 * expert_zero + 0.05 * impossible_duration
    consistency_score = round(max(0.5, 1.0 - penalty), 4)
    return ConsistencyResult(
        consistency_score=consistency_score,
        is_honeypot=False,
        expert_zero_count=expert_zero,
        impossible_duration_count=impossible_duration,
        reasons=reasons,
    )
