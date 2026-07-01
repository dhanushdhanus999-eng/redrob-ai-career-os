"""Canonical Track 1 ranking entrypoint — network-free, CPU-only, < 5 minutes.

Satisfies the submission spec §10.3 single-command requirement:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Hard constraints honoured (spec §3): **no network**, **no GPU**, ≤ 16 GB RAM,
≤ 5 minutes wall-clock for the ranking step. This script loads *no* embedding
model, cross-encoder, or hosted LLM — it ranks with BM25 lexical recall plus
deterministic structured scoring (skills, role relevance, experience, behaviour,
career trajectory, location) and an internal-consistency honeypot filter. That
keeps it comfortably inside the compute budget and makes a network call
impossible by construction.

The richer offline research pipeline (dense retrieval, cross-encoder, optional
LLM analysis) lives in ``scripts/generate_submission.py`` and is NOT used to
produce the official submission.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.explainer import build_reasoning
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.consistency import score_consistency
from src.utils.role_relevance import score_career_trajectory, score_role_relevance
from src.utils.skill_ontology import SkillMatcher

# Shared, dependency-free Track-1 JD constants (no torch/faiss import cost).
from src.utils.track1_spec import (
    INDIA_LOCATION_TOKENS,
    TRACK1_MAX_YEARS,
    TRACK1_MIN_YEARS,
    TRACK1_MUST_HAVE_GROUPS,
    TRACK1_MUST_HAVE_SKILLS,
    TRACK1_NICE_TO_HAVE_SKILLS,
    TRACK1_JOB_TITLE,
)

TOP_K = 100
DEFAULT_RECALL_K = 1500
PROCESSED_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "challenge_candidates.parquet"
PROCESSED_JOB = PROJECT_ROOT / "data" / "processed" / "challenge_jobs.csv"
BM25_INDEX = PROJECT_ROOT / "outputs" / "models" / "bm25_demo_index.pkl"


# ── small numeric helpers (kept local so rank.py has no heavy dependencies) ───

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def score_experience(cand_years: float, min_years: float, max_years: float) -> float:
    if not min_years and not max_years:
        return 0.5
    if min_years and cand_years < min_years:
        return max(0.0, cand_years / max(min_years, 1.0))
    if max_years and cand_years > max_years:
        return max(0.45, 1.0 - (cand_years - max_years) / max(max_years * 2.0, 1.0))
    return 1.0


def score_location(candidate_location: str) -> float:
    loc = str(candidate_location).lower().strip()
    if any(t in loc for t in INDIA_LOCATION_TOKENS):
        return 1.0
    if not loc or loc in ("nan", "none", "not specified"):
        return 0.55
    return 0.25


def score_behavior(row: pd.Series) -> tuple[float, float]:
    """Return (behavioural_composite, recency) in [0, 1]."""
    completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
    response_rate = _safe_float(row.get("recruiter_response_rate"))
    github = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
    saved = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
    search = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
    assessment = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
    open_to_work = 1.0 if bool(row.get("open_to_work_flag")) else 0.0

    recency = 0.4
    raw_last = _safe_text(row.get("last_active"))
    if raw_last:
        try:
            active_date = datetime.fromisoformat(raw_last[:10]).date()
            days_since = max((date.today() - active_date).days, 0)
            recency = max(0.0, 1.0 - days_since / 180.0)
        except ValueError:
            pass

    beh = float(np.clip(
        0.25 * completeness + 0.20 * recency + 0.20 * response_rate
        + 0.10 * open_to_work + 0.10 * github + 0.05 * saved
        + 0.05 * search + 0.05 * assessment,
        0.0, 1.0,
    ))
    return beh, recency


# ── data loading ──────────────────────────────────────────────────────────────

def load_candidates(path: Path | None) -> pd.DataFrame:
    """Load candidates from a parquet, JSONL, or JSONL.GZ file.

    The organiser provides ``candidates.jsonl`` at Stage 3; the repo ships a
    pre-flattened parquet for local runs. Both are supported.
    """
    if path is None:
        if PROCESSED_CANDIDATES.exists():
            print(f"Loading candidates from {PROCESSED_CANDIDATES}")
            return pd.read_parquet(PROCESSED_CANDIDATES)
        raise FileNotFoundError(
            "No --candidates given and data/processed/challenge_candidates.parquet is missing."
        )

    path = Path(path)
    if path.suffix == ".parquet":
        print(f"Loading candidates from {path}")
        return pd.read_parquet(path)

    # Raw JSONL(.gz) — flatten with the same logic that built the parquet.
    from src.data.challenge_bundle import flatten_candidate_record, stream_candidate_records

    print(f"Flattening candidate records from {path} …")
    rows = [flatten_candidate_record(rec) for rec in stream_candidate_records(path)]
    return pd.DataFrame.from_records(rows)


def load_job_text() -> str:
    if PROCESSED_JOB.exists():
        job = pd.read_csv(PROCESSED_JOB).iloc[0]
        parts = [
            _safe_text(job.get(col))
            for col in ("job_title", "responsibilities", "required_skills",
                        "preferred_skills", "excluded_profiles", "ideal_candidate",
                        "job_description")
        ]
        text = " ".join(p for p in parts if p)
        if text.strip():
            return text
    return TRACK1_JOB_TITLE + " " + " ".join(TRACK1_MUST_HAVE_SKILLS)


# ── ranking ────────────────────────────────────────────────────────────────────

def rank(candidates: pd.DataFrame, *, recall_k: int) -> pd.DataFrame:
    skill_matcher = SkillMatcher()
    must_skills = TRACK1_MUST_HAVE_SKILLS
    nice_skills = TRACK1_NICE_TO_HAVE_SKILLS
    min_years, max_years = TRACK1_MIN_YEARS, TRACK1_MAX_YEARS

    candidates = candidates.copy()
    candidates["_cid"] = candidates["candidate_id"].astype(str)
    lookup = candidates.set_index("_cid", drop=False)

    ids = candidates["candidate_id"].astype(str).tolist()

    # ── Acquire the BM25 index. Tokenisation is PRE-COMPUTATION (spec §10.3 lets
    #    it exceed 5 min); loading the saved index is part of the ranking step. ─
    bm25 = BM25Retriever()
    index_was_built = False
    t_load = time.perf_counter()
    if BM25_INDEX.exists():
        print(f"Loading precomputed BM25 index from {BM25_INDEX} …")
        bm25.load(BM25_INDEX)
        if set(bm25.candidate_ids) != set(ids):
            print("  Index candidate set does not match input — rebuilding (pre-compute) …")
            bm25.build_index(documents=candidates["profile_text"].astype(str).tolist(), candidate_ids=ids)
            bm25.save(BM25_INDEX)
            index_was_built = True
    else:
        print(f"No precomputed index — building (one-off pre-computation) over {len(ids):,} candidates …")
        bm25.build_index(documents=candidates["profile_text"].astype(str).tolist(), candidate_ids=ids)
        BM25_INDEX.parent.mkdir(parents=True, exist_ok=True)
        bm25.save(BM25_INDEX)
        index_was_built = True
    load_time = time.perf_counter() - t_load

    # ── Timed ranking step. Index *load* counts; a fresh *build* is pre-compute. ─
    t_rank = time.perf_counter() - (0.0 if index_was_built else load_time)
    job_text = load_job_text()
    extra = [s for s in must_skills if s.lower() not in job_text.lower()]
    query = job_text + (" " + " ".join(extra) if extra else "")

    recall = bm25.retrieve(query, top_k=recall_k)
    print(f"  BM25 recall: {len(recall)} candidates")
    if not recall:
        raise RuntimeError("BM25 returned no candidates.")

    raw = np.asarray([s for _, s in recall], dtype=float)
    bm25_max = float(raw.max()) or 1e-9

    scored: list[dict] = []
    honeypots = 0
    for cid, raw_bm25 in recall:
        row = lookup.loc[str(cid)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        retrieval_norm = float(raw_bm25) / bm25_max
        cand_skills = [s.strip() for s in _safe_text(row.get("skills")).split(",") if s.strip()]
        cand_years = _safe_float(row.get("total_experience"))

        must_match = skill_matcher.match_score(must_skills, cand_skills)
        nice_match = skill_matcher.match_score(nice_skills, cand_skills)
        # Must-have signal is driven by *capability-group* coverage (a candidate
        # needs one vector DB, not all eight), with a light depth term so richer
        # profiles still break ties. This rewards breadth across the JD's real
        # capability areas rather than redundant tokens within one.
        group_cov = skill_matcher.group_coverage(TRACK1_MUST_HAVE_GROUPS, cand_skills)
        must_score = 0.85 * group_cov["coverage"] + 0.15 * must_match["composite_score"]
        skill_score = must_score * 0.75 + nice_match["composite_score"] * 0.25

        exp_score = score_experience(cand_years, min_years, max_years)
        beh_score, _recency = score_behavior(row)
        role_score = score_role_relevance(_safe_text(row.get("current_role")), _safe_text(row.get("headline")))
        career_score = score_career_trajectory(_safe_text(row.get("career_history_text")))
        location_score = score_location(_safe_text(row.get("location")) or _safe_text(row.get("country")))
        consistency = score_consistency(_safe_text(row.get("skills_detailed")), cand_years)

        # No-semantic 9-signal formula (matches generate_submission's --no-semantic branch).
        overall = (
            0.18 * retrieval_norm
            + 0.24 * skill_score
            + 0.13 * role_score
            + 0.11 * exp_score
            + 0.09 * beh_score
            + 0.16 * career_score
            + 0.04 * location_score
            + 0.05 * consistency.consistency_score
        )
        if consistency.is_honeypot:
            overall *= 0.05
            honeypots += 1

        nice_matched = sorted(
            set(nice_match.get("matched_skills", [])) | (set(must_match.get("matched_skills", [])) & set(nice_skills))
        )
        scored.append({
            "candidate_id": str(cid),
            "overall": overall,
            "matched_must": list(must_match.get("matched_skills", [])),
            "covered_groups": group_cov["covered_groups"],
            "n_groups_total": group_cov["n_total"],
            "matched_nice": nice_matched,
            "cand_years": cand_years,
            "current_title": _safe_text(row.get("current_role")),
            "location": _safe_text(row.get("location")) or _safe_text(row.get("country")),
            "beh_score": beh_score,
            "response_rate": _safe_float(row.get("recruiter_response_rate")),
            "github_score": min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0),
            "open_to_work": bool(row.get("open_to_work_flag")),
            "notice_period": _safe_float(row.get("notice_period_days")),
            "is_honeypot": consistency.is_honeypot,
        })

    # Deterministic order: score desc, then candidate_id asc (spec tie-break rule).
    scored.sort(key=lambda x: (-x["overall"], x["candidate_id"]))
    print(f"Honeypots detected in recall: {honeypots} | in top {TOP_K}: "
          f"{sum(1 for s in scored[:TOP_K] if s['is_honeypot'])}")

    top = scored[:TOP_K]
    anchor = top[0]["overall"] or 1e-9
    rows: list[dict] = []
    for r, s in enumerate(top, start=1):
        rows.append({
            "candidate_id": s["candidate_id"],
            "rank": r,
            "score": round(min(1.0, s["overall"] / anchor), 6),
            "reasoning": build_reasoning(
                rank=r,
                job_title=TRACK1_JOB_TITLE,
                job_min_years=min_years,
                job_max_years=max_years,
                matched_must=s["matched_must"],
                matched_nice=s["matched_nice"],
                n_must_total=len(must_skills),
                matched_groups=s["covered_groups"],
                n_groups_total=s["n_groups_total"],
                current_title=s["current_title"],
                cand_years_exp=s["cand_years"],
                location=s["location"],
                behavioral_score=s["beh_score"],
                response_rate=s["response_rate"],
                github_score=s["github_score"],
                open_to_work=s["open_to_work"],
                notice_period_days=int(s["notice_period"]) if s["notice_period"] else None,
            ),
        })

    rank_elapsed = time.perf_counter() - t_rank
    print(f"Ranking step (recall + scoring, excl. index pre-compute): {rank_elapsed:.1f}s  (budget: 300s)")
    if rank_elapsed > 300:
        print("WARNING: ranking step exceeded the 5-minute Stage-3 compute budget!")
    return pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Path to candidates.jsonl / .jsonl.gz / .parquet "
                             "(default: data/processed/challenge_candidates.parquet)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "submissions" / "submission.csv",
                        help="Output CSV path")
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_K)
    args = parser.parse_args()

    t_total = time.perf_counter()
    candidates = load_candidates(args.candidates)
    submission = rank(candidates, recall_k=args.recall_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    elapsed = time.perf_counter() - t_total

    print("\n" + "=" * 60)
    print(f"Submission saved : {args.out}")
    print(f"Rows             : {len(submission)}")
    print(f"Score range      : {submission['score'].min():.4f} – {submission['score'].max():.4f}")
    print(f"Ranking wall-clock: {elapsed:.1f}s  (budget: 300s)")
    if elapsed > 300:
        print("WARNING: exceeded the 5-minute Stage-3 compute budget!")
    print("=" * 60)
    print(submission[["rank", "candidate_id", "score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
