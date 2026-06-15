"""Official India Runs Track 1 submission generator.

Pipeline stages:
  Stage 0: Load BM25 index (built from 100K candidate pool)
  Stage 1: BM25 recall — top recall_k candidates
  Stage 1b: Semantic similarity scoring (BGE-large-en-v1.5, optional)
  Stage 2: Multi-signal scoring (9 signals, sums to 1.0):
             BM25 norm   × 0.13
             Semantic    × 0.18  (if model available, else redistributed)
             Skill match × 0.18  (expanded ontology — 60+ synonyms, 13 families)
             Role score  × 0.12  (AI Engineer → 1.0, HR Manager → 0.1)
             Exp fit     × 0.10
             Behavioral  × 0.08
             Career      × 0.12  (production-evidence; consulting penalty baked in)
             Location    × 0.04
             Consistency × 0.05  (honeypots hard-capped — Stage-3 DQ guard)
  Stage 2b: LTR blend (opt-in --ltr; 40% minority signal, partially circular)
  Stage 3: Cross-encoder re-rank of top 50 (optional)
  Stage 4: LLM listwise re-rank — OFFLINE ONLY, forbidden for the submission
  Output:  100-row CSV: candidate_id, rank, score, reasoning

NOTE: the canonical, network-free, ≤5-minute submission entrypoint is `rank.py`
at the repo root. This script is the richer research/analysis pipeline.

Usage:
    python scripts/generate_submission.py                    # full research pipeline
    python scripts/generate_submission.py --no-semantic      # skip BGE (faster)
    python scripts/generate_submission.py --no-crossencoder  # skip cross-encoder
    python scripts/generate_submission.py --ltr              # blend LTR (off by default)
    python scripts/generate_submission.py --validate         # validate after run
    python scripts/generate_submission.py --recall-k 2000    # wider recall
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.baselines.common import build_candidate_documents, load_phase2_bundle
from src.data.challenge_bundle import discover_challenge_bundle, load_candidate_id_set
from src.data.schema import combine_text_values
from src.eval.submission import validate_track1_submission
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser
from src.ranking.explainer import explain_ranking
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.consistency import score_consistency
from src.utils.paths import MODELS_DIR, SUBMISSIONS_DIR, ensure_project_dirs
from src.utils.role_relevance import score_career_trajectory, score_role_relevance
from src.utils.skill_ontology import SkillMatcher

BM25_INDEX = MODELS_DIR / "bm25_demo_index.pkl"
DENSE_INDEX = MODELS_DIR / "dense_demo_index"
TOP_K = 100


# ── Track 1 JD constants — shared, dependency-free source of truth ──────────
# Extracted directly from the DOCX job description for precision (the rule-based
# parser misses most skills due to the prose format). Defined in
# src/utils/track1_spec.py so rank.py can reuse them without import cost.
from src.utils.track1_spec import (  # noqa: E402
    TRACK1_JOB_SENIORITY,
    TRACK1_JOB_TITLE,
    TRACK1_MAX_YEARS,
    TRACK1_MIN_YEARS,
    TRACK1_MUST_HAVE_SKILLS,
    TRACK1_NICE_TO_HAVE_SKILLS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def score_behavior(row: pd.Series) -> float:
    """Composite behavioral/activity score in [0, 1]."""
    completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
    if completeness == 0.0:
        completeness = _safe_float(row.get("profile_completeness"))

    response_rate = _safe_float(row.get("recruiter_response_rate"))
    github        = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
    saved         = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
    search        = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
    assessment    = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
    open_to_work  = 1.0 if bool(row.get("open_to_work_flag")) else 0.0

    recency = 0.4
    raw_last = _safe_text(row.get("last_active"))
    if raw_last:
        try:
            active_date = datetime.fromisoformat(raw_last[:10]).date()
            days_since = max((date.today() - active_date).days, 0)
            recency = max(0.0, 1.0 - days_since / 180.0)
        except ValueError:
            pass

    return round(float(np.clip(
        0.25 * completeness
        + 0.20 * recency
        + 0.20 * response_rate
        + 0.10 * open_to_work
        + 0.10 * github
        + 0.05 * saved
        + 0.05 * search
        + 0.05 * assessment,
        0.0, 1.0,
    )), 6)


def score_location(candidate_location: str) -> float:
    """Score candidate location fit for a Pune/Noida hybrid role in India."""
    loc = str(candidate_location).lower().strip()
    india_tokens = (
        "india", "pune", "noida", "bangalore", "bengaluru",
        "hyderabad", "mumbai", "delhi", "chennai", "gurugram",
        "gurgaon", "kolkata", "ahmedabad", "jaipur", "kochi",
    )
    if any(t in loc for t in india_tokens):
        return 1.0
    if not loc or loc in ("nan", "none", "not specified", ""):
        return 0.55
    return 0.25


def _load_ltr_model():
    import pickle
    from src.utils.paths import MODELS_DIR
    ltr_path = MODELS_DIR / "ltr_model.pkl"
    if not ltr_path.exists():
        return None, []
    try:
        with ltr_path.open("rb") as f:
            payload = pickle.load(f)
        print(f"  LTR model loaded from {ltr_path}")
        return payload["model"], payload["feature_cols"]
    except Exception as exc:
        print(f"  LTR model load failed ({exc}) — using hardcoded weights")
        return None, []


def _ltr_rescore(
    scored: list[dict],
    ltr_model,
    feature_cols: list[str],
    *,
    blend: float = 0.4,
) -> list[dict]:
    """Blend LTR predictions with the hand-tuned 8-signal formula.

    The LTR model is trained on *pseudo*-labels derived from role/experience
    signals that are also its own features, so it is partially circular (see
    docs/ablation.md). We therefore *blend* rather than override: the interpretable
    formula stays the ranker of record (weight ``1 - blend``) and LTR contributes
    a minority signal (weight ``blend``). Both are min-max normalised across the
    recall set before mixing so the scales are comparable.
    """
    if ltr_model is None or not scored:
        return scored
    X = pd.DataFrame(
        [{col: s.get(col, 0.0) for col in feature_cols} for s in scored]
    ).fillna(0.0).values
    ltr_scores = np.asarray(ltr_model.predict(X), dtype=float)

    lo, hi = float(ltr_scores.min()), float(ltr_scores.max())
    ltr_norm = (ltr_scores - lo) / (hi - lo) if hi > lo else np.full(len(ltr_scores), 0.5)

    formula = np.asarray([s["overall"] for s in scored], dtype=float)
    f_lo, f_hi = float(formula.min()), float(formula.max())
    formula_norm = (formula - f_lo) / (f_hi - f_lo) if f_hi > f_lo else np.full(len(formula), 0.5)

    for s, fn, ln in zip(scored, formula_norm, ltr_norm):
        s["overall"] = float((1.0 - blend) * fn + blend * ln)
    return sorted(scored, key=lambda x: x["overall"], reverse=True)


def score_experience(cand_years: float, min_years: float, max_years: float) -> float:
    """Score candidate years-of-experience against the JD range."""
    if not min_years and not max_years:
        return 0.5
    if min_years and cand_years < min_years:
        return round(max(0.0, cand_years / max(min_years, 1.0)), 6)
    if max_years and cand_years > max_years:
        extra = cand_years - max_years
        return round(max(0.45, 1.0 - extra / max(max_years * 2.0, 1.0)), 6)
    return 1.0


# ── Optional: semantic similarity model ──────────────────────────────────────

def _load_semantic_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading semantic model ({model_name})…")
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        print(f"  Semantic model unavailable ({exc}) — semantic scores set to 0.5.")
        return None


def _semantic_scores(
    model,
    job_text: str,
    candidate_texts: list[str],
    batch_size: int = 64,
) -> list[float]:
    if model is None:
        return [0.5] * len(candidate_texts)

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    job_emb = model.encode([job_text], normalize_embeddings=True, show_progress_bar=False)
    scores: list[float] = []
    for i in range(0, len(candidate_texts), batch_size):
        batch = candidate_texts[i : i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        scores.extend(cosine_similarity(job_emb, embs)[0].tolist())
    return scores


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_submission(
    *,
    recall_k: int = 2000,
    use_semantic: bool = True,
    use_dense: bool = True,
    use_cross_encoder: bool = True,
    use_ltr: bool = False,
    use_llm_rerank: bool = False,
) -> pd.DataFrame:
    ensure_project_dirs()

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading candidate pool…")
    bundle = load_phase2_bundle(require_labels=False)
    candidates = bundle.candidates.copy()
    cand_id_col = bundle.candidate_schema.candidate_id
    candidates["_cid"] = candidates[cand_id_col].astype(str)
    lookup = candidates.set_index("_cid", drop=False)

    cand_parser   = CandidateProfileParser()
    skill_matcher = SkillMatcher()

    # Use hardcoded Track 1 skills — far more accurate than rule-based parser
    must_skills  = TRACK1_MUST_HAVE_SKILLS
    nice_skills  = TRACK1_NICE_TO_HAVE_SKILLS
    min_years    = TRACK1_MIN_YEARS
    max_years    = TRACK1_MAX_YEARS
    job_title    = TRACK1_JOB_TITLE
    job_seniority = TRACK1_JOB_SENIORITY

    # Build JD text from the processed job file (used for retrieval + semantic)
    job_row  = bundle.jobs.iloc[0]
    job_text = combine_text_values(job_row, bundle.job_schema.text_columns)
    if not job_text.strip():
        job_text = " ".join(must_skills)   # fallback

    print(f"  Job: '{job_title}'")
    print(f"  Must-have skills: {len(must_skills)}, Nice-to-have: {len(nice_skills)}")

    # ── Stage 0: BM25 + Dense indices ───────────────────────────────────────
    ids: list[str] | None = None
    docs: list[str] | None = None

    bm25 = BM25Retriever()
    if BM25_INDEX.exists():
        print(f"Loading BM25 index from {BM25_INDEX}…")
        bm25.load(BM25_INDEX)
    else:
        print(f"Building BM25 index ({len(bundle.candidates):,} candidates)…")
        ids, docs = build_candidate_documents(bundle)
        bm25.build_index(documents=docs, candidate_ids=ids)
        BM25_INDEX.parent.mkdir(parents=True, exist_ok=True)
        bm25.save(BM25_INDEX)

    # ── Stage 0b: Dense index (skipped when --no-dense) ─────────────────────
    extra_terms = [s for s in must_skills if s.lower() not in job_text.lower()]
    skill_query = job_text + (" " + " ".join(extra_terms) if extra_terms else "")

    t0 = time.perf_counter()
    if use_dense:
        dense = DenseRetriever()
        _dense_meta = DENSE_INDEX.parent / (DENSE_INDEX.name + ".meta.pkl")
        if _dense_meta.exists():
            print(f"Loading dense index from {DENSE_INDEX}…")
            dense.load(DENSE_INDEX)
        else:
            if ids is None:
                print("Building candidate documents for dense index…")
                ids, docs = build_candidate_documents(bundle)
            print(f"Building dense index ({len(ids):,} candidates) — ~10 min on CPU…")
            dense.build_index(documents=docs, candidate_ids=ids)
            DENSE_INDEX.parent.mkdir(parents=True, exist_ok=True)
            dense.save(DENSE_INDEX)

        retriever = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        recall_results = retriever.retrieve(skill_query, top_k=recall_k, recall_k=recall_k)
        print(f"Hybrid recall (BM25+dense): {len(recall_results)} candidates in {(time.perf_counter()-t0)*1000:.0f} ms")
    else:
        print("Dense retrieval skipped (--no-dense). Using BM25-only recall…")
        recall_results = bm25.retrieve(skill_query, top_k=recall_k)
        print(f"BM25 recall: {len(recall_results)} candidates in {(time.perf_counter()-t0)*1000:.0f} ms")

    if not recall_results:
        raise RuntimeError("BM25 recall returned 0 results — rebuild the index.")

    raw_scores = np.asarray([s for _, s in recall_results], dtype=float)
    score_max  = float(raw_scores.max()) or 1e-9

    # ── Stage 1b: Semantic similarity ────────────────────────────────────────
    sem_model = _load_semantic_model() if use_semantic else None
    recalled_profile_texts: list[str] = []
    for cid, _ in recall_results:
        try:
            row = lookup.loc[str(cid)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            text = _safe_text(row.get("profile_text")) or _safe_text(row.get("headline"))
        except KeyError:
            text = ""
        recalled_profile_texts.append(text)

    print("Computing semantic similarity scores…")
    raw_sem = _semantic_scores(sem_model, skill_query, recalled_profile_texts)
    sem_max = max(raw_sem) or 1e-9

    # ── Stage 2: Multi-signal scoring ────────────────────────────────────────
    print(f"Multi-signal scoring {len(recall_results)} candidates…")
    scored: list[dict] = []

    for idx, (cid, raw_retrieval) in enumerate(recall_results):
        retrieval_norm = float(raw_retrieval) / score_max
        sem_norm       = float(raw_sem[idx]) / sem_max

        try:
            row = lookup.loc[str(cid)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            row = pd.Series(dtype=object)

        parsed_cand    = cand_parser.parse_row(row, bundle.candidate_schema)
        cand_skills    = list(parsed_cand.get("skills") or [])
        cand_years     = _safe_float(
            parsed_cand.get("total_experience_years"),
            default=_safe_float(row.get("total_experience")),
        )
        cand_seniority = str(parsed_cand.get("seniority", "")).lower()
        cand_location  = _safe_text(row.get("location")) or _safe_text(row.get("country"))

        # Skill matching: must-have (75%) + nice-to-have (25%)
        must_match  = skill_matcher.match_score(must_skills, cand_skills)
        nice_match  = skill_matcher.match_score(nice_skills, cand_skills)
        skill_score = must_match["composite_score"] * 0.75 + nice_match["composite_score"] * 0.25

        exp_score      = score_experience(cand_years, min_years, max_years)

        # Pre-compute behavioral sub-signals for LTR feature vector
        completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
        if completeness == 0.0:
            completeness = _safe_float(row.get("profile_completeness"))
        response_rate = _safe_float(row.get("recruiter_response_rate"))
        github_score  = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
        saved         = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
        search_app    = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
        assessment    = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
        open_to_work  = 1.0 if bool(row.get("open_to_work_flag")) else 0.0
        recency       = 0.4
        raw_last = _safe_text(row.get("last_active"))
        if raw_last:
            try:
                active_date = datetime.fromisoformat(raw_last[:10]).date()
                days_since = max((date.today() - active_date).days, 0)
                recency = max(0.0, 1.0 - days_since / 180.0)
            except ValueError:
                pass

        beh_score      = score_behavior(row)
        role_score     = score_role_relevance(
            _safe_text(row.get("current_role")),
            _safe_text(row.get("headline")),
        )
        career_score   = score_career_trajectory(_safe_text(row.get("career_history_text")))
        location_score = score_location(cand_location)

        # Internal-consistency / honeypot check (see src/utils/consistency.py).
        # Honeypots are forced to tier 0 in the hidden ground truth and a >10%
        # honeypot rate in the top 100 is an automatic Stage-3 disqualification.
        consistency = score_consistency(
            _safe_text(row.get("skills_detailed")), cand_years
        )

        # Weighted formula — 9 signals, sums to 1.0. Re-balanced away from raw
        # keyword/skill matching (which the JD explicitly calls a trap) toward
        # production-evidence (career trajectory) and profile plausibility.
        # retrieval 0.13, semantic 0.18, skill 0.18, role 0.12, exp 0.10,
        # behavioral 0.08, career 0.12, location 0.04, consistency 0.05
        if use_semantic:
            overall = (
                0.13 * retrieval_norm
                + 0.18 * sem_norm
                + 0.18 * skill_score
                + 0.12 * role_score
                + 0.10 * exp_score
                + 0.08 * beh_score
                + 0.12 * career_score
                + 0.04 * location_score
                + 0.05 * consistency.consistency_score
            )
        else:
            # Redistribute semantic weight when model unavailable
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

        # Hard cap: a detected honeypot cannot reach the shortlist regardless of
        # how many keywords it stuffs into the profile.
        if consistency.is_honeypot:
            overall *= 0.05

        nice_matched = sorted(
            set(nice_match.get("matched_skills", [])) | (set(must_match.get("matched_skills", [])) & set(nice_skills))
        )

        scored.append({
            "candidate_id":   str(cid),
            "retrieval_norm": retrieval_norm,
            "sem_norm":       sem_norm,
            "skill_score":    skill_score,
            "exp_score":      exp_score,
            "beh_score":      beh_score,
            "role_score":     role_score,
            "career_score":   career_score,
            "location_score": location_score,
            "consistency_score": consistency.consistency_score,
            "is_honeypot":    consistency.is_honeypot,
            "overall":        overall,
            "matched_skills": list(must_match.get("matched_skills", [])),
            "matched_nice":   nice_matched,
            "missing_skills": list(must_match.get("missing_skills", [])),
            "cand_years":     cand_years,
            "current_title":  _safe_text(row.get("current_role")),
            "location":       cand_location,
            "notice_period":  _safe_float(row.get("notice_period_days")),
            "seniority_match": job_seniority == cand_seniority,
            # LTR feature vector (needed by _ltr_rescore)
            "must_composite":  must_match.get("composite_score", 0.0),
            "nice_composite":  nice_match.get("composite_score", 0.0),
            "must_exact_cov":  must_match.get("exact_coverage", 0.0),
            "nice_exact_cov":  nice_match.get("exact_coverage", 0.0),
            "n_must_matched":  must_match.get("n_exact_matched", 0),
            "n_must_missing":  must_match.get("n_missing", 0),
            "completeness":    completeness,
            "response_rate":   response_rate,
            "recency":         recency,
            "github_score":    github_score,
            "saved":           saved,
            "search_app":      search_app,
            "assessment":      assessment,
            "open_to_work":    open_to_work,
        })

    scored.sort(key=lambda x: x["overall"], reverse=True)

    # Preserve the 8-signal scores before LTR overrides them — used for the
    # submitted score column so the values reflect actual signal quality.
    for s in scored:
        s["overall_8signal"] = s["overall"]

    # Stage 2b: LTR reranking (opt-in blend; off by default)
    # The pseudo-label LTR model is partially circular, so it is NOT the ranker
    # of record. With --ltr it contributes a minority (40%) blended signal.
    if use_ltr:
        ltr_model, ltr_feature_cols = _load_ltr_model()
        if ltr_model is not None:
            print("Blending LTR predictions (40%) with the 8-signal formula…")
            scored = _ltr_rescore(scored, ltr_model, ltr_feature_cols, blend=0.4)

    top5_roles = [
        _safe_text(lookup.loc[s["candidate_id"]].get("current_role"))
        if s["candidate_id"] in lookup.index else "?"
        for s in scored[:5]
    ]
    print(f"Top-5 roles after Stage 2: {top5_roles}")
    honeypots_in_top100 = sum(1 for s in scored[:TOP_K] if s.get("is_honeypot"))
    print(f"Honeypots detected in recall set: {sum(1 for s in scored if s.get('is_honeypot'))} "
          f"| in top {TOP_K}: {honeypots_in_top100}")

    # ── Stage 3: Cross-encoder re-rank of top 50 ─────────────────────────────
    if use_cross_encoder:
        print("Cross-encoder re-ranking top 50…")
        try:
            from src.ranking.cross_encoder import CrossEncoderReranker
            ce_reranker = CrossEncoderReranker()
            top50_pairs: list[tuple[str, str]] = []
            for s in scored[:50]:
                cid = s["candidate_id"]
                try:
                    profile_text = _safe_text(lookup.loc[cid].get("profile_text"))
                except KeyError:
                    profile_text = ""
                top50_pairs.append((cid, profile_text))

            ce_ranked = ce_reranker.rerank(job_text, top50_pairs, top_k=50)
            ce_order  = {cid: i for i, (cid, _) in enumerate(ce_ranked)}
            top50_rescored = sorted(
                scored[:50],
                key=lambda s: ce_order.get(s["candidate_id"], 99),
            )
            scored = top50_rescored + scored[50:]
            top3 = [s["candidate_id"] for s in scored[:3]]
            print(f"  Cross-encoder done — top-3: {top3}")
        except Exception as exc:
            print(f"  Cross-encoder skipped ({exc})")

    # ── Stage 4: LLM listwise re-rank of top 30 ──────────────────────────────
    # WARNING: this calls the local Ollama server (network). The competition's
    # Stage-3 reproduction runs with the network OFF, so this MUST NOT be used to
    # produce the submitted CSV — it exists only for offline analysis/comparison.
    # The canonical, network-free submission entrypoint is `rank.py`.
    if use_llm_rerank:
        print("WARNING: --llm-rerank makes network calls and is FORBIDDEN for the")
        print("         official submission (Stage-3 runs with no network). Use only offline.")
        print("LLM listwise re-ranking top 30 (local Ollama)…")
        try:
            from src.ranking.llm_reranker import LLMReranker
            reranker = LLMReranker()
            if not reranker.enabled:
                print("  Ollama server not reachable — skipping LLM re-rank.")
            else:
                top30_pairs: list[tuple[str, str]] = []
                for s in scored[:30]:
                    cid = s["candidate_id"]
                    try:
                        profile_text = _safe_text(lookup.loc[cid].get("profile_text"))
                    except KeyError:
                        profile_text = ""
                    top30_pairs.append((cid, profile_text))

                reranked = reranker.rerank(job_text, top30_pairs, top_k=30)
                reranked_order = {cid: i for i, (cid, _) in enumerate(reranked)}
                top30 = sorted(
                    scored[:30],
                    key=lambda s: reranked_order.get(s["candidate_id"], 99),
                )
                scored = top30 + scored[30:]
                print(f"  LLM re-rank done — top-3: {[s['candidate_id'] for s in scored[:3]]}")
        except Exception as exc:
            print(f"  LLM re-rank failed ({exc})")

    # ── Build submission rows ─────────────────────────────────────────────────
    top = scored[:TOP_K]
    # Use the pre-LTR 8-signal scores for the submitted score column.
    # LTR + cross-encoder determine ORDER; the 8-signal formula provides
    # interpretable, well-spread score values to assign to each rank position.
    signal_pool = sorted([s["overall_8signal"] for s in top], reverse=True)
    anchor = signal_pool[0] if signal_pool else 1.0

    rows: list[dict] = []
    for rank, s in enumerate(top, start=1):
        normalised_score = round(signal_pool[rank - 1] / max(anchor, 1e-9), 6)
        # Only surface semantic alignment text when an actual semantic model was used
        sem_score_for_explain = s["sem_norm"] if use_semantic else 0.0
        reasoning = explain_ranking(
            rank=rank,
            job_title=job_title,
            must_skills=must_skills,
            nice_skills=nice_skills,
            candidate_skills=s["matched_skills"] + s["missing_skills"],
            cand_years_exp=s["cand_years"],
            job_min_years=min_years,
            job_max_years=max_years,
            seniority_match=s["seniority_match"],
            behavioral_score=s["beh_score"],
            semantic_score=sem_score_for_explain,
            current_title=s.get("current_title", ""),
            location=s.get("location", ""),
            matched_must=s["matched_skills"],
            matched_nice=s.get("matched_nice", []),
            response_rate=s.get("response_rate"),
            github_score=s.get("github_score", 0.0),
            open_to_work=bool(s.get("open_to_work")),
            notice_period_days=int(s["notice_period"]) if s.get("notice_period") else None,
        )
        rows.append({
            "candidate_id": s["candidate_id"],
            "rank":         rank,
            "score":        normalised_score,
            "reasoning":    reasoning,
        })

    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--recall-k", type=int, default=2000)
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip semantic similarity scoring (faster, lower quality)")
    parser.add_argument("--no-dense", action="store_true",
                        help="Skip dense retrieval — use BM25-only recall (no BGE model needed)")
    parser.add_argument("--no-crossencoder", action="store_true",
                        help="Skip cross-encoder re-ranking")
    parser.add_argument("--ltr", action="store_true",
                        help="Blend the (partially circular) pseudo-label LTR model at 40%% (off by default)")
    parser.add_argument("--llm-rerank", action="store_true",
                        help="OFFLINE ONLY — network LLM re-rank, forbidden for the official submission")
    parser.add_argument("--validate", action="store_true",
                        help="Validate submission against released candidate pool")
    parser.add_argument("--output", type=Path,
                        default=SUBMISSIONS_DIR / "final_submission.csv")
    args = parser.parse_args()

    t_total = time.perf_counter()
    submission = build_submission(
        recall_k=args.recall_k,
        use_semantic=not args.no_semantic,
        use_dense=not args.no_dense,
        use_cross_encoder=not args.no_crossencoder,
        use_ltr=args.ltr,
        use_llm_rerank=args.llm_rerank,
    )
    elapsed = time.perf_counter() - t_total
    print(f"\nGenerated {len(submission)} rows in {elapsed:.1f}s")

    if args.validate:
        print("\nValidating against released candidate pool…")
        try:
            bundle = discover_challenge_bundle()
            valid_ids = load_candidate_id_set(bundle)
            issues = validate_track1_submission(submission, valid_candidate_ids=valid_ids)
        except FileNotFoundError:
            issues = validate_track1_submission(submission)

        if issues:
            print("VALIDATION FAILED:")
            for issue in issues:
                print(f"  - {issue}")
            raise SystemExit(1)
        print("  Validation passed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Submission saved : {args.output}")
    print(f"Rows             : {len(submission)}")
    print(f"Score range      : {submission['score'].min():.4f} – {submission['score'].max():.4f}")
    print(f"\nTop 10 candidates:")
    print(submission[["rank", "candidate_id", "score"]].head(10).to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
