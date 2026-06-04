"""India Runs Track 1 interactive ranking demo.

Paste a job description and get ranked candidates with transparent score
breakdowns. The demo uses the released public bundle and avoids pretending that
hidden evaluation labels or trained LTR artifacts exist locally.

Run a core smoke test:
    python app/demo.py --smoke

Run the Gradio UI:
    python app/demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import build_candidate_documents, load_phase2_bundle
from src.data.schema import combine_text_values
from src.parsing.candidate_parser import CandidateProfileParser
from src.parsing.jd_parser import JobDescriptionParser
from src.ranking.explainer import explain_ranking
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.paths import MODELS_DIR
from src.utils.skill_ontology import SKILL_FAMILIES, SKILL_SYNONYMS, SkillMatcher, normalize_skill


BM25_DEMO_INDEX = MODELS_DIR / "bm25_demo_index.pkl"
DEFAULT_RECALL_K = 300
SCORE_COLUMNS = ["BM25", "Skill", "Experience Fit", "Behavioral"]


EXAMPLE_JD = """Senior AI Engineer - Founding Team

We are hiring a senior engineer to own the intelligence layer of a talent
platform. The role needs production experience with Python, embeddings-based
retrieval, vector databases, hybrid search, ranking evaluation, and LLM-powered
reranking.

Requirements:
- 5-9 years of applied ML or search/recommendation experience
- Strong Python and production engineering judgment
- Hands-on retrieval systems using BGE, E5, OpenAI embeddings, FAISS, Qdrant,
  Pinecone, Weaviate, Milvus, OpenSearch, or Elasticsearch
- Experience evaluating ranking systems with NDCG, MAP, MRR, A/B tests, and
  recruiter feedback loops
- Bonus: LoRA, PEFT, learning-to-rank, marketplace products, or HR-tech
"""


DEMO_TERMS = {
    "A/B Testing",
    "Artificial Intelligence",
    "BGE",
    "CI/CD",
    "Docker",
    "E5",
    "Elasticsearch",
    "Embeddings",
    "FAISS",
    "FastAPI",
    "Feature Engineering",
    "Google Cloud Platform",
    "Hybrid Search",
    "Kubernetes",
    "Large Language Models",
    "Learning to Rank",
    "LightGBM",
    "LoRA",
    "Machine Learning",
    "MAP",
    "Milvus",
    "MRR",
    "Natural Language Processing",
    "NDCG",
    "OpenAI Embeddings",
    "OpenSearch",
    "PEFT",
    "Pinecone",
    "Python",
    "Qdrant",
    "QLoRA",
    "Ranking",
    "Recommendation Systems",
    "Retrieval",
    "Sentence Transformers",
    "Vector Databases",
    "Weaviate",
}


@dataclass(frozen=True)
class CandidateScore:
    """Score details for one retrieved candidate."""

    candidate_id: str
    bm25_score: float
    skill_score: float
    experience_score: float
    behavioral_score: float
    overall_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    rationale: str


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_skills(value: object) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    skills: list[str] = []
    seen: set[str] = set()
    for raw_part in text.replace(";", ",").replace("|", ",").split(","):
        part = raw_part.strip(" .:-")
        if not part:
            continue
        canonical = normalize_skill(part)
        lowered = canonical.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        skills.append(canonical)
    return skills


def _term_variants() -> list[tuple[str, str]]:
    terms = set(DEMO_TERMS)
    terms.update(SKILL_SYNONYMS)
    terms.update(SKILL_SYNONYMS.values())
    for family_skills in SKILL_FAMILIES.values():
        terms.update(family_skills)

    variants: list[tuple[str, str]] = []
    for term in sorted(terms, key=len, reverse=True):
        canonical = normalize_skill(term)
        variants.append((term.lower(), canonical))
    return variants


TERM_VARIANTS = _term_variants()


def extract_skill_mentions(text: str, *, max_skills: int = 16) -> list[str]:
    """Extract known technical and ranking terms from a job description."""
    lowered = f" {_safe_text(text).lower()} "
    found: list[str] = []
    seen: set[str] = set()

    for variant, canonical in TERM_VARIANTS:
        if not variant or canonical.lower() in seen:
            continue
        if variant in lowered:
            seen.add(canonical.lower())
            found.append(canonical)
        if len(found) >= max_skills:
            break

    return found


def score_experience(candidate_years: float, min_years: float | None, max_years: float | None) -> float:
    """Score candidate experience against a parsed JD range."""
    if not min_years and not max_years:
        return 0.5
    if min_years and candidate_years < min_years:
        return round(max(0.0, candidate_years / max(min_years, 1.0)), 4)
    if max_years and candidate_years > max_years:
        extra_years = candidate_years - max_years
        return round(max(0.45, 1.0 - extra_years / max(max_years * 2.0, 1.0)), 4)
    return 1.0


def score_behavior(row: pd.Series) -> float:
    """Combine public behavioral/activity signals into a 0-1 score."""
    completeness = _safe_float(row.get("profile_completeness_score")) / 100.0
    if completeness == 0.0:
        completeness = _safe_float(row.get("profile_completeness"))

    response_rate = _safe_float(row.get("recruiter_response_rate"))
    github_activity = min(_safe_float(row.get("github_activity_score")) / 100.0, 1.0)
    saved_by_recruiters = min(_safe_float(row.get("saved_by_recruiters_30d")) / 10.0, 1.0)
    search_appearance = min(_safe_float(row.get("search_appearance_30d")) / 500.0, 1.0)
    assessment_avg = min(_safe_float(row.get("skill_assessment_avg")) / 100.0, 1.0)
    open_to_work = 1.0 if bool(row.get("open_to_work_flag")) else 0.0

    recency = 0.4
    raw_last_active = _safe_text(row.get("last_active"))
    if raw_last_active:
        try:
            active_date = datetime.fromisoformat(raw_last_active[:10]).date()
            days_since_active = max((date.today() - active_date).days, 0)
            recency = max(0.0, 1.0 - days_since_active / 180.0)
        except ValueError:
            recency = 0.4

    score = (
        0.25 * completeness
        + 0.20 * recency
        + 0.20 * response_rate
        + 0.10 * open_to_work
        + 0.10 * github_activity
        + 0.05 * saved_by_recruiters
        + 0.05 * search_appearance
        + 0.05 * assessment_avg
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def score_breakdown_chart(result_df: pd.DataFrame | None) -> go.Figure:
    """Build a grouped bar chart for the top candidates."""
    figure = go.Figure()
    if result_df is None or result_df.empty:
        figure.update_layout(title="Score Breakdown", height=320)
        return figure

    top_rows = result_df.head(5)
    for column in SCORE_COLUMNS:
        figure.add_trace(
            go.Bar(
                name=column,
                x=top_rows["Candidate ID"],
                y=top_rows[column],
                text=[f"{value:.2f}" for value in top_rows[column]],
                textposition="outside",
            )
        )
    figure.update_layout(
        title="Score Breakdown - Top 5 Candidates",
        barmode="group",
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        height=340,
        margin=dict(l=32, r=24, t=56, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


class DemoRankingEngine:
    """Runtime state for the local/Hugging Face demo."""

    def __init__(self) -> None:
        self.bundle = load_phase2_bundle(require_labels=False)
        self.jobs = self.bundle.jobs
        self.candidates = self.bundle.candidates.copy()
        candidate_id_col = self.bundle.candidate_schema.candidate_id
        self.candidates["_candidate_id_str"] = self.candidates[candidate_id_col].astype(str)
        self.candidate_lookup = self.candidates.set_index("_candidate_id_str", drop=False)
        self.job_parser = JobDescriptionParser()
        self.candidate_parser = CandidateProfileParser()
        self.skill_matcher = SkillMatcher()
        self.bm25 = self._load_or_build_bm25()

    def _load_or_build_bm25(self) -> BM25Retriever:
        retriever = BM25Retriever()
        if BM25_DEMO_INDEX.exists():
            try:
                retriever.load(BM25_DEMO_INDEX)
                return retriever
            except Exception:
                BM25_DEMO_INDEX.unlink(missing_ok=True)

        candidate_ids, documents = build_candidate_documents(self.bundle)
        retriever.build_index(documents=documents, candidate_ids=candidate_ids)
        retriever.save(BM25_DEMO_INDEX)
        return retriever

    def default_job_description(self) -> str:
        if self.jobs.empty:
            return EXAMPLE_JD
        first_job = self.jobs.iloc[0]
        job_text = combine_text_values(first_job, self.bundle.job_schema.text_columns)
        return job_text or EXAMPLE_JD

    def parse_job(self, job_description: str) -> dict[str, Any]:
        parsed = self.job_parser.parse_text(job_description)
        extracted_skills = extract_skill_mentions(job_description)
        if extracted_skills:
            parsed["must_have_skills"] = extracted_skills
        return parsed

    def _candidate_row(self, candidate_id: str) -> pd.Series:
        selected = self.candidate_lookup.loc[str(candidate_id)]
        if isinstance(selected, pd.DataFrame):
            return selected.iloc[0]
        return selected

    def _score_candidate(
        self,
        *,
        rank_position: int,
        parsed_job: dict[str, Any],
        candidate_id: str,
        raw_bm25_score: float,
        bm25_norm: float,
    ) -> CandidateScore:
        row = self._candidate_row(candidate_id)
        parsed_candidate = self.candidate_parser.parse_row(row, self.bundle.candidate_schema)

        candidate_skills = parsed_candidate.get("skills") or _split_skills(row.get("skills"))
        must_skills = list(parsed_job.get("must_have_skills", []))
        nice_skills = list(parsed_job.get("nice_to_have_skills", []))
        skill_match = self.skill_matcher.match_score(must_skills, candidate_skills)

        candidate_years = _safe_float(
            parsed_candidate.get("total_experience_years"),
            default=_safe_float(row.get("total_experience")),
        )
        experience = score_experience(
            candidate_years,
            parsed_job.get("min_years_experience"),
            parsed_job.get("max_years_experience"),
        )
        behavioral = score_behavior(row)

        overall = (
            0.38 * bm25_norm
            + 0.27 * skill_match["composite_score"]
            + 0.17 * experience
            + 0.18 * behavioral
        )
        rationale = explain_ranking(
            rank=rank_position,
            job_title=str(parsed_job.get("title", "")),
            must_skills=must_skills,
            nice_skills=nice_skills,
            candidate_skills=list(candidate_skills),
            cand_years_exp=candidate_years,
            job_min_years=_safe_float(parsed_job.get("min_years_experience")),
            seniority_match=(
                str(parsed_job.get("seniority", "")).lower()
                == str(parsed_candidate.get("seniority", "")).lower()
            ),
            behavioral_score=behavioral,
            semantic_score=bm25_norm,
        )

        return CandidateScore(
            candidate_id=str(candidate_id),
            bm25_score=round(bm25_norm, 4),
            skill_score=float(skill_match["composite_score"]),
            experience_score=experience,
            behavioral_score=behavioral,
            overall_score=round(float(overall), 4),
            matched_skills=list(skill_match.get("matched_skills", [])),
            missing_skills=list(skill_match.get("missing_skills", [])),
            rationale=rationale,
        )

    def rank(self, job_description: str, top_k: int = 10) -> tuple[pd.DataFrame, str, go.Figure]:
        job_description = _safe_text(job_description)
        if not job_description:
            empty = pd.DataFrame()
            return empty, "Paste a job description to start ranking.", score_breakdown_chart(empty)

        start_time = time.perf_counter()
        parsed_job = self.parse_job(job_description)
        recall_k = max(DEFAULT_RECALL_K, min(1000, int(top_k) * 40))
        recall_results = self.bm25.retrieve(job_description, top_k=recall_k)
        if not recall_results:
            empty = pd.DataFrame()
            return empty, "No candidates retrieved for this job description.", score_breakdown_chart(empty)

        raw_scores = np.asarray([score for _, score in recall_results], dtype=float)
        score_min = float(raw_scores.min())
        score_max = float(raw_scores.max())
        score_range = score_max - score_min

        scored_candidates: list[CandidateScore] = []
        for index, (candidate_id, raw_score) in enumerate(recall_results, start=1):
            bm25_norm = 0.5 if score_range <= 0 else (float(raw_score) - score_min) / score_range
            scored_candidates.append(
                self._score_candidate(
                    rank_position=index,
                    parsed_job=parsed_job,
                    candidate_id=candidate_id,
                    raw_bm25_score=float(raw_score),
                    bm25_norm=bm25_norm,
                )
            )

        ranked = sorted(scored_candidates, key=lambda item: item.overall_score, reverse=True)[: int(top_k)]
        output_rows: list[dict[str, object]] = []
        for rank, score in enumerate(ranked, start=1):
            row = self._candidate_row(score.candidate_id)
            skills = _split_skills(row.get("skills"))
            output_rows.append(
                {
                    "Rank": rank,
                    "Candidate ID": score.candidate_id,
                    "Current Role": _safe_text(row.get("current_role")),
                    "Company": _safe_text(row.get("current_company")),
                    "Location": _safe_text(row.get("location")) or _safe_text(row.get("country")),
                    "Experience (yrs)": round(_safe_float(row.get("total_experience")), 1),
                    "Overall": score.overall_score,
                    "BM25": score.bm25_score,
                    "Skill": score.skill_score,
                    "Experience Fit": score.experience_score,
                    "Behavioral": score.behavioral_score,
                    "Matched Skills": ", ".join(score.matched_skills[:5]) or "-",
                    "Missing Skills": ", ".join(score.missing_skills[:5]) or "-",
                    "Top Profile Skills": ", ".join(skills[:6]) or "-",
                    "Why This Rank": score.rationale,
                }
            )

        result_df = pd.DataFrame(output_rows)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        skills_preview = ", ".join(parsed_job.get("must_have_skills", [])[:6]) or "none inferred"
        status = (
            f"Ranked {len(result_df)} candidates from {len(recall_results)} recalled profiles "
            f"in {elapsed_ms:.0f} ms. Parsed skills: {skills_preview}. "
            "Scoring uses BM25 + skill + experience + behavioral signals; hidden labels are not local."
        )
        return result_df, status, score_breakdown_chart(result_df)


_ENGINE: DemoRankingEngine | None = None


def get_engine() -> DemoRankingEngine:
    """Return the lazily initialized demo engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DemoRankingEngine()
    return _ENGINE


def rank_candidates(job_description: str, top_k: int = 10) -> tuple[pd.DataFrame, str, go.Figure]:
    """Gradio-compatible ranking function."""
    return get_engine().rank(job_description, top_k=int(top_k))


def build_demo() -> Any:
    """Create the Gradio Blocks app."""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Gradio is not installed. Install demo dependencies with "
            "`pip install -r app/requirements.txt` and rerun `python app/demo.py`."
        ) from exc

    default_jd = get_engine().default_job_description()
    with gr.Blocks(
        title="India Runs - Intelligent Candidate Discovery",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# India Runs - Intelligent Candidate Discovery
Paste a job description to rank the released 100,000-candidate pool with transparent explanations.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                jd_input = gr.Textbox(
                    label="Job Description",
                    value=default_jd,
                    lines=18,
                    placeholder="Paste a full JD here...",
                )
                top_k_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=5,
                    label="Candidates to show",
                )
                rank_button = gr.Button("Rank Candidates", variant="primary", size="lg")
            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Pipeline Status", interactive=False)
                breakdown_chart = gr.Plot(label="Score Breakdown")
                results_table = gr.Dataframe(
                    label="Ranked Candidates",
                    interactive=False,
                    wrap=True,
                    max_height=620,
                )

        rank_button.click(
            fn=rank_candidates,
            inputs=[jd_input, top_k_slider],
            outputs=[results_table, status_box, breakdown_chart],
        )
        gr.Markdown(
            """
Pipeline: JD parsing -> BM25 recall -> skill/experience/behavioral scoring -> explanations.
LTR, cross-encoder, and LLM reranking modules are implemented in the repo and remain ready for
trained artifacts or organizer labels.
"""
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the India Runs demo.")
    parser.add_argument("--smoke", action="store_true", help="Run one ranking pass without Gradio.")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    args = parser.parse_args()

    if args.smoke:
        result_df, status, _ = rank_candidates(EXAMPLE_JD, top_k=5)
        print(status)
        print(result_df[["Rank", "Candidate ID", "Current Role", "Overall"]].to_string(index=False))
        return

    demo = build_demo()
    demo.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
