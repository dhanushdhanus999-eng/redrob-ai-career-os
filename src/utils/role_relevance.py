"""Role relevance scoring and negative profile detection for Track 1.

The JD explicitly names excluded profile types and describes an ideal candidate.
These functions translate those requirements into numeric signals that prevent
HR Managers, Accountants, and non-technical profiles from floating to the top
purely on behavioural signals.
"""

from __future__ import annotations

import re
from typing import Sequence


# ── Tokens that indicate AI/ML production experience ─────────────────────────

STRONG_POSITIVE_ROLE_TOKENS: frozenset[str] = frozenset({
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "research scientist",
    "applied scientist",
    "deep learning engineer",
    "nlp engineer",
    "computer vision engineer",
    "search engineer",
    "information retrieval",
    "ranking engineer",
    "recommendation engineer",
    "mlops engineer",
    "ml platform engineer",
    "ai researcher",
    "ml researcher",
    "data scientist",
    "applied ml",
    "applied ai",
})

# Generic engineering titles — neutral by themselves, need skill confirmation
NEUTRAL_ROLE_TOKENS: frozenset[str] = frozenset({
    "software engineer",
    "software developer",
    "backend engineer",
    "backend developer",
    "full stack engineer",
    "platform engineer",
    "infrastructure engineer",
    "data engineer",
    "analytics engineer",
    "research engineer",
    "scientist",
})

# Titles explicitly called out as anti-patterns in the JD
NEGATIVE_ROLE_TOKENS: frozenset[str] = frozenset({
    "hr manager",
    "hr coordinator",
    "hr business partner",
    "human resources",
    "talent acquisition",
    "talent manager",
    "recruiter",
    "recruitment",
    "accountant",
    "finance manager",
    "financial analyst",
    "chief financial",
    "ca ",          # Chartered Accountant
    "mechanical engineer",
    "civil engineer",
    "structural engineer",
    "electrical engineer",
    "chemical engineer",
    "manufacturing engineer",
    "content writer",
    "copywriter",
    "technical writer",
    "graphic designer",
    "ui designer",
    "ux designer",
    "visual designer",
    "sales executive",
    "sales manager",
    "business development",
    "account manager",
    "account executive",
    "marketing manager",
    "digital marketing",
    "brand manager",
    "seo specialist",
    "customer support",
    "customer success",
    "customer service",
    "operations manager",
    "supply chain",
    "logistics",
    "project manager",       # general PM — NOT technical PM
    "program manager",
    "business analyst",
    "operations analyst",
    "data entry",
    "office manager",
    "executive assistant",
    "teacher",
    "professor",
    "lecturer",
    "doctor",
    "physician",
    "nurse",
    "lawyer",
    "attorney",
})

# Consulting / services firms explicitly named in the JD as excluded backgrounds
CONSULTING_FIRM_TOKENS: frozenset[str] = frozenset({
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl technologies",
    "hcl tech",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "niit technologies",
    "mindtree",
    "l&t infotech",
    "ltimindtree",
    "birlasoft",
    "mastech",
    "kpit technologies",
})

# Career history tokens that signal genuine AI/ML product experience
PRODUCT_AI_CAREER_TOKENS: frozenset[str] = frozenset({
    "vector",
    "embedding",
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "llm",
    "transformer",
    "bert",
    "gpt",
    "neural",
    "pytorch",
    "tensorflow",
    "sklearn",
    "scikit",
    "mlflow",
    "kubeflow",
    "feature store",
    "model serving",
    "model deployment",
    "inference",
    "fine-tuning",
    "fine tuning",
    "rag",
    "reranking",
    "rerank",
    "ndcg",
    "mrr",
    "a/b test",
    "recall@",
    "precision@",
    "faiss",
    "qdrant",
    "pinecone",
    "weaviate",
    "milvus",
    "lora",
    "peft",
    "langchain",
    "llamaindex",
    "production ml",
    "ml pipeline",
    "data pipeline",
    "model training",
    "experimentation",
    "annotation",
    "labeling",
})


_YEAR_PATTERN = re.compile(r"\b(20\d{2}|199\d)\b")

# Tokens that suggest a distinct company in career text
_COMPANY_TOKENS: tuple[str, ...] = (
    " inc.", " ltd.", " pvt.", " technologies", " systems",
    " solutions", " labs", " corp.", " consulting", " services",
    " software", " tech ", " group", " digital",
)


def _estimate_job_hops(career_history_text: str) -> float:
    """Estimate a title-chasing penalty from career history text.

    Returns 0.0 when no signal, up to 0.20 when the pattern strongly suggests
    frequent short-tenure moves (many company tokens relative to years span).
    """
    lowered = career_history_text.lower()

    years_found = _YEAR_PATTERN.findall(lowered)
    if len(years_found) < 2:
        return 0.0

    year_ints = sorted(int(y) for y in set(years_found))
    career_span = year_ints[-1] - year_ints[0]
    if career_span <= 0:
        return 0.0

    company_count = sum(1 for token in _COMPANY_TOKENS if token in lowered)
    hops_per_year = company_count / career_span

    if hops_per_year > 1.2:
        return 0.20    # strong title-chaser signal
    if hops_per_year > 0.75:
        return 0.10    # moderate signal
    return 0.0


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def score_role_relevance(current_role: str, headline: str = "") -> float:
    """Return a role-relevance score in [0, 1].

    1.0 — confirmed AI/ML engineering title
    0.6 — neutral software/data engineering title (needs skill confirmation)
    0.1 — explicitly excluded title (HR, Accountant, etc.)
    0.3 — unrecognised / other
    """
    combined = _normalise(f"{current_role} {headline}")

    # Hard negative check first
    for token in NEGATIVE_ROLE_TOKENS:
        if token in combined:
            return 0.1

    # Strong positive AI/ML titles
    for token in STRONG_POSITIVE_ROLE_TOKENS:
        if token in combined:
            return 1.0

    # Neutral engineering titles — partial credit
    for token in NEUTRAL_ROLE_TOKENS:
        if token in combined:
            return 0.6

    return 0.3


def score_career_trajectory(career_history_text: str) -> float:
    """Score how much of the career history involves product AI/ML work.

    Returns 0–1. Higher = stronger AI/ML product-company trajectory.
    Penalises consulting-firm-heavy backgrounds per the JD's explicit note.
    """
    if not career_history_text:
        return 0.3

    lowered = _normalise(career_history_text)

    # Consulting background penalty
    consulting_hits = sum(1 for firm in CONSULTING_FIRM_TOKENS if firm in lowered)
    if consulting_hits >= 2:
        consulting_penalty = 0.40
    elif consulting_hits == 1:
        consulting_penalty = 0.20
    else:
        consulting_penalty = 0.0

    # Title-chasing penalty — frequent short-tenure job hops
    consulting_penalty = min(consulting_penalty + _estimate_job_hops(career_history_text), 0.55)

    # Count AI/ML product signals across the full career history
    ai_hits = sum(1 for token in PRODUCT_AI_CAREER_TOKENS if token in lowered)
    ai_score = min(ai_hits / 6.0, 1.0)   # saturates at 6 distinct signals

    return float(max(0.0, round(ai_score - consulting_penalty, 4)))
