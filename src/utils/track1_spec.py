"""Track 1 job-description constants — the released 'Senior AI Engineer' JD.

These are extracted directly from the DOCX job description. They live in a
dependency-free module so both the lightweight ranking entrypoint (``rank.py``)
and the heavier research pipeline (``scripts/generate_submission.py``) share one
source of truth without pulling in torch / faiss / sentence-transformers.
"""

from __future__ import annotations

TRACK1_MUST_HAVE_SKILLS: list[str] = [
    "Python",
    "Embeddings",
    "Sentence Transformers",
    "BGE Embeddings",
    "E5 Embeddings",
    "OpenAI Embeddings",
    "Vector Databases",
    "Qdrant",
    "Pinecone",
    "Weaviate",
    "Milvus",
    "FAISS",
    "OpenSearch",
    "Elasticsearch",
    "Hybrid Search",
    "Dense Retrieval",
    "Semantic Search",
    "NDCG",
    "MRR",
    "MAP",
    "A/B Testing",
    "Retrieval Augmented Generation",
    "Large Language Models",
    "Ranking",
    "Reranking",
]

TRACK1_NICE_TO_HAVE_SKILLS: list[str] = [
    "LoRA",
    "QLoRA",
    "PEFT",
    "Fine-Tuning",
    "Learning to Rank",
    "LambdaRank",
    "LightGBM",
    "XGBoost",
    "LangChain",
    "LlamaIndex",
    "MLflow",
    "Kubeflow",
    "FastAPI",
    "Distributed Systems",
    "Hugging Face",
    "PyTorch",
    "Anthropic SDK",
    "Open Source",
]

# ── Must-have *capability groups* ────────────────────────────────────────────
# The 25 must-have skills above are granular tools, and many are mutually
# substitutable — a candidate needs *one* vector DB, not all eight; *one*
# embedding stack, not all five. Scoring and reporting a flat "N / 25" therefore
# undersells strong specialists (the JD's single best candidate matches only a
# handful of the 25 tokens by name). The JD is really asking for coverage across
# a small set of *capabilities*. We group the 25 tokens into the capability areas
# the JD actually cares about; a candidate "covers" a group if they match ANY
# member (exact / same-family / fuzzy). Reporting "5 / 7 core capabilities" is
# both more accurate and far more legible to a Stage-4 human reviewer than
# "5 / 25 must-have skills". Order is preserved (Python 3.7+ dicts are ordered).
TRACK1_MUST_HAVE_GROUPS: dict[str, list[str]] = {
    "Python": ["Python"],
    "Embeddings": [
        "Embeddings",
        "Sentence Transformers",
        "BGE Embeddings",
        "E5 Embeddings",
        "OpenAI Embeddings",
    ],
    "Vector databases": [
        "Vector Databases",
        "Qdrant",
        "Pinecone",
        "Weaviate",
        "Milvus",
        "FAISS",
        "OpenSearch",
        "Elasticsearch",
    ],
    "Retrieval & search": ["Hybrid Search", "Dense Retrieval", "Semantic Search"],
    "Ranking & reranking": ["Ranking", "Reranking"],
    "Retrieval metrics & A/B testing": ["NDCG", "MRR", "MAP", "A/B Testing"],
    "RAG & LLMs": ["Retrieval Augmented Generation", "Large Language Models"],
}

# Shared India-location tokens — one source of truth so the *score* (rank.py) and
# the *reasoning text* (explainer.py) can never disagree about whether a city is in
# India. (A drifted copy previously printed "based in Jaipur … outside India" on
# the top candidate.) Substring match against a lowercased location string.
INDIA_LOCATION_TOKENS: tuple[str, ...] = (
    "india", "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "chennai", "gurugram", "gurgaon", "kolkata",
    "ahmedabad", "jaipur", "kochi", "pune-noida",
)

TRACK1_MIN_YEARS = 5.0
TRACK1_MAX_YEARS = 9.0
TRACK1_JOB_TITLE = "Senior AI Engineer - Founding Team"
TRACK1_JOB_SENIORITY = "senior"
