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

TRACK1_MIN_YEARS = 5.0
TRACK1_MAX_YEARS = 9.0
TRACK1_JOB_TITLE = "Senior AI Engineer - Founding Team"
TRACK1_JOB_SENIORITY = "senior"
