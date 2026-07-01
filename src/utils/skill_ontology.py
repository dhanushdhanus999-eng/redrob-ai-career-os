"""Skill normalisation and graded matching utilities."""

from __future__ import annotations

import re
from typing import Dict

from rapidfuzz import fuzz, process

from src.utils.failure_fixes import ADDITIONAL_SKILL_SYNONYMS
from src.utils.text_utils import clean_text


SKILL_SYNONYMS: Dict[str, str] = {
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "angularjs": "Angular",
    "angular js": "Angular",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "py": "Python",
    "python3": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "tf": "TensorFlow",
    "torch": "PyTorch",
    "aws": "Amazon Web Services",
    "azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform",
    "rest": "REST API",
    "restful": "REST API",
    "cicd": "CI/CD",
    "ci/cd": "CI/CD",
    "llm": "Large Language Models",
    # ── Vector databases ──────────────────────────────────────────────────────
    "vector db": "Vector Databases",
    "vector database": "Vector Databases",
    "vectordb": "Vector Databases",
    "vector store": "Vector Databases",
    "chroma": "ChromaDB",
    "chromadb": "ChromaDB",
    "pgvector": "pgvector",
    "pg vector": "pgvector",
    "qdrant": "Qdrant",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "milvus": "Milvus",
    "lancedb": "LanceDB",
    "vespa": "Vespa",
    "opensearch": "OpenSearch",
    # ── Embeddings & retrieval ────────────────────────────────────────────────
    "bge": "BGE Embeddings",
    "bge-large": "BGE Embeddings",
    "bge large": "BGE Embeddings",
    "e5": "E5 Embeddings",
    "e5-large": "E5 Embeddings",
    "e5 large": "E5 Embeddings",
    "ada-002": "OpenAI Embeddings",
    "text-embedding": "OpenAI Embeddings",
    "openai embedding": "OpenAI Embeddings",
    "sentence transformer": "Sentence Transformers",
    "sentence-transformer": "Sentence Transformers",
    "sbert": "Sentence Transformers",
    "hybrid search": "Hybrid Search",
    "dense retrieval": "Dense Retrieval",
    "dense search": "Dense Retrieval",
    "semantic search": "Semantic Search",
    "bm25": "BM25",
    "reranking": "Reranking",
    "re-ranking": "Reranking",
    "cross encoder": "Cross-Encoder",
    "cross-encoder": "Cross-Encoder",
    "ann search": "ANN Search",
    "approximate nearest neighbor": "ANN Search",
    "embedding drift": "Embeddings",
    "index refresh": "Vector Databases",
    # ── LLM fine-tuning ───────────────────────────────────────────────────────
    "lora": "LoRA",
    "lo-ra": "LoRA",
    "qlora": "QLoRA",
    "q-lora": "QLoRA",
    "peft": "PEFT",
    "sft": "Supervised Fine-Tuning",
    "rlhf": "RLHF",
    "dpo": "DPO",
    "instruct tuning": "Supervised Fine-Tuning",
    "fine tuning": "Fine-Tuning",
    "fine-tuning": "Fine-Tuning",
    "finetuning": "Fine-Tuning",
    # ── LLM frameworks & inference ────────────────────────────────────────────
    "langchain": "LangChain",
    "lang chain": "LangChain",
    "llamaindex": "LlamaIndex",
    "llama index": "LlamaIndex",
    "llama-index": "LlamaIndex",
    "vllm": "vLLM",
    "tgi": "TGI",
    "text generation inference": "TGI",
    "ollama": "Ollama",
    "openai api": "OpenAI API",
    "anthropic": "Anthropic SDK",
    "anthropic sdk": "Anthropic SDK",
    "gpt-4": "OpenAI API",
    "chatgpt": "OpenAI API",
    "llama": "Llama",
    "llama2": "Llama",
    "llama3": "Llama",
    "mistral": "Mistral",
    "gemini": "Gemini",
    "cohere": "Cohere",
    # ── Ranking & evaluation ──────────────────────────────────────────────────
    "ltr": "Learning to Rank",
    "learning-to-rank": "Learning to Rank",
    "lambdarank": "LambdaRank",
    "lambda rank": "LambdaRank",
    "ndcg": "NDCG",
    "mrr": "MRR",
    "mean reciprocal rank": "MRR",
    "mean average precision": "MAP",
    "precision@k": "Precision@K",
    "recall@k": "Recall@K",
    "a/b testing": "A/B Testing",
    "ab testing": "A/B Testing",
    "a/b test": "A/B Testing",
    "online evaluation": "A/B Testing",
    "offline evaluation": "Ranking Evaluation",
    "ranking evaluation": "Ranking Evaluation",
    "retrieval evaluation": "Ranking Evaluation",
    # ── MLOps & infrastructure ────────────────────────────────────────────────
    "mlflow": "MLflow",
    "ml flow": "MLflow",
    "kubeflow": "Kubeflow",
    "wandb": "Weights & Biases",
    "weights and biases": "Weights & Biases",
    "weights & biases": "Weights & Biases",
    "feature store": "Feature Store",
    "feast": "Feast",
    "bentoml": "BentoML",
    "triton": "Triton Inference Server",
    "triton inference": "Triton Inference Server",
    "ray": "Ray",
    "ray serve": "Ray",
    "celery": "Celery",
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
}
SKILL_SYNONYMS.update(ADDITIONAL_SKILL_SYNONYMS)

SKILL_FAMILIES: Dict[str, list[str]] = {
    "Python Ecosystem": [
        "Python",
        "FastAPI",
        "Flask",
        "Django",
        "NumPy",
        "Pandas",
        "SQLAlchemy",
    ],
    "JavaScript Ecosystem": [
        "JavaScript",
        "TypeScript",
        "React",
        "Vue.js",
        "Angular",
        "Node.js",
        "Next.js",
    ],
    "ML/AI": [
        "Machine Learning",
        "Deep Learning",
        "PyTorch",
        "TensorFlow",
        "scikit-learn",
        "XGBoost",
        "LightGBM",
    ],
    "Data Engineering": [
        "Spark",
        "Kafka",
        "Airflow",
        "dbt",
        "Databricks",
    ],
    "Cloud": [
        "Amazon Web Services",
        "Google Cloud Platform",
        "Microsoft Azure",
    ],
    "Databases": [
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Elasticsearch",
    ],
    "DevOps": [
        "Docker",
        "Kubernetes",
        "Terraform",
        "CI/CD",
        "Ansible",
        "Jenkins",
    ],
    "NLP": [
        "Natural Language Processing",
        "Large Language Models",
        "Hugging Face",
        "spaCy",
        "NLTK",
    ],
    "Vector Databases": [
        "Vector Databases",
        "Qdrant",
        "Pinecone",
        "Weaviate",
        "Milvus",
        "ChromaDB",
        "pgvector",
        "FAISS",
        "LanceDB",
        "Vespa",
        "OpenSearch",
        "ANN Search",
    ],
    "Embeddings & Retrieval": [
        "Sentence Transformers",
        "BGE Embeddings",
        "E5 Embeddings",
        "OpenAI Embeddings",
        "Dense Retrieval",
        "Hybrid Search",
        "Semantic Search",
        "BM25",
        "Reranking",
        "Cross-Encoder",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
    "LLM Fine-tuning": [
        "LoRA",
        "QLoRA",
        "PEFT",
        "Supervised Fine-Tuning",
        "RLHF",
        "DPO",
        "Fine-Tuning",
        "Hugging Face",
    ],
    "Ranking & Evaluation": [
        "Learning to Rank",
        "LambdaRank",
        "LightGBM",
        "XGBoost",
        "NDCG",
        "MRR",
        "MAP",
        "A/B Testing",
        "Ranking Evaluation",
        "Precision@K",
        "Recall@K",
    ],
    "LLM Frameworks": [
        "Large Language Models",
        "LangChain",
        "LlamaIndex",
        "vLLM",
        "TGI",
        "OpenAI API",
        "Anthropic SDK",
        "Ollama",
        "Generative AI",
        "Llama",
        "Mistral",
        "Gemini",
        "Cohere",
    ],
    "MLOps": [
        "MLflow",
        "Kubeflow",
        "Weights & Biases",
        "Feature Store",
        "Feast",
        "BentoML",
        "Triton Inference Server",
        "Ray",
        "CI/CD",
        "Docker",
        "Kubernetes",
        "FastAPI",
    ],
}

_SKILL_TO_FAMILY: Dict[str, str] = {
    skill: family
    for family, skills in SKILL_FAMILIES.items()
    for skill in skills
}
_CANONICAL_LOOKUP: Dict[str, str] = {
    re.sub(r"\s+", " ", canonical.lower()): canonical
    for canonical in {value for value in SKILL_SYNONYMS.values()} | set(_SKILL_TO_FAMILY)
}


def normalize_skill(skill: str) -> str:
    """Normalise a raw skill string into its canonical form when known."""
    if not isinstance(skill, str):
        return ""

    cleaned = re.sub(r"\s+", " ", clean_text(skill).lower())
    if not cleaned:
        return ""
    if cleaned in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[cleaned]
    if cleaned in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[cleaned]
    return clean_text(skill)


def normalize_skills_list(skills: list[str]) -> list[str]:
    """Normalise and deduplicate a list of skills while preserving order."""
    seen: set[str] = set()
    normalised: list[str] = []
    for skill in skills:
        canonical = normalize_skill(skill)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalised.append(canonical)
    return normalised


def skills_in_same_family(skill_a: str, skill_b: str) -> bool:
    """Return True when two skills belong to the same technology family."""
    family_a = _SKILL_TO_FAMILY.get(skill_a)
    family_b = _SKILL_TO_FAMILY.get(skill_b)
    return family_a is not None and family_a == family_b


class SkillMatcher:
    """Compute graded skill overlap features between jobs and candidate profiles."""

    def __init__(self, fuzzy_threshold: int = 80) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    def match_score(
        self,
        required_skills: list[str],
        candidate_skills: list[str],
        *,
        credit_family_match: float = 0.5,
        credit_fuzzy_match: float = 0.7,
    ) -> dict:
        """Return exact, family, fuzzy, and composite coverage statistics."""
        required = normalize_skills_list(required_skills)
        candidate = normalize_skills_list(candidate_skills)
        if not required:
            return self._empty_result()

        candidate_set = set(candidate)
        exact_matched: list[str] = []
        family_matched: list[str] = []
        fuzzy_matched: list[str] = []
        missing: list[str] = []

        for skill in required:
            if skill in candidate_set:
                exact_matched.append(skill)
                continue
            if any(skills_in_same_family(skill, other) for other in candidate):
                family_matched.append(skill)
                continue
            if candidate:
                fuzzy_match = process.extractOne(
                    skill,
                    candidate,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=self.fuzzy_threshold,
                )
                if fuzzy_match is not None:
                    fuzzy_matched.append(skill)
                    continue
            missing.append(skill)

        total = len(required)
        exact_cov = len(exact_matched) / total
        family_cov = (len(exact_matched) + len(family_matched)) / total
        fuzzy_cov = (len(exact_matched) + len(family_matched) + len(fuzzy_matched)) / total
        composite = (
            exact_cov
            + (len(family_matched) / total) * credit_family_match
            + (len(fuzzy_matched) / total) * credit_fuzzy_match
        )
        extra_skills = [skill for skill in candidate if skill not in set(required)]

        return {
            "exact_coverage": round(exact_cov, 4),
            "family_coverage": round(family_cov, 4),
            "fuzzy_coverage": round(fuzzy_cov, 4),
            "composite_score": round(min(composite, 1.0), 4),
            "n_exact_matched": len(exact_matched),
            "n_family_matched": len(family_matched),
            "n_fuzzy_matched": len(fuzzy_matched),
            "n_missing": len(missing),
            "matched_skills": exact_matched,
            "missing_skills": missing,
            "extra_skills_count": len(extra_skills),
        }

    def group_coverage(
        self,
        groups: Dict[str, list[str]],
        candidate_skills: list[str],
    ) -> dict:
        """Score capability *coverage* across a set of skill groups.

        Each group is a capability area (e.g. "Vector databases") whose members
        are substitutable tools. A group counts as covered when the candidate
        matches ANY member — exact, same-family, or fuzzy — so a candidate who
        knows one vector DB gets full credit for that capability rather than a
        diluted 1/8. Returns the covered group names, the coverage fraction, and
        the concrete member skills that triggered each hit (for explanations).
        """
        covered: list[str] = []
        matched_members: Dict[str, list[str]] = {}
        for group_name, members in groups.items():
            result = self.match_score(members, candidate_skills)
            n_hit = (
                result["n_exact_matched"]
                + result["n_family_matched"]
                + result["n_fuzzy_matched"]
            )
            if n_hit > 0:
                covered.append(group_name)
                # matched_skills holds the exact hits; fall back to the whole
                # group name when only a family/fuzzy match fired.
                matched_members[group_name] = result["matched_skills"] or [group_name]

        total = len(groups)
        coverage = len(covered) / total if total else 0.0
        return {
            "covered_groups": covered,
            "n_covered": len(covered),
            "n_total": total,
            "coverage": round(coverage, 4),
            "matched_members": matched_members,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "exact_coverage": 0.0,
            "family_coverage": 0.0,
            "fuzzy_coverage": 0.0,
            "composite_score": 0.0,
            "n_exact_matched": 0,
            "n_family_matched": 0,
            "n_fuzzy_matched": 0,
            "n_missing": 0,
            "matched_skills": [],
            "missing_skills": [],
            "extra_skills_count": 0,
        }
