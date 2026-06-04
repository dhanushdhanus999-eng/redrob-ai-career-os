"""LLM listwise reranking with disk caching and graceful fallback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.utils.paths import CACHE_DIR


LLM_CACHE_DIR = CACHE_DIR / "llm_rerank"
DEFAULT_SYSTEM_PROMPT = """\
You are a senior AI engineering recruiter evaluating candidates for the role of
Senior AI Engineer (Founding Team) at Redrob AI, a Series A AI-native talent platform.

=== ROLE REQUIREMENTS ===
The role demands PRODUCTION experience (not tutorial-level or demo-level) with:
- Embeddings-based retrieval systems deployed to real users (BGE, E5, sentence-transformers,
  OpenAI embeddings) — embedding drift, index refresh, retrieval-quality regression
- Vector databases in production: Qdrant, Pinecone, Weaviate, Milvus, FAISS, OpenSearch
- Hybrid search infrastructure (lexical + dense + RRF fusion)
- Ranking evaluation: NDCG, MAP, MRR, A/B tests, recruiter feedback loops
- LLM integration in products — not just API wrappers, actual system ownership
- Python with production engineering judgment

Nice to have: LoRA/QLoRA/PEFT fine-tuning, learning-to-rank models, HR-tech exposure,
distributed systems, open-source contributions in AI/ML.

=== EXPLICITLY EXCLUDED PROFILES (rank these LAST) ===
- HR Managers, Talent Acquisition, Recruiters — even if they list AI skills
- Accountants, Finance Managers, Civil/Mechanical/Electrical Engineers
- Content Writers, Graphic Designers, Marketing Managers
- Sales Executives, Business Development, Customer Support, Operations Managers
- General Project Managers, Business Analysts (non-technical)
- Candidates from consulting/services only: TCS, Infosys, Wipro, Accenture, Cognizant —
  the JD explicitly excludes "consulting-only" backgrounds
- Framework enthusiasts who build demos but have not shipped ranking/search to real users
- Title-chasers who switch companies every 1-2 years for title bumps

=== IDEAL CANDIDATE ===
6-8 years total, of which 4-5 are in applied ML/AI at product companies (not services).
Has shipped at least one end-to-end ranking, search, or recommendation system to real users.
Strong opinions on retrieval (hybrid vs dense), evaluation (offline vs online), and LLM
integration backed by systems they actually built — not blog posts or tutorials.

=== INSTRUCTIONS ===
Rank candidates from BEST to WORST fit. Heavily penalise excluded profiles even if they
list many AI skills — the skills are likely superficial. A genuine ML Engineer with 4
directly relevant skills outranks an HR Manager with 9 generic AI skills.

Return ONLY valid JSON — no extra text:
{
  "ranked_ids": ["id1", "id2", "id3", ...],
  "reasoning": "One sentence explaining why your top choice ranks first."
}
"""


def _cache_key(job_text: str, candidate_ids: list[str]) -> str:
    content = f"{job_text}|{','.join(candidate_ids)}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class LLMReranker:
    """Use a cached LLM call to rerank the final shortlist when available."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        client: object | None = None,
        cache_dir: Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir or LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if client is not None:
            self.client = client
            return

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self.client = None
            return

        try:
            import anthropic
        except ImportError:
            self.client = None
            return

        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _cache_path(self, job_text: str, candidate_ids: list[str]) -> Path:
        return self.cache_dir / f"{_cache_key(job_text, candidate_ids)}.json"

    def _normalize_result(
        self,
        ranked_ids: list[str],
        original_candidates: list[tuple[str, str]],
        reasoning: str,
    ) -> list[tuple[str, str]]:
        original_ids = [candidate_id for candidate_id, _ in original_candidates]
        known_ids = set(original_ids)
        normalized_ids = [candidate_id for candidate_id in ranked_ids if candidate_id in known_ids]
        normalized_ids.extend(
            candidate_id
            for candidate_id in original_ids
            if candidate_id not in set(normalized_ids)
        )
        return [(candidate_id, reasoning) for candidate_id in normalized_ids]

    def rerank(
        self,
        job_text: str,
        candidates: list[tuple[str, str]],
        *,
        top_k: int = 30,
    ) -> list[tuple[str, str]]:
        """Return reranked candidate IDs plus a shared reasoning string."""
        candidates = list(candidates[:top_k])
        if not candidates:
            return []

        cache_path = self._cache_path(job_text, [candidate_id for candidate_id, _ in candidates])
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return self._normalize_result(
                list(payload.get("ranked_ids", [])),
                candidates,
                str(payload.get("reasoning", "")),
            )

        if not self.enabled:
            return [(candidate_id, "") for candidate_id, _ in candidates]

        candidate_block = "\n".join(
            f"[{candidate_id}] {candidate_text[:350]}"
            for candidate_id, candidate_text in candidates
        )
        user_prompt = (
            "JOB DESCRIPTION:\n"
            f"{job_text[:1200]}\n\n"
            "CANDIDATES:\n"
            f"{candidate_block}\n\n"
            "Return the ranked candidate IDs as JSON."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = getattr(response.content[0], "text", "").strip()
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            payload = json.loads(cleaned)
        except Exception:
            return [(candidate_id, "") for candidate_id, _ in candidates]

        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return self._normalize_result(
            list(payload.get("ranked_ids", [])),
            candidates,
            str(payload.get("reasoning", "")),
        )
