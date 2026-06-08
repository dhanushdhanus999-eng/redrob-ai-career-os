"""LLM listwise reranking with Gemini race-runner, disk caching, and graceful fallback.

Three Gemini models are called simultaneously; whichever responds first is used
and the others are cancelled. This eliminates single-model rate-spike failures.

Race order: gemini-2.5-flash → gemini-2.5-flash-lite → gemini-2.5-pro
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from src.utils.paths import CACHE_DIR


LLM_CACHE_DIR = CACHE_DIR / "llm_rerank"

_RACE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

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


# ── Gemini async helpers ──────────────────────────────────────────────────────

async def _call_gemini(model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Call one Gemini model asynchronously and return the text response."""
    import google.generativeai as genai  # imported lazily

    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
    )
    response = await model.generate_content_async(user_prompt)
    return response.text


async def _race_gemini(system_prompt: str, user_prompt: str) -> str:
    """Race three Gemini models; return the first successful response."""
    tasks = [
        asyncio.create_task(_call_gemini(name, system_prompt, user_prompt))
        for name in _RACE_MODELS
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        return next(iter(done)).result()
    except Exception:
        # Cancel all remaining tasks on any error
        for task in tasks:
            task.cancel()
        raise


def _run_race(system_prompt: str, user_prompt: str) -> str:
    """Synchronous wrapper — runs the async race in a new event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing loop (e.g. Jupyter / Gradio): use nest_asyncio
            import nest_asyncio  # optional dep
            nest_asyncio.apply()
            return loop.run_until_complete(_race_gemini(system_prompt, user_prompt))
        return loop.run_until_complete(_race_gemini(system_prompt, user_prompt))
    except RuntimeError:
        return asyncio.run(_race_gemini(system_prompt, user_prompt))


# ── Main class ────────────────────────────────────────────────────────────────

class LLMReranker:
    """Use a cached Gemini LLM call to rerank the final shortlist when available."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        race_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        """``race_fn`` overrides the Gemini race-runner — used to inject a fake
        model call in tests or to swap in an alternative LLM provider without
        touching the caching/normalisation logic."""
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir or LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._race_fn = race_fn or _run_race

        if race_fn is not None:
            self._ready = True
            return

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._ready = True
            except ImportError:
                self._ready = False
        else:
            self._ready = False

    @property
    def enabled(self) -> bool:
        return self._ready

    def _cache_path(self, job_text: str, candidate_ids: list[str]) -> Path:
        return self.cache_dir / f"{_cache_key(job_text, candidate_ids)}.json"

    def _normalize_result(
        self,
        ranked_ids: list[str],
        original_candidates: list[tuple[str, str]],
        reasoning: str,
    ) -> list[tuple[str, str]]:
        original_ids = [cid for cid, _ in original_candidates]
        known_ids = set(original_ids)
        normalized_ids = [cid for cid in ranked_ids if cid in known_ids]
        normalized_ids.extend(
            cid for cid in original_ids if cid not in set(normalized_ids)
        )
        return [(cid, reasoning) for cid in normalized_ids]

    def rerank(
        self,
        job_text: str,
        candidates: list[tuple[str, str]],
        *,
        top_k: int = 30,
    ) -> list[tuple[str, str]]:
        """Return reranked (candidate_id, reasoning) pairs using the Gemini race-runner."""
        candidates = list(candidates[:top_k])
        if not candidates:
            return []

        candidate_ids = [cid for cid, _ in candidates]
        cache_path = self._cache_path(job_text, candidate_ids)

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return self._normalize_result(
                list(payload.get("ranked_ids", [])),
                candidates,
                str(payload.get("reasoning", "")),
            )

        if not self.enabled:
            return [(cid, "") for cid, _ in candidates]

        candidate_block = "\n".join(
            f"[{cid}] {text[:900]}"
            for cid, text in candidates
        )
        user_prompt = (
            "JOB DESCRIPTION:\n"
            f"{job_text[:2500]}\n\n"
            "CANDIDATES:\n"
            f"{candidate_block}\n\n"
            "Return the ranked candidate IDs as JSON."
        )

        try:
            raw_text = self._race_fn(self.system_prompt, user_prompt).strip()
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            payload = json.loads(cleaned)
        except Exception:
            return [(cid, "") for cid, _ in candidates]

        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return self._normalize_result(
            list(payload.get("ranked_ids", [])),
            candidates,
            str(payload.get("reasoning", "")),
        )
