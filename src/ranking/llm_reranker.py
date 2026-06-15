"""LLM listwise reranking via a local Ollama server, with disk caching and fallback.

A single local model served by Ollama (default ``qwen2.5:7b``) is called over HTTP to
rerank the final shortlist. No hosted-LLM provider (Gemini / OpenAI / Anthropic) is used
anywhere — the only LLM dependency is the local Ollama host.

Note on competition compliance: this still issues a network request to the Ollama host,
so — exactly like the previous hosted-LLM version — it remains OFFLINE-RESEARCH-ONLY and is
never part of the network-free `rank.py` submission path.

Configuration (env, both optional):
- ``OLLAMA_BASE_URL``  default ``http://100.96.26.32:11434``
- ``OLLAMA_MODEL``     default ``qwen2.5:7b``   (alternatives on the host: ``qwen3.5:9b``,
                       ``llama3:8b``, ``mistral:latest``, ``deepseek-coder-v2:latest`` …)
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from src.utils.paths import CACHE_DIR


LLM_CACHE_DIR = CACHE_DIR / "llm_rerank"

DEFAULT_OLLAMA_BASE_URL = "http://100.96.26.32:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

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


# ── Ollama HTTP helpers ───────────────────────────────────────────────────────

def _call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    base_url: str,
    timeout: float,
) -> str:
    """Call one local Ollama chat model and return the text response.

    Uses ``format: "json"`` and ``temperature: 0`` so the model returns a single
    deterministic JSON object suitable for parsing downstream.
    """
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["message"]["content"]


def _ollama_available(base_url: str, *, timeout: float = 3.0) -> bool:
    """Return True if the Ollama server answers ``/api/tags`` quickly."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ── Main class ────────────────────────────────────────────────────────────────

class LLMReranker:
    """Rerank the final shortlist with a cached local-Ollama LLM call when available."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        call_fn: Callable[[str, str], str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        """``call_fn`` overrides the Ollama call — used to inject a fake model
        response in tests without touching the caching/normalisation logic."""
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir or LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.timeout = timeout
        self._call_fn = call_fn
        # Health check is lazy (see ``enabled``) so construction never blocks on
        # the network — the cache-hit path returns without ever reaching Ollama.
        self._ready: bool | None = True if call_fn is not None else None

    @property
    def enabled(self) -> bool:
        if self._call_fn is not None:
            return True
        if self._ready is None:
            self._ready = _ollama_available(self.base_url)
        return self._ready

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        if self._call_fn is not None:
            return self._call_fn(system_prompt, user_prompt)
        return _call_ollama(
            self.model,
            system_prompt,
            user_prompt,
            base_url=self.base_url,
            timeout=self.timeout,
        )

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
        """Return reranked (candidate_id, reasoning) pairs using the local Ollama model."""
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
            raw_text = self._call(self.system_prompt, user_prompt).strip()
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
