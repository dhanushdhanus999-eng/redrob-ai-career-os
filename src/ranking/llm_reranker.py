"""LLM listwise reranking with disk caching and graceful fallback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.utils.paths import CACHE_DIR


LLM_CACHE_DIR = CACHE_DIR / "llm_rerank"
DEFAULT_SYSTEM_PROMPT = """\
You are an expert talent matching system.
Rank the candidate IDs from BEST to WORST fit for the role.

Consider:
1. Skill match
2. Experience and domain fit
3. Seniority alignment
4. Career trajectory relevance

Return only valid JSON:
{
  "ranked_ids": ["id1", "id2"],
  "reasoning": "One sentence explaining the top choice."
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
