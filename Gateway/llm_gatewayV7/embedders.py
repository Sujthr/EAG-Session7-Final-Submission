"""Embedding provider — OpenAI text-embedding-3-small (1536-dim).

Single provider, no key rotation needed. Rate limits are generous on OpenAI
(3000 RPM on tier 1) so we keep the EmbedRateState machinery for correctness
but it will almost never trigger.

EMBED_DIM is fixed at 1536 for the lifetime of a FAISS index. If you ever
need to rebuild the index with a different model, delete state/index.faiss
and state/index_ids.json first.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Literal

import httpx


TaskType = Literal["retrieval_document", "retrieval_query"]
EMBED_DIM = 1536  # text-embedding-3-small native dimension
MAX_INPUT_CHARS = 8000
BACKOFF_STEPS = [5, 10, 15]


class EmbedderError(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class EmbedRateState:
    """Per-embedder rate state with backoff."""

    def __init__(self, rpm: int, cooldown: float):
        self.rpm = rpm
        self.cooldown = cooldown
        self.calls_minute: deque[float] = deque()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""
        self.backoff_step = 0

    def _gc(self) -> None:
        cutoff = time.time() - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()

    def can_use(self) -> tuple[bool, str]:
        self._gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        if self.cooldown > 0:
            wait = self.cooldown - (now - self.last_call)
            if wait > 0:
                return False, f"cooldown ({wait:.1f}s)"
        if self.rpm > 0 and len(self.calls_minute) >= self.rpm:
            return False, f"RPM limit ({self.rpm}/min)"
        return True, ""

    def record(self) -> None:
        now = time.time()
        self.calls_minute.append(now)
        self.last_call = now
        self.backoff_step = 0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    def mark_failure(self, reason: str) -> None:
        idx = min(self.backoff_step, len(BACKOFF_STEPS) - 1)
        secs = BACKOFF_STEPS[idx]
        self.backoff_step += 1
        self.unavailable_until = time.time() + secs
        self.unavailable_reason = reason[:80]

    def snapshot(self) -> dict:
        self._gc()
        now = time.time()
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": self.rpm,
            "cooldown_s": self.cooldown,
            "cooldown_remaining": max(0.0, self.cooldown - (now - self.last_call)) if self.last_call else 0.0,
            "backoff_remaining": max(0.0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
            "backoff_step": self.backoff_step,
        }


class EmbeddingProvider:
    name: str = ""
    model: str = ""
    state: EmbedRateState

    async def embed(self, text: str, task_type: TaskType) -> dict:
        raise NotImplementedError


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embeddings via text-embedding-3-small (1536-dim)."""
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        # OpenAI tier-1 allows 3000 RPM for embeddings; set conservatively.
        self.state = EmbedRateState(rpm=3000, cooldown=0.0)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def embed(self, text: str, task_type: TaskType) -> dict:
        body = {"model": self.model, "input": text}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.openai.com/v1/embeddings",
                             headers=self._headers(), json=body)
        if r.status_code != 200:
            raise EmbedderError(
                f"openai HTTP {r.status_code}: {r.text[:200]}", status=r.status_code
            )
        d = r.json()
        vec = (d.get("data") or [{}])[0].get("embedding") or []
        if not vec:
            raise EmbedderError(f"openai returned no embedding: {str(d)[:200]}")
        return {"embedding": vec, "model": self.model, "dim": len(vec)}


def build_embedders() -> tuple[list[EmbeddingProvider], list[str]]:
    """Return (ordered list of embedders, ordered list of names). OpenAI only."""
    k = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")
    if not k:
        raise RuntimeError("OPENAI_API_KEY not set in Gateway/.env")
    embedder = OpenAIEmbedder(k, model)
    print(f"[embedders] OpenAI embedder active: {model} ({EMBED_DIM}-dim)")
    return [embedder], [embedder.name]


async def embed_with_failover(
    embedders: list[EmbeddingProvider],
    text: str,
    task_type: TaskType,
    explicit: str | None = None,
) -> tuple[str, dict, list, int]:
    """Run with per-provider rate-state gating. Returns (name, result, attempts, latency_ms)."""
    attempts: list[dict] = []
    candidates = embedders
    if explicit:
        candidates = [e for e in embedders if e.name == explicit]
        if not candidates:
            raise EmbedderError(f"unknown embedder '{explicit}'", status=400)

    last_err: Exception | None = None
    t0 = time.time()
    for e in candidates:
        ok, why = e.state.can_use()
        if not ok:
            attempts.append({"provider": e.name, "reason": why})
            if explicit:
                raise EmbedderError(f"{e.name} unavailable: {why}", status=429)
            continue
        try:
            out = await e.embed(text, task_type)
            e.state.record()
            latency = int((time.time() - t0) * 1000)
            return e.name, out, attempts, latency
        except Exception as exc:
            last_err = exc
            reason = str(exc)[:200]
            e.state.mark_failure(reason)
            attempts.append({"provider": e.name, "reason": reason})
            if explicit:
                raise
    raise EmbedderError(
        f"all embedders unavailable. attempts={attempts}. last_error={last_err}",
        status=503,
    )
