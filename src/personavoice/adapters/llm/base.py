"""LLM adapter base class.

Concrete backends (ollama, vllm, mlx_lm) subclass this and override `stream_chat`.
`chat` is a convenience that concatenates the streamed tokens for the turn-based path.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from ...models import CheckResult, Msg, Persona, Role

# Seconds between warm-up retries while a server backend is still loading its model.
_WARMUP_RETRY_S = 2.0


class LLMAdapter:
    """The persona brain. `stream_chat` yields reply tokens for a message history."""

    name: str = "base"
    stage: str = "llm"
    implemented: bool = False  # real backends set this True (drops the "stub" warning)

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        """Yield reply tokens. Real backends implement this as an async generator."""
        raise NotImplementedError(f"{self.name}.stream_chat is not implemented yet")

    async def chat(self, messages: list[Msg], persona: Persona) -> str:
        """Convenience: collect the streamed tokens into a single reply string."""
        chunks: list[str] = []
        async for tok in self.stream_chat(messages, persona):
            chunks.append(tok)
        return "".join(chunks)

    async def warmup(self, persona: Persona) -> None:
        """Warm the brain so the first reply's first token is fast.

        Streams a single token for a trivial prompt: for a server backend (vLLM / LM Studio)
        this opens the HTTP client and triggers the server's first (slow) prefill; for an
        in-process model it loads the weights. No-op for stub adapters.

        A server backend may still be loading its model when the worker starts (the agent's
        prewarm runs right after `docker compose up -d vllm`), so transient failures are
        retried until the server answers or the wait budget (`PERSONAVOICE_LLM_WARMUP_WAIT`
        seconds, default 180) is spent — at which point the last error is raised for the caller
        to log. An in-process model succeeds on the first attempt.
        """
        if not self.implemented:
            return
        deadline = time.monotonic() + float(os.getenv("PERSONAVOICE_LLM_WARMUP_WAIT", "180"))
        while True:
            try:
                await self._warm_once(persona)
                return
            except Exception:
                if time.monotonic() >= deadline:
                    raise  # give up; the caller logs it and the first turn pays the load
                await asyncio.sleep(_WARMUP_RETRY_S)

    async def _warm_once(self, persona: Persona) -> None:
        """One warm-up attempt: stream a single token, then close the stream."""
        stream = self.stream_chat([Msg(role=Role.user, content="Hi")], persona)
        try:
            async for _tok in stream:
                break  # one token proves the path is warm
        finally:
            # Concrete backends implement stream_chat as an async generator; close it so an
            # early break tears down the underlying HTTP stream rather than leaking it.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    def check(self) -> CheckResult:
        warnings = [] if self.implemented else ["stub adapter — implemented in a later milestone"]
        return CheckResult(
            stage=self.stage,
            adapter=self.name,
            ok=True,
            detail=f"loaded (model={self.model!r})",
            warnings=warnings,
        )
