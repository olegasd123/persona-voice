"""LLM adapter base class.

Concrete backends (ollama, vllm, mlx_lm) subclass this and override `stream_chat`.
`chat` is a convenience that concatenates the streamed tokens for the turn-based path.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from ...models import CheckResult, Msg, Persona, Role

# Seconds between warm-up retries while a server backend is still loading its model.
_WARMUP_RETRY_S = 2.0


@runtime_checkable
class Tool(Protocol):
    """The minimal tool contract the LLM adapter needs.

    Declared here, in the adapter layer, so a backend can run the tool-call loop without importing
    the orchestrator's registry — `orchestrator/tools.ToolSpec` satisfies this structurally, so
    the dependency only ever points orchestrator → adapters (no cycle).
    """

    @property
    def name(self) -> str:
        """The tool's unique name (matches the `function.name` the model calls)."""
        ...

    def schema(self) -> dict[str, Any]:
        """The OpenAI `tools` entry (a ``{"type": "function", ...}`` object) sent to the model."""
        ...

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with parsed JSON arguments and return the result text."""
        ...


class LLMAdapter:
    """The persona brain. `stream_chat` yields reply tokens for a message history."""

    name: str = "base"
    stage: str = "llm"
    implemented: bool = False  # real backends set this True (drops the "stub" warning)
    # Whether this backend can route tool/function calls. Backends that can't leave
    # it False; `stream_chat_with_tools` then degrades to a plain reply (tools ignored), keeping
    # the BACKEND switch honest where a backend lacks the capability.
    supports_tools: bool = False

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        """Yield reply tokens. Real backends implement this as an async generator."""
        raise NotImplementedError(f"{self.name}.stream_chat is not implemented yet")

    async def stream_chat_with_tools(
        self, messages: list[Msg], persona: Persona, tools: Sequence[Tool]
    ) -> AsyncIterator[str]:
        """Stream a reply, first resolving any tool calls the model makes.

        Default implementation **ignores tools** and just streams the reply, so a backend that
        can't do function calling (or any persona with no tools) behaves exactly as `stream_chat`.
        Tool-capable backends (the OpenAI-compatible path) override this with the model → call →
        execute → feed-back loop, then stream the follow-up reply for TTS.
        """
        async for tok in self.stream_chat(messages, persona):
            yield tok

    async def chat(self, messages: list[Msg], persona: Persona) -> str:
        """Convenience: collect the streamed tokens into a single reply string."""
        chunks: list[str] = []
        async for tok in self.stream_chat(messages, persona):
            chunks.append(tok)
        return "".join(chunks)

    async def chat_with_tools(
        self, messages: list[Msg], persona: Persona, tools: Sequence[Tool]
    ) -> str:
        """Tool-aware `chat`: collect the (post-tool) streamed tokens into one reply string."""
        chunks: list[str] = []
        async for tok in self.stream_chat_with_tools(messages, persona, tools):
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
