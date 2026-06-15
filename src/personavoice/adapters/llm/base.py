"""LLM adapter base class.

Concrete backends (ollama, vllm, mlx_lm) subclass this and override `stream_chat`.
`chat` is a convenience that concatenates the streamed tokens for the turn-based M1 path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...models import CheckResult, Msg, Persona


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

    def check(self) -> CheckResult:
        warnings = [] if self.implemented else ["stub adapter — implemented in a later milestone"]
        return CheckResult(
            stage=self.stage,
            adapter=self.name,
            ok=True,
            detail=f"loaded (model={self.model!r})",
            warnings=warnings,
        )
