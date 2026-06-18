"""Ollama LLM (Mac / M4 Max dev).

Talks to a local Ollama server's `/api/chat` endpoint with streaming enabled, yielding
reply tokens as they arrive. The base class's `chat()` collects them for the turn-based
pipeline. `httpx` is imported lazily so the adapter constructs without the `mac` extra;
the request/response plumbing is split into pure helpers so it's unit-testable offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...models import Msg, Persona
from .base import LLMAdapter

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/chat"


def _build_options(persona: Persona) -> dict[str, Any]:
    """Map persona LLM settings onto Ollama's generation options."""
    return {
        "temperature": persona.llm.temperature,
        "top_p": persona.llm.top_p,
        "num_predict": persona.llm.max_tokens,
    }


def _build_payload(
    model: str | None, messages: list[Msg], persona: Persona, *, stream: bool = True
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "stream": stream,
        "options": _build_options(persona),
    }


def _token_from_line(line: str) -> str | None:
    """Extract the content token from one streamed JSONL line, or None if there is none.

    Ollama streams one JSON object per line: `{"message": {"content": "..."}, "done": ...}`.
    Blank lines and lines without message content (e.g. the final `done` record) yield None.
    """
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    content = obj.get("message", {}).get("content")
    return content or None


class OllamaLLM(LLMAdapter):
    name = "ollama"
    implemented = True

    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "httpx is not installed; install the Mac extra: `pip install -e '.[mac]'`"
            ) from exc

        base_url = self.options.get("base_url", _DEFAULT_BASE_URL)
        timeout = self.options.get("timeout", _DEFAULT_TIMEOUT)
        payload = _build_payload(self.model, messages, persona, stream=True)

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", _chat_url(base_url), json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                token = _token_from_line(line)
                if token:
                    yield token
