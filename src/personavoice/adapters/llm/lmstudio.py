"""LM Studio LLM (Mac / M4 Max dev).

Talks to LM Studio's OpenAI-compatible server (`/v1/chat/completions`) with streaming
enabled. Many local models (e.g. Qwen3, gpt-oss) are *reasoning* models: their stream
carries the thinking trace in `delta.reasoning_content` and the spoken reply in
`delta.content`. We deliberately yield only `content` so the persona never speaks its
reasoning out loud.

`extra_body` (a YAML option) is merged into the request, so model-specific knobs like
`reasoning_effort: low` or `chat_template_kwargs` are configurable without code changes.
`httpx` is imported lazily; the request/response plumbing is pure and unit-testable offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...models import Msg, Persona
from .base import LLMAdapter

_DEFAULT_BASE_URL = "http://localhost:1234/v1"
_DEFAULT_TIMEOUT = 120.0


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _build_payload(
    model: str | None,
    messages: list[Msg],
    persona: Persona,
    *,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "stream": stream,
        "temperature": persona.llm.temperature,
        "top_p": persona.llm.top_p,
        "max_tokens": persona.llm.max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    return payload


def _token_from_sse_line(line: str) -> str | None:
    """Extract the spoken-content token from one SSE line, or None.

    OpenAI streaming sends `data: {json}` lines plus blank separators and a final
    `data: [DONE]`. Reasoning deltas (`reasoning_content`) and role-only deltas yield None
    so only the actual reply is spoken.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    obj = json.loads(payload)
    choices = obj.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("delta") or {}).get("content")
    return content or None


class LMStudioLLM(LLMAdapter):
    name = "lmstudio"
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
        extra_body = self.options.get("extra_body")
        payload = _build_payload(self.model, messages, persona, stream=True, extra_body=extra_body)

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", _chat_url(base_url), json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                token = _token_from_sse_line(line)
                if token:
                    yield token
