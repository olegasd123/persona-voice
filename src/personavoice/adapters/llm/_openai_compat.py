"""Shared OpenAI-compatible chat-completions streaming.

LM Studio (Mac dev) and vLLM (CUDA prod) both expose an OpenAI-compatible
`/v1/chat/completions` endpoint, so they share one streaming implementation and differ
only in their default base URL. Many local models (Qwen3, gpt-oss) are *reasoning* models
whose stream carries the thinking trace in `delta.reasoning_content` and the spoken reply
in `delta.content`; we deliberately yield only `content` so the persona never speaks its
reasoning out loud.

`extra_body` (a YAML option) is merged into the request body, so model-specific knobs
(`reasoning_effort: low`, `chat_template_kwargs`, vLLM `guided_*`, ...) are configurable
without code changes. `httpx` is imported lazily; the request/response plumbing is pure and
unit-testable offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...models import Msg, Persona
from .base import LLMAdapter

_DEFAULT_TIMEOUT = 120.0


def chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def build_payload(
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


def token_from_sse_line(line: str) -> str | None:
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


class OpenAICompatLLM(LLMAdapter):
    """Base for backends exposing an OpenAI-compatible `/v1/chat/completions` stream.

    Subclasses set `name`, `implemented = True`, and `default_base_url`. The active base
    URL can still be overridden per-deployment via the `base_url` adapter option.
    """

    default_base_url = "http://localhost:8000/v1"

    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "httpx is not installed; install a backend extra: "
                "`pip install -e '.[mac]'` or `pip install -e '.[cuda]'`"
            ) from exc

        base_url = self.options.get("base_url", self.default_base_url)
        timeout = self.options.get("timeout", _DEFAULT_TIMEOUT)
        extra_body = self.options.get("extra_body")
        payload = build_payload(self.model, messages, persona, stream=True, extra_body=extra_body)

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", chat_url(base_url), json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                token = token_from_sse_line(line)
                if token:
                    yield token
