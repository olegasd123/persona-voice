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
from pathlib import Path
from typing import Any

from ...models import Msg, Persona
from .base import LLMAdapter

_DEFAULT_TIMEOUT = 120.0


def chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def lora_request_model(persona: Persona) -> str | None:
    """The served LoRA name to request for a persona, or None.

    A persona's `llm.lora` is a path/name to the trained adapter; vLLM serves it under a name
    registered with `--lora-modules <name>=<path>`, and a request selects it via the `model`
    field. By convention the served name is the adapter's basename, so a backend launched with
    `--lora-modules hr_interviewer=/adapters/hr_interviewer` matches `lora: adapters/hr_interviewer`.
    """
    lora = persona.llm.lora
    if not lora:
        return None
    return Path(lora).name


def build_payload(
    model: str | None,
    messages: list[Msg],
    persona: Persona,
    *,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
    lora: str | None = None,
) -> dict[str, Any]:
    # On a LoRA-capable backend the request `model` selects the served adapter by name;
    # otherwise it's the base model the stage was configured with.
    payload: dict[str, Any] = {
        "model": lora or model,
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
    URL can still be overridden per-deployment via the `base_url` adapter option. Backends that
    can hot-load LoRA adapters (vLLM) set `supports_lora = True` so a persona's `llm.lora`
    routes the request to the served adapter.
    """

    default_base_url = "http://localhost:8000/v1"
    supports_lora = False

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
        lora = lora_request_model(persona) if self.supports_lora else None
        payload = build_payload(
            self.model, messages, persona, stream=True, extra_body=extra_body, lora=lora
        )

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", chat_url(base_url), json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                token = token_from_sse_line(line)
                if token:
                    yield token
