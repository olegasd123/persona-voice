"""LM Studio adapter: pure request/response helpers (no network), plus a mocked SSE stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.adapters.llm._openai_compat import (
    build_payload,
    chat_url,
    token_from_sse_line,
)
from personavoice.models import Msg, Role
from personavoice.persona import load_persona


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}"


def test_chat_url_appends_endpoint() -> None:
    assert chat_url("http://localhost:1234/v1") == "http://localhost:1234/v1/chat/completions"
    assert chat_url("http://host/v1/") == "http://host/v1/chat/completions"


def test_build_payload_maps_persona_and_merges_extra_body(config_dir: Path) -> None:
    persona = _companion(config_dir)
    messages = [Msg(role=Role.system, content="sys"), Msg(role=Role.user, content="hi")]
    payload = build_payload(
        "openai/gpt-oss-20b",
        messages,
        persona,
        stream=True,
        extra_body={"reasoning_effort": "low"},
    )

    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["stream"] is True
    assert payload["temperature"] == persona.llm.temperature
    assert payload["top_p"] == persona.llm.top_p
    assert payload["max_tokens"] == persona.llm.max_tokens
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["reasoning_effort"] == "low"  # extra_body merged in


def test_token_from_sse_line_returns_content_only() -> None:
    assert token_from_sse_line(_sse({"choices": [{"delta": {"content": "Hi"}}]})) == "Hi"
    # reasoning trace must NOT be spoken
    assert (
        token_from_sse_line(_sse({"choices": [{"delta": {"reasoning_content": "think"}}]})) is None
    )
    # role-only / empty / done / non-data lines
    assert token_from_sse_line(_sse({"choices": [{"delta": {"role": "assistant"}}]})) is None
    assert token_from_sse_line(_sse({"choices": [{"delta": {"content": ""}}]})) is None
    assert token_from_sse_line("data: [DONE]") is None
    assert token_from_sse_line("") is None
    assert token_from_sse_line(": keep-alive comment") is None


async def test_stream_chat_yields_content_skips_reasoning(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    httpx = pytest.importorskip("httpx")
    from personavoice.adapters.llm.lmstudio import LMStudioLLM

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant"}}]}),
        _sse({"choices": [{"delta": {"reasoning_content": "let me think"}}]}),
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " there"}}]}),
        "data: [DONE]",
    ]
    body = ("\n\n".join(lines) + "\n\n").encode()

    def handler(request: object) -> object:
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args: object, **kwargs: object) -> object:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    adapter = LMStudioLLM(model="openai/gpt-oss-20b")
    persona = _companion(config_dir)
    messages = [Msg(role=Role.user, content="hi")]

    tokens = [tok async for tok in adapter.stream_chat(messages, persona)]
    assert tokens == ["Hello", " there"]
    assert await adapter.chat(messages, persona) == "Hello there"
