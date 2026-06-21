"""Ollama adapter: pure request/response helpers (no network), plus a mocked stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.adapters.llm.ollama import (
    _build_options,
    _build_payload,
    _chat_url,
    _token_from_line,
)
from personavoice.models import Msg, Role
from personavoice.persona import load_persona


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


def test_chat_url_normalizes_trailing_slash() -> None:
    assert _chat_url("http://localhost:11434") == "http://localhost:11434/api/chat"
    assert _chat_url("http://host:11434/") == "http://host:11434/api/chat"


def test_build_options_maps_persona_settings(config_dir: Path) -> None:
    persona = _companion(config_dir)
    opts = _build_options(persona)
    assert opts["temperature"] == persona.llm.temperature
    assert opts["top_p"] == persona.llm.top_p
    assert opts["num_predict"] == persona.llm.max_tokens


def test_build_payload_shape(config_dir: Path) -> None:
    persona = _companion(config_dir)
    messages = [Msg(role=Role.system, content="sys"), Msg(role=Role.user, content="hi")]
    payload = _build_payload("qwen2.5:7b-instruct", messages, persona, stream=True)

    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "options" in payload


def test_token_from_line() -> None:
    assert _token_from_line('{"message": {"content": "Hel"}, "done": false}') == "Hel"
    assert _token_from_line('{"message": {"content": ""}, "done": true}') is None
    assert _token_from_line('{"done": true}') is None
    assert _token_from_line("   ") is None


async def test_stream_chat_yields_tokens(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    httpx = pytest.importorskip("httpx")
    from personavoice.adapters.llm.ollama import OllamaLLM

    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " there"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    body = ("\n".join(lines) + "\n").encode()

    def handler(request: object) -> object:
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args: object, **kwargs: object) -> object:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    adapter = OllamaLLM(model="qwen2.5:7b-instruct")
    persona = _companion(config_dir)
    messages = [Msg(role=Role.user, content="hi")]

    tokens = [tok async for tok in adapter.stream_chat(messages, persona)]
    assert tokens == ["Hello", " there"]
    # base chat() collects the stream into the full reply.
    assert await adapter.chat(messages, persona) == "Hello there"
