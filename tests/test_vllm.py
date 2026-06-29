"""vLLM adapter: shares OpenAI-compatible streaming with LM Studio, differs in base URL."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from personavoice.adapters.llm._openai_compat import chat_url, trace_enabled
from personavoice.adapters.llm.vllm import VLLMAdapter
from personavoice.models import Msg, Role
from personavoice.obs import TRACE
from personavoice.persona import load_persona


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}"


def test_vllm_defaults() -> None:
    adapter = VLLMAdapter(model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert adapter.name == "vllm"
    assert adapter.implemented is True
    assert adapter.default_base_url == "http://localhost:8000/v1"
    assert adapter.check().ok
    assert adapter.check().warnings == []  # implemented -> no stub warning


async def test_vllm_streams_content_against_configured_url(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    httpx = pytest.importorskip("httpx")

    lines = [
        _sse({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),  # must be skipped
        _sse({"choices": [{"delta": {"content": "Sure"}}]}),
        _sse({"choices": [{"delta": {"content": ", done."}}]}),
        "data: [DONE]",
    ]
    body = ("\n\n".join(lines) + "\n\n").encode()
    seen: dict[str, str] = {}

    def handler(request: object) -> object:
        seen["url"] = str(request.url)  # type: ignore[attr-defined]
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args: object, **kwargs: object) -> object:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    # Override the default base URL via options, as cuda.yaml does with ${PERSONAVOICE_LLM_BASE_URL}.
    adapter = VLLMAdapter(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        options={"base_url": "http://vllm:8000/v1"},
    )
    persona = _companion(config_dir)
    messages = [Msg(role=Role.user, content="hi")]

    tokens = [tok async for tok in adapter.stream_chat(messages, persona)]
    assert tokens == ["Sure", ", done."]
    assert seen["url"] == chat_url("http://vllm:8000/v1")


def test_trace_enabled_follows_log_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(TRACE, logger="personavoice.llm"):
        assert trace_enabled() is True  # TRACE turns the dump on
    with caplog.at_level(logging.DEBUG, logger="personavoice.llm"):
        assert trace_enabled() is True  # so does the louder DEBUG
    with caplog.at_level(logging.INFO, logger="personavoice.llm"):
        assert trace_enabled() is False  # INFO and quieter leave it off


def _mock_httpx(monkeypatch: pytest.MonkeyPatch, body: bytes):
    httpx = pytest.importorskip("httpx")
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    real_async_client = httpx.AsyncClient

    def patched_client(*args: object, **kwargs: object) -> object:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)


async def test_trace_logs_request_and_response_when_enabled(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    body = (_sse({"choices": [{"delta": {"content": "Hello"}}]}) + "\n\ndata: [DONE]\n\n").encode()
    _mock_httpx(monkeypatch, body)

    adapter = VLLMAdapter(model="m", options={"base_url": "http://vllm:8000/v1"})
    persona = _companion(config_dir)

    with caplog.at_level(TRACE, logger="personavoice.llm"):
        tokens = [
            tok async for tok in adapter.stream_chat([Msg(role=Role.user, content="hi")], persona)
        ]

    assert tokens == ["Hello"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "request →" in text  # the outgoing payload was logged
    assert "response ←" in text and "Hello" in text  # the reassembled reply was logged


async def test_trace_silent_by_default(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    body = (_sse({"choices": [{"delta": {"content": "Hi"}}]}) + "\n\ndata: [DONE]\n\n").encode()
    _mock_httpx(monkeypatch, body)

    adapter = VLLMAdapter(model="m", options={"base_url": "http://vllm:8000/v1"})
    persona = _companion(config_dir)

    with caplog.at_level(logging.INFO, logger="personavoice.llm"):
        _ = [tok async for tok in adapter.stream_chat([Msg(role=Role.user, content="hi")], persona)]

    # No trace from our logger at INFO (it only fires at TRACE/DEBUG); other loggers (httpx) may log.
    assert [r for r in caplog.records if r.name == "personavoice.llm"] == []
