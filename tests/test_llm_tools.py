"""Tool / function calling over the OpenAI-compatible stream.

Covers the pure SSE/tool-call parsing and the full model → tool → result → answer loop with a
mocked transport, so the wire protocol is exercised offline (no LM Studio / vLLM needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.adapters.llm._openai_compat import (
    StreamDelta,
    ToolCall,
    _execute_tool,
    _ToolCallBuffer,
    parse_stream_delta,
)
from personavoice.adapters.llm.lmstudio import LMStudioLLM
from personavoice.models import Msg, Role
from personavoice.orchestrator.tools import ToolSpec
from personavoice.persona import load_persona


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}"


def _body(*objs: dict) -> bytes:
    return ("\n\n".join(_sse(o) for o in objs) + "\n\ndata: [DONE]\n\n").encode()


# --- pure parsing ---------------------------------------------------------------------


def test_parse_stream_delta_content_and_tool_calls() -> None:
    content = parse_stream_delta(_sse({"choices": [{"delta": {"content": "Hi"}}]}))
    assert content == StreamDelta(content="Hi", tool_calls=None)

    frag = [{"index": 0, "id": "c1", "function": {"name": "get_x", "arguments": "{}"}}]
    tool = parse_stream_delta(_sse({"choices": [{"delta": {"tool_calls": frag}}]}))
    assert tool is not None and tool.content is None and tool.tool_calls == frag

    # reasoning-only / role-only / done / blank → empty or None
    assert parse_stream_delta(_sse({"choices": [{"delta": {"reasoning_content": "x"}}]})) == (
        StreamDelta(content=None, tool_calls=None)
    )
    assert parse_stream_delta("data: [DONE]") is None
    assert parse_stream_delta("") is None


def test_tool_call_buffer_reassembles_fragments() -> None:
    buf = _ToolCallBuffer()
    assert buf.empty()
    buf.add([{"index": 0, "id": "call_1", "function": {"name": "get_current_time"}}])
    buf.add([{"index": 0, "function": {"arguments": '{"timezone"'}}])
    buf.add([{"index": 0, "function": {"arguments": ': "UTC"}'}}])
    calls = buf.calls()
    assert calls == [
        ToolCall(id="call_1", name="get_current_time", arguments='{"timezone": "UTC"}')
    ]


def test_tool_call_buffer_handles_two_parallel_calls_and_missing_id() -> None:
    buf = _ToolCallBuffer()
    buf.add([{"index": 0, "function": {"name": "a", "arguments": "{}"}}])  # no id
    buf.add([{"index": 1, "id": "c2", "function": {"name": "b", "arguments": "{}"}}])
    calls = buf.calls()
    assert [(c.id, c.name) for c in calls] == [("call_0", "a"), ("c2", "b")]  # id synthesized


def test_tool_call_buffer_drops_nameless_call() -> None:
    buf = _ToolCallBuffer()
    buf.add([{"index": 0, "function": {"arguments": "{}"}}])  # never got a name
    assert buf.calls() == []


# --- _execute_tool (safe error strings) -----------------------------------------------


def _tool(name: str, handler) -> ToolSpec:
    return ToolSpec(name=name, description="", parameters={}, handler=handler)


async def test_execute_tool_happy_path_parses_args() -> None:
    seen: dict = {}

    def handler(args: dict) -> str:
        seen.update(args)
        return "RESULT"

    by_name = {"t": _tool("t", handler)}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments='{"x": 1}'), 5.0)
    assert out == "RESULT"
    assert seen == {"x": 1}


async def test_execute_tool_empty_arguments_is_empty_dict() -> None:
    by_name = {"t": _tool("t", lambda a: f"got {a}")}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments=""), 5.0)
    assert out == "got {}"


async def test_execute_tool_unknown_name() -> None:
    out = await _execute_tool({}, ToolCall(id="1", name="ghost", arguments="{}"), 5.0)
    assert "unknown tool" in out and "ghost" in out


async def test_execute_tool_bad_json() -> None:
    by_name = {"t": _tool("t", lambda a: "x")}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments="{not json"), 5.0)
    assert "could not parse arguments" in out


async def test_execute_tool_non_object_arguments() -> None:
    by_name = {"t": _tool("t", lambda a: "x")}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments="[1,2]"), 5.0)
    assert "must be a JSON object" in out


async def test_execute_tool_handler_error_is_caught() -> None:
    def boom(_args: dict) -> str:
        raise RuntimeError("kaboom")

    by_name = {"t": _tool("t", boom)}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments="{}"), 5.0)
    assert "failed" in out and "kaboom" in out


async def test_execute_tool_timeout() -> None:
    import asyncio

    async def slow(_args: dict) -> str:
        await asyncio.sleep(10)
        return "never"

    by_name = {"t": _tool("t", slow)}
    out = await _execute_tool(by_name, ToolCall(id="1", name="t", arguments="{}"), 0.01)
    assert "timed out" in out


# --- the full streaming tool loop -----------------------------------------------------


def _mock_httpx(monkeypatch, bodies: list[bytes], captured: list[dict]):
    """Patch httpx.AsyncClient to a MockTransport that returns `bodies` in order."""
    httpx = pytest.importorskip("httpx")
    state = {"n": 0}

    def handler(request: object) -> object:
        captured.append(json.loads(request.content))  # type: ignore[attr-defined]
        body = bodies[min(state["n"], len(bodies) - 1)]
        state["n"] += 1
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> object:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)


def _recording_time_tool(record: list[dict]) -> ToolSpec:
    return ToolSpec(
        name="get_current_time",
        description="time",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda args: record.append(args) or "12:00 UTC",  # type: ignore[func-returns-value]
    )


async def test_tool_loop_executes_then_streams_answer(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    pytest.importorskip("httpx")
    record: list[dict] = []
    captured: list[dict] = []
    tool_call = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": '{"timezone": "UTC"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    answer = {"choices": [{"delta": {"content": "It is noon."}}]}
    _mock_httpx(monkeypatch, [_body(tool_call), _body(answer)], captured)

    adapter = LMStudioLLM(model="m")
    tools = [_recording_time_tool(record)]
    tokens = [
        tok
        async for tok in adapter.stream_chat_with_tools(
            [Msg(role=Role.user, content="what time is it?")], _companion(config_dir), tools
        )
    ]

    assert "".join(tokens) == "It is noon."
    # The tool ran with the parsed args.
    assert record == [{"timezone": "UTC"}]
    # First request offered the tool schema; the second carried the call + tool result back.
    assert captured[0]["tools"][0]["function"]["name"] == "get_current_time"
    follow_up = captured[1]["messages"]
    assert follow_up[-2]["role"] == "assistant" and follow_up[-2]["tool_calls"][0]["id"] == "call_1"
    assert follow_up[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "12:00 UTC"}


async def test_no_tools_passthrough_omits_schema(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    captured: list[dict] = []
    _mock_httpx(monkeypatch, [_body({"choices": [{"delta": {"content": "Hi."}}]})], captured)

    adapter = LMStudioLLM(model="m")
    tokens = [
        tok
        async for tok in adapter.stream_chat_with_tools(
            [Msg(role=Role.user, content="hi")], _companion(config_dir), []
        )
    ]
    assert tokens == ["Hi."]
    assert "tools" not in captured[0]  # no schemas sent when the persona has no tools


async def test_tool_loop_caps_iterations_then_answers_without_tools(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    pytest.importorskip("httpx")
    monkeypatch.setenv("PERSONAVOICE_TOOL_MAX_ITERS", "1")
    record: list[dict] = []
    captured: list[dict] = []
    tool_call = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }
    fallback = {"choices": [{"delta": {"content": "I checked."}}]}
    _mock_httpx(monkeypatch, [_body(tool_call), _body(fallback)], captured)

    adapter = LMStudioLLM(model="m")
    tokens = [
        tok
        async for tok in adapter.stream_chat_with_tools(
            [Msg(role=Role.user, content="time?")],
            _companion(config_dir),
            [_recording_time_tool(record)],
        )
    ]

    assert "".join(tokens) == "I checked."
    assert record == [{}]  # the one allowed tool call ran
    # The cap forces a final completion *without* tools so the user still hears an answer.
    assert "tools" not in captured[1]


async def test_content_first_response_ignores_stray_tool_calls(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    # If a response streams content (the answer), a later tool-call delta in the same response is
    # not executed — content wins and the turn ends.
    record: list[dict] = []
    captured: list[dict] = []
    mixed = {
        "choices": [{"delta": {"content": "Answer."}}],
    }
    _mock_httpx(monkeypatch, [_body(mixed)], captured)

    adapter = LMStudioLLM(model="m")
    tokens = [
        tok
        async for tok in adapter.stream_chat_with_tools(
            [Msg(role=Role.user, content="hi")],
            _companion(config_dir),
            [_recording_time_tool(record)],
        )
    ]
    assert tokens == ["Answer."]
    assert record == []  # no tool executed
    assert len(captured) == 1  # single round-trip


def test_base_adapter_ignores_tools(config_dir: Path) -> None:
    # A backend that can't route tools degrades to a plain reply (no schemas, no loop).
    from personavoice.adapters.llm.ollama import OllamaLLM

    assert OllamaLLM.supports_tools is False
    assert LMStudioLLM.supports_tools is True
