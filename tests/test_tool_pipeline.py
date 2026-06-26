"""Tool wiring through the pipelines.

Verifies that a persona that declares tools routes through the adapter's tool path (and the
tools actually run), while a tool-free persona — or a missing registry — takes the plain path
unchanged. The wire protocol itself is covered in `test_llm_tools.py`; here we use a fake
tool-capable LLM to assert the orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from personavoice.adapters.factory import Backend
from personavoice.adapters.llm.base import LLMAdapter, Tool
from personavoice.models import Msg, Persona
from personavoice.orchestrator import Pipeline, StreamingPipeline
from personavoice.orchestrator.tools import ToolRegistry, ToolSpec
from personavoice.persona import load_persona

from .fakes import FakeSTT, FakeTTS, make_persona


def _companion(config_dir: Path) -> Persona:
    return load_persona(config_dir / "personas" / "companion.yaml")


class ToolFakeLLM(LLMAdapter):
    """A tool-capable fake: records which path ran and executes whatever tools it's handed."""

    name = "tool_fake"
    supports_tools = True

    def __init__(self, reply: str = "It is noon.") -> None:
        super().__init__()
        self._reply = reply
        self.plain_called = False
        self.tool_path_called = False
        self.received_tools: list[str] | None = None

    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        self.plain_called = True
        yield self._reply

    async def stream_chat_with_tools(
        self, messages: list[Msg], persona: Persona, tools: Sequence[Tool]
    ) -> AsyncIterator[str]:
        self.tool_path_called = True
        self.received_tools = [t.name for t in tools]
        for tool in tools:
            await tool.run({})
        yield self._reply


def _backend(llm: ToolFakeLLM) -> Backend:
    return Backend(name="fake", stt=FakeSTT("hi"), llm=llm, tts=FakeTTS())


def _recording_registry(record: list[str]) -> ToolRegistry:
    spec = ToolSpec(
        name="get_current_time",
        description="time",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda args: record.append("ran") or "12:00 UTC",  # type: ignore[func-returns-value]
    )
    return ToolRegistry({"get_current_time": spec})


# --- streaming pipeline ---------------------------------------------------------------


async def test_streaming_persona_with_tools_uses_tool_path(config_dir: Path) -> None:
    record: list[str] = []
    llm = ToolFakeLLM()
    pipe = StreamingPipeline(
        _backend(llm), _companion(config_dir), tools=_recording_registry(record)
    )

    chunks = [c async for c in pipe.stream_response("what time is it?")]

    assert llm.tool_path_called and not llm.plain_called
    assert llm.received_tools == ["get_current_time"]
    assert record == ["ran"]  # the tool actually executed
    assert b"".join(chunks).endswith(b"It is noon.")  # the follow-up reply was voiced


async def test_streaming_tool_free_persona_uses_plain_path(config_dir: Path) -> None:
    record: list[str] = []
    llm = ToolFakeLLM()
    # make_persona() declares no tools, so even with a registry present the plain path runs.
    pipe = StreamingPipeline(_backend(llm), make_persona(), tools=_recording_registry(record))

    async for _ in pipe.stream_response("hi"):
        pass

    assert llm.plain_called and not llm.tool_path_called
    assert record == []


async def test_streaming_no_registry_uses_plain_path(config_dir: Path) -> None:
    llm = ToolFakeLLM()
    # Persona declares tools, but no registry is wired → plain path (graceful).
    pipe = StreamingPipeline(_backend(llm), _companion(config_dir))

    async for _ in pipe.stream_response("hi"):
        pass

    assert llm.plain_called and not llm.tool_path_called


async def test_streaming_unknown_tool_name_falls_back_to_plain(config_dir: Path) -> None:
    record: list[str] = []
    llm = ToolFakeLLM()
    persona = make_persona().model_copy(update={"tools": ["ghost"]})
    pipe = StreamingPipeline(_backend(llm), persona, tools=_recording_registry(record))

    async for _ in pipe.stream_response("hi"):
        pass

    # "ghost" resolves to nothing, so there are no specs → plain path, nothing executed.
    assert llm.plain_called and not llm.tool_path_called
    assert record == []


# --- turn-based pipeline --------------------------------------------------------------


async def test_turn_pipeline_persona_with_tools_uses_tool_path(config_dir: Path) -> None:
    record: list[str] = []
    llm = ToolFakeLLM(reply="Right now it is noon.")
    pipe = Pipeline(_backend(llm), _companion(config_dir), tools=_recording_registry(record))

    result = await pipe.run_turn(b"audio")

    assert llm.tool_path_called and not llm.plain_called
    assert record == ["ran"]
    assert result.reply == "Right now it is noon."


async def test_turn_pipeline_tool_free_persona_uses_plain_path(config_dir: Path) -> None:
    record: list[str] = []
    llm = ToolFakeLLM()
    pipe = Pipeline(_backend(llm), make_persona(), tools=_recording_registry(record))

    await pipe.run_turn(b"audio")

    assert llm.plain_called and not llm.tool_path_called
    assert record == []
