"""The tool registry + built-in tools (Feature D, orchestrator/tools.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from personavoice.adapters.llm.base import Tool
from personavoice.orchestrator.tools import (
    ToolRegistry,
    ToolSpec,
    default_tool_registry,
    get_current_time_tool,
)


def _echo_tool(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="echo back",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda args: f"echo:{args}",
    )


# --- ToolSpec -------------------------------------------------------------------------


def test_toolspec_satisfies_adapter_tool_protocol() -> None:
    # The orchestrator's ToolSpec is what the adapter's structural `Tool` protocol expects,
    # so the adapter never has to import the registry.
    assert isinstance(_echo_tool(), Tool)


def test_toolspec_schema_is_openai_function_shape() -> None:
    schema = _echo_tool("get_x").schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_x"
    assert schema["function"]["description"] == "echo back"
    assert schema["function"]["parameters"]["type"] == "object"


async def test_toolspec_run_normalizes_sync_and_async_handlers() -> None:
    sync = ToolSpec(name="s", description="", parameters={}, handler=lambda a: "sync")

    async def _async_handler(_args: dict) -> str:
        return "async"

    asy = ToolSpec(name="a", description="", parameters={}, handler=_async_handler)
    assert await sync.run({}) == "sync"
    assert await asy.run({}) == "async"


async def test_toolspec_run_coerces_non_string_result() -> None:
    spec = ToolSpec(name="n", description="", parameters={}, handler=lambda a: 42)
    assert await spec.run({}) == "42"


# --- ToolRegistry ---------------------------------------------------------------------


def test_select_resolves_declared_names_in_order() -> None:
    reg = ToolRegistry({"a": _echo_tool("a"), "b": _echo_tool("b")})
    specs = reg.select(["b", "a"])
    assert [s.name for s in specs] == ["b", "a"]


def test_select_skips_unknown_and_dedupes() -> None:
    reg = ToolRegistry({"a": _echo_tool("a")})
    specs = reg.select(["a", "ghost", "a"])
    assert [s.name for s in specs] == ["a"]  # unknown dropped, duplicate collapsed


def test_select_empty_for_none_or_empty() -> None:
    reg = default_tool_registry()
    assert reg.select(None) == []
    assert reg.select([]) == []


def test_unknown_reports_unregistered_names() -> None:
    reg = ToolRegistry({"a": _echo_tool("a")})
    assert reg.unknown(["a", "ghost"]) == ["ghost"]
    assert reg.unknown([]) == []


def test_registry_membership_and_names() -> None:
    reg = default_tool_registry()
    assert "get_current_time" in reg
    assert reg.names() == ["get_current_time"]
    assert len(reg) == 1


# --- built-in get_current_time --------------------------------------------------------


def _fixed_clock() -> datetime:
    return datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC)


async def test_get_current_time_defaults_to_utc() -> None:
    spec = get_current_time_tool(now=_fixed_clock)
    out = await spec.run({})
    assert "2026-06-25T12:00:00+00:00" in out
    assert "UTC" in out


async def test_get_current_time_converts_known_timezone() -> None:
    pytest.importorskip("tzdata")  # needs the IANA db on Windows
    spec = get_current_time_tool(now=_fixed_clock)
    out = await spec.run({"timezone": "America/New_York"})
    # 12:00 UTC is 08:00 in EDT (June).
    assert "08:00:00" in out
    assert "America/New_York" in out


async def test_get_current_time_unknown_timezone_falls_back_to_utc() -> None:
    spec = get_current_time_tool(now=_fixed_clock)
    out = await spec.run({"timezone": "Mars/Olympus"})
    assert "UTC" in out
    assert "unknown timezone" in out
    assert "2026-06-25T12:00:00+00:00" in out


def test_default_registry_clock_is_injectable() -> None:
    reg = default_tool_registry(now=_fixed_clock)
    assert reg.get("get_current_time") is not None
