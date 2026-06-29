"""Tool / function calling — the tool registry and built-in tools.

The brain is otherwise a closed conversationalist. This module lets a persona *call out*
mid-turn: the teacher can look something up, the companion can check the time. It owns the
two pure pieces — the **catalog of callable tools** (`ToolRegistry`) and the **tool contract**
(`ToolSpec`: a JSON-schema description plus an async handler) — while the *protocol* of getting
the model to emit a call and feeding the result back lives in the LLM adapter
(`adapters/llm/_openai_compat.py`). Keeping them apart means the adapter depends only on a tiny
structural `Tool` interface (`adapters/llm/base.Tool`), not on this module, so there's no import
cycle (orchestrator → adapters, never the reverse).

A persona opts in by listing tool names in its `tools:` field (empty by default, so an unchanged
persona is a pure conversationalist and the LLM is never sent any tool schema). `select()` maps
those names to the concrete specs the pipeline hands the adapter; an unknown name is skipped with
a warning so a persona that references an unconfigured tool still runs.

Everything here is pure (no network, no ML deps) and the one shipped tool (`get_current_time`)
takes an injectable clock, so the whole surface is unit-testable offline — the same discipline as
`emotion.py` / `chunker.py`. Side-effecting or networked tools (weather, web search) are
deliberately *not* shipped here: they must be added behind explicit gating — any
side-effecting tool stays gated.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("personavoice.tools")

# A tool handler takes the parsed JSON arguments and returns the result text the model will see.
# Sync or async — `ToolSpec.run` normalizes both, so a trivial tool stays a plain function.
ToolHandler = Callable[[dict[str, Any]], "str | Awaitable[str]"]


@dataclass(frozen=True)
class ToolSpec:
    """One callable tool: its OpenAI-style schema plus the handler that runs it.

    `parameters` is a JSON Schema object describing the call arguments (``{"type": "object",
    ...}``); it is sent to the model verbatim. `handler` receives the parsed arguments dict and
    returns the result string fed back into the conversation. Structurally satisfies the adapter's
    `Tool` protocol (`name` / `schema()` / `run()`), so the adapter never imports this module.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        """The OpenAI `tools` entry for this tool (a ``{"type": "function", ...}`` object)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, arguments: dict[str, Any]) -> str:
        """Invoke the handler (sync or async) and return its result text.

        The handler's own exceptions propagate to the caller (the adapter's tool loop), which
        turns them into a safe error result the model can recover from rather than failing the
        turn — keeping that policy in one place.
        """
        result = self.handler(arguments)
        if inspect.isawaitable(result):
            result = await result
        return str(result)


class ToolRegistry:
    """The set of tools available to a deployment, keyed by name.

    The pipeline calls `select(persona.tools)` to get the subset a persona opted into and hands
    those specs to the LLM adapter. Unknown names are dropped (logged once) so a stale persona
    reference degrades gracefully instead of crashing the turn.
    """

    def __init__(self, tools: Mapping[str, ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = dict(tools or {})

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        """All registered tool names, sorted (stable for `--check` / listings)."""
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def select(self, names: list[str] | None) -> list[ToolSpec]:
        """Resolve a persona's declared tool names to specs, in declared order.

        Unknown names are skipped with a warning (a persona may reference a tool that isn't
        configured on this backend); duplicates are de-duplicated, preserving first position.
        """
        out: list[ToolSpec] = []
        seen: set[str] = set()
        for name in names or []:
            if name in seen:
                continue
            spec = self._tools.get(name)
            if spec is None:
                logger.warning("persona references unknown tool %r; skipping", name)
                continue
            seen.add(name)
            out.append(spec)
        return out

    def unknown(self, names: list[str] | None) -> list[str]:
        """The declared names that aren't registered (for `--check` validation)."""
        return [name for name in (names or []) if name not in self._tools]


# --------------------------------------------------------------------------------------
# Built-in tools (safe, side-effect-free, offline)
# --------------------------------------------------------------------------------------


# Default clock: a timezone-aware UTC `datetime`. Injectable so tests are deterministic.
def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _make_get_current_time(now: Callable[[], datetime]) -> ToolHandler:
    """Build the `get_current_time` handler bound to a clock.

    Returns the current date/time as an ISO-8601 string. An optional IANA `timezone` argument
    converts into that zone; an unknown or unavailable zone (e.g. no `tzdata` installed) falls
    back to UTC with a note rather than erroring, so the tool is robust everywhere.
    """

    def handler(arguments: dict[str, Any]) -> str:
        current = now()
        tz_name = arguments.get("timezone")
        if isinstance(tz_name, str) and tz_name.strip():
            tz_name = tz_name.strip()
            try:
                from zoneinfo import ZoneInfo

                current = current.astimezone(ZoneInfo(tz_name))
            except Exception:
                stamp = current.isoformat(timespec="seconds")
                return f"The current time is {stamp} (UTC; unknown timezone {tz_name!r})."
            return f"The current time is {current.isoformat(timespec='seconds')} ({tz_name})."
        return f"The current time is {current.isoformat(timespec='seconds')} (UTC)."

    return handler


def get_current_time_tool(now: Callable[[], datetime] | None = None) -> ToolSpec:
    """The built-in clock tool: side-effect-free, offline, deterministic with an injected clock."""
    return ToolSpec(
        name="get_current_time",
        description=(
            "Get the current date and time. Optionally pass an IANA timezone name "
            "(e.g. 'America/New_York'); defaults to UTC. Use it whenever the user asks what "
            "time or date it is."
        ),
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone name, e.g. 'Europe/London'. Optional; omit for UTC."
                    ),
                }
            },
            "required": [],
        },
        handler=_make_get_current_time(now or _utc_now),
    )


def default_tool_registry(*, now: Callable[[], datetime] | None = None) -> ToolRegistry:
    """The registry the agent and demos build: the bundled safe tools.

    `now` is forwarded to the clock tool so tests can pin the time. Only side-effect-free,
    offline tools live here; anything networked or side-effecting is added deliberately elsewhere.
    """
    registry = ToolRegistry()
    registry.register(get_current_time_tool(now))
    return registry
