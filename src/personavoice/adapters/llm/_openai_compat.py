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

**Tool / function calling (Feature D).** When a persona opts into tools, `stream_chat_with_tools`
runs the loop: stream a completion *with the tool schemas*; if the model emits tool calls (and no
spoken content), execute each tool, append the call + result to the conversation, and loop;
otherwise the content streams straight through as the spoken reply. Nothing is voiced during the
tool round-trips — TTS is deferred until the follow-up text streams. The streamed tool-call
fragments (which arrive split across SSE chunks) are reassembled by `_ToolCallBuffer`; the parsing
is pure and offline-testable.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...models import Msg, Persona
from .base import LLMAdapter, Tool

_DEFAULT_TIMEOUT = 120.0
# Tool loop bounds (latency-sensitive). Overridable per-deployment via env; an out-of-range or
# unparsable value falls back to the default rather than failing the turn.
_DEFAULT_TOOL_MAX_ITERS = 4
_DEFAULT_TOOL_TIMEOUT = 10.0


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


def _chat_payload(
    model: str | None,
    wire_messages: list[dict[str, Any]],
    persona: Persona,
    *,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
    lora: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a chat-completions request body from already-wire-format messages.

    Used directly by the tool loop (which appends raw `assistant`/`tool` messages the `Msg` model
    can't express); `build_payload` is the `Msg`-list front door onto it.
    """
    # On a LoRA-capable backend the request `model` selects the served adapter by name;
    # otherwise it's the base model the stage was configured with.
    payload: dict[str, Any] = {
        "model": lora or model,
        "messages": wire_messages,
        "stream": stream,
        "temperature": persona.llm.temperature,
        "top_p": persona.llm.top_p,
        "max_tokens": persona.llm.max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if extra_body:
        payload.update(extra_body)
    return payload


def _to_wire(messages: list[Msg]) -> list[dict[str, Any]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


def build_payload(
    model: str | None,
    messages: list[Msg],
    persona: Persona,
    *,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
    lora: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _chat_payload(
        model,
        _to_wire(messages),
        persona,
        stream=stream,
        extra_body=extra_body,
        lora=lora,
        tools=tools,
    )


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


# --------------------------------------------------------------------------------------
# Tool / function calling (Feature D)
# --------------------------------------------------------------------------------------


@dataclass
class StreamDelta:
    """One parsed streaming chunk: spoken content and/or tool-call fragments."""

    content: str | None = None
    # Raw `delta.tool_calls` fragments (each a dict with `index`/`id`/`function.{name,arguments}`).
    tool_calls: list[dict[str, Any]] | None = None


def parse_stream_delta(line: str) -> StreamDelta | None:
    """Parse one SSE line into a `StreamDelta`, or None for blanks / `[DONE]` / non-data lines.

    Like `token_from_sse_line` but also surfaces `delta.tool_calls` so the tool loop can tell a
    spoken reply from a function call. Reasoning-only / role-only deltas yield an empty delta.
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
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    tool_calls = delta.get("tool_calls")
    return StreamDelta(content=content or None, tool_calls=tool_calls or None)


@dataclass
class ToolCall:
    """A fully reassembled tool call: a stable id, the tool name, and raw JSON argument text."""

    id: str
    name: str
    arguments: str


@dataclass
class _ToolCallBuffer:
    """Reassemble streamed `tool_calls` fragments into complete `ToolCall`s.

    Each call is keyed by its stream `index`; the `id`/`name` arrive in the first fragment and the
    `arguments` JSON streams in pieces, so we concatenate per index and keep first-seen order.
    """

    _slots: dict[int, dict[str, str]] = field(default_factory=dict)
    _order: list[int] = field(default_factory=list)

    def add(self, fragments: list[dict[str, Any]]) -> None:
        for frag in fragments:
            idx = frag.get("index", 0)
            slot = self._slots.get(idx)
            if slot is None:
                slot = {"id": "", "name": "", "arguments": ""}
                self._slots[idx] = slot
                self._order.append(idx)
            if frag.get("id"):
                slot["id"] = frag["id"]
            fn = frag.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]  # usually whole; concatenate defensively
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    def empty(self) -> bool:
        return not self._slots

    def calls(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for idx in self._order:
            slot = self._slots[idx]
            # A call with no name is unusable; the model never gives a usable result, so drop it.
            if not slot["name"]:
                continue
            # Some backends omit the id on a single call; synthesize a stable one so the
            # assistant/tool message pair still links correctly.
            call_id = slot["id"] or f"call_{idx}"
            out.append(ToolCall(id=call_id, name=slot["name"], arguments=slot["arguments"]))
        return out


def _assistant_tool_calls_message(calls: list[ToolCall]) -> dict[str, Any]:
    """The assistant turn that *requested* the tools (content null, tool_calls listed)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments or "{}"},
            }
            for c in calls
        ],
    }


async def _execute_tool(by_name: dict[str, Tool], call: ToolCall, timeout_s: float) -> str:
    """Run one tool call, returning a result string — including a *safe error string* on failure.

    A hallucinated tool name, bad argument JSON, a timeout, or a handler exception all become a
    short error message fed back to the model instead of raising, so a single tool problem lets the
    model recover (or apologize) rather than killing the turn.
    """
    tool = by_name.get(call.name)
    if tool is None:
        return f"Error: unknown tool {call.name!r}."
    raw = call.arguments.strip()
    try:
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for tool {call.name!r}."
    if not isinstance(args, dict):
        return f"Error: arguments for tool {call.name!r} must be a JSON object."
    try:
        return await asyncio.wait_for(tool.run(args), timeout_s)
    except TimeoutError:
        return f"Error: tool {call.name!r} timed out after {timeout_s:g}s."
    except Exception as exc:  # a misbehaving tool must not kill the turn
        return f"Error: tool {call.name!r} failed: {exc}."


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value >= 1 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


class OpenAICompatLLM(LLMAdapter):
    """Base for backends exposing an OpenAI-compatible `/v1/chat/completions` stream.

    Subclasses set `name`, `implemented = True`, and `default_base_url`. The active base
    URL can still be overridden per-deployment via the `base_url` adapter option. Backends that
    can hot-load LoRA adapters (vLLM) set `supports_lora = True` so a persona's `llm.lora`
    routes the request to the served adapter.
    """

    default_base_url = "http://localhost:8000/v1"
    supports_lora = False
    # The OpenAI tools API is part of `/v1/chat/completions`, so both LM Studio and vLLM can
    # route function calls (vLLM needs `--enable-auto-tool-choice --tool-call-parser ...`).
    supports_tools = True

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

    async def stream_chat_with_tools(
        self, messages: list[Msg], persona: Persona, tools: Sequence[Tool]
    ) -> AsyncIterator[str]:
        """Stream a reply, resolving tool calls first (Feature D).

        With no tools this is exactly `stream_chat`. Otherwise each iteration streams a completion
        that includes the tool schemas: if spoken content comes back it streams through as the
        reply (done); if the model instead emits tool calls, they're executed and appended, and the
        loop continues. A `max_iters` cap bounds the round-trips; if it's hit, one final completion
        *without* tools is streamed so the user still hears an answer.
        """
        if not tools:
            async for tok in self.stream_chat(messages, persona):
                yield tok
            return

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
        max_iters = _env_int("PERSONAVOICE_TOOL_MAX_ITERS", _DEFAULT_TOOL_MAX_ITERS)
        tool_timeout = _env_float("PERSONAVOICE_TOOL_TIMEOUT", _DEFAULT_TOOL_TIMEOUT)

        by_name: dict[str, Tool] = {t.name: t for t in tools}
        schemas = [t.schema() for t in tools]
        wire = _to_wire(messages)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for _ in range(max_iters):
                payload = _chat_payload(
                    self.model,
                    wire,
                    persona,
                    stream=True,
                    extra_body=extra_body,
                    lora=lora,
                    tools=schemas,
                )
                buffer = _ToolCallBuffer()
                spoke = False
                async with client.stream("POST", chat_url(base_url), json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        delta = parse_stream_delta(line)
                        if delta is None:
                            continue
                        if delta.content:
                            spoke = True
                            yield delta.content
                        elif delta.tool_calls and not spoke:
                            buffer.add(delta.tool_calls)
                # The model spoke (terminal answer) or asked for nothing actionable → we're done.
                if spoke or buffer.empty():
                    return
                calls = buffer.calls()
                if not calls:
                    return
                wire.append(_assistant_tool_calls_message(calls))
                for call in calls:
                    result = await _execute_tool(by_name, call, tool_timeout)
                    wire.append({"role": "tool", "tool_call_id": call.id, "content": result})

            # Iteration cap reached: take the tool results we have and ask for a plain spoken
            # answer (no tools), so a runaway tool loop still ends in something the user hears.
            final = _chat_payload(
                self.model, wire, persona, stream=True, extra_body=extra_body, lora=lora
            )
            async with client.stream("POST", chat_url(base_url), json=final) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    token = token_from_sse_line(line)
                    if token:
                        yield token
