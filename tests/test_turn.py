"""Barge-in: the TurnController cancels an in-flight response when the user speaks (M3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from personavoice.orchestrator import StreamingPipeline, TurnController
from personavoice.persona import load_persona

from .fakes import make_backend


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


async def _collect_sink(out: list[bytes]):
    async def sink(chunk: bytes) -> None:
        out.append(chunk)

    return sink


async def test_drives_audio_to_sink_until_done() -> None:
    async def audio() -> AsyncIterator[bytes]:
        for b in (b"a", b"b", b"c"):
            yield b

    out: list[bytes] = []
    ctrl = TurnController(await _collect_sink(out))
    ctrl.begin(audio())
    await ctrl.join()

    assert out == [b"a", b"b", b"c"]
    assert not ctrl.speaking


async def test_interrupt_cancels_in_flight_response_and_closes_generator() -> None:
    closed = asyncio.Event()
    started = asyncio.Event()

    async def slow_audio() -> AsyncIterator[bytes]:
        try:
            started.set()
            yield b"first"
            await asyncio.sleep(10)  # would block forever without barge-in
            yield b"never"
        finally:
            closed.set()  # generator cleanup must run on cancellation

    out: list[bytes] = []
    ctrl = TurnController(await _collect_sink(out))
    ctrl.begin(slow_audio())
    await started.wait()
    await asyncio.sleep(0)  # let the first chunk reach the sink

    assert ctrl.interrupt() is True
    await ctrl.join()

    assert out == [b"first"]  # only what played before barge-in
    assert closed.is_set()  # the streaming generator was torn down
    assert not ctrl.speaking


async def test_begin_interrupts_the_previous_response() -> None:
    async def long_audio(tag: bytes) -> AsyncIterator[bytes]:
        yield tag
        await asyncio.sleep(10)

    out: list[bytes] = []
    ctrl = TurnController(await _collect_sink(out))
    ctrl.begin(long_audio(b"old"))
    await asyncio.sleep(0)
    ctrl.begin(long_audio(b"new"))  # new utterance supersedes the old one
    await asyncio.sleep(0)
    ctrl.interrupt()
    await ctrl.join()

    assert b"old" in out and b"new" in out


async def test_interrupt_when_idle_is_a_noop() -> None:
    ctrl = TurnController(await _collect_sink([]))
    assert ctrl.interrupt() is False
    await ctrl.join()  # nothing in flight — returns immediately


async def test_barge_in_commits_partial_reply_to_history(config_dir: Path) -> None:
    # A reply that streams slowly; barge-in mid-way should still record what was spoken.
    gate = asyncio.Event()

    async def tokens():
        yield "First sentence. "
        gate.set()
        await asyncio.sleep(10)
        yield "Second sentence."

    backend = make_backend()

    async def stream_chat(messages, persona):
        async for t in tokens():
            yield t

    backend.llm.stream_chat = stream_chat  # type: ignore[method-assign]
    pipe = StreamingPipeline(backend, _companion(config_dir))

    out: list[bytes] = []
    ctrl = TurnController(await _collect_sink(out))
    ctrl.begin(pipe.stream_response("hi"))
    await gate.wait()
    await asyncio.sleep(0)
    ctrl.interrupt()
    await ctrl.join()

    # The user turn + the partial reply that was actually spoken are in history.
    assert pipe.history[0].content == "hi"
    assert pipe.history[1].content == "First sentence."
    assert out == [b"RIFF" + b"First sentence."]
