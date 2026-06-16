"""Turn-taking & barge-in control (M3).

In a live conversation the user can start talking while the persona is still speaking
(*barge-in*). The fix is simple in shape: drive each response as a single asyncio task, and
when the user starts a new utterance, cancel it. Cancellation propagates into the streaming
response generator, which closes the in-flight LLM (httpx) and TTS streams.

`TurnController` is backend-agnostic — it drives any `AsyncIterator[bytes]` (audio chunks)
into an async `sink` callback — so the barge-in semantics are unit-testable with fakes,
independent of LiveKit. `agent.py` wires VAD speech-start to `interrupt()`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

AudioSink = Callable[[bytes], Awaitable[None]]


class TurnController:
    """Runs one response at a time and cancels it on barge-in."""

    def __init__(self, sink: AudioSink) -> None:
        self._sink = sink
        self._task: asyncio.Task[None] | None = None

    @property
    def speaking(self) -> bool:
        """True while a response is actively streaming to the sink."""
        return self._task is not None and not self._task.done()

    def begin(self, audio: AsyncIterator[bytes]) -> asyncio.Task[None]:
        """Interrupt any in-flight response, then start streaming `audio` to the sink."""
        self.interrupt()
        self._task = asyncio.create_task(self._drive(audio))
        return self._task

    def interrupt(self) -> bool:
        """Cancel the in-flight response (barge-in). Returns True if one was speaking."""
        if self.speaking:
            assert self._task is not None
            self._task.cancel()
            return True
        return False

    async def join(self) -> None:
        """Await the current response, swallowing the cancellation from a barge-in."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _drive(self, audio: AsyncIterator[bytes]) -> None:
        try:
            async for chunk in audio:
                await self._sink(chunk)
        finally:
            # Ensure the generator's own finally-blocks run (close LLM/TTS streams) even
            # when we're cancelled mid-flight.
            aclose = getattr(audio, "aclose", None)
            if aclose is not None:
                await aclose()
