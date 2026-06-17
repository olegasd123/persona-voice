"""Streaming STT → LLM → TTS pipeline (M3).

The M1 `Pipeline` waits for the whole reply before speaking; this streaming variant pipes
LLM tokens into the sentence chunker and on into TTS, so audio starts as soon as the first
sentence is ready. That's the core of "stream everything": it cuts *perceived* latency
even though total compute is unchanged.

    user text ──▶ LLM.stream_chat ──▶ stream_sentences ──▶ TTS.stream_tts ──▶ wav chunks
                       │ (tokens)            │ (sentences)         │ (audio)

It depends only on the adapter base classes, so it runs with the real backend or with the
test fakes. STT-in is handled upstream (by LiveKit VAD in `agent.py`, or by a one-shot
`transcribe` in the demo); `stream_response` takes already-transcribed user text.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from ..adapters.factory import Backend
from ..memory import ConversationMemory
from ..models import Msg, Persona, Role
from ..persona.prompt import build_messages
from ..voice.registry import VoiceRegistry
from .chunker import stream_sentences
from .pipeline import voice_ref_for


@dataclass
class StreamMetrics:
    """Latency landmarks for one streamed response (seconds from turn start)."""

    first_token: float | None = None  # time to the LLM's first token
    first_audio: float | None = None  # time to the first audio chunk (perceived latency)
    total: float | None = None  # time to finish speaking the whole reply
    reply: str = ""  # the full reply text, assembled from the token stream


class StreamingPipeline:
    """Runs one persona conversation, streaming each reply sentence-by-sentence.

    Keeps an in-memory `history` like the M1 pipeline so multi-turn sessions have context.
    `stream_response` is a cancellable async generator: cancelling the task that drives it
    (see `TurnController`) tears down the in-flight LLM and TTS streams for barge-in.
    """

    def __init__(
        self,
        backend: Backend,
        persona: Persona,
        voices: VoiceRegistry | None = None,
        *,
        memory: ConversationMemory | None = None,
        user_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.persona = persona
        self.voices = voices
        self.history: list[Msg] = []
        # Cross-session memory (M8): consent-gated, no-op until a user opts in. The session id
        # ties this run's recorded turns together so recall can exclude the live session.
        self.memory = memory
        self.user_id = user_id
        self.session_id: str | None = (
            memory.start_session(user_id, persona.id) if memory is not None and user_id else None
        )

    async def stream_response(
        self,
        user_text: str,
        *,
        history: list[Msg] | None = None,
        metrics: StreamMetrics | None = None,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield WAV audio chunks for the persona's reply to `user_text`.

        Records latency landmarks into `metrics` (if given) and appends the user turn and
        full reply to `self.history` when using the internal history.

        `on_sentence`, if given, is awaited with each chunked sentence as it's pulled into
        TTS (i.e. just before that sentence is voiced). The live agent uses this to publish
        a growing assistant transcript over WebRTC in step with the spoken audio; it's a
        no-op for the offline demo/tests. Exceptions from the callback are swallowed so a
        transcript-publish hiccup never stops the audio.
        """
        use_internal = history is None
        history = self.history if use_internal else history

        memory_context = await self._recall(user_text)
        messages = build_messages(
            self.persona, history=history, user_input=user_text, memory_context=memory_context
        )
        voice = voice_ref_for(self.persona, self.backend, self.voices)
        collected: list[str] = []
        t0 = time.perf_counter()

        async def _tokens() -> AsyncIterator[str]:
            async for tok in self.backend.llm.stream_chat(messages, self.persona):
                if metrics is not None and metrics.first_token is None:
                    metrics.first_token = time.perf_counter() - t0
                collected.append(tok)
                yield tok

        async def _tap(it: AsyncIterator[str]) -> AsyncIterator[str]:
            async for sentence in it:
                if on_sentence is not None:
                    # A transcript-publish hiccup must not mute the audio.
                    with contextlib.suppress(Exception):
                        await on_sentence(sentence)
                yield sentence

        sentences = _tap(stream_sentences(_tokens()))
        try:
            async for audio in self.backend.tts.stream_tts(sentences, voice):
                if metrics is not None and metrics.first_audio is None:
                    metrics.first_audio = time.perf_counter() - t0
                yield audio
        finally:
            # Runs on normal completion *and* on cancellation (barge-in): record what we
            # produced and commit the (possibly partial) reply to history so context stays
            # consistent with what the user actually heard.
            reply = "".join(collected).strip()
            if metrics is not None:
                metrics.total = time.perf_counter() - t0
                metrics.reply = reply
            if use_internal:
                self.history.append(Msg(role=Role.user, content=user_text))
                self.history.append(Msg(role=Role.assistant, content=reply))
            await self._remember(user_text, reply)

    async def _recall(self, user_text: str) -> str | None:
        """Memory block to inject for this turn (None when memory is off / no consent)."""
        if self.memory is None or not self.user_id:
            return None
        return await self.memory.recall(
            self.user_id, user_text, persona=self.persona, session_id=self.session_id
        )

    async def _remember(self, user_text: str, reply: str) -> None:
        """Persist this exchange to cross-session memory (no-op when memory is off)."""
        if self.memory is None or not self.user_id or self.session_id is None:
            return
        await self.memory.record_user(self.user_id, self.session_id, self.persona, user_text)
        if reply:
            await self.memory.record_assistant(self.user_id, self.session_id, self.persona, reply)
