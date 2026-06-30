"""Streaming STT → LLM → TTS pipeline.

The turn-based `Pipeline` waits for the whole reply before speaking; this streaming variant pipes
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
from ..emotion import split_leading_emotion
from ..memory import ConversationMemory
from ..models import Msg, Persona, SessionOptions
from ..persona.prompt import build_messages
from ..safety import Moderator
from ..voice.registry import VoiceRegistry
from .chunker import chunk_kwargs_from_env, stream_sentences
from .pipeline import _PipelineBase
from .tools import ToolRegistry


@dataclass
class StreamMetrics:
    """Latency landmarks for one streamed response (seconds from turn start)."""

    first_token: float | None = None  # time to the LLM's first token
    first_audio: float | None = None  # time to the first audio chunk (perceived latency)
    total: float | None = None  # time to finish speaking the whole reply
    reply: str = ""  # the full reply text, assembled from the token stream


class StreamingPipeline(_PipelineBase):
    """Runs one persona conversation, streaming each reply sentence-by-sentence.

    Keeps an in-memory `history` like the turn-based pipeline so multi-turn sessions have context.
    `stream_response` is a cancellable async generator: cancelling the task that drives it
    (see `TurnController`) tears down the in-flight LLM and TTS streams for barge-in.

    Beyond the shared `_PipelineBase` scaffolding it adds the per-sentence output guard
    (`_guard`), TTS chunk sizing, and consent-gated cross-session memory.
    """

    def __init__(
        self,
        backend: Backend,
        persona: Persona,
        voices: VoiceRegistry | None = None,
        *,
        options: SessionOptions | None = None,
        moderator: Moderator | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        user_id: str | None = None,
        chunk_kwargs: dict[str, int | None] | None = None,
        dynamic_emotion: bool | None = None,
    ) -> None:
        super().__init__(
            backend,
            persona,
            voices,
            options=options,
            moderator=moderator,
            tools=tools,
            dynamic_emotion=dynamic_emotion,
        )
        # TTS chunk-sizing knobs (latency); resolved from the environment by default so
        # the live agent and demos pick up `PERSONAVOICE_TTS_*` without extra wiring.
        self.chunk_kwargs = chunk_kwargs if chunk_kwargs is not None else chunk_kwargs_from_env()
        # Cross-session memory: consent-gated, no-op until a user opts in. The session id
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

        # Input moderation: a flagged/crisis utterance short-circuits to a safe spoken reply,
        # skipping the LLM (and memory recall) entirely.
        canned = await self._safe_input_reply(user_text)
        if canned is None:
            memory_context = await self._recall(user_text)
            messages = build_messages(
                self.persona,
                history=history,
                user_input=user_text,
                memory_context=memory_context,
                options=self.options,
                dynamic_emotion=self.dynamic_emotion,
            )
        else:
            messages = []
        voice = self._resolve_voice()
        collected: list[str] = []
        # `spoken` records the sentences actually voiced once the output guard is engaged; on a
        # breach it lets the turn commit the safe text (not the raw tokens) to history/memory.
        spoken: list[str] = []
        moderated = False
        demeanor = self._demeanor
        t0 = time.perf_counter()

        tool_specs = self._tool_specs()

        async def _raw() -> AsyncIterator[str]:
            """The reply token stream — a canned safe reply, or the LLM — timing the first token.

            When the persona declares tools, the LLM stream resolves any tool calls first (no
            audio is produced during the tool round-trips); the follow-up reply then streams as
            usual. With no tools this is the plain `stream_chat` path.
            """
            if canned is not None:
                if metrics is not None and metrics.first_token is None:
                    metrics.first_token = time.perf_counter() - t0
                yield canned
                return
            if tool_specs:
                token_stream = self.backend.llm.stream_chat_with_tools(
                    messages, self.persona, tool_specs
                )
            else:
                token_stream = self.backend.llm.stream_chat(messages, self.persona)
            async for tok in token_stream:
                if metrics is not None and metrics.first_token is None:
                    metrics.first_token = time.perf_counter() - t0
                yield tok

        async def _collect(it: AsyncIterator[str]) -> AsyncIterator[str]:
            """Tap the (post-tag) tokens to assemble the spoken reply for history/metrics."""
            async for tok in it:
                collected.append(tok)
                yield tok

        async def _tap(it: AsyncIterator[str]) -> AsyncIterator[str]:
            async for sentence in it:
                if on_sentence is not None:
                    # A transcript-publish hiccup must not mute the audio.
                    with contextlib.suppress(Exception):
                        await on_sentence(sentence)
                yield sentence

        async def _guard(it: AsyncIterator[str]) -> AsyncIterator[str]:
            """Output bound (streaming): moderate each sentence just before it is voiced.

            Clean sentences pass straight through; the first sentence that breaches the bound
            (the `rude` cap, harassment, threats) is replaced with the safe line and the stream
            stops — the streaming analogue of the turn-based `_bound_output`, run per sentence so
            it never buffers the whole reply. Runs ahead of `_tap` so the published transcript
            shows the safe line, not the breaching sentence.
            """
            nonlocal moderated
            moderator = self.moderator
            assert moderator is not None  # only wired in when a guard is configured
            async for sentence in it:
                verdict = await moderator.check_output(sentence, demeanor=demeanor)
                if verdict.flagged and verdict.replacement is not None:
                    moderated = True
                    spoken.append(verdict.replacement)
                    # Tear down the in-flight LLM/TTS source stream so we stop generating the
                    # rest of the reply (mirrors TurnController's teardown on barge-in).
                    aclose = getattr(it, "aclose", None)
                    if aclose is not None:
                        with contextlib.suppress(Exception):
                            await aclose()
                    yield verdict.replacement
                    return
                spoken.append(sentence)
                yield sentence

        try:
            # Per-utterance emotion: peel a leading `[emotion]` tag off the reply
            # (buffering only its leading window) and apply it to this turn's voice — before TTS
            # captures `voice`. Skipped for a canned safe reply (no LLM, no tag) and when off, so
            # the original passthrough is untouched. Inside `try` so a first-token error still
            # commits the (empty) turn in `finally`, as before.
            body = _raw()
            if self.dynamic_emotion and canned is None:
                emotion, body = await split_leading_emotion(body)
                if emotion is not None:
                    voice = voice.model_copy(update={"emotion": emotion})
            sentence_stream: AsyncIterator[str] = stream_sentences(
                _collect(body),
                max_chunk_chars=self.chunk_kwargs.get("max_chunk_chars") or 240,
                first_chunk_chars=self.chunk_kwargs.get("first_chunk_chars"),
            )
            # Output moderation runs per sentence, before TTS — skipped for our own canned safe
            # reply and when no guard is configured, so the default path is unchanged.
            if self.moderator is not None and canned is None:
                sentence_stream = _guard(sentence_stream)
            sentences = _tap(sentence_stream)
            async for audio in self.backend.tts.stream_tts(sentences, voice):
                if metrics is not None and metrics.first_audio is None:
                    metrics.first_audio = time.perf_counter() - t0
                yield audio
        finally:
            # Runs on normal completion *and* on cancellation (barge-in): record what we
            # produced and commit the (possibly partial) reply to history so context stays
            # consistent with what the user actually heard.
            # A breached turn voiced the safe replacement, not the raw tokens — commit what the
            # user actually heard (clean sentences + the safe line) so history, memory, and the
            # metrics record stay consistent with the audio.
            reply = " ".join(spoken).strip() if moderated else "".join(collected).strip()
            if metrics is not None:
                metrics.total = time.perf_counter() - t0
                metrics.reply = reply
            if use_internal:
                self._commit_history(user_text, reply)
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
