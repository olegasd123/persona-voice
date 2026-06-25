"""LiveKit Agents worker: live, streaming, barge-in conversation.

This is the production entrypoint. LiveKit handles the WebRTC transport and ships Silero
VAD for endpointing; everything model-side reuses the tested streaming core:

    mic ─WebRTC▶ VAD endpoint ─▶ STT.transcribe ─▶ StreamingPipeline.stream_response
                                                          │ wav chunks
        speaker ◀─WebRTC─ rtc.AudioSource ◀── TurnController ◀──┘

`TurnController` makes barge-in one line: when VAD reports the user started talking, we
`interrupt()` the in-flight response (tearing down LLM+TTS) and flush the output queue.

LiveKit and its plugins are **heavy and only used at runtime**, so they're imported lazily
— `server --check`, the unit tests, and the offline demos never need the `livekit` extra.
The pure pieces (wav↔PCM conversion in `audio.py`, the chunker/pipeline/turn controller)
are unit-tested; the live browser run is validated against a running LiveKit server (see
README "Live LiveKit agent"), the analog of the on-GPU validation step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from ..adapters.factory import Backend, build_backend
from ..audio import pcm16_to_wav, wav_to_pcm16
from ..memory import ConversationMemory
from ..models import Persona, SessionOptions
from ..obs import configure_logging, turn_metrics_from_stream
from ..persona.registry import PersonaRegistry
from ..persona.store import UserPersonaStore
from ..safety import Moderator, moderator_from_env
from ..server.config import (
    Settings,
    build_conversation_memory,
    load_backend_config,
    load_user_persona_store,
    load_voice_registry,
)
from ..voice.registry import VoiceRegistry
from .endpointing import vad_load_kwargs
from .pipeline import voice_ref_for
from .streaming import StreamingPipeline, StreamMetrics
from .tools import ToolRegistry, default_tool_registry
from .turn import TurnController

# LiveKit's runtime types (`rtc.AudioSource`, `rtc.Track`, ...) are only present when the
# `livekit` extra is installed, so they're annotated as `Any` to keep this module importable
# (and type-checkable) without it.
logger = logging.getLogger("personavoice.agent")

# Output track format. 24 kHz mono matches Kokoro/Chatterbox output and is a common WebRTC
# rate; per-sentence TTS wavs are resampled to this before being captured as frames.
_OUT_SAMPLE_RATE = 24000
_FRAME_MS = 20  # frame size pushed to the AudioSource (WebRTC-typical)

# A bare persona id is an identifier (matches our persona file stems); anything else
# (JSON arrays, stray punctuation) is rejected so it can't be mistaken for an id.
_PERSONA_ID_RE = re.compile(r"[A-Za-z0-9_-]+")

# Publishes one assistant-transcript update: (segment_id, full_text, is_final). Injected into
# `PersonaAgent` so the LiveKit-specific construction (`make_transcript_publisher`) stays out
# of the testable turn logic; `None` (no transport, e.g. unit tests) skips publishing.
TranscriptPublisher = Callable[[str, str, bool], Coroutine[Any, Any, None]]


def _require_livekit() -> tuple[Any, Any, Any]:
    """Import the LiveKit runtime lazily, with a clear install hint if it's missing."""
    try:
        from livekit import agents, rtc
        from livekit.plugins import silero
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "livekit is not installed; install the streaming extra: `pip install -e '.[livekit]'`"
        ) from exc
    return agents, rtc, silero


def _vad_event_type(agents: Any, rtc: Any) -> Any:
    """Return the VAD event enum across LiveKit SDK versions."""
    agents_vad = getattr(agents, "vad", None)
    event_type = getattr(agents_vad, "VADEventType", None) or getattr(rtc, "VADEventType", None)
    if event_type is None:
        raise RuntimeError("LiveKit VAD event type is unavailable; update livekit-agents")
    return event_type


class PersonaAgent:
    """Drives one participant's conversation: VAD → STT → streaming reply, with barge-in.

    Owns the per-session `StreamingPipeline` (so history accumulates) and a
    `TurnController` bound to an `rtc.AudioSource` sink that captures TTS audio onto the
    published WebRTC track.
    """

    def __init__(
        self,
        backend: Backend,
        persona: Persona,
        source: Any,
        voices: VoiceRegistry | None = None,
        *,
        options: SessionOptions | None = None,
        moderator: Moderator | None = None,
        tools: ToolRegistry | None = None,
        publish_transcript: TranscriptPublisher | None = None,
        memory: ConversationMemory | None = None,
        user_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._persona = persona
        self._source = source
        self._memory = memory
        # `_explicit_options` are the per-session overrides the client set (metadata / data
        # message); `_options` is them merged over the persona's `session_defaults`, so a custom
        # persona's authored defaults (e.g. a tutor that defaults to CEFR A1) apply unless the
        # client overrides them. Recomputed on a persona switch.
        self._explicit_options = options or SessionOptions()
        self._options = self._effective_options(persona)
        self._pipeline = StreamingPipeline(
            backend,
            persona,
            voices,
            options=self._options,
            moderator=moderator,
            tools=tools,
            memory=memory,
            user_id=user_id,
        )
        self._turn = TurnController(self._capture_wav)
        self._frame_samples = max(1, _OUT_SAMPLE_RATE * _FRAME_MS // 1000)
        # Publishes the assistant's spoken words back as a live transcript (the client renders
        # LiveKit `TranscriptionEvent`s). `None` (no transport, e.g. unit tests) skips it.
        self._publish_transcript = publish_transcript
        # Strong refs to in-flight best-effort "final transcript" publishes (see `_speak`).
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def persona(self) -> Persona:
        return self._persona

    def set_persona(self, persona: Persona) -> None:
        """Hot-swap the persona mid-session, keeping the conversation history.

        Interrupts any in-flight reply so the next user turn is answered (and voiced) by the
        new persona; the shared `StreamingPipeline.history` carries over so the conversation
        continues rather than resetting. The effective options are recomputed over the new
        persona's `session_defaults` (the client's explicit overrides still win).
        """
        if persona.id == self._persona.id:
            return
        self._turn.interrupt()
        self._persona = persona
        self._pipeline.persona = persona
        new_options = self._effective_options(persona)
        if new_options != self._options:
            self._options = new_options
            self._pipeline.options = new_options
        logger.info("persona switched to %s mid-session", persona.id)

    @property
    def options(self) -> SessionOptions:
        return self._options

    def _effective_options(self, persona: Persona) -> SessionOptions:
        """Explicit session overrides layered over the persona's authored option defaults."""
        return self._explicit_options.merged_over(persona.session_defaults)

    def set_options(self, options: SessionOptions) -> None:
        """Apply per-session option changes mid-call (voice / CEFR / demeanor).

        The given options are *merged over* the current explicit overrides, so a data message
        that only sets `demeanor` keeps the existing voice/CEFR; the result is then layered over
        the persona's `session_defaults`. The in-flight reply is interrupted so the next user
        turn is answered under the new options (the pipeline reads them per turn).
        """
        new_explicit = options.merged_over(self._explicit_options)
        new_effective = new_explicit.merged_over(self._persona.session_defaults)
        if new_explicit == self._explicit_options and new_effective == self._options:
            return
        self._turn.interrupt()
        self._explicit_options = new_explicit
        self._options = new_effective
        self._pipeline.options = new_effective
        logger.info(
            "session options updated mid-session: %s", new_effective.model_dump(exclude_none=True)
        )

    async def _capture_wav(self, wav: bytes) -> None:
        """Sink: push one sentence's WAV onto the WebRTC track as 20 ms PCM frames."""
        _, rtc, _ = _require_livekit()
        pcm = wav_to_pcm16(wav, _OUT_SAMPLE_RATE)
        stride = self._frame_samples * 2  # 2 bytes per int16 sample
        for off in range(0, len(pcm), stride):
            data = pcm[off : off + stride]
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=_OUT_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(data) // 2,
            )
            # capture_frame paces to realtime, so cancellation (barge-in) lands here.
            await self._source.capture_frame(frame)

    def on_user_speech_started(self) -> None:
        """Barge-in: the user started talking — stop the assistant immediately."""
        if self._turn.interrupt():
            logger.info("barge-in: interrupted assistant mid-response")
            # Drop audio already queued in the source so playback stops now, not later.
            clear = getattr(self._source, "clear_queue", None)
            if clear is not None:
                clear()

    async def on_user_utterance(self, utterance_pcm: bytes, sample_rate: int) -> None:
        """A complete user utterance (VAD-segmented PCM) → transcribe → streamed reply.

        STT failures are caught and logged rather than propagated: one bad utterance must not
        kill the per-track consumer task (graceful error recovery) — the session stays
        live for the next turn.
        """
        wav = pcm16_to_wav(utterance_pcm, sample_rate)
        t0 = time.perf_counter()
        try:
            transcript = await self._backend.stt.transcribe(wav)
        except Exception:
            logger.exception("STT failed for an utterance; skipping this turn")
            return
        stt_s = time.perf_counter() - t0
        text = transcript.text.strip()
        if not text:
            return
        logger.info("user: %s", text)
        self._turn.begin(self._speak(text, stt_s))

    async def _speak(self, user_text: str, stt_s: float | None = None) -> Any:
        """Stream the reply audio while publishing the assistant transcript in step with it.

        Each chunked sentence grows a single transcription segment (keyed by `seg_id`) so the
        client updates one bubble in place rather than appending fragments. The final marker
        is published fire-and-forget from the `finally` so a barge-in cancellation — which
        tears this generator down mid-flight — still flips the (partial) line to final.

        A `StreamMetrics` is captured per turn and logged on completion (observability).
        A non-cancellation error mid-stream is logged and swallowed so the worker survives;
        a `CancelledError` (barge-in) is recorded as `interrupted` and re-raised so the turn
        controller's teardown still runs.
        """
        seg_id = uuid.uuid4().hex
        parts: list[str] = []
        publish = self._publish_transcript
        metrics = StreamMetrics()
        interrupted = False
        error: str | None = None

        async def _on_sentence(sentence: str) -> None:
            if publish is None:
                return
            cleaned = sentence.strip()
            if cleaned:
                parts.append(cleaned)
                await publish(seg_id, " ".join(parts), False)

        try:
            async for audio in self._pipeline.stream_response(
                user_text, metrics=metrics, on_sentence=_on_sentence
            ):
                yield audio
        except (asyncio.CancelledError, GeneratorExit):
            # Barge-in: cancellation arrives either as CancelledError (suspended in the LLM/TTS
            # stream) or GeneratorExit (suspended in the sink, torn down via aclose). Either way
            # record it as an interruption and re-raise so the turn controller's teardown runs.
            interrupted = True
            raise
        except Exception as exc:  # graceful recovery: keep the session alive after a failure
            error = repr(exc)
            logger.exception("turn failed during streaming")
        finally:
            turn_metrics_from_stream(
                self._persona.id,
                user_text,
                metrics,
                stt_s=stt_s,
                interrupted=interrupted,
                error=error,
            ).log(logger)
            if publish is not None and parts:
                # Fire-and-forget: on barge-in this generator is being cancelled, so awaiting
                # here would just re-raise — schedule the final marker as its own task.
                task = asyncio.create_task(publish(seg_id, " ".join(parts), True))
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)

    async def aclose(self) -> None:
        self._turn.interrupt()
        await self._turn.join()
        for task in list(self._pending):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._memory is not None:
            # Flush any in-flight profile distillation before the worker tears down.
            await self._memory.aclose()


def _load_vad(silero: Any) -> Any:
    """Load Silero VAD with the operator's endpointing knobs, tolerating old SDKs.

    `vad_load_kwargs()` reads `PERSONAVOICE_VAD_*`; if a kwarg isn't accepted by the installed
    `livekit-plugins-silero`, fall back to the stock defaults rather than crashing the worker.
    """
    kwargs = vad_load_kwargs()
    if not kwargs:
        return silero.VAD.load()
    try:
        return silero.VAD.load(**kwargs)
    except TypeError:
        logger.warning("silero VAD rejected tuning kwargs %s; using defaults", sorted(kwargs))
        return silero.VAD.load()


async def _consume_track(
    agent: PersonaAgent, track: Any, silero: Any, vad: Any | None = None
) -> None:
    """Run VAD over a participant's audio track and feed utterances to the agent.

    Reuses a prewarmed Silero VAD when one is passed (loaded once at worker startup); falls
    back to loading one on demand so the function still works outside the worker (e.g. tests).
    """
    agents, rtc, _ = _require_livekit()
    vad_event_type = _vad_event_type(agents, rtc)
    if vad is None:
        vad = _load_vad(silero)
    vad_stream = vad.stream()
    audio_stream = rtc.AudioStream(track)

    async def _pump_frames() -> None:
        async for ev in audio_stream:
            vad_stream.push_frame(ev.frame)

    pump = asyncio.create_task(_pump_frames())
    try:
        async for ev in vad_stream:
            if ev.type == vad_event_type.START_OF_SPEECH:
                agent.on_user_speech_started()
            elif ev.type == vad_event_type.END_OF_SPEECH:
                # `ev.frames` holds the buffered speech; concatenate to one PCM utterance.
                pcm = b"".join(bytes(f.data) for f in ev.frames)
                sr = ev.frames[0].sample_rate if ev.frames else 16000
                await agent.on_user_utterance(pcm, sr)
    finally:
        pump.cancel()
        await vad_stream.aclose()


def make_transcript_publisher(local_participant: Any, track_sid: str | None) -> TranscriptPublisher:
    """Build the `TranscriptPublisher` that pushes assistant transcripts over WebRTC.

    Lives here (not in `PersonaAgent`) so the LiveKit-specific `rtc.Transcription` construction
    stays out of the testable turn logic. Each call publishes one segment (growing `text`,
    flipped to `final` at the end); failures are swallowed so a transcript hiccup never breaks
    the turn. If the track sid is unknown, returns a no-op.
    """

    async def _noop(_seg_id: str, _text: str, _is_final: bool) -> None:
        return None

    if local_participant is None or not track_sid:
        return _noop

    async def _publish(seg_id: str, text: str, is_final: bool) -> None:
        _, rtc, _ = _require_livekit()
        try:
            segment = rtc.TranscriptionSegment(
                id=seg_id,
                text=text,
                start_time=0,
                end_time=0,
                final=is_final,
                language="en",
            )
            await local_participant.publish_transcription(
                rtc.Transcription(
                    participant_identity=local_participant.identity,
                    track_sid=track_sid,
                    segments=[segment],
                )
            )
        except Exception:  # transcript publishing must never break the turn
            logger.debug("failed to publish assistant transcript", exc_info=True)

    return _publish


def _data_text(data: Any) -> str | None:
    """Decode a LiveKit data message to text, across SDK shapes.

    Newer LiveKit passes a `DataPacket` (with a `.data` bytes attribute); older versions
    pass the raw `bytes` directly. Either way we want UTF-8 text to parse a persona id from.
    """
    payload = getattr(data, "data", data)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload).decode("utf-8", "ignore")
    if isinstance(payload, str):
        return payload
    return None


def persona_id_from_metadata(meta: str | None) -> str | None:
    """Extract a persona id from one metadata string, or None.

    Accepts a JSON object carrying a `persona` key (`{"persona": "hr_interviewer"}`, how a
    client typically tags a room/job) or a bare persona id (`"hr_interviewer"`). Anything
    empty or unparsable yields None so the caller can fall through to the next source.
    """
    if not meta or not meta.strip():
        return None
    meta = meta.strip()
    if meta.startswith("{"):
        try:
            obj = json.loads(meta)
        except json.JSONDecodeError:
            return None
        pid = obj.get("persona") if isinstance(obj, dict) else None
        return pid.strip() if isinstance(pid, str) and pid.strip() else None
    return meta if _PERSONA_ID_RE.fullmatch(meta) else None


def resolve_persona_id(sources: list[str | None], default: str) -> str:
    """First persona id found across `sources` (highest priority first), else `default`."""
    for src in sources:
        pid = persona_id_from_metadata(src)
        if pid:
            return pid
    return default


def user_id_from_metadata(meta: str | None) -> str | None:
    """Extract a memory user id from a JSON metadata string's `user` key, or None.

    Memory is keyed by a stable user id so it persists across sessions. The client tags the
    room/job with `{"user": "...", "persona": "..."}`; only the JSON form carries a user id
    (a bare string is treated as a persona id, see `persona_id_from_metadata`).
    """
    if not meta or not meta.strip() or not meta.strip().startswith("{"):
        return None
    try:
        obj = json.loads(meta.strip())
    except json.JSONDecodeError:
        return None
    uid = obj.get("user") if isinstance(obj, dict) else None
    return uid.strip() if isinstance(uid, str) and uid.strip() else None


def resolve_user_id(sources: list[str | None]) -> str | None:
    """First memory user id found across `sources` (highest priority first), else None."""
    for src in sources:
        uid = user_id_from_metadata(src)
        if uid:
            return uid
    return None


def resolve_session_options(sources: list[str | None]) -> SessionOptions:
    """Merge `SessionOptions` across metadata sources (highest priority first).

    Each source can set a different subset of fields (voice / cefr / demeanor); they're
    merged so a higher-priority source's set fields win, matching how the client may put some
    options on the job and others on the room. The analog of `persona_id_from_metadata` for
    the session-options rail.
    """
    merged = SessionOptions()
    for src in reversed(sources):  # apply lowest priority first so the highest wins
        merged = SessionOptions.from_metadata(src).merged_over(merged)
    return merged


# How long to wait for the caller to appear so we can read its token metadata. The agent is
# dispatched *because* a participant joined, so it's normally already present; this just closes
# the brief window before the initial room state syncs. Override with PERSONAVOICE_PARTICIPANT_WAIT.
_PARTICIPANT_WAIT_S = float(os.getenv("PERSONAVOICE_PARTICIPANT_WAIT", "10") or "10")


def _participant_metadata(room: Any) -> str | None:
    """The first remote participant's token metadata, or None.

    The token server embeds the per-call selection (persona / voice / cefr / demeanor / user) in
    the caller's LiveKit access-token metadata (see `server/token_server.py`), which surfaces
    here as the participant's `metadata` once it has joined the room. This is the *reliable*
    selection channel: room/job metadata are empty under automatic dispatch, and the client's
    on-connect data message can race the agent joining and be dropped. Tolerant of a room with
    no participants (returns None) so the callers fall through to room/job metadata.
    """
    participants = getattr(room, "remote_participants", None) or {}
    for participant in participants.values():
        meta = getattr(participant, "metadata", None)
        if isinstance(meta, str) and meta.strip():
            return meta
    return None


async def _await_participant(ctx: Any) -> None:
    """Best-effort wait for the caller to join so its token metadata is readable.

    Uses `JobContext.wait_for_participant` when present, bounded by a timeout so a caller that
    never appears can't hang the job. Tolerates SDKs without the method (and any wait error):
    we then just read whatever room state has already synced after `connect`.
    """
    waiter = getattr(ctx, "wait_for_participant", None)
    if waiter is None:
        return
    with contextlib.suppress(Exception):
        await asyncio.wait_for(waiter(), timeout=_PARTICIPANT_WAIT_S)


def _default_persona_id(registry: PersonaRegistry) -> str:
    """The fallback persona: `PERSONAVOICE_PERSONA` if it's known, else the first registered."""
    env = os.getenv("PERSONAVOICE_PERSONA")
    if env and env in registry:
        return env
    return registry.ids()[0] if len(registry) else "companion"


def _lookup_persona(
    registry: PersonaRegistry,
    user_store: UserPersonaStore | None,
    user_id: str | None,
    persona_id: str,
) -> Persona | None:
    """Resolve a persona id to a `Persona`, or None if unknown.

    Curated personas win on id clash (so a user can't shadow a built-in); failing that, the
    caller's own custom persona (from the multi-user store) is consulted when a user id is known.
    """
    if persona_id in registry:
        return registry.get(persona_id)
    if user_store is not None and user_id:
        return user_store.get(user_id, persona_id)
    return None


def _select_persona(
    ctx: Any,
    registry: PersonaRegistry,
    override: str | None,
    *,
    user_store: UserPersonaStore | None = None,
    user_id: str | None = None,
) -> Persona:
    """Pick the persona for a job: explicit override → job/room metadata → default.

    A custom persona (resolved against the caller's `user_id` in the user store) is honored
    alongside the curated registry; an unknown id falls back to the default curated persona.
    Priority: explicit override → the caller's token metadata → job/room metadata → default.
    """
    default = _default_persona_id(registry)
    room = getattr(ctx, "room", None)
    job_meta = getattr(getattr(ctx, "job", None), "metadata", None)
    room_meta = getattr(room, "metadata", None)
    part_meta = _participant_metadata(room)
    persona_id = resolve_persona_id([override, part_meta, job_meta, room_meta], default)
    persona = _lookup_persona(registry, user_store, user_id, persona_id)
    if persona is None:
        logger.warning("requested persona %r is unknown; using %r", persona_id, default)
        persona = registry.get(default)
    return persona


# Process-global cache of the warmed backend + registries. On Windows the LiveKit worker uses a
# THREAD executor, so every job runner lives in *this* process; caching here means the heavy
# models load exactly once even when the idle pool spawns a replacement runner (which would
# otherwise prewarm a second copy and double GPU memory — fatal on a 16 GB card). The lock
# serializes the two runners that can initialize at once (the boot idle runner + its refill).
_WARM_LOCK = threading.Lock()
_WARMED: dict[str, Any] = {}


def prewarm(proc: Any) -> None:
    """LiveKit worker prewarm hook: load every heavy model once, before any job is dispatched.

    The first turn was dominated by lazy model loading (STT/TTS weights, CUDA kernels, the LLM
    server's first prefill) happening *during* the turn. This runs at worker startup instead:
    it builds the backend + registries and runs a tiny dummy inference through each stage so
    everything is warm by the time a caller speaks — moving the cold start off the first turn
    (and off every later session this process serves). The warmed objects are cached in
    `proc.userdata` (for `entrypoint` to reuse) and process-globally (so the idle pool's refill
    runner reuses them instead of loading a second copy). Warm-up failures are logged, never
    raised: a miss only means that one stage pays its load on its first turn, as it did before.
    """
    configure_logging()
    proc.userdata.update(_ensure_warm())


def _ensure_warm() -> dict[str, Any]:
    """Build + warm the backend once per process; later runners reuse the cached objects."""
    with _WARM_LOCK:
        if _WARMED:
            logger.info("prewarm: reusing models already loaded in this process")
            return _WARMED
        _, _, silero = _require_livekit()
        settings = Settings.load()
        backend = build_backend(load_backend_config(settings))
        voices = load_voice_registry(settings)
        registry = PersonaRegistry(settings.personas_dir)
        user_personas = load_user_persona_store(settings)
        vad = _load_vad(silero)
        logger.info("prewarm: loading models (backend=%s)…", backend.name)
        # No event loop is running during prewarm, so drive the async warm-ups with asyncio.run.
        asyncio.run(_warmup_backend(backend, voices, registry))
        _WARMED.update(
            backend=backend,
            voices=voices,
            registry=registry,
            user_personas=user_personas,
            vad=vad,
        )
        return _WARMED


async def _warmup_backend(
    backend: Backend, voices: VoiceRegistry, registry: PersonaRegistry
) -> None:
    """Run one tiny inference through each stage to force model load (best-effort, timed)."""
    try:
        persona = registry.get(_default_persona_id(registry))
        voice = voice_ref_for(persona, backend, voices)
    except Exception as exc:
        logger.warning("prewarm: no persona/voice to warm with; skipping (%s)", exc)
        return
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("stt", lambda: backend.stt.warmup()),
        ("llm", lambda: backend.llm.warmup(persona)),
        ("tts", lambda: backend.tts.warmup(voice)),
    ]
    for name, start in stages:
        t0 = time.perf_counter()
        try:
            await start()
            logger.info("prewarm: %s warm (%.1fs)", name, time.perf_counter() - t0)
        except Exception as exc:
            # Expected when e.g. vLLM never came up — log one concise line, not a stack trace.
            logger.warning(
                "prewarm: %s warm-up failed after %.1fs; its first turn will pay the load (%s)",
                name,
                time.perf_counter() - t0,
                exc,
            )


async def entrypoint(ctx: Any, *, persona_id: str | None = None) -> None:
    """LiveKit Agents job entrypoint: connect, publish a track, converse until disconnect.

    The persona and session options (voice / CEFR / demeanor) are selected per job: an explicit
    `persona_id` wins, then the caller's token metadata (`{"persona": "...", "voice": ...}` the
    token server embedded), then job/room metadata, then `PERSONAVOICE_PERSONA`/the first
    registered persona. A client can also switch persona or options mid-call by sending a data
    message (see `_data_text`); the registry hot-reloads so edited persona files take effect too.
    """
    agents, rtc, silero = _require_livekit()

    settings = Settings.load()
    # Reuse the models loaded by `prewarm` (proc.userdata, else the process-global cache) so the
    # first turn is warm; fall back to building them here when nothing was prewarmed (e.g. tests).
    proc_data = getattr(getattr(ctx, "proc", None), "userdata", None) or _WARMED or {}
    backend = proc_data.get("backend") or build_backend(load_backend_config(settings))
    voices = proc_data.get("voices")
    if voices is None:
        voices = load_voice_registry(settings)
    registry = proc_data.get("registry")
    if registry is None:
        registry = PersonaRegistry(settings.personas_dir)
    user_personas = proc_data.get("user_personas")
    if user_personas is None:
        user_personas = load_user_persona_store(settings)
    vad = proc_data.get("vad")

    # Connect first, then read the caller's *token* metadata. The token server embeds the
    # per-call selection (`{"persona","voice","cefr","demeanor","user"}`) in the client's
    # LiveKit access token (see `server/token_server.py`), which surfaces as the participant's
    # metadata once it's in the room. Under automatic dispatch the room/job metadata are empty
    # and the client's on-connect data message races the agent joining, so this is the reliable
    # channel; room/job metadata stay as a lower-priority fallback (explicit dispatch).
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    await _await_participant(ctx)

    job_meta = getattr(getattr(ctx, "job", None), "metadata", None)
    room_meta = getattr(getattr(ctx, "room", None), "metadata", None)
    part_meta = _participant_metadata(getattr(ctx, "room", None))
    # Highest priority first: the caller's own token, then explicit-dispatch job/room metadata.
    meta_sources = [part_meta, job_meta, room_meta]

    # The user id (`{"user": "..."}`) scopes custom-persona resolution and keys memory.
    meta_user_id = resolve_user_id(meta_sources)
    persona = _select_persona(
        ctx, registry, persona_id, user_store=user_personas, user_id=meta_user_id
    )
    logger.info("agent starting (backend=%s persona=%s)", backend.name, persona.id)

    # Cross-session memory: keyed by that stable user id, or the remote participant's identity.
    # Without one we run stateless; the facade is also dormant until that user grants consent
    # (see `personavoice-memory`).
    remote_ids = [p.identity for p in getattr(ctx.room, "remote_participants", {}).values()]
    user_id = meta_user_id or (remote_ids[0] if remote_ids else None)
    memory = build_conversation_memory(settings, backend, persona) if user_id else None
    if memory is not None:
        logger.info(
            "memory enabled for user %r (persona memory=%s)", user_id, persona.memory.enabled
        )

    # Per-session overrides (voice / CEFR / demeanor) the caller chose, plus the moderation
    # guard (no-op unless PERSONAVOICE_MODERATION is set).
    options = resolve_session_options(meta_sources)
    if options.any_set():
        logger.info("session options: %s", options.model_dump(exclude_none=True))
    moderator = moderator_from_env()
    # Tool / function calling (Feature D): the bundled safe tools. A persona only calls the tools
    # it lists in `persona.tools`, so this is inert for the tool-free personas.
    tools = default_tool_registry()

    source = rtc.AudioSource(_OUT_SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("assistant-voice", source)
    publication = await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    publisher = make_transcript_publisher(
        ctx.room.local_participant, getattr(publication, "sid", None)
    )
    agent = PersonaAgent(
        backend,
        persona,
        source,
        voices,
        options=options,
        moderator=moderator,
        tools=tools,
        publish_transcript=publisher,
        memory=memory,
        user_id=user_id,
    )
    consumers: set[asyncio.Task[None]] = set()  # keep strong refs so tasks aren't GC'd

    @ctx.room.on("track_subscribed")
    def _on_track(track: Any, *_: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            task = asyncio.create_task(_consume_track(agent, track, silero, vad))
            consumers.add(task)
            task.add_done_callback(consumers.discard)

    @ctx.room.on("data_received")
    def _on_data(data: Any, *_: Any) -> None:
        text = _data_text(data)
        pid = persona_id_from_metadata(text)
        if pid:
            registry.reload()  # pick up any edits to a curated persona file before swapping
            if user_personas is not None:
                user_personas.reload()  # and any custom persona created since the call began
            target = _lookup_persona(registry, user_personas, user_id, pid)
            if target is not None:
                agent.set_persona(target)
            else:
                logger.warning("ignoring data request to switch to unknown persona %r", pid)
        # A data message can also change session options (voice / CEFR / demeanor) mid-call.
        options = SessionOptions.from_metadata(text)
        if options.any_set():
            agent.set_options(options)

    # Stay alive until the job is cancelled (participant leaves / worker shuts down).
    try:
        await asyncio.Event().wait()
    finally:
        await agent.aclose()


def run() -> None:
    """CLI shim: hand control to the LiveKit Agents worker runtime.

    Reads `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` from the environment
    (see `.env.example`). Blocks; LiveKit manages the worker lifecycle and dispatches jobs.
    LiveKit's CLI takes a subcommand (`start` for prod, `dev` for hot-reload, `connect` to
    join a room); default to `start` when none is given so `personavoice --serve` works.
    """
    configure_logging()
    agents, _, _ = _require_livekit()
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.argv = [sys.argv[0], "start"]
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Single shared GPU: keep exactly one warm runner so the models load once. The prod
            # default (one per CPU, up to 4) would prewarm several runners and multiply VRAM —
            # an OOM on a 16 GB card (each runner holds its own STT+TTS copy).
            num_idle_processes=1,
            # Loading weights — and downloading them on first run — far exceeds the 10 s default;
            # allow generously (override with PERSONAVOICE_PREWARM_TIMEOUT).
            initialize_process_timeout=float(os.getenv("PERSONAVOICE_PREWARM_TIMEOUT", "600")),
        )
    )


if __name__ == "__main__":  # pragma: no cover - launched as a worker, not in tests
    run()
