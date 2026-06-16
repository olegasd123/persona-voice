"""LiveKit Agents worker: live, streaming, barge-in conversation (M3).

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
README "Live streaming server (M3)"), the analog of M2's on-4080 step.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from ..adapters.factory import Backend, build_backend
from ..audio import pcm16_to_wav, wav_to_pcm16
from ..models import Persona
from ..persona.registry import PersonaRegistry
from ..server.config import Settings, load_backend_config
from .streaming import StreamingPipeline
from .turn import TurnController

# LiveKit's runtime types (`rtc.AudioSource`, `rtc.Track`, ...) are only present when the
# `livekit` extra is installed, so they're annotated as `Any` to keep this module importable
# (and type-checkable) without it.
logger = logging.getLogger("personavoice.agent")

# Output track format. 24 kHz mono matches Kokoro/Orpheus output and is a common WebRTC
# rate; per-sentence TTS wavs are resampled to this before being captured as frames.
_OUT_SAMPLE_RATE = 24000
_FRAME_MS = 20  # frame size pushed to the AudioSource (WebRTC-typical)


def _require_livekit() -> tuple[Any, Any, Any]:
    """Import the LiveKit runtime lazily, with a clear install hint if it's missing."""
    try:
        from livekit import agents, rtc
        from livekit.plugins import silero
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "livekit is not installed; install the streaming extra: "
            "`pip install -e '.[livekit]'`"
        ) from exc
    return agents, rtc, silero


class PersonaAgent:
    """Drives one participant's conversation: VAD → STT → streaming reply, with barge-in.

    Owns the per-session `StreamingPipeline` (so history accumulates) and a
    `TurnController` bound to an `rtc.AudioSource` sink that captures TTS audio onto the
    published WebRTC track.
    """

    def __init__(self, backend: Backend, persona: Persona, source: Any) -> None:
        self._backend = backend
        self._persona = persona
        self._source = source
        self._pipeline = StreamingPipeline(backend, persona)
        self._turn = TurnController(self._capture_wav)
        self._frame_samples = max(1, _OUT_SAMPLE_RATE * _FRAME_MS // 1000)

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
        """A complete user utterance (VAD-segmented PCM) → transcribe → streamed reply."""
        wav = pcm16_to_wav(utterance_pcm, sample_rate)
        transcript = await self._backend.stt.transcribe(wav)
        text = transcript.text.strip()
        if not text:
            return
        logger.info("user: %s", text)
        self._turn.begin(self._pipeline.stream_response(text))

    async def aclose(self) -> None:
        self._turn.interrupt()
        await self._turn.join()


async def _consume_track(agent: PersonaAgent, track: Any, silero: Any) -> None:
    """Run VAD over a participant's audio track and feed utterances to the agent."""
    _, rtc, _ = _require_livekit()
    vad = silero.VAD.load()
    vad_stream = vad.stream()
    audio_stream = rtc.AudioStream(track)

    async def _pump_frames() -> None:
        async for ev in audio_stream:
            vad_stream.push_frame(ev.frame)

    pump = asyncio.create_task(_pump_frames())
    try:
        async for ev in vad_stream:
            if ev.type == rtc.VADEventType.START_OF_SPEECH:
                agent.on_user_speech_started()
            elif ev.type == rtc.VADEventType.END_OF_SPEECH:
                # `ev.frames` holds the buffered speech; concatenate to one PCM utterance.
                pcm = b"".join(bytes(f.data) for f in ev.frames)
                sr = ev.frames[0].sample_rate if ev.frames else 16000
                await agent.on_user_utterance(pcm, sr)
    finally:
        pump.cancel()
        await vad_stream.aclose()


def _resolve_persona(settings: Settings, persona_id: str) -> Persona:
    return PersonaRegistry(settings.personas_dir).get(persona_id)


async def entrypoint(ctx: Any, *, persona_id: str = "companion") -> None:
    """LiveKit Agents job entrypoint: connect, publish a track, converse until disconnect."""
    agents, rtc, silero = _require_livekit()

    settings = Settings.load()
    backend = build_backend(load_backend_config(settings))
    persona = _resolve_persona(settings, persona_id)
    logger.info("agent starting (backend=%s persona=%s)", backend.name, persona.id)

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(_OUT_SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("assistant-voice", source)
    await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    agent = PersonaAgent(backend, persona, source)
    consumers: set[asyncio.Task[None]] = set()  # keep strong refs so tasks aren't GC'd

    @ctx.room.on("track_subscribed")
    def _on_track(track: Any, *_: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            task = asyncio.create_task(_consume_track(agent, track, silero))
            consumers.add(task)
            task.add_done_callback(consumers.discard)

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
    agents, _, _ = _require_livekit()
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.argv = [sys.argv[0], "start"]
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":  # pragma: no cover - launched as a worker, not in tests
    run()
