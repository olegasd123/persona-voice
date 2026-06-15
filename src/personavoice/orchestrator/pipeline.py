"""The turn-based STT → LLM → TTS pipeline (M1).

This is the "walking skeleton": one audio turn in, one audio turn out, no streaming and no
barge-in (those arrive with LiveKit in M3). It depends only on the adapter base classes, so
it runs with the real Mac backend or with fakes in tests.

    wav bytes ──▶ STT.transcribe ──▶ LLM.chat ──▶ TTS.synthesize ──▶ wav bytes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..adapters.factory import Backend
from ..models import Msg, Persona, Role, Transcript, VoiceRef
from ..persona.prompt import build_messages


@dataclass
class TurnResult:
    """Everything one turn produced, plus per-stage timings (seconds)."""

    transcript: Transcript
    reply: str
    audio: bytes  # WAV-encoded reply
    timings: dict[str, float] = field(default_factory=dict)


def voice_ref_for(persona: Persona, backend: Backend) -> VoiceRef:
    """Build the VoiceRef a persona should speak with on the active backend."""
    return VoiceRef(
        id=persona.voice.ref,
        emotion=persona.voice.emotion,
        backend=backend.tts.name,
    )


class Pipeline:
    """Runs one persona conversation over a fixed backend.

    Keeps an in-memory `history` of the conversation so multi-turn CLI sessions have
    context. Pass `history=...` to `run_turn` to override it for a one-off turn.
    """

    def __init__(self, backend: Backend, persona: Persona) -> None:
        self.backend = backend
        self.persona = persona
        self.history: list[Msg] = []

    async def run_turn(self, audio_in: bytes, history: list[Msg] | None = None) -> TurnResult:
        """STT → LLM → TTS for one audio turn. Updates `self.history` when using it."""
        use_internal = history is None
        history = self.history if use_internal else history

        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        transcript = await self.backend.stt.transcribe(audio_in)
        t_stt = time.perf_counter()
        timings["stt"] = t_stt - t0

        messages = build_messages(self.persona, history=history, user_input=transcript.text)
        reply = await self.backend.llm.chat(messages, self.persona)
        t_llm = time.perf_counter()
        timings["llm"] = t_llm - t_stt

        voice = voice_ref_for(self.persona, self.backend)
        audio_out = await self.backend.tts.synthesize(reply, voice)
        t_tts = time.perf_counter()
        timings["tts"] = t_tts - t_llm
        timings["total"] = t_tts - t0

        if use_internal:
            self.history.append(Msg(role=Role.user, content=transcript.text))
            self.history.append(Msg(role=Role.assistant, content=reply))

        return TurnResult(transcript=transcript, reply=reply, audio=audio_out, timings=timings)
