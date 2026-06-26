"""The turn-based STT → LLM → TTS pipeline.

This is the "walking skeleton": one audio turn in, one audio turn out, no streaming and no
barge-in (those arrive with the LiveKit agent). It depends only on the adapter base classes, so
it runs with the real Mac backend or with fakes in tests.

    wav bytes ──▶ STT.transcribe ──▶ LLM.chat ──▶ TTS.synthesize ──▶ wav bytes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..adapters.factory import Backend
from ..emotion import dynamic_emotion_enabled, split_emotion_hint
from ..models import Msg, Persona, Role, SessionOptions, Transcript, VoiceRef
from ..persona.prompt import build_messages
from ..safety import Moderator
from ..voice.registry import VoiceRegistry
from .tools import ToolRegistry

logger = logging.getLogger("personavoice.pipeline")


@dataclass
class TurnResult:
    """Everything one turn produced, plus per-stage timings (seconds)."""

    transcript: Transcript
    reply: str
    audio: bytes  # WAV-encoded reply
    timings: dict[str, float] = field(default_factory=dict)


def voice_ref_for(
    persona: Persona,
    backend: Backend,
    voices: VoiceRegistry | None = None,
    *,
    voice_choice: str | None = None,
) -> VoiceRef:
    """Build the VoiceRef a persona should speak with on the active backend.

    An explicit `voice_choice` (a session-level voice override) wins when it resolves to a
    speakable voice on this backend — independent of any per-persona assignment. Failing that
    (or when no choice is given), a clone/fine-tune assigned to this persona wins on a cloning
    backend; otherwise the persona's logical voice ref resolves to a backend-native preset so
    personas sound distinct. Without a registry, the raw ref is passed through (adapters then
    fall back to their default voice).
    """
    if voices is not None:
        supports_cloning = getattr(backend.tts, "supports_cloning", False)
        if voice_choice:
            ref = voices.resolve_choice(
                voice_choice,
                backend.tts.name,
                supports_cloning=supports_cloning,
                default_emotion=persona.voice.emotion,
            )
            if ref is not None:
                return ref
            logger.warning(
                "voice %r not available on %s; using persona default",
                voice_choice,
                backend.tts.name,
            )
        return voices.resolve_for_persona(
            persona, backend.tts.name, supports_cloning=supports_cloning
        )
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

    def __init__(
        self,
        backend: Backend,
        persona: Persona,
        voices: VoiceRegistry | None = None,
        *,
        options: SessionOptions | None = None,
        moderator: Moderator | None = None,
        tools: ToolRegistry | None = None,
        dynamic_emotion: bool | None = None,
    ) -> None:
        self.backend = backend
        self.persona = persona
        self.voices = voices
        # Per-session overrides (voice / cefr / demeanor). None = persona defaults.
        self.options = options
        # Optional input/output guard. None = no moderation (behavior unchanged).
        self.moderator = moderator
        # Tool / function calling. None = no tools; a persona only calls tools it
        # lists in `persona.tools`. See `StreamingPipeline` for the streaming counterpart.
        self.tools = tools
        # Per-utterance emotion: off by default (the env toggle), so behavior is
        # unchanged unless an operator opts in. See `StreamingPipeline.dynamic_emotion`.
        self.dynamic_emotion = (
            dynamic_emotion if dynamic_emotion is not None else dynamic_emotion_enabled()
        )
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

        # Input moderation: a flagged utterance (crisis especially) short-circuits the LLM
        # with a safe reply instead of running the normal turn.
        emotion: str | None = None
        reply = await self._safe_input_reply(transcript.text)
        if reply is None:
            messages = build_messages(
                self.persona,
                history=history,
                user_input=transcript.text,
                options=self.options,
                dynamic_emotion=self.dynamic_emotion,
            )
            # When the persona declares tools, run the model → tool → result loop and use the
            # follow-up reply; otherwise a plain single-shot completion.
            specs = (
                self.tools.select(self.persona.tools)
                if self.tools is not None and self.persona.tools
                else []
            )
            if specs:
                reply = await self.backend.llm.chat_with_tools(messages, self.persona, specs)
            else:
                reply = await self.backend.llm.chat(messages, self.persona)
            # Per-utterance emotion: strip the leading `[emotion]` tag off the reply
            # before it reaches the output guard, TTS, history, or the transcript.
            if self.dynamic_emotion:
                emotion, reply = split_emotion_hint(reply)
            reply = await self._bound_output(reply)
        t_llm = time.perf_counter()
        timings["llm"] = t_llm - t_stt

        voice = voice_ref_for(
            self.persona, self.backend, self.voices, voice_choice=self._voice_choice
        )
        if emotion is not None:
            voice = voice.model_copy(update={"emotion": emotion})
        audio_out = await self.backend.tts.synthesize(reply, voice)
        t_tts = time.perf_counter()
        timings["tts"] = t_tts - t_llm
        timings["total"] = t_tts - t0

        if use_internal:
            self.history.append(Msg(role=Role.user, content=transcript.text))
            self.history.append(Msg(role=Role.assistant, content=reply))

        return TurnResult(transcript=transcript, reply=reply, audio=audio_out, timings=timings)

    @property
    def _voice_choice(self) -> str | None:
        return self.options.voice if self.options else None

    async def _safe_input_reply(self, user_text: str) -> str | None:
        """A safe canned reply when the input is flagged, else None (run the normal turn)."""
        if self.moderator is None:
            return None
        verdict = await self.moderator.check_input(user_text)
        if verdict.flagged and verdict.replacement is not None:
            return verdict.replacement
        return None

    async def _bound_output(self, reply: str) -> str:
        """Replace a reply that breaches the output bound (the `rude` cap, abuse, threats)."""
        if self.moderator is None:
            return reply
        demeanor = self.options.demeanor if self.options else None
        verdict = await self.moderator.check_output(reply, demeanor=demeanor)
        if verdict.flagged and verdict.replacement is not None:
            return verdict.replacement
        return reply
