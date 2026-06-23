"""The turn-based pipeline (STT -> LLM -> TTS), exercised with fake adapters."""

from __future__ import annotations

from pathlib import Path

from personavoice.models import Demeanor, Role, SessionOptions, VoiceDef
from personavoice.orchestrator import Pipeline
from personavoice.persona import load_persona
from personavoice.safety import KeywordModerator
from personavoice.voice import VoiceRegistry

from .fakes import make_backend


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


async def test_run_turn_flows_through_all_stages(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend(stt_text="what is your name", llm_reply="I'm your companion.")
    pipe = Pipeline(backend, persona)

    result = await pipe.run_turn(b"audio-bytes")

    # STT received the audio; transcript and reply propagated through to TTS.
    assert backend.stt.received == b"audio-bytes"  # type: ignore[attr-defined]
    assert result.transcript.text == "what is your name"
    assert result.reply == "I'm your companion."
    assert backend.tts.last_text == "I'm your companion."  # type: ignore[attr-defined]
    assert result.audio == b"RIFF" + b"I'm your companion."


async def test_run_turn_builds_persona_messages(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend(stt_text="hello", llm_reply="hi")
    pipe = Pipeline(backend, persona)

    await pipe.run_turn(b"x")

    messages = backend.llm.last_messages  # type: ignore[attr-defined]
    assert messages[0].role == Role.system
    assert persona.system_prompt.strip()[:20] in messages[0].content
    assert messages[-1].role == Role.user
    assert messages[-1].content == "hello"


async def test_voice_ref_carries_persona_voice(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend()
    pipe = Pipeline(backend, persona)

    await pipe.run_turn(b"x")

    voice = backend.tts.last_voice  # type: ignore[attr-defined]
    assert voice.id == persona.voice.ref
    assert voice.emotion == persona.voice.emotion
    assert voice.backend == "fake_tts"


async def test_timings_present(config_dir: Path) -> None:
    pipe = Pipeline(make_backend(), _companion(config_dir))
    result = await pipe.run_turn(b"x")
    assert set(result.timings) == {"stt", "llm", "tts", "total"}
    assert all(v >= 0 for v in result.timings.values())


async def test_history_accumulates_across_turns(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend(stt_text="hello", llm_reply="hi")
    pipe = Pipeline(backend, persona)

    await pipe.run_turn(b"x")
    assert [m.role for m in pipe.history] == [Role.user, Role.assistant]

    await pipe.run_turn(b"y")
    # Second turn sees the first turn's user+assistant messages in its context.
    second_messages = backend.llm.last_messages  # type: ignore[attr-defined]
    assert len(second_messages) == 4  # system + prior user/assistant + new user
    assert len(pipe.history) == 4


async def test_explicit_history_does_not_mutate_internal(config_dir: Path) -> None:
    pipe = Pipeline(make_backend(), _companion(config_dir))
    await pipe.run_turn(b"x", history=[])
    assert pipe.history == []  # explicit history bypasses the internal accumulator


# --- session options (voice override) + moderation ------------------------------------


async def test_voice_choice_overrides_persona_voice(config_dir: Path) -> None:
    backend = make_backend()
    voices = VoiceRegistry({"alt": VoiceDef(presets={"fake_tts": "alt_preset"})})
    pipe = Pipeline(backend, _companion(config_dir), voices, options=SessionOptions(voice="alt"))
    await pipe.run_turn(b"x")
    assert backend.tts.last_voice.id == "alt_preset"  # type: ignore[attr-defined]


async def test_unavailable_voice_choice_falls_back_to_persona_default(config_dir: Path) -> None:
    backend = make_backend()
    voices = VoiceRegistry({"companion_soft": VoiceDef(presets={"fake_tts": "af_heart"})})
    pipe = Pipeline(backend, _companion(config_dir), voices, options=SessionOptions(voice="ghost"))
    await pipe.run_turn(b"x")
    # The unknown choice can't resolve, so the persona's own voice is used.
    assert backend.tts.last_voice.id == "af_heart"  # type: ignore[attr-defined]


async def test_moderation_input_short_circuits_the_llm(config_dir: Path) -> None:
    backend = make_backend(stt_text="I want to die", llm_reply="THIS SHOULD NOT BE SPOKEN")
    pipe = Pipeline(backend, _companion(config_dir), moderator=KeywordModerator())
    result = await pipe.run_turn(b"x")
    assert "988" in result.reply  # the calm crisis reply
    assert backend.llm.last_messages is None  # type: ignore[attr-defined]  # LLM bypassed


async def test_moderation_bounds_abusive_output(config_dir: Path) -> None:
    backend = make_backend(stt_text="hi", llm_reply="Honestly, you're an idiot.")
    pipe = Pipeline(
        backend,
        _companion(config_dir),
        options=SessionOptions(demeanor=Demeanor.rude),
        moderator=KeywordModerator(),
    )
    result = await pipe.run_turn(b"x")
    assert "idiot" not in result.reply  # replaced with the softened line
