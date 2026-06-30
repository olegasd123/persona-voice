"""The streaming pipeline (LLM tokens -> sentences -> per-sentence TTS)."""

from __future__ import annotations

from pathlib import Path

from personavoice.models import Role, SessionOptions, VoiceDef
from personavoice.orchestrator import StreamingPipeline, StreamMetrics
from personavoice.persona import load_persona
from personavoice.safety import KeywordModerator
from personavoice.voice import VoiceRegistry

from .fakes import FakeLLM, FakeSTT, FakeTTS, make_backend


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


async def test_streams_one_audio_chunk_per_sentence(config_dir: Path) -> None:
    backend = make_backend(llm_reply="Hello there. How are you? Good.")
    pipe = StreamingPipeline(backend, _companion(config_dir))

    chunks = [c async for c in pipe.stream_response("hi")]

    # TTS was called once per sentence, not once for the whole reply.
    assert backend.tts.chunks == ["Hello there.", "How are you?", "Good."]  # type: ignore[attr-defined]
    assert chunks == [b"RIFF" + s.encode() for s in backend.tts.chunks]  # type: ignore[attr-defined]


async def test_metrics_capture_first_token_and_audio(config_dir: Path) -> None:
    backend = make_backend(llm_reply="One. Two.")
    pipe = StreamingPipeline(backend, _companion(config_dir))
    metrics = StreamMetrics()

    async for _ in pipe.stream_response("hi", metrics=metrics):
        pass

    assert metrics.first_token is not None
    assert metrics.first_audio is not None
    assert metrics.total is not None
    assert metrics.first_token <= metrics.first_audio <= metrics.total
    assert metrics.reply == "One. Two."


async def test_history_accumulates_full_reply(config_dir: Path) -> None:
    backend = make_backend(stt_text="ignored", llm_reply="A reply.")
    pipe = StreamingPipeline(backend, _companion(config_dir))

    async for _ in pipe.stream_response("hello"):
        pass

    assert [(m.role, m.content) for m in pipe.history] == [
        (Role.user, "hello"),
        (Role.assistant, "A reply."),
    ]
    # A second turn sees the first turn in its context.
    async for _ in pipe.stream_response("again"):
        pass
    assert backend.llm.last_messages[-1].content == "again"  # type: ignore[attr-defined]
    assert len(pipe.history) == 4


async def test_explicit_history_does_not_mutate_internal(config_dir: Path) -> None:
    pipe = StreamingPipeline(make_backend(), _companion(config_dir))
    async for _ in pipe.stream_response("hi", history=[]):
        pass
    assert pipe.history == []


async def test_on_sentence_taps_each_sentence_in_order(config_dir: Path) -> None:
    backend = make_backend(llm_reply="One. Two. Three.")
    pipe = StreamingPipeline(backend, _companion(config_dir))

    seen: list[str] = []

    async def on_sentence(sentence: str) -> None:
        seen.append(sentence)

    audio = [c async for c in pipe.stream_response("hi", on_sentence=on_sentence)]

    # The tap sees each sentence, in order, and audio still flows for every one.
    assert seen == ["One.", "Two.", "Three."]
    assert len(audio) == 3


async def test_on_sentence_callback_error_does_not_break_audio(config_dir: Path) -> None:
    backend = make_backend(llm_reply="Hello there. How are you?")
    pipe = StreamingPipeline(backend, _companion(config_dir))

    async def boom(_sentence: str) -> None:
        raise RuntimeError("transcript publish failed")

    # A throwing callback must not stop synthesis — every sentence is still voiced.
    audio = [c async for c in pipe.stream_response("hi", on_sentence=boom)]
    assert audio == [b"RIFF" + s.encode() for s in backend.tts.chunks]  # type: ignore[attr-defined]


async def test_builds_persona_system_prompt(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend(llm_reply="hi")
    pipe = StreamingPipeline(backend, persona)

    async for _ in pipe.stream_response("hello"):
        pass

    messages = backend.llm.last_messages  # type: ignore[attr-defined]
    assert messages[0].role == Role.system
    assert persona.system_prompt.strip()[:20] in messages[0].content


# --- session options (voice override) + moderation ------------------------------------


async def test_options_thread_into_system_prompt(config_dir: Path) -> None:
    from personavoice.models import Demeanor

    backend = make_backend(llm_reply="hi")
    pipe = StreamingPipeline(
        backend, _companion(config_dir), options=SessionOptions(demeanor=Demeanor.rude)
    )
    async for _ in pipe.stream_response("hello"):
        pass
    assert "brusque" in backend.llm.last_messages[0].content  # type: ignore[attr-defined]


async def test_voice_choice_override(config_dir: Path) -> None:
    backend = make_backend(llm_reply="Hi.")
    voices = VoiceRegistry({"alt": VoiceDef(presets={"fake_tts": "alt_preset"})})
    pipe = StreamingPipeline(
        backend, _companion(config_dir), voices, options=SessionOptions(voice="alt")
    )
    async for _ in pipe.stream_response("hi"):
        pass
    assert backend.tts.last_voice.id == "alt_preset"  # type: ignore[attr-defined]


# --- dynamic emotion ------------------------------------------------------


async def test_dynamic_emotion_strips_tag_and_applies_to_voice(config_dir: Path) -> None:
    backend = make_backend(llm_reply="[excited] Hello there.")
    pipe = StreamingPipeline(backend, _companion(config_dir), dynamic_emotion=True)
    metrics = StreamMetrics()

    async for _ in pipe.stream_response("hi", metrics=metrics):
        pass

    # The tag is parsed off: TTS, the assembled reply, and history never see it...
    assert backend.tts.chunks == ["Hello there."]  # type: ignore[attr-defined]
    assert metrics.reply == "Hello there."
    assert pipe.history[-1].content == "Hello there."
    # ...and it drives the voice for this turn.
    assert backend.tts.last_voice.emotion == "excited"  # type: ignore[attr-defined]


async def test_dynamic_emotion_off_leaves_tag_and_voice_untouched(config_dir: Path) -> None:
    # With dynamic emotion off (default), a stray tag is just spoken text and the voice is unchanged.
    backend = make_backend(llm_reply="[excited] Hello there.")
    pipe = StreamingPipeline(backend, _companion(config_dir))

    async for _ in pipe.stream_response("hi"):
        pass

    assert "".join(backend.tts.chunks) == "[excited] Hello there."  # type: ignore[attr-defined]
    assert backend.tts.last_voice.emotion != "excited"  # type: ignore[attr-defined]


async def test_dynamic_emotion_directive_only_when_enabled(config_dir: Path) -> None:
    needle = "emotion tag in square brackets"

    on = make_backend(llm_reply="hi")
    async for _ in StreamingPipeline(
        on, _companion(config_dir), dynamic_emotion=True
    ).stream_response("hello"):
        pass
    assert needle in on.llm.last_messages[0].content  # type: ignore[attr-defined]

    off = make_backend(llm_reply="hi")
    async for _ in StreamingPipeline(off, _companion(config_dir)).stream_response("hello"):
        pass
    assert needle not in off.llm.last_messages[0].content  # type: ignore[attr-defined]


async def test_moderation_input_short_circuits(config_dir: Path) -> None:
    backend = make_backend(llm_reply="THIS SHOULD NOT BE SPOKEN")
    pipe = StreamingPipeline(backend, _companion(config_dir), moderator=KeywordModerator())

    chunks = [c async for c in pipe.stream_response("I want to die")]

    spoken = b"".join(chunks).decode()  # the fake wav carries the spoken text
    assert "988" in spoken  # the calm crisis reply was voiced
    assert backend.llm.last_messages is None  # type: ignore[attr-defined]  # LLM bypassed
    # The canned reply is committed to history like a normal turn.
    assert [m.role for m in pipe.history] == [Role.user, Role.assistant]
    assert "988" in pipe.history[-1].content


async def test_moderation_bounds_abusive_output(config_dir: Path) -> None:
    from personavoice.adapters.factory import Backend
    from personavoice.models import Demeanor

    # A clean opening sentence, then an abusive one, then more the user must never hear.
    backend = Backend(
        name="fake",
        stt=FakeSTT("hi"),
        llm=FakeLLM(
            "ignored",
            tokens=["Fine, here it is. ", "You're an idiot. ", "Anything else? "],
        ),
        tts=FakeTTS(),
    )
    pipe = StreamingPipeline(
        backend,
        _companion(config_dir),
        options=SessionOptions(demeanor=Demeanor.rude),
        moderator=KeywordModerator(),
    )

    chunks = [c async for c in pipe.stream_response("go on then")]
    spoken = b"".join(chunks).decode()

    # The clean sentence is voiced; the abuse is replaced in place and the rest never reached.
    assert backend.tts.chunks[0] == "Fine, here it is."  # type: ignore[attr-defined]
    assert "idiot" not in spoken
    assert "Anything else" not in spoken
    assert "respectful" in spoken  # the softened line was voiced in place of the abuse
    # Generation was torn down at the breach rather than running to the end.
    assert backend.llm.completed is False  # type: ignore[attr-defined]
    # History commits the safe text the user actually heard, not the raw tokens.
    assert "idiot" not in pipe.history[-1].content
    assert pipe.history[-1].content.startswith("Fine, here it is.")


async def test_moderation_clean_output_passes_through(config_dir: Path) -> None:
    # With a guard configured but a clean reply, output is identical to no moderation.
    reply = "Hello there. How are you?"
    guarded = make_backend(llm_reply=reply)
    pipe = StreamingPipeline(guarded, _companion(config_dir), moderator=KeywordModerator())
    chunks = [c async for c in pipe.stream_response("hi")]

    plain = make_backend(llm_reply=reply)
    plain_pipe = StreamingPipeline(plain, _companion(config_dir))
    plain_chunks = [c async for c in plain_pipe.stream_response("hi")]

    assert guarded.tts.chunks == plain.tts.chunks  # type: ignore[attr-defined]
    assert chunks == plain_chunks
    assert pipe.history[-1].content == plain_pipe.history[-1].content == reply
