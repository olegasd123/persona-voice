"""The M3 streaming pipeline (LLM tokens -> sentences -> per-sentence TTS)."""

from __future__ import annotations

from pathlib import Path

from personavoice.models import Role
from personavoice.orchestrator import StreamingPipeline, StreamMetrics
from personavoice.persona import load_persona

from .fakes import make_backend


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


async def test_builds_persona_system_prompt(config_dir: Path) -> None:
    persona = _companion(config_dir)
    backend = make_backend(llm_reply="hi")
    pipe = StreamingPipeline(backend, persona)

    async for _ in pipe.stream_response("hello"):
        pass

    messages = backend.llm.last_messages  # type: ignore[attr-defined]
    assert messages[0].role == Role.system
    assert persona.system_prompt.strip()[:20] in messages[0].content
