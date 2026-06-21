"""The LiveKit streaming agent's backend-side logic (no livekit/WebRTC needed).

The WebRTC/VAD glue needs a running LiveKit server (validated live, like the on-GPU validation
step); these cover the parts that are pure: barge-in dispatch and the
VAD-utterance → STT → streaming-reply routing, with a fake AudioSource and fake backend.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice.adapters.factory import Backend
from personavoice.models import Msg, Persona, Transcript
from personavoice.orchestrator import agent
from personavoice.orchestrator.turn import TurnController
from personavoice.persona import load_persona

from .fakes import FakeLLM, FakeSTT, FakeTTS, make_backend


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


def _hr(config_dir: Path):
    return load_persona(config_dir / "personas" / "hr_interviewer.yaml")


class FakeSource:
    """Stand-in for `rtc.AudioSource` — records barge-in queue flushes and frames."""

    def __init__(self) -> None:
        self.cleared = False
        self.frames: list[object] = []

    def clear_queue(self) -> None:
        self.cleared = True

    async def capture_frame(self, frame: object) -> None:
        self.frames.append(frame)


def test_require_livekit_raises_clean_error_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy `from livekit import ...` to fail regardless of whether the extra is
    # installed in this env (setting a module to None in sys.modules makes import raise).
    import sys

    monkeypatch.setitem(sys.modules, "livekit", None)
    with pytest.raises(RuntimeError, match="livekit"):
        agent._require_livekit()


def test_vad_event_type_prefers_agents_vad() -> None:
    event_type = SimpleNamespace(START_OF_SPEECH=object(), END_OF_SPEECH=object())
    agents = SimpleNamespace(vad=SimpleNamespace(VADEventType=event_type))
    rtc = SimpleNamespace(VADEventType=object())

    assert agent._vad_event_type(agents, rtc) is event_type


def test_vad_event_type_falls_back_to_rtc() -> None:
    event_type = SimpleNamespace(START_OF_SPEECH=object(), END_OF_SPEECH=object())
    agents = SimpleNamespace(vad=SimpleNamespace())
    rtc = SimpleNamespace(VADEventType=event_type)

    assert agent._vad_event_type(agents, rtc) is event_type


async def test_barge_in_interrupts_and_flushes_source(config_dir: Path) -> None:
    source = FakeSource()
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), source)

    async def slow():  # never reaches the (livekit) sink before we interrupt
        await asyncio.sleep(10)
        yield b"x"

    ag._turn.begin(slow())
    await asyncio.sleep(0)

    ag.on_user_speech_started()
    assert source.cleared is True  # queued playback dropped immediately
    await ag._turn.join()


async def test_barge_in_when_idle_is_a_noop(config_dir: Path) -> None:
    source = FakeSource()
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), source)
    ag.on_user_speech_started()
    assert source.cleared is False


async def test_utterance_transcribes_then_streams_reply(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="hello", llm_reply="Hi there. Bye.")
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)  # bypass the livekit AudioSource sink

    pcm = b"\x00\x00" * 1600  # 0.1 s of silence as 16-bit PCM
    await ag.on_user_utterance(pcm, 16000)
    await ag._turn.join()

    # STT text drove the reply, streamed sentence-by-sentence.
    assert spoken == [b"RIFF" + b"Hi there.", b"RIFF" + b"Bye."]
    assert ag._pipeline.history[0].content == "hello"


# --- graceful error recovery + per-turn metrics ---------------------------------


class RaisingSTT(FakeSTT):
    async def transcribe(self, audio: bytes) -> Transcript:
        raise RuntimeError("stt backend exploded")


class RaisingLLM(FakeLLM):
    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        yield "Hello there. "  # one good sentence, then a failure mid-stream
        raise RuntimeError("llm stream exploded")


async def test_stt_failure_is_swallowed_and_session_survives(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = Backend(name="fake", stt=RaisingSTT(), llm=FakeLLM(), tts=FakeTTS())
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)

    # Must not raise — a bad utterance is logged and skipped.
    await ag.on_user_utterance(b"\x00\x00" * 1600, 16000)
    await ag._turn.join()
    assert spoken == []
    assert ag._pipeline.history == []


async def test_streaming_failure_is_recovered_and_logs_metrics(
    config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = Backend(name="fake", stt=FakeSTT("hi"), llm=RaisingLLM(), tts=FakeTTS())
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)
    with caplog.at_level(logging.WARNING, logger="personavoice.agent"):
        await ag.on_user_utterance(b"\x00\x00" * 1600, 16000)
        await ag._turn.join()

    # The good first sentence was spoken before the failure; the worker did not crash.
    assert spoken == [b"RIFF" + b"Hello there."]
    # A turn-metrics record was logged at WARNING (it carries the error).
    metric_logs = [r for r in caplog.records if getattr(r, "turn", None) is not None]
    assert metric_logs and metric_logs[-1].turn["error"] is not None


async def test_empty_utterance_is_ignored(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="   ", llm_reply="should not run")
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)

    await ag.on_user_utterance(b"\x00\x00" * 100, 16000)
    await ag._turn.join()
    assert spoken == []  # blank transcript → no reply, nothing spoken
    assert ag._pipeline.history == []


# --- assistant transcript publishing --------------------------------------------------


async def test_speak_publishes_growing_then_final_transcript(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="hi", llm_reply="Hello there. How are you?")
    published: list[tuple[str, str, bool]] = []

    async def publish(seg_id: str, text: str, is_final: bool) -> None:
        published.append((seg_id, text, is_final))

    ag = agent.PersonaAgent(
        backend, _companion(config_dir), FakeSource(), publish_transcript=publish
    )

    async def sink(_wav: bytes) -> None:
        pass

    ag._turn = TurnController(sink)
    await ag.on_user_utterance(b"\x00\x00" * 1600, 16000)
    await ag._turn.join()
    # The fire-and-forget final publish is scheduled in `finally`; let it run.
    await asyncio.gather(*list(ag._pending))

    texts = [(text, is_final) for _id, text, is_final in published]
    # One segment grows sentence-by-sentence (interim, not final), then a final marker.
    assert texts == [
        ("Hello there.", False),
        ("Hello there. How are you?", False),
        ("Hello there. How are you?", True),
    ]
    # All updates share one segment id so the client updates a single bubble in place.
    assert len({seg_id for seg_id, _t, _f in published}) == 1


async def test_no_publisher_means_no_transcript_but_audio_flows(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="hi", llm_reply="A. B.")
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())  # no publisher

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)
    await ag.on_user_utterance(b"\x00\x00" * 1600, 16000)
    await ag._turn.join()

    assert spoken == [b"RIFF" + b"A.", b"RIFF" + b"B."]
    assert ag._pending == set()  # nothing scheduled when there's no publisher


def test_make_transcript_publisher_is_noop_without_track_sid() -> None:
    # Returned unconditionally (so the agent always has a callable); a missing sid → no-op.
    pub = agent.make_transcript_publisher(SimpleNamespace(identity="agent"), None)
    assert asyncio.run(pub("seg", "hi", True)) is None


def test_make_transcript_publisher_builds_and_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[object] = []

    class FakeLP:
        identity = "assistant-1"

        async def publish_transcription(self, transcription: object) -> None:
            sent.append(transcription)

    # Fake the lazy livekit import so we can assert the rtc.Transcription is shaped right.
    fake_rtc = SimpleNamespace(
        TranscriptionSegment=lambda **kw: SimpleNamespace(**kw),
        Transcription=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(agent, "_require_livekit", lambda: (None, fake_rtc, None))

    pub = agent.make_transcript_publisher(FakeLP(), "TR_track123")
    asyncio.run(pub("seg-1", "Hello there.", True))

    assert len(sent) == 1
    transcription = sent[0]
    assert transcription.participant_identity == "assistant-1"
    assert transcription.track_sid == "TR_track123"
    seg = transcription.segments[0]
    assert (seg.id, seg.text, seg.final, seg.language) == ("seg-1", "Hello there.", True, "en")


# --- persona selection (switch via API) -----------------------------------------------


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ('{"persona": "hr_interviewer"}', "hr_interviewer"),  # JSON object
        ("hr_interviewer", "hr_interviewer"),  # bare id
        ('  {"persona":"pm_interviewer"}  ', "pm_interviewer"),  # whitespace tolerated
        ("", None),  # blank
        ("   ", None),
        (None, None),
        ("{not json}", None),  # unparsable JSON → None, not a crash
        ('{"other": "x"}', None),  # JSON without a persona key
        ('["companion"]', None),  # JSON non-object
        ('{"persona": ""}', None),  # empty persona value
    ],
)
def test_persona_id_from_metadata(meta: str | None, expected: str | None) -> None:
    assert agent.persona_id_from_metadata(meta) == expected


def test_resolve_persona_id_priority_and_default() -> None:
    # First non-empty source wins; later ones are ignored.
    assert (
        agent.resolve_persona_id([None, "", "hr_interviewer", "companion"], "x") == "hr_interviewer"
    )
    # All empty → default.
    assert agent.resolve_persona_id([None, ""], "companion") == "companion"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b'{"persona": "hr_interviewer"}', "hr_interviewer"),  # raw bytes (older SDK)
        ("companion", "companion"),  # str
        (123, None),  # unexpected type → None
    ],
)
def test_data_text(data: object, expected: str | None) -> None:
    assert agent.persona_id_from_metadata(agent._data_text(data)) == expected


def test_data_text_unwraps_datapacket() -> None:
    class FakePacket:  # newer LiveKit passes a DataPacket with a `.data` attribute
        data = b'{"persona": "pm_interviewer"}'

    assert agent._data_text(FakePacket()) == '{"persona": "pm_interviewer"}'


# --- mid-session hot-swap (switch without restart) ------------------------------------


def test_set_persona_swaps_pipeline_and_keeps_history(config_dir: Path) -> None:
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), FakeSource())
    ag._pipeline.history.append(object())  # type: ignore[arg-type]  # a prior turn

    ag.set_persona(_hr(config_dir))

    assert ag.persona.id == "hr_interviewer"
    assert ag._pipeline.persona.id == "hr_interviewer"
    assert len(ag._pipeline.history) == 1  # conversation continues, not reset


async def test_set_persona_interrupts_in_flight_reply(config_dir: Path) -> None:
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), FakeSource())

    async def slow():
        await asyncio.sleep(10)
        yield b"x"

    ag._turn.begin(slow())
    await asyncio.sleep(0)
    assert ag._turn.speaking is True

    ag.set_persona(_hr(config_dir))
    await ag._turn.join()  # let the cancellation land
    assert ag._turn.speaking is False  # barge-out: old persona stopped talking


def test_set_same_persona_is_a_noop(config_dir: Path) -> None:
    companion = _companion(config_dir)
    ag = agent.PersonaAgent(make_backend(), companion, FakeSource())
    ag.set_persona(_companion(config_dir))  # same id, different instance
    assert ag.persona is companion  # unchanged
