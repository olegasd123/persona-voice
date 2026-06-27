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
from personavoice.persona.registry import PersonaRegistry
from personavoice.persona.store import UserPersonaStore

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


# --- session options (mid-session) ----------------------------------------------------


def test_resolve_session_options_merges_across_sources() -> None:
    from personavoice.models import CEFRLevel, Demeanor

    # Highest priority (first) source wins per field; fields combine across sources.
    opts = agent.resolve_session_options(
        ['{"demeanor": "rude"}', '{"cefr": "B1", "demeanor": "kind"}']
    )
    assert opts.demeanor is Demeanor.rude  # first source wins for demeanor
    assert opts.cefr is CEFRLevel.b1  # only the second source set cefr


def test_resolve_session_options_empty() -> None:
    from personavoice.models import SessionOptions

    assert agent.resolve_session_options([None, "", "not json"]) == SessionOptions()


def test_set_options_merges_and_propagates(config_dir: Path) -> None:
    from personavoice.models import CEFRLevel, Demeanor, SessionOptions

    ag = agent.PersonaAgent(
        make_backend(),
        _companion(config_dir),
        FakeSource(),
        options=SessionOptions(cefr=CEFRLevel.b1),
    )
    ag.set_options(SessionOptions(demeanor=Demeanor.rude))
    # The new field is applied; the prior field is kept; the pipeline sees the merged options.
    assert ag.options.demeanor is Demeanor.rude
    assert ag.options.cefr is CEFRLevel.b1
    assert ag._pipeline.options == ag.options


async def test_set_options_interrupts_in_flight_reply(config_dir: Path) -> None:
    from personavoice.models import Demeanor, SessionOptions

    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), FakeSource())

    async def slow():  # type: ignore[no-untyped-def]
        await asyncio.sleep(10)
        yield b"x"

    ag._turn.begin(slow())
    await asyncio.sleep(0)
    assert ag._turn.speaking is True

    ag.set_options(SessionOptions(demeanor=Demeanor.rude))
    await ag._turn.join()
    assert ag._turn.speaking is False


def test_set_options_noop_when_unchanged(config_dir: Path) -> None:
    from personavoice.models import Demeanor, SessionOptions

    ag = agent.PersonaAgent(
        make_backend(),
        _companion(config_dir),
        FakeSource(),
        options=SessionOptions(demeanor=Demeanor.rude),
    )
    before = ag.options
    ag.set_options(SessionOptions(demeanor=Demeanor.rude))  # same value
    assert ag.options == before


# --- custom personas (multi-user store) -----------------------------------------------


def _custom_persona(persona_id: str, *, name: str = "Custom", **session_defaults: str) -> Persona:
    body: dict[str, object] = {
        "id": persona_id,
        "name": name,
        "system_prompt": "You are a custom persona.",
        "llm": {"base_model": "qwen2.5-7b-instruct"},
        "voice": {"ref": "voices/companion_soft"},
    }
    if session_defaults:
        body["session_defaults"] = session_defaults
    return Persona.model_validate(body)


def _store(tmp_path: Path, users: dict[str, dict[str, Persona]]) -> UserPersonaStore:
    # In-memory (constructor users=) so no file is written during the test.
    return UserPersonaStore(tmp_path / "u.json", users=users)


def test_lookup_persona_resolves_custom(config_dir: Path, tmp_path: Path) -> None:
    reg = PersonaRegistry(config_dir / "personas")
    store = _store(tmp_path, {"alice": {"french-tutor": _custom_persona("french-tutor")}})
    assert agent._lookup_persona(reg, store, "alice", "french-tutor").id == "french-tutor"
    assert agent._lookup_persona(reg, store, "bob", "french-tutor") is None  # other user
    assert agent._lookup_persona(reg, store, "alice", "companion").id == "companion"  # curated
    assert agent._lookup_persona(reg, store, "alice", "ghost") is None


def test_lookup_persona_curated_wins_on_clash(config_dir: Path, tmp_path: Path) -> None:
    reg = PersonaRegistry(config_dir / "personas")
    store = _store(tmp_path, {"alice": {"companion": _custom_persona("companion", name="Evil")}})
    # The built-in companion wins; the user's same-id persona can't shadow it.
    assert agent._lookup_persona(reg, store, "alice", "companion").name != "Evil"


def test_select_persona_picks_custom_from_metadata(config_dir: Path, tmp_path: Path) -> None:
    reg = PersonaRegistry(config_dir / "personas")
    store = _store(tmp_path, {"alice": {"french-tutor": _custom_persona("french-tutor")}})
    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata='{"persona": "french-tutor", "user": "alice"}'),
        room=SimpleNamespace(metadata=None),
    )
    persona = agent._select_persona(ctx, reg, None, user_store=store, user_id="alice")
    assert persona.id == "french-tutor"


def test_select_persona_unknown_custom_falls_back(config_dir: Path, tmp_path: Path) -> None:
    reg = PersonaRegistry(config_dir / "personas")
    store = _store(tmp_path, {})
    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata='{"persona": "ghost", "user": "alice"}'),
        room=SimpleNamespace(metadata=None),
    )
    persona = agent._select_persona(ctx, reg, None, user_store=store, user_id="alice")
    assert persona.id == agent._default_persona_id(reg)


# --- caller token metadata (the reliable selection channel) ---------------------------


def _room_with_participants(*metas: str | None) -> SimpleNamespace:
    """A fake `ctx.room` whose remote participants carry the given token metadata strings."""
    parts = {f"p{i}": SimpleNamespace(identity=f"p{i}", metadata=m) for i, m in enumerate(metas)}
    return SimpleNamespace(metadata=None, remote_participants=parts)


def test_participant_metadata_picks_first_nonempty() -> None:
    room = _room_with_participants("", '{"persona": "hr_interviewer", "voice": "hr_warm"}')
    assert agent._participant_metadata(room) == '{"persona": "hr_interviewer", "voice": "hr_warm"}'


def test_participant_metadata_none_without_participants() -> None:
    assert agent._participant_metadata(SimpleNamespace()) is None  # no remote_participants attr
    assert agent._participant_metadata(SimpleNamespace(remote_participants={})) is None
    assert agent._participant_metadata(_room_with_participants(None, "")) is None


def test_select_persona_prefers_caller_token_metadata(config_dir: Path, tmp_path: Path) -> None:
    # Persona rides the caller's token metadata (room/job empty under automatic dispatch).
    reg = PersonaRegistry(config_dir / "personas")
    store = _store(tmp_path, {})
    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata=None),
        room=_room_with_participants('{"persona": "hr_interviewer"}'),
    )
    persona = agent._select_persona(ctx, reg, None, user_store=store, user_id=None)
    assert persona.id == "hr_interviewer"


def test_session_options_resolve_from_caller_token_metadata() -> None:
    from personavoice.models import CEFRLevel

    # The voice / cefr the user "Applied" arrive in the token metadata and must be honored.
    room = _room_with_participants('{"voice": "hr_warm", "cefr": "B1"}')
    opts = agent.resolve_session_options([agent._participant_metadata(room), None, None])
    assert opts.voice == "hr_warm"
    assert opts.cefr is CEFRLevel.b1


# --- persona-authored session defaults ------------------------------------------------


def test_persona_session_defaults_apply(config_dir: Path) -> None:
    from personavoice.models import CEFRLevel

    persona = _custom_persona("tutor", cefr="a1")  # baked-in default
    ag = agent.PersonaAgent(make_backend(), persona, FakeSource())
    assert ag.options.cefr is CEFRLevel.a1
    assert ag._pipeline.options.cefr is CEFRLevel.a1


def test_explicit_options_override_persona_defaults(config_dir: Path) -> None:
    from personavoice.models import CEFRLevel, Demeanor, SessionOptions

    persona = _custom_persona("tutor", cefr="a1", demeanor="kind")
    ag = agent.PersonaAgent(
        make_backend(), persona, FakeSource(), options=SessionOptions(cefr=CEFRLevel.c2)
    )
    assert ag.options.cefr is CEFRLevel.c2  # explicit wins
    assert ag.options.demeanor is Demeanor.kind  # persona default fills the gap


def test_set_persona_recomputes_session_defaults(config_dir: Path) -> None:
    from personavoice.models import CEFRLevel

    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), FakeSource())
    assert ag.options.cefr is None  # companion has no session defaults
    ag.set_persona(_custom_persona("tutor", cefr="a1"))
    assert ag.options.cefr is CEFRLevel.a1  # the new persona's default now applies


def test_explicit_options_survive_persona_switch(config_dir: Path) -> None:
    from personavoice.models import CEFRLevel, Demeanor, SessionOptions

    ag = agent.PersonaAgent(
        make_backend(),
        _companion(config_dir),
        FakeSource(),
        options=SessionOptions(demeanor=Demeanor.rude),
    )
    ag.set_persona(_custom_persona("tutor", cefr="a1"))
    assert ag.options.demeanor is Demeanor.rude  # explicit override preserved across the switch
    assert ag.options.cefr is CEFRLevel.a1  # new persona's default added underneath


# --- semantic endpointing -------------------------------------------------


class QueueSTT(FakeSTT):
    """Returns successive transcripts, one per utterance, so a turn can be fed in fragments."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = list(texts)

    async def transcribe(self, audio: bytes) -> Transcript:
        text = self._texts.pop(0) if self._texts else ""
        return Transcript(text=text, is_final=True, language="en")


def _endpointing_agent(
    config_dir: Path, texts: list[str], *, grace_s: float
) -> tuple[agent.PersonaAgent, list[bytes]]:
    backend = Backend(name="fake", stt=QueueSTT(texts), llm=FakeLLM("Okay."), tts=FakeTTS())
    ag = agent.PersonaAgent(
        backend, _companion(config_dir), FakeSource(), semantic_endpointing=True, grace_s=grace_s
    )
    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)  # bypass the livekit AudioSource sink
    return ag, spoken


async def _utterance(ag: agent.PersonaAgent) -> None:
    await ag.on_user_utterance(b"\x00\x00" * 1600, 16000)


async def test_unfinished_utterance_is_held_then_merged(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    # Big grace so the timer never fires — the continuation arrives first.
    ag, spoken = _endpointing_agent(
        config_dir, ["I went to the store and", "bought milk."], grace_s=60.0
    )

    await _utterance(ag)  # ends on "and" → held, nothing answered yet
    assert spoken == []
    assert ag._held == "I went to the store and"
    assert ag._pipeline.history == []

    await _utterance(ag)  # completes the thought → one merged turn
    await ag._turn.join()
    assert spoken == [b"RIFF" + b"Okay."]
    assert ag._held is None
    assert ag._pipeline.history[0].content == "I went to the store and bought milk."


async def test_complete_utterance_answers_immediately(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    ag, spoken = _endpointing_agent(config_dir, ["What time is it?"], grace_s=60.0)
    await _utterance(ag)
    await ag._turn.join()
    assert spoken == [b"RIFF" + b"Okay."]
    assert ag._held is None
    assert ag._flush_handle is None  # nothing held, no timer armed


async def test_grace_window_flushes_held_utterance(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    # Tiny grace: with no continuation, the held fragment is answered as-is once it elapses.
    ag, spoken = _endpointing_agent(config_dir, ["I'm thinking of"], grace_s=0.01)
    await _utterance(ag)
    assert spoken == []  # held, not yet answered
    await asyncio.sleep(0.05)  # let the grace timer fire
    await ag._turn.join()
    assert spoken == [b"RIFF" + b"Okay."]
    assert ag._held is None
    assert ag._pipeline.history[0].content == "I'm thinking of"


async def test_resume_cancels_pending_flush_then_merges(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    ag, spoken = _endpointing_agent(config_dir, ["I'm thinking of", "the blue one."], grace_s=60.0)
    await _utterance(ag)
    assert ag._flush_handle is not None  # a flush is armed while holding

    ag.on_user_speech_started()  # the user resumed before the grace elapsed
    assert ag._flush_handle is None  # so the flush is cancelled, not fired
    assert spoken == []

    await _utterance(ag)  # their continuation completes the thought
    await ag._turn.join()
    assert spoken == [b"RIFF" + b"Okay."]
    assert ag._pipeline.history[0].content == "I'm thinking of the blue one."


async def test_endpointing_off_answers_each_utterance(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    # Default (off): an utterance that *looks* unfinished is still answered immediately.
    backend = Backend(name="fake", stt=FakeSTT("I went and"), llm=FakeLLM("Okay."), tts=FakeTTS())
    ag = agent.PersonaAgent(
        backend, _companion(config_dir), FakeSource(), semantic_endpointing=False
    )
    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)
    await _utterance(ag)
    await ag._turn.join()
    assert spoken == [b"RIFF" + b"Okay."]
    assert ag._held is None


# --- post-session feedback report (teardown hook) -------------------------------------


class _ReportLLM:
    """Fake LLM whose `.chat` returns a fixed rubric JSON (the slice the builder uses)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, messages: object, persona: object) -> str:
        return self.reply


def _report_memory(tmp_path: Path):
    from personavoice.memory import ConversationMemory, MemoryStore

    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    llm = _ReportLLM('{"summary": "Nice work.", "scores": [{"name": "Clarity", "score": 4}]}')
    return ConversationMemory(store, llm=llm), store


async def test_aclose_finalizes_report(config_dir: Path, tmp_path: Path) -> None:
    mem, store = _report_memory(tmp_path)
    ag = agent.PersonaAgent(
        make_backend(),
        _companion(config_dir),
        FakeSource(),
        memory=mem,
        user_id="alice",
        session_reports=True,  # force on (companion is a "general" persona)
    )
    # Seed the live conversation history the report is built from.
    ag._pipeline.history = [
        Msg(role="user", content="I led a migration project."),
        Msg(role="assistant", content="What was the result?"),
        Msg(role="user", content="We cut costs by a third."),
    ]
    session_id = ag._pipeline.session_id
    assert session_id is not None

    await ag.aclose()
    assert store.load_report("alice", session_id)["summary"] == "Nice work."


async def test_aclose_skips_report_when_disabled(config_dir: Path, tmp_path: Path) -> None:
    mem, store = _report_memory(tmp_path)
    ag = agent.PersonaAgent(
        make_backend(),
        _companion(config_dir),
        FakeSource(),
        memory=mem,
        user_id="alice",
        session_reports=False,
    )
    ag._pipeline.history = [Msg(role="user", content="hello there friend")]
    session_id = ag._pipeline.session_id
    await ag.aclose()
    assert session_id is not None
    assert store.load_report("alice", session_id) is None
