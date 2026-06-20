"""Adapter warm-up + worker prewarm wiring (no real models, no GPU).

The live agent prewarms every stage at startup so the first turn isn't stuck behind lazy
model loading. These tests pin that contract: a real (`implemented`) adapter's `warmup`
drives one inference through its normal path; a stub adapter's `warmup` is a no-op (it must
not call the unimplemented inference method); and `_warmup_backend` warms all three stages
best-effort, so one stage failing doesn't starve the others.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from personavoice.adapters.factory import Backend
from personavoice.adapters.llm.base import LLMAdapter
from personavoice.adapters.stt.base import STTAdapter
from personavoice.adapters.tts.base import TTSAdapter
from personavoice.models import Msg, Persona, Role, VoiceRef
from personavoice.orchestrator.agent import _warmup_backend
from personavoice.persona import PersonaRegistry

from .fakes import FakeLLM, FakeSTT, FakeTTS, make_persona


# `implemented = True` opts these fakes into the base-class warm path (which calls the real
# inference method); the shipped fakes leave it False to model unimplemented stubs.
class WarmSTT(FakeSTT):
    implemented = True


class WarmLLM(FakeLLM):
    implemented = True


class WarmTTS(FakeTTS):
    implemented = True


def test_stt_warmup_runs_one_transcription() -> None:
    stt = WarmSTT()
    asyncio.run(stt.warmup())
    assert stt.received is not None  # a (silent) clip was transcribed


def test_llm_warmup_streams_one_token() -> None:
    llm = WarmLLM()
    asyncio.run(llm.warmup(make_persona()))
    assert llm.last_messages is not None
    assert len(llm.last_messages) == 1  # exactly the warm-up prompt
    assert llm.last_messages[0].role == Role.user
    assert not llm.completed  # the stream is closed after the first token, not exhausted


def test_tts_warmup_synthesizes_a_phrase() -> None:
    tts = WarmTTS()
    asyncio.run(tts.warmup(VoiceRef(id="voices/test")))
    assert tts.last_text == "Ready."


def test_llm_warmup_retries_until_server_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    # vLLM may still be loading its model when the agent starts; warm-up must keep retrying
    # rather than give up on the first connection error.
    async def _no_wait(_seconds: float) -> None:  # don't actually sleep between retries
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_wait)

    class FlakyLLM(WarmLLM):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def stream_chat(
            self, messages: list[Msg], persona: Persona
        ) -> AsyncIterator[str]:
            self.attempts += 1
            if self.attempts < 3:  # "server not ready yet" on the first two tries
                raise ConnectionError("server disconnected")
            yield "ok"

    llm = FlakyLLM()
    asyncio.run(llm.warmup(make_persona()))
    assert llm.attempts == 3  # retried past the transient failures, then succeeded


def test_llm_warmup_gives_up_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_wait)
    monkeypatch.setenv("PERSONAVOICE_LLM_WARMUP_WAIT", "0")  # no budget → raise immediately

    class DeadLLM(WarmLLM):
        async def stream_chat(
            self, messages: list[Msg], persona: Persona
        ) -> AsyncIterator[str]:
            raise ConnectionError("server down")
            yield  # pragma: no cover - makes this an async generator

    with pytest.raises(ConnectionError):
        asyncio.run(DeadLLM().warmup(make_persona()))


def test_stub_adapters_warmup_is_a_noop() -> None:
    # Base adapters are unimplemented (implemented=False); warming them must not raise by
    # calling their NotImplementedError inference methods.
    asyncio.run(STTAdapter().warmup())
    asyncio.run(LLMAdapter().warmup(make_persona()))
    asyncio.run(TTSAdapter().warmup(VoiceRef(id="voices/test")))


def test_warmup_backend_warms_all_stages(config_dir) -> None:  # type: ignore[no-untyped-def]
    backend = Backend(name="fake", stt=WarmSTT(), llm=WarmLLM(), tts=WarmTTS())
    registry = PersonaRegistry(config_dir / "personas")
    asyncio.run(_warmup_backend(backend, None, registry))
    assert backend.stt.received is not None
    assert backend.llm.last_messages is not None
    assert backend.tts.last_text == "Ready."


def test_warmup_backend_is_best_effort_when_a_stage_fails(config_dir) -> None:  # type: ignore[no-untyped-def]
    class BoomSTT(WarmSTT):
        async def warmup(self) -> None:
            raise RuntimeError("boom")

    backend = Backend(name="fake", stt=BoomSTT(), llm=WarmLLM(), tts=WarmTTS())
    registry = PersonaRegistry(config_dir / "personas")
    # The STT failure is swallowed; LLM + TTS still warm.
    asyncio.run(_warmup_backend(backend, None, registry))
    assert backend.llm.last_messages is not None
    assert backend.tts.last_text == "Ready."
