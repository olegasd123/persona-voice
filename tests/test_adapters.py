"""Adapter factory + base-class contract."""

from __future__ import annotations

import pytest

from personavoice.adapters.factory import (
    UnknownAdapterError,
    build_backend,
)
from personavoice.adapters.protocols import LLMProtocol, STTProtocol, TTSProtocol
from personavoice.adapters.tts.base import TTSAdapter
from personavoice.models import BackendConfig, StageConfig, VoiceRef


def _backend_config(backend: str) -> BackendConfig:
    return BackendConfig(
        backend=backend,
        stt=StageConfig(adapter="whisper_mlx" if backend == "mac" else "faster_whisper"),
        llm=StageConfig(adapter="ollama" if backend == "mac" else "vllm"),
        tts=StageConfig(adapter="kokoro" if backend == "mac" else "orpheus"),
    )


@pytest.mark.parametrize("backend", ["mac", "cuda"])
def test_factory_builds_and_satisfies_protocols(backend: str) -> None:
    be = build_backend(_backend_config(backend))
    assert isinstance(be.stt, STTProtocol)
    assert isinstance(be.llm, LLMProtocol)
    assert isinstance(be.tts, TTSProtocol)
    for adapter in (be.stt, be.llm, be.tts):
        assert adapter.check().ok


def test_factory_rejects_unknown_adapter() -> None:
    cfg = BackendConfig(
        backend="mac",
        stt=StageConfig(adapter="bogus"),
        llm=StageConfig(adapter="ollama"),
        tts=StageConfig(adapter="kokoro"),
    )
    with pytest.raises(UnknownAdapterError):
        build_backend(cfg)


@pytest.mark.parametrize("backend", ["mac", "cuda"])
def test_stt_stream_is_not_used_directly(backend: str) -> None:
    # Live STT endpointing is VAD-driven in the live agent, not the base `stream`, which
    # stays unimplemented and must fail loudly if called.
    be = build_backend(_backend_config(backend))
    with pytest.raises(NotImplementedError):
        be.stt.stream(iter(()))  # type: ignore[arg-type]


async def test_tts_stream_tts_default_synthesizes_each_chunk() -> None:
    # The base `stream_tts` default speaks each sentence chunk as it arrives (skipping
    # blanks), so every backend gets streaming for free on top of its one-shot synthesize.
    class _OneShotTTS(TTSAdapter):
        name = "oneshot"

        async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
            return b"<" + text.encode() + b">"

    async def chunks():
        for c in ["First.", "   ", "Second."]:
            yield c

    out = [b async for b in _OneShotTTS().stream_tts(chunks(), VoiceRef(id="x"))]
    assert out == [b"<First.>", b"<Second.>"]


def test_cuda_adapters_are_implemented() -> None:
    # The CUDA cascade is real, not stubs: no "stub adapter" warning.
    be = build_backend(_backend_config("cuda"))
    for adapter in (be.stt, be.llm, be.tts):
        assert adapter.implemented is True
        assert not any("stub adapter" in w for w in adapter.check().warnings)
