"""Adapter factory + base-class contract."""

from __future__ import annotations

import pytest

from personavoice.adapters.factory import (
    UnknownAdapterError,
    build_backend,
)
from personavoice.adapters.protocols import LLMProtocol, STTProtocol, TTSProtocol
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
def test_streaming_not_implemented_until_m3(backend: str) -> None:
    # Turn-based transcribe/synthesize land in M1/M2; live streaming arrives in M3, so the
    # base streaming methods must still fail loudly on every backend.
    be = build_backend(_backend_config(backend))
    with pytest.raises(NotImplementedError):
        be.stt.stream(iter(()))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        be.tts.stream_tts(iter(()), VoiceRef(id="x"))  # type: ignore[arg-type]


def test_cuda_adapters_are_implemented() -> None:
    # After M2 the CUDA cascade is real, not stubs: no "stub adapter" warning.
    be = build_backend(_backend_config("cuda"))
    for adapter in (be.stt, be.llm, be.tts):
        assert adapter.implemented is True
        assert not any("stub adapter" in w for w in adapter.check().warnings)
