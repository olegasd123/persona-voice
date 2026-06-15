"""Adapter factory + base-class contract."""

from __future__ import annotations

import pytest

from personavoice.adapters.factory import (
    UnknownAdapterError,
    build_backend,
)
from personavoice.adapters.protocols import LLMProtocol, STTProtocol, TTSProtocol
from personavoice.models import BackendConfig, StageConfig


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


async def test_stub_stream_raises_not_implemented() -> None:
    be = build_backend(_backend_config("mac"))
    with pytest.raises(NotImplementedError):
        await be.stt.transcribe(b"")
