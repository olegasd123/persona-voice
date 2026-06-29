"""Backend factory: map adapter names from backend YAML to adapter classes.

`build_backend(BackendConfig)` returns a `Backend` bundle holding the three constructed
adapters. Construction is cheap (no model loading) so it works in `server --check` on any
machine; the heavy work happens lazily inside each adapter's stream methods.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import BackendConfig, StageConfig
from .llm.base import LLMAdapter
from .llm.lmstudio import LMStudioLLM
from .llm.mlx_lm import MLXLMAdapter
from .llm.ollama import OllamaLLM
from .llm.vllm import VLLMAdapter
from .stt.base import STTAdapter
from .stt.faster_whisper import FasterWhisperSTT
from .stt.parakeet import ParakeetSTT
from .stt.whisper_mlx import WhisperMLXSTT
from .tts.base import TTSAdapter
from .tts.chatterbox import ChatterboxTTS
from .tts.kokoro import KokoroTTS

STT_ADAPTERS: dict[str, type[STTAdapter]] = {
    cls.name: cls for cls in (WhisperMLXSTT, FasterWhisperSTT, ParakeetSTT)
}
LLM_ADAPTERS: dict[str, type[LLMAdapter]] = {
    cls.name: cls for cls in (LMStudioLLM, OllamaLLM, VLLMAdapter, MLXLMAdapter)
}
TTS_ADAPTERS: dict[str, type[TTSAdapter]] = {cls.name: cls for cls in (KokoroTTS, ChatterboxTTS)}


@dataclass
class Backend:
    """A fully-constructed cascade for one machine."""

    name: str
    stt: STTAdapter
    llm: LLMAdapter
    tts: TTSAdapter


class UnknownAdapterError(ValueError):
    """Raised when backend YAML names an adapter that isn't registered."""


def _build(stage: str, table: dict, cfg: StageConfig):
    try:
        cls = table[cfg.adapter]
    except KeyError:
        known = ", ".join(sorted(table)) or "(none)"
        raise UnknownAdapterError(
            f"unknown {stage} adapter {cfg.adapter!r}; known: {known}"
        ) from None
    return cls(model=cfg.model, options=cfg.options)


def build_backend(config: BackendConfig) -> Backend:
    return Backend(
        name=config.backend,
        stt=_build("stt", STT_ADAPTERS, config.stt),
        llm=_build("llm", LLM_ADAPTERS, config.llm),
        tts=_build("tts", TTS_ADAPTERS, config.tts),
    )
