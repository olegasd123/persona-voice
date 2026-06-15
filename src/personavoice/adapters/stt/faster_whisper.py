"""faster-whisper STT (CUDA / RTX 4080). Implemented in M2."""

from __future__ import annotations

from .base import STTAdapter


class FasterWhisperSTT(STTAdapter):
    name = "faster_whisper"
