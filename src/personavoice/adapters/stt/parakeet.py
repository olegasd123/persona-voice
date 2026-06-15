"""NVIDIA Parakeet STT (CUDA / RTX 4080) — alternative to faster-whisper. Implemented in M2."""

from __future__ import annotations

from .base import STTAdapter


class ParakeetSTT(STTAdapter):
    name = "parakeet"
