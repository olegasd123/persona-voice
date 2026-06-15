"""Chatterbox TTS (CUDA; MPS on Mac) — emotion-exaggeration control, zero-shot cloning. Implemented in M2/M5."""

from __future__ import annotations

from .base import TTSAdapter


class ChatterboxTTS(TTSAdapter):
    name = "chatterbox"
    supports_cloning = True
