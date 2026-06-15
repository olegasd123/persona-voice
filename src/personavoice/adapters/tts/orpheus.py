"""Orpheus TTS (CUDA / RTX 4080) — expressive, emotion tags, zero-shot cloning. Implemented in M2/M5."""

from __future__ import annotations

from .base import TTSAdapter


class OrpheusTTS(TTSAdapter):
    name = "orpheus"
    supports_cloning = True
