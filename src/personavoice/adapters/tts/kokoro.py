"""Kokoro TTS (Mac dev) — fast, no cloning. Default dev voice. Implemented in M1."""

from __future__ import annotations

from .base import TTSAdapter


class KokoroTTS(TTSAdapter):
    name = "kokoro"
    supports_cloning = False
