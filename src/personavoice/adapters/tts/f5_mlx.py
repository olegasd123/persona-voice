"""f5-tts-mlx TTS (Mac) — zero-shot voice cloning on Apple Silicon. Implemented in M5."""

from __future__ import annotations

from .base import TTSAdapter


class F5MLXTTS(TTSAdapter):
    name = "f5_mlx"
    supports_cloning = True
