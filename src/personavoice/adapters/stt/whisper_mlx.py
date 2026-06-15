"""mlx-whisper STT (Mac / M4 Max). Implemented in M1."""

from __future__ import annotations

from .base import STTAdapter


class WhisperMLXSTT(STTAdapter):
    name = "whisper_mlx"
