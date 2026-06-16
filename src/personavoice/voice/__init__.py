"""Voice registry + cloning. The registry (M4) maps logical voices to backend presets;
zero-shot cloning lands in M5 and fine-tuning in M9."""

from .registry import VoiceError, VoiceRegistry

__all__ = ["VoiceError", "VoiceRegistry"]
