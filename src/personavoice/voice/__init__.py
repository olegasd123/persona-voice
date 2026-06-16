"""Voice registry + cloning. The registry (M4) maps logical voices to backend presets;
zero-shot cloning (M5) clones a voice from a sample and assigns it to a persona; fine-tuning
lands in M9."""

from .clone import ClonedVoice, CloneError, ClonesStore, VoiceCloner, validate_sample
from .registry import VoiceError, VoiceRegistry

__all__ = [
    "CloneError",
    "ClonedVoice",
    "ClonesStore",
    "VoiceCloner",
    "VoiceError",
    "VoiceRegistry",
    "validate_sample",
]
