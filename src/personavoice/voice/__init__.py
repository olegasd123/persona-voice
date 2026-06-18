"""Voice registry + cloning + fine-tuning. The registry (M4) maps logical voices to backend
presets; zero-shot cloning (M5) clones a voice from a sample and assigns it to a persona;
fine-tuning (M9) trains a high-fidelity checkpoint and assigns it, taking precedence over a
clone."""

from .clone import ClonedVoice, CloneError, ClonesStore, VoiceCloner, validate_sample
from .finetuned import FinetunedVoice, FinetunedVoiceError, FinetunedVoicesStore
from .registry import VoiceError, VoiceRegistry

__all__ = [
    "CloneError",
    "ClonedVoice",
    "ClonesStore",
    "FinetunedVoice",
    "FinetunedVoiceError",
    "FinetunedVoicesStore",
    "VoiceCloner",
    "VoiceError",
    "VoiceRegistry",
    "validate_sample",
]
