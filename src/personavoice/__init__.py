"""Persona-Voice: a modular speech-to-speech persona system (STT -> LLM -> TTS).

The package is organized as swappable backends behind stable adapter interfaces so the
same orchestration runs on the M4 Max (MLX / Ollama / Kokoro) and the RTX 4080
(faster-whisper / vLLM / Orpheus). See README.md.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
