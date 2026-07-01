"""Persona-Voice: a modular speech-to-speech persona system (STT -> LLM -> TTS).

The package is organized as swappable backends behind stable adapter interfaces so the
same orchestration runs on Apple silicon (MLX / Ollama / Kokoro) and a CUDA GPU
(faster-whisper / vLLM / Chatterbox). See README.md.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
