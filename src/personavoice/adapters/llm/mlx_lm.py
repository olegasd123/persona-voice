"""mlx-lm LLM (Mac / M4 Max) — in-process alternative to Ollama. Implemented in M1/M2."""

from __future__ import annotations

from .base import LLMAdapter


class MLXLMAdapter(LLMAdapter):
    name = "mlx_lm"
