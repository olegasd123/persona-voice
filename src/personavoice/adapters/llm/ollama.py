"""Ollama LLM (Mac / M4 Max dev). Implemented in M1."""

from __future__ import annotations

from .base import LLMAdapter


class OllamaLLM(LLMAdapter):
    name = "ollama"
