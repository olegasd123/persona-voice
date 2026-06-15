"""vLLM (OpenAI-compatible) LLM (CUDA / RTX 4080 prod). Implemented in M2."""

from __future__ import annotations

from .base import LLMAdapter


class VLLMAdapter(LLMAdapter):
    name = "vllm"
