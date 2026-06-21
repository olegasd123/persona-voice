"""LM Studio LLM (Mac / M4 Max dev).

Talks to LM Studio's OpenAI-compatible server at `http://localhost:1234/v1`. The shared
streaming + reasoning-trace handling lives in `_openai_compat`; this subclass only pins the
default base URL (override per-deployment with the `base_url` option). See that module for
why only `delta.content` is spoken.
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatLLM


class LMStudioLLM(OpenAICompatLLM):
    name = "lmstudio"
    implemented = True
    default_base_url = "http://localhost:1234/v1"
