"""vLLM LLM (CUDA / RTX 4080).

vLLM serves an OpenAI-compatible `/v1/chat/completions` endpoint, so it reuses the shared
streaming logic in `_openai_compat` and only pins the default base URL (`:8000/v1`). The
server runs as its own process/container — see `docker-compose.yml` — e.g.:

    vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --max-model-len 8192

Persona `temperature` / `top_p` / `max_tokens` map onto the standard OpenAI sampling
params; vLLM-specific knobs (`guided_json`, `chat_template_kwargs`, ...) go through
`extra_body` in `config/backends/cuda.yaml`.

Per-persona LoRA hot-swap: launch vLLM with `--enable-lora --lora-modules
hr_interviewer=/adapters/hr_interviewer ...`, and a persona whose `llm.lora` basename matches a
served module name is routed to that adapter automatically (the request `model` becomes the
LoRA name). `supports_lora = True` enables this; LM Studio leaves it off.
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatLLM


class VLLMAdapter(OpenAICompatLLM):
    name = "vllm"
    implemented = True
    default_base_url = "http://localhost:8000/v1"
    supports_lora = True
