"""M7 inference hot-swap: a persona's `llm.lora` routes to the served adapter on vLLM."""

from __future__ import annotations

from personavoice.adapters.llm._openai_compat import (
    build_payload,
    lora_request_model,
)
from personavoice.adapters.llm.lmstudio import LMStudioLLM
from personavoice.adapters.llm.vllm import VLLMAdapter
from personavoice.models import Msg, Role

from .fakes import make_persona


def _messages() -> list[Msg]:
    return [Msg(role=Role.user, content="hello")]


def test_lora_request_model_is_basename() -> None:
    assert lora_request_model(make_persona(lora="adapters/hr_interviewer")) == "hr_interviewer"
    assert lora_request_model(make_persona(lora="hr_interviewer")) == "hr_interviewer"


def test_lora_request_model_none_when_unset() -> None:
    assert lora_request_model(make_persona(lora=None)) is None


def test_build_payload_uses_base_model_without_lora() -> None:
    payload = build_payload("base-model", _messages(), make_persona())
    assert payload["model"] == "base-model"


def test_build_payload_lora_overrides_model() -> None:
    payload = build_payload("base-model", _messages(), make_persona(), lora="hr_interviewer")
    assert payload["model"] == "hr_interviewer"


def test_vllm_supports_lora_lmstudio_does_not() -> None:
    assert VLLMAdapter.supports_lora is True
    assert LMStudioLLM.supports_lora is False


def test_payload_still_carries_sampling_params() -> None:
    persona = make_persona()
    payload = build_payload("m", _messages(), persona, lora="x")
    assert payload["temperature"] == persona.llm.temperature
    assert payload["max_tokens"] == persona.llm.max_tokens
