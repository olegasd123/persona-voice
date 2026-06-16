"""Core data models shared across the cascade.

These are the typed values that flow between stages (`Transcript`, `Msg`, audio bytes)
and the declarative config objects (`Persona`, `VoiceRef`, `BackendConfig`). Keeping them
in one place lets every adapter depend on the contract without importing each other.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Conversation values (flow between stages)
# --------------------------------------------------------------------------------------


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Msg(BaseModel):
    """A single chat message in the LLM conversation."""

    role: Role
    content: str


class Transcript(BaseModel):
    """STT output. `is_final` distinguishes streamed partials from finalized text."""

    text: str
    is_final: bool = False
    confidence: float | None = None
    language: str | None = None
    # Word/segment timing in seconds, when the backend provides it.
    start: float | None = None
    end: float | None = None


class VoiceRef(BaseModel):
    """A reference to a TTS voice — either a built-in preset or a cloned voice."""

    id: str
    name: str | None = None
    # Source sample used to create a clone (~10 s wav), if any.
    sample_path: str | None = None
    # Cached speaker embedding / conditioning produced by `clone_voice`.
    embedding_path: str | None = None
    # Transcript of `sample_path`, for reference-text cloning backends (e.g. F5).
    ref_text: str | None = None
    emotion: str | None = None
    # Backend that produced/owns this voice (e.g. "kokoro", "orpheus").
    backend: str | None = None


class VoiceDef(BaseModel):
    """One entry in the voice registry (config/voices.yaml).

    A *logical* voice (e.g. `companion_soft`) that personas reference, mapped to a
    backend-native preset per TTS adapter so the personas sound distinct on each backend.
    `sample` is a clone source used by cloning-capable backends (Chatterbox/F5) in M5; until
    then a backend with no `presets` entry falls back to its own default voice.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    emotion: str | None = None
    # tts adapter name ("kokoro" | "orpheus" | ...) -> that backend's preset id.
    presets: dict[str, str] = Field(default_factory=dict)
    # Path to a ~10 s clone sample (M5), resolved by cloning backends.
    sample: str | None = None


# --------------------------------------------------------------------------------------
# Persona config (loaded from config/personas/*.yaml)
# --------------------------------------------------------------------------------------


class TurnStyle(StrEnum):
    concise = "concise"
    balanced = "balanced"
    verbose = "verbose"


class LLMSettings(BaseModel):
    base_model: str
    lora: str | None = None  # adapter path; null until trained (M7)
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512


class VoiceSettings(BaseModel):
    ref: str  # VoiceRef id or a path under the voices/ registry
    emotion: str = "neutral"


class BehaviorSettings(BaseModel):
    turn_style: TurnStyle = TurnStyle.balanced
    follow_up_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class MemorySettings(BaseModel):
    enabled: bool = False
    scope: str = "per_user"


class Persona(BaseModel):
    """A persona = system prompt + (optional) LoRA + voice + behavior knobs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    system_prompt: str
    llm: LLMSettings
    voice: VoiceSettings
    behavior: BehaviorSettings = BehaviorSettings()
    memory: MemorySettings = MemorySettings()


# --------------------------------------------------------------------------------------
# Backend config (loaded from config/backends/{mac,cuda}.yaml)
# --------------------------------------------------------------------------------------


class StageConfig(BaseModel):
    """Config for one stage (stt/llm/tts): which adapter, which model, and free-form
    adapter options (device, compute_type, base_url, ...)."""

    adapter: str
    model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str  # "mac" | "cuda"
    stt: StageConfig
    llm: StageConfig
    tts: StageConfig


# --------------------------------------------------------------------------------------
# Adapter self-check result (used by `server --check`)
# --------------------------------------------------------------------------------------


class CheckResult(BaseModel):
    stage: str  # "stt" | "llm" | "tts"
    adapter: str
    ok: bool
    detail: str = ""
    warnings: list[str] = Field(default_factory=list)
