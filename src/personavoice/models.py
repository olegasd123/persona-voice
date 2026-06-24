"""Core data models shared across the cascade.

These are the typed values that flow between stages (`Transcript`, `Msg`, audio bytes)
and the declarative config objects (`Persona`, `VoiceRef`, `BackendConfig`). Keeping them
in one place lets every adapter depend on the contract without importing each other.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("personavoice.models")

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
    # Transcript of `sample_path`, when a cloning backend needs one.
    ref_text: str | None = None
    emotion: str | None = None
    # Backend that produced/owns this voice (e.g. "kokoro", "chatterbox").
    backend: str | None = None
    # Fine-tuned checkpoint a cloning backend should load instead of its base weights
    # (voice fine-tuning). None → use the backend's base model.
    model_path: str | None = None


class VoiceDef(BaseModel):
    """One entry in the voice registry (config/voices.yaml).

    A *logical* voice (e.g. `companion_soft`) that personas reference, mapped to a
    backend-native preset per TTS adapter so the personas sound distinct on each backend.
    `sample` is a clone source used by cloning-capable backends; until then a backend with
    no `presets` entry falls back to its own default voice.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    emotion: str | None = None
    # tts adapter name ("kokoro" | "chatterbox" | ...) -> that backend's preset id.
    presets: dict[str, str] = Field(default_factory=dict)
    # Path to a ~10 s clone sample, resolved by cloning backends.
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
    lora: str | None = None  # adapter path; null until trained
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
    scope: str = "per_user"  # "per_user" | "per_user_persona"
    # How many relevant past turns to retrieve into the prompt each turn (RAG).
    top_k: int = Field(default=4, ge=0)
    # Distill the rolling profile every N recorded turns (0 = manual/`consolidate` only).
    summarize_every: int = Field(default=6, ge=0)


class Persona(BaseModel):
    """A persona = system prompt + (optional) LoRA + voice + behavior knobs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    # One-line blurb for client persona pickers (the system prompt is too long to show).
    description: str = ""
    system_prompt: str
    llm: LLMSettings
    voice: VoiceSettings
    behavior: BehaviorSettings = BehaviorSettings()
    memory: MemorySettings = MemorySettings()
    # Per-session option defaults baked into the persona (custom personas set these via N3's
    # authoring routes). The agent layers an explicit `SessionOptions` over these; an unset
    # field falls through to the persona's authored behavior. Empty for the curated personas.
    # Forward ref + `Persona.model_rebuild()` below, since `SessionOptions` is defined later.
    session_defaults: SessionOptions = Field(default_factory=lambda: SessionOptions())


# --------------------------------------------------------------------------------------
# Session options (per-conversation overrides on top of a persona)
# --------------------------------------------------------------------------------------


class CEFRLevel(StrEnum):
    """Common European Framework of Reference language-proficiency levels."""

    a1 = "A1"
    a2 = "A2"
    b1 = "B1"
    b2 = "B2"
    c1 = "C1"
    c2 = "C2"


class Demeanor(StrEnum):
    """How the persona carries itself for this session. `natural` = persona as authored."""

    kind = "kind"
    natural = "natural"
    rude = "rude"


class SessionOptions(BaseModel):
    """Per-conversation overrides layered on top of a persona.

    All fields are optional; `None` means "use the persona's default". These are decided
    before a call (carried in room/job metadata) and can be swapped mid-call. They reuse the
    persona-selection rail: token metadata → agent → optional data message.
    """

    model_config = ConfigDict(extra="ignore")

    voice: str | None = None  # voice-library id (preset | clone | finetuned)
    cefr: CEFRLevel | None = None
    demeanor: Demeanor | None = None

    # Case-insensitive enums: clients/CLI flags send "b1"/"KIND"; the enum values are
    # canonical ("B1"/"kind"). Normalize before validation so casing never drops a value.
    @field_validator("cefr", mode="before")
    @classmethod
    def _normalize_cefr(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @field_validator("demeanor", mode="before")
    @classmethod
    def _normalize_demeanor(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @classmethod
    def from_metadata(cls, meta: str | None) -> SessionOptions:
        """Parse session options from a JSON metadata string, tolerating junk.

        The client tags the room/job with
        `{"persona": "...", "voice": "...", "cefr": "B1", "demeanor": "kind"}`. Missing keys
        are fine (the field stays None); an unknown enum value is dropped (logged once) rather
        than raising, so a stray value never blocks a session. Anything that isn't a JSON
        object yields empty options.
        """
        if not meta or not meta.strip() or not meta.strip().startswith("{"):
            return cls()
        try:
            obj = json.loads(meta.strip())
        except json.JSONDecodeError:
            return cls()
        if not isinstance(obj, dict):
            return cls()
        # Validate each field on its own so one bad value (e.g. cefr="Z9") drops only that
        # field instead of discarding the whole options object.
        kept: dict[str, Any] = {}
        for key in ("voice", "cefr", "demeanor"):
            value = obj.get(key)
            if value is None:
                continue
            try:
                cls.model_validate({key: value})
            except ValidationError:
                logger.warning("dropping invalid session option %s=%r", key, value)
                continue
            kept[key] = value
        return cls.model_validate(kept)

    def merged_over(self, base: SessionOptions) -> SessionOptions:
        """Return `base` with this object's set (non-None) fields taking precedence."""
        return base.model_copy(update={k: v for k, v in self.model_dump().items() if v is not None})

    def any_set(self) -> bool:
        """True when at least one override is set (vs. an all-None "use the defaults")."""
        return any(v is not None for v in self.model_dump().values())


# `Persona.session_defaults` forward-references `SessionOptions` (defined just above); finish
# building the deferred Persona schema now that the name resolves.
Persona.model_rebuild()


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
