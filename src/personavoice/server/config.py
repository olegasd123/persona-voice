"""Settings + config loading.

`Settings` reads environment (and a local `.env`) for the backend selection and paths.
The loader functions turn `config/backends/{backend}.yaml` and `config/personas/*.yaml`
into validated models. Nothing here loads ML weights.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from .._env import expand_env_vars
from ..memory import ConversationMemory, MemoryStore, cipher_from_key
from ..models import BackendConfig, Persona
from ..persona.loader import load_personas
from ..persona.store import UserPersonaStore
from ..voice.clone import ClonesStore
from ..voice.finetuned import FinetunedVoicesStore
from ..voice.registry import VoiceRegistry

VALID_BACKENDS = ("mac", "cuda")


class ConfigError(ValueError):
    """Configuration is missing or invalid."""


class Settings:
    """Process-level settings resolved from environment / `.env`."""

    def __init__(
        self,
        *,
        backend: str,
        config_dir: Path,
        models_dir: Path,
        clones_dir: Path | None = None,
        finetuned_dir: Path | None = None,
        memory_dir: Path | None = None,
        memory_key: str | None = None,
        memory_max_turns: int = 2000,
        user_personas_path: Path | None = None,
    ) -> None:
        self.backend = backend
        self.config_dir = config_dir
        self.models_dir = models_dir
        self._clones_dir = clones_dir
        self._finetuned_dir = finetuned_dir
        self._memory_dir = memory_dir
        self.memory_key = memory_key
        self.memory_max_turns = memory_max_turns
        self._user_personas_path = user_personas_path

    @property
    def backends_dir(self) -> Path:
        return self.config_dir / "backends"

    @property
    def personas_dir(self) -> Path:
        return self.config_dir / "personas"

    @property
    def voices_path(self) -> Path:
        return self.config_dir / "voices.yaml"

    @property
    def clones_dir(self) -> Path:
        """Where cloned voices + their manifest live. Defaults under the models dir."""
        return self._clones_dir or (self.models_dir / "clones")

    @property
    def finetuned_dir(self) -> Path:
        """Where fine-tuned voices + their manifest live. Defaults under the models dir."""
        return self._finetuned_dir or (self.models_dir / "finetuned")

    @property
    def memory_dir(self) -> Path:
        """Where per-user conversation memory lives. Defaults under the models dir."""
        return self._memory_dir or (self.models_dir / "memory")

    @property
    def user_personas_path(self) -> Path:
        """The writable JSON file holding user-authored personas. Defaults under models dir."""
        return self._user_personas_path or (self.models_dir / "user_personas.json")

    @classmethod
    def load(cls, *, backend: str | None = None, env_file: str | Path | None = ".env") -> Settings:
        """Resolve settings from env/.env, with an optional `backend` override."""
        if env_file is not None and Path(env_file).is_file():
            load_dotenv(env_file)

        resolved = (backend or os.getenv("BACKEND") or "mac").strip().lower()
        if resolved not in VALID_BACKENDS:
            raise ConfigError(
                f"invalid BACKEND {resolved!r}; expected one of {', '.join(VALID_BACKENDS)}"
            )

        config_dir = Path(os.getenv("PERSONAVOICE_CONFIG_DIR", "config")).expanduser()
        models_dir = Path(os.getenv("PERSONAVOICE_MODELS_DIR", "models")).expanduser()
        clones_env = os.getenv("PERSONAVOICE_CLONES_DIR")
        clones_dir = Path(clones_env).expanduser() if clones_env else None
        finetuned_env = os.getenv("PERSONAVOICE_FINETUNED_DIR")
        finetuned_dir = Path(finetuned_env).expanduser() if finetuned_env else None
        memory_env = os.getenv("PERSONAVOICE_MEMORY_DIR")
        memory_dir = Path(memory_env).expanduser() if memory_env else None
        memory_key = os.getenv("PERSONAVOICE_MEMORY_KEY") or None
        memory_max_turns = _memory_max_turns()
        user_personas_env = os.getenv("PERSONAVOICE_USER_PERSONAS")
        user_personas_path = Path(user_personas_env).expanduser() if user_personas_env else None
        return cls(
            backend=resolved,
            config_dir=config_dir,
            models_dir=models_dir,
            clones_dir=clones_dir,
            finetuned_dir=finetuned_dir,
            memory_dir=memory_dir,
            memory_key=memory_key,
            memory_max_turns=memory_max_turns,
            user_personas_path=user_personas_path,
        )


def _memory_max_turns() -> int:
    """Transcript retention cap from `PERSONAVOICE_MEMORY_MAX_TURNS` (default 2000, 0 = keep
    everything). The facade floors the cap at its recall window and skips training-opted-in
    users, so the default is safe: it bounds file growth without changing recall behavior.
    Garbage values fall back to the default rather than silently disabling retention.
    """
    raw = (os.getenv("PERSONAVOICE_MEMORY_MAX_TURNS") or "2000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        logging.getLogger("personavoice.config").warning(
            "PERSONAVOICE_MEMORY_MAX_TURNS=%r is not an integer; using 2000", raw
        )
        return 2000


def max_sessions() -> int:
    """Per-worker concurrent-session cap, from `PERSONAVOICE_MAX_SESSIONS` (default 1, min 1).

    Canonical reader shared by the agent worker (which *enforces* it via admission control) and the
    token server + `--check` (which *report* it). The default of 1 is the safe number for a single
    GPU: a second concurrent caller runs in a second process that loads its own STT+TTS copy and
    OOMs a 16 GB card. Garbage / non-positive values fall back to 1 rather than disabling the gate.
    """
    raw = (os.getenv("PERSONAVOICE_MAX_SESSIONS") or "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        logging.getLogger("personavoice.config").warning(
            "PERSONAVOICE_MAX_SESSIONS=%r is not an integer; using 1", raw
        )
        return 1


def load_backend_config(settings: Settings) -> BackendConfig:
    path = settings.backends_dir / f"{settings.backend}.yaml"
    if not path.is_file():
        raise ConfigError(f"backend config not found: {path}")
    try:
        raw = yaml.safe_load(expand_env_vars(path.read_text())) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        config = BackendConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid backend config: {exc}") from exc
    if config.backend != settings.backend:
        raise ConfigError(
            f"{path}: declares backend {config.backend!r} but file is for {settings.backend!r}"
        )
    return config


def load_all_personas(settings: Settings) -> dict[str, Persona]:
    return load_personas(settings.personas_dir)


def load_clones_store(settings: Settings) -> ClonesStore:
    """Load the cloned-voice catalog (tolerates a missing manifest → empty store)."""
    return ClonesStore.load(settings.clones_dir)


def load_finetuned_store(settings: Settings) -> FinetunedVoicesStore:
    """Load the fine-tuned-voice catalog (tolerates a missing manifest → empty store)."""
    return FinetunedVoicesStore.load(settings.finetuned_dir)


def load_voice_registry(settings: Settings) -> VoiceRegistry:
    """Load the voice registry with the clone + fine-tuned catalogs attached.

    Tolerates missing `voices.yaml` / manifests; the attached stores let `resolve_for_persona`
    honor per-persona assignments on a cloning backend — a fine-tuned voice taking precedence
    over a clone over the static preset.
    """
    return VoiceRegistry.load(
        settings.voices_path,
        clones=load_clones_store(settings),
        finetuned=load_finetuned_store(settings),
    )


def load_user_persona_store(settings: Settings) -> UserPersonaStore:
    """Load the multi-user custom-persona store (tolerates a missing file → empty store)."""
    return UserPersonaStore.load(settings.user_personas_path)


def load_memory_store(settings: Settings) -> MemoryStore:
    """Open the per-user memory store, encrypted if `PERSONAVOICE_MEMORY_KEY` is set."""
    return MemoryStore(settings.memory_dir, cipher=cipher_from_key(settings.memory_key))


def build_conversation_memory(
    settings: Settings, backend: object, persona: Persona | None = None
) -> ConversationMemory:
    """Build the runtime memory facade bound to the backend's LLM for distillation.

    Retrieval / consolidation knobs come from `persona.memory` when a persona is given (the
    facade is single-knobbed; the agent's primary persona seeds them). `backend` is an
    `adapters.factory.Backend`; typed loosely to avoid a heavy import here.
    """
    store = load_memory_store(settings)
    llm = getattr(backend, "llm", None)
    mem = persona.memory if persona is not None else None
    return ConversationMemory(
        store,
        llm=llm,
        k=mem.top_k if mem else 4,
        summarize_every=mem.summarize_every if mem else 6,
        max_transcript_turns=settings.memory_max_turns,
    )
