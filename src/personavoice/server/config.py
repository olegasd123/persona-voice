"""Settings + config loading.

`Settings` reads environment (and a local `.env`) for the backend selection and paths.
The loader functions turn `config/backends/{backend}.yaml` and `config/personas/*.yaml`
into validated models. Nothing here loads ML weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from .._env import expand_env_vars
from ..models import BackendConfig, Persona
from ..persona.loader import load_personas
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
    ) -> None:
        self.backend = backend
        self.config_dir = config_dir
        self.models_dir = models_dir

    @property
    def backends_dir(self) -> Path:
        return self.config_dir / "backends"

    @property
    def personas_dir(self) -> Path:
        return self.config_dir / "personas"

    @property
    def voices_path(self) -> Path:
        return self.config_dir / "voices.yaml"

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
        return cls(backend=resolved, config_dir=config_dir, models_dir=models_dir)


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


def load_voice_registry(settings: Settings) -> VoiceRegistry:
    """Load the voice registry (tolerates a missing `voices.yaml`)."""
    return VoiceRegistry.load(settings.voices_path)
