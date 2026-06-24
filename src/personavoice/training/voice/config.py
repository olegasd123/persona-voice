"""Voice fine-tune config + plan builders.

`VoiceTrainConfig` is the one declarative description of a voice fine-tune.
`build_voice_train_plan` turns it into a Chatterbox trainer command and a written trainer
config. Chatterbox is MIT, but it ships no official finetune CLI — point `trainer_script` at
the community trainer used by your training image.

Voice fine-tuning is CUDA-centric (the plan's training track). Building the plan is pure (no
model libs, no disk), so it's fully unit-tested; `finetune.py` writes any config and runs the
commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

VALID_ENGINES = ("chatterbox",)


class VoiceTrainConfigError(ValueError):
    """A voice-train config is missing, malformed, or fails validation."""


class VoiceTrainConfig(BaseModel):
    """Everything needed to launch one voice fine-tune, engine-agnostic."""

    model_config = ConfigDict(extra="forbid")

    voice_name: str  # the fine-tuned voice id (registry name, output subdir)
    engine: str = "chatterbox"
    base_model: str = "ResembleAI/chatterbox"
    data_dir: str  # holds metadata.csv (+ wavs)
    output_dir: str  # where the trained checkpoint lands
    speaker: str | None = None  # target speaker label (provenance)
    # Shared hyperparameter vocabulary (mapped per engine below).
    epochs: float = 50.0
    learning_rate: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 1
    warmup_updates: int = 200
    save_every: int = 1000
    max_seconds: float = 30.0  # drop/clip audio longer than this
    seed: int = 0
    lora: bool = False  # parameter-efficient fine-tune where the engine supports it
    # Chatterbox trainer entrypoint (no official CLI ships).
    trainer_script: str = "training/voice/chatterbox_finetune.py"
    # Free-form passthrough merged into the trainer config.
    extra: dict[str, Any] = Field(default_factory=dict)

    def validate_engine(self) -> None:
        if self.engine not in VALID_ENGINES:
            raise VoiceTrainConfigError(
                f"invalid engine {self.engine!r}; expected one of {VALID_ENGINES}"
            )


@dataclass
class VoiceTrainPlan:
    """A ready-to-run voice fine-tune invocation for one engine."""

    engine: str
    tool: str
    command: list[str]  # the trainer argv
    config_filename: str | None = None  # where finetune.py should write `config` (if any)
    config: dict[str, Any] | None = None
    extra_argv: list[str] = field(default_factory=list)


def chatterbox_config(cfg: VoiceTrainConfig) -> dict[str, Any]:
    """A trainer config the community Chatterbox finetune script reads."""
    config: dict[str, Any] = {
        "voice_name": cfg.voice_name,
        "base_checkpoint": cfg.base_model,
        "metadata": str(Path(cfg.data_dir) / "metadata.csv"),
        "output_dir": cfg.output_dir,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "grad_accumulation_steps": cfg.grad_accum,
        "max_audio_seconds": cfg.max_seconds,
        "use_lora": cfg.lora,
        "seed": cfg.seed,
    }
    config.update(cfg.extra)
    return config


def chatterbox_finetune_command(cfg: VoiceTrainConfig, config_path: str | Path) -> list[str]:
    return ["python", cfg.trainer_script, "--config", str(config_path)]


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def build_voice_train_plan(cfg: VoiceTrainConfig) -> VoiceTrainPlan:
    """Turn a `VoiceTrainConfig` into a runnable Chatterbox plan."""
    cfg.validate_engine()
    config_path = str(Path(cfg.output_dir) / f"{cfg.voice_name}_finetune.chatterbox.yaml")
    return VoiceTrainPlan(
        engine="chatterbox",
        tool=cfg.trainer_script,
        command=chatterbox_finetune_command(cfg, config_path),
        config_filename=config_path,
        config=chatterbox_config(cfg),
    )


# --------------------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------------------


def from_voice(
    voice_name: str,
    *,
    engine: str,
    data_dir: str | Path,
    output_dir: str | Path,
    **overrides: Any,
) -> VoiceTrainConfig:
    """Seed a config from a voice name + dirs (the CLI's common case)."""
    fields: dict[str, Any] = {
        "voice_name": voice_name,
        "engine": engine,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
    }
    fields.update(overrides)
    return VoiceTrainConfig(**fields)


def load_config(path: str | Path) -> VoiceTrainConfig:
    """Load a `VoiceTrainConfig` from YAML (training/voice/configs/*.yaml)."""
    path = Path(path)
    if not path.is_file():
        raise VoiceTrainConfigError(f"voice-train config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise VoiceTrainConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return VoiceTrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise VoiceTrainConfigError(f"{path}: invalid voice-train config: {exc}") from exc
