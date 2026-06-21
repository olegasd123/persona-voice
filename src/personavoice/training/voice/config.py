"""Voice fine-tune config + plan builders.

`VoiceTrainConfig` is the one declarative description of a voice fine-tune; `build_voice_train_plan`
turns it into an engine-specific `VoiceTrainPlan` — the dataset-prep command (when an engine needs
one), the trainer command, and any written config — for either:

  - **f5**         → `f5-tts_finetune-cli` (the canonical F5-TTS trainer; mature, well-documented).
                     Its default checkpoint weights are CC-BY-NC, so a fine-tuned F5 voice is for
                     dev / personal personas, not commercial redistribution. Dataset prep runs
                     `f5-tts_prepare_csv_wavs` first.
  - **chatterbox** → a configurable trainer script (`trainer_script`); Chatterbox is MIT, so it's
                     the clean-license path, but it ships no official finetune CLI — point
                     `trainer_script` at the community trainer (see the voice README).

These plan-builders are the declarative *intent* (and what's unit-tested). The verified
end-to-end CUDA runner is `training/voice/run_finetune.sh` (in the Blackwell trainer image):
f5-tts ≥1.1 renamed the prep step to the `f5_tts.train.datasets.prepare_csv_wavs` *module*
(which wants a CSV with a `audio_file|text` header and absolute wav paths) and fixes its data/
ckpt roots by `dataset_name`, so the runner builds that CSV, fetches the pretrained pinyin
vocab, runs prepare + finetune, and copies the checkpoint back out — the bits a flat argv can't
capture. See `training/voice/README.md`.

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

VALID_ENGINES = ("f5", "chatterbox")


class VoiceTrainConfigError(ValueError):
    """A voice-train config is missing, malformed, or fails validation."""


class VoiceTrainConfig(BaseModel):
    """Everything needed to launch one voice fine-tune, engine-agnostic."""

    model_config = ConfigDict(extra="forbid")

    voice_name: str  # the fine-tuned voice id (registry name, output subdir)
    engine: str = "f5"  # "f5" -> f5-tts_finetune-cli; "chatterbox" -> trainer_script
    # f5: the base exp_name ("F5TTS_v1_Base" / "F5TTS_Base" / "E2TTS_Base").
    # chatterbox: a base checkpoint dir/id to fine-tune from.
    base_model: str = "F5TTS_v1_Base"
    data_dir: str  # holds metadata.csv (+ wavs); F5 prepares an arrow dataset alongside
    output_dir: str  # where the trained checkpoint lands
    dataset_name: str | None = None  # F5 registers datasets by name (default: voice_name)
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
    tokenizer: str = "pinyin"  # f5 tokenizer (pinyin handles English; char/custom also valid)
    # chatterbox-only: the community trainer entrypoint (no official CLI ships).
    trainer_script: str = "training/voice/chatterbox_finetune.py"
    # Free-form passthrough: f5 -> extra argv appended; chatterbox -> merged into its config.
    extra: dict[str, Any] = Field(default_factory=dict)

    def validate_engine(self) -> None:
        if self.engine not in VALID_ENGINES:
            raise VoiceTrainConfigError(
                f"invalid engine {self.engine!r}; expected one of {VALID_ENGINES}"
            )

    @property
    def resolved_dataset_name(self) -> str:
        return self.dataset_name or self.voice_name


@dataclass
class VoiceTrainPlan:
    """A ready-to-run voice fine-tune invocation for one engine."""

    engine: str  # "f5" | "chatterbox"
    tool: str  # the executable that must be available (e.g. "f5-tts_finetune-cli")
    command: list[str]  # the trainer argv
    prepare_command: list[str] | None = None  # dataset-prep argv to run first (f5), else None
    config_filename: str | None = None  # where finetune.py should write `config` (if any)
    config: dict[str, Any] | None = None  # a trainer config file (chatterbox), else None
    extra_argv: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# F5-TTS (f5-tts_finetune-cli)
# --------------------------------------------------------------------------------------


def _extra_argv(extra: dict[str, Any]) -> list[str]:
    """Render an `extra` dict as `--key value` argv (bare flags for `True`)."""
    argv: list[str] = []
    for key, value in extra.items():
        flag = f"--{key}"
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def f5_prepare_command(cfg: VoiceTrainConfig) -> list[str]:
    """`f5-tts_prepare_csv_wavs <data_dir> <prepared_dir>` — build F5's arrow dataset.

    F5 reads `<data_dir>/metadata.csv` (+ wavs) and writes the tokenized dataset it trains on.
    """
    prepared = str(Path(cfg.data_dir) / f"{cfg.resolved_dataset_name}_prepared")
    return ["f5-tts_prepare_csv_wavs", cfg.data_dir, prepared]


def f5_finetune_command(cfg: VoiceTrainConfig) -> list[str]:
    """Build the `f5-tts_finetune-cli` argv for one voice fine-tune."""
    command = [
        "f5-tts_finetune-cli",
        "--exp_name",
        cfg.base_model,
        "--dataset_name",
        cfg.resolved_dataset_name,
        "--learning_rate",
        str(cfg.learning_rate),
        "--batch_size_per_gpu",
        str(cfg.batch_size),
        "--grad_accumulation_steps",
        str(cfg.grad_accum),
        "--epochs",
        str(int(cfg.epochs)),
        "--num_warmup_updates",
        str(cfg.warmup_updates),
        "--save_per_updates",
        str(cfg.save_every),
        "--tokenizer",
        cfg.tokenizer,
        "--finetune",
    ]
    command.extend(_extra_argv(cfg.extra))
    return command


# --------------------------------------------------------------------------------------
# Chatterbox (configurable community trainer)
# --------------------------------------------------------------------------------------


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
    """Turn a `VoiceTrainConfig` into the engine-specific runnable plan."""
    cfg.validate_engine()
    if cfg.engine == "f5":
        return VoiceTrainPlan(
            engine="f5",
            tool="f5-tts_finetune-cli",
            command=f5_finetune_command(cfg),
            prepare_command=f5_prepare_command(cfg),
        )
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
