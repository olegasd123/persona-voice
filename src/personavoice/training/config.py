"""LoRA training config + plan builders.

`LoRATrainConfig` is the one declarative description of a persona fine-tune. `build_train_plan`
turns it into a backend-specific `TrainPlan` — the trainer config dict, the config filename, and
the command to run — for either:

  - **mac**  → `mlx_lm lora -c <config>.yaml` (light LoRA on Apple silicon).
  - **cuda** → `llamafactory-cli train <config>.yaml` (QLoRA on the 4080; `quantize_4bit`).

Building the plan is pure (no model libs, no disk), so it's fully unit-tested; `train.py` writes
the config and runs the command. `from_persona` seeds a config from a `Persona` (base model,
ids, output path) so the CLI needs only a persona id + a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import Persona

VALID_TRAIN_BACKENDS = ("mac", "cuda")


class TrainConfigError(ValueError):
    """A training config is missing, malformed, or fails validation."""


class LoRATrainConfig(BaseModel):
    """Everything needed to launch one persona LoRA fine-tune, backend-agnostic."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    base_model: str
    backend: str = "mac"  # "mac" -> mlx-lm; "cuda" -> LLaMA-Factory QLoRA
    # mac: a directory holding train.jsonl/valid.jsonl. cuda: the dataset_info.json dir.
    data_dir: str
    adapter_dir: str  # where the trained adapter is written
    # LoRA hyperparameters (shared vocabulary; mapped per backend below).
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    num_layers: int = 16  # how many top transformer layers get adapters (-1 = all)
    # Optimization.
    learning_rate: float = 1e-4
    batch_size: int = 4
    iters: int = 600  # mac: training iterations
    epochs: float = 3.0  # cuda: training epochs
    max_seq_len: int = 2048
    steps_per_report: int = 10
    save_every: int = 100
    seed: int = 0
    # CUDA-only.
    quantize_4bit: bool = True  # QLoRA (bitsandbytes 4-bit) to fit the 4080
    template: str = "qwen"  # LLaMA-Factory chat template (qwen / llama3 / ...)
    # Free-form passthrough merged into the generated trainer config (escape hatch).
    extra: dict[str, Any] = Field(default_factory=dict)

    def validate_backend(self) -> None:
        if self.backend not in VALID_TRAIN_BACKENDS:
            raise TrainConfigError(
                f"invalid backend {self.backend!r}; expected one of {VALID_TRAIN_BACKENDS}"
            )


@dataclass
class TrainPlan:
    """A ready-to-run training invocation for one backend."""

    tool: str  # "mlx_lm" | "llama_factory"
    config_filename: str  # where train.py should write `config`
    config: dict[str, Any]  # the trainer's config file contents
    command: list[str]  # argv to launch (references the written config)


# --------------------------------------------------------------------------------------
# mlx-lm (Mac)
# --------------------------------------------------------------------------------------


def mlx_lm_config(cfg: LoRATrainConfig) -> dict[str, Any]:
    """The YAML mlx-lm's `lora` subcommand reads (`-c config.yaml`)."""
    config: dict[str, Any] = {
        "model": cfg.base_model,
        "train": True,
        "fine_tune_type": "lora",
        "data": cfg.data_dir,
        "adapter_path": cfg.adapter_dir,
        "num_layers": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "iters": cfg.iters,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_len,
        "steps_per_report": cfg.steps_per_report,
        "save_every": cfg.save_every,
        "seed": cfg.seed,
        # mlx-lm's `scale` is the LoRA alpha multiplier; rank/dropout map directly.
        "lora_parameters": {"rank": cfg.rank, "scale": cfg.alpha, "dropout": cfg.dropout},
    }
    config.update(cfg.extra)
    return config


def mlx_lm_command(config_path: str | Path) -> list[str]:
    return ["python", "-m", "mlx_lm", "lora", "-c", str(config_path)]


# --------------------------------------------------------------------------------------
# LLaMA-Factory (CUDA / 4080, QLoRA)
# --------------------------------------------------------------------------------------


def llama_factory_config(cfg: LoRATrainConfig) -> dict[str, Any]:
    """The YAML `llamafactory-cli train` reads. `dataset` names an entry the user registers
    in `<data_dir>/dataset_info.json` pointing at the ShareGPT file (see the training README)."""
    config: dict[str, Any] = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": cfg.base_model,
        "finetuning_type": "lora",
        "lora_rank": cfg.rank,
        "lora_alpha": int(cfg.alpha),
        "lora_dropout": cfg.dropout,
        "lora_target": "all",
        "dataset": f"{cfg.persona_id}_persona",
        "dataset_dir": cfg.data_dir,
        "template": cfg.template,
        "cutoff_len": cfg.max_seq_len,
        "output_dir": cfg.adapter_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": cfg.steps_per_report,
        "save_steps": cfg.save_every,
        "bf16": True,
        "seed": cfg.seed,
    }
    if cfg.quantize_4bit:
        config["quantization_bit"] = 4
        config["quantization_method"] = "bnb"
    config.update(cfg.extra)
    return config


def llama_factory_command(config_path: str | Path) -> list[str]:
    return ["llamafactory-cli", "train", str(config_path)]


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def build_train_plan(cfg: LoRATrainConfig) -> TrainPlan:
    """Turn a `LoRATrainConfig` into the backend-specific runnable plan."""
    cfg.validate_backend()
    name = f"{cfg.persona_id}_lora.{cfg.backend}.yaml"
    config_path = str(Path(cfg.adapter_dir) / name)
    if cfg.backend == "mac":
        return TrainPlan(
            tool="mlx_lm",
            config_filename=config_path,
            config=mlx_lm_config(cfg),
            command=mlx_lm_command(config_path),
        )
    return TrainPlan(
        tool="llama_factory",
        config_filename=config_path,
        config=llama_factory_config(cfg),
        command=llama_factory_command(config_path),
    )


# --------------------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------------------


def from_persona(
    persona: Persona,
    *,
    backend: str,
    data_dir: str | Path,
    adapter_dir: str | Path,
    base_model: str | None = None,
    **overrides: Any,
) -> LoRATrainConfig:
    """Seed a config from a persona (base model, ids, output path) + overrides."""
    fields: dict[str, Any] = {
        "persona_id": persona.id,
        "base_model": base_model or persona.llm.base_model,
        "backend": backend,
        "data_dir": str(data_dir),
        "adapter_dir": str(adapter_dir),
    }
    fields.update(overrides)
    return LoRATrainConfig(**fields)


def load_config(path: str | Path) -> LoRATrainConfig:
    """Load a `LoRATrainConfig` from YAML (training/persona_lora/configs/*.yaml)."""
    path = Path(path)
    if not path.is_file():
        raise TrainConfigError(f"training config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise TrainConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return LoRATrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise TrainConfigError(f"{path}: invalid training config: {exc}") from exc
