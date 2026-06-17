"""Run a persona LoRA fine-tune (M7).

`prepare` is pure-ish: it builds the backend `TrainPlan` and writes the trainer config file,
returning the config path + command without launching anything (so `--dry-run` and tests stop
here). `run_training` then shells out to the trainer (`mlx_lm` on Mac, `llamafactory-cli` on
CUDA) via an injectable `runner`, so the heavy, machine-specific step is the only thing that
touches a GPU / training lib — everything above it is unit-tested offline.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

from .config import LoRATrainConfig, TrainPlan, build_train_plan

logger = logging.getLogger("personavoice.training.train")

# argv -> return code. Defaults to a real subprocess; tests inject a fake.
Runner = Callable[[list[str]], int]

_TOOL_EXECUTABLE = {"mlx_lm": "python", "llama_factory": "llamafactory-cli"}


class TrainingError(RuntimeError):
    """The trainer is unavailable or exited non-zero."""


def _default_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def prepare(cfg: LoRATrainConfig) -> TrainPlan:
    """Build the plan and write its trainer config to disk; do not launch the trainer."""
    plan = build_train_plan(cfg)
    config_path = Path(plan.config_filename)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(plan.config, sort_keys=False))
    logger.info("wrote %s trainer config -> %s", plan.tool, config_path)
    return plan


def _tool_available(plan: TrainPlan) -> bool:
    if plan.tool == "mlx_lm":
        import importlib.util

        return importlib.util.find_spec("mlx_lm") is not None
    return shutil.which(_TOOL_EXECUTABLE.get(plan.tool, plan.tool)) is not None


def run_training(
    cfg: LoRATrainConfig,
    *,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> TrainPlan:
    """Prepare the config and (unless `dry_run`) launch the trainer.

    Raises `TrainingError` if the backend trainer isn't installed, or it exits non-zero.
    """
    plan = prepare(cfg)
    if dry_run:
        logger.info("dry run; not launching: %s", " ".join(plan.command))
        return plan

    if not _tool_available(plan):
        raise TrainingError(
            f"the {plan.tool} trainer is not available. "
            + (
                "install the Mac LoRA extra: `pip install -e '.[train]'`"
                if plan.tool == "mlx_lm"
                else "install LLaMA-Factory in the CUDA training image (see the training README)"
            )
        )

    run = runner or _default_runner
    logger.info("launching: %s", " ".join(plan.command))
    code = run(plan.command)
    if code != 0:
        raise TrainingError(f"{plan.tool} exited with code {code}")
    return plan
