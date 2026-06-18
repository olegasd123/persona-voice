"""Run a voice fine-tune.

`prepare` is pure-ish: it builds the engine `VoiceTrainPlan` and writes any trainer config
file, returning the plan without launching anything (so `--dry-run` and tests stop here).
`run_finetune` then shells out — first the dataset-prep step (F5 builds its arrow dataset),
then the trainer — via an injectable `runner`, so the heavy, GPU-bound step is the only thing
that touches a training lib; everything above it is unit-tested offline.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

from .config import VoiceTrainConfig, VoiceTrainPlan, build_voice_train_plan

logger = logging.getLogger("personavoice.training.voice.finetune")

# argv -> return code. Defaults to a real subprocess; tests inject a fake.
Runner = Callable[[list[str]], int]


class VoiceTrainingError(RuntimeError):
    """The voice trainer is unavailable or exited non-zero."""


def _default_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def prepare(cfg: VoiceTrainConfig) -> VoiceTrainPlan:
    """Build the plan and write any trainer config to disk; do not launch the trainer."""
    plan = build_voice_train_plan(cfg)
    if plan.config is not None and plan.config_filename is not None:
        config_path = Path(plan.config_filename)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(plan.config, sort_keys=False), encoding="utf-8")
        logger.info("wrote %s trainer config -> %s", plan.engine, config_path)
    return plan


def _tool_available(plan: VoiceTrainPlan) -> bool:
    if plan.engine == "chatterbox":
        # The trainer is a script path; available if it exists or is on PATH.
        return Path(plan.tool).is_file() or shutil.which(plan.tool) is not None
    return shutil.which(plan.tool) is not None


def _unavailable_message(plan: VoiceTrainPlan) -> str:
    if plan.engine == "f5":
        return (
            "the f5-tts_finetune-cli trainer is not available; install F5-TTS in the CUDA "
            "training image (`pip install f5-tts`) — see training/voice/README.md"
        )
    return (
        f"the Chatterbox trainer script {plan.tool!r} was not found; point `trainer_script` at "
        "the community Chatterbox finetune trainer — see training/voice/README.md"
    )


def run_finetune(
    cfg: VoiceTrainConfig,
    *,
    dry_run: bool = False,
    skip_prepare: bool = False,
    runner: Runner | None = None,
) -> VoiceTrainPlan:
    """Prepare the dataset + config and (unless `dry_run`) launch the fine-tune.

    Runs `plan.prepare_command` first (F5's dataset build) unless `skip_prepare`. Raises
    `VoiceTrainingError` if the trainer isn't installed, or any step exits non-zero.
    """
    plan = prepare(cfg)
    if dry_run:
        if plan.prepare_command and not skip_prepare:
            logger.info("dry run; would prepare: %s", " ".join(plan.prepare_command))
        logger.info("dry run; not launching: %s", " ".join(plan.command))
        return plan

    if not _tool_available(plan):
        raise VoiceTrainingError(_unavailable_message(plan))

    run = runner or _default_runner
    if plan.prepare_command and not skip_prepare:
        logger.info("preparing dataset: %s", " ".join(plan.prepare_command))
        code = run(plan.prepare_command)
        if code != 0:
            raise VoiceTrainingError(f"dataset prep exited with code {code}")

    logger.info("launching: %s", " ".join(plan.command))
    code = run(plan.command)
    if code != 0:
        raise VoiceTrainingError(f"{plan.engine} trainer exited with code {code}")
    return plan
