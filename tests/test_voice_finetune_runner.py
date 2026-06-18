"""Voice fine-tune runner: prepare, prepare+train ordering, dry-run, failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice.training.voice import finetune as ft_mod
from personavoice.training.voice.config import VoiceTrainConfig
from personavoice.training.voice.finetune import (
    VoiceTrainingError,
    prepare,
    run_finetune,
)


def _cfg(tmp_path: Path, *, engine: str = "f5") -> VoiceTrainConfig:
    return VoiceTrainConfig(
        voice_name="my_voice",
        engine=engine,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
    )


def test_prepare_f5_writes_no_config(tmp_path: Path) -> None:
    plan = prepare(_cfg(tmp_path, engine="f5"))
    assert plan.config_filename is None
    assert plan.prepare_command is not None


def test_prepare_chatterbox_writes_config(tmp_path: Path) -> None:
    plan = prepare(_cfg(tmp_path, engine="chatterbox"))
    written = Path(plan.config_filename)
    assert written.is_file()
    loaded = yaml.safe_load(written.read_text())
    assert loaded["voice_name"] == "my_voice"
    assert loaded["metadata"].endswith("metadata.csv")


def test_dry_run_launches_nothing(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    run_finetune(_cfg(tmp_path), dry_run=True, runner=lambda c: calls.append(c) or 0)
    assert calls == []


def test_run_prepares_then_trains_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    calls: list[list[str]] = []
    run_finetune(_cfg(tmp_path, engine="f5"), runner=lambda c: calls.append(c) or 0)
    # F5: dataset prep runs first, then the trainer.
    assert calls[0][0] == "f5-tts_prepare_csv_wavs"
    assert calls[1][0] == "f5-tts_finetune-cli"


def test_skip_prepare_runs_only_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    calls: list[list[str]] = []
    run_finetune(
        _cfg(tmp_path, engine="f5"), skip_prepare=True, runner=lambda c: calls.append(c) or 0
    )
    assert len(calls) == 1 and calls[0][0] == "f5-tts_finetune-cli"


def test_prepare_failure_aborts_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    calls: list[list[str]] = []

    def runner(c: list[str]) -> int:
        calls.append(c)
        return 3 if c[0] == "f5-tts_prepare_csv_wavs" else 0

    with pytest.raises(VoiceTrainingError, match="dataset prep exited with code 3"):
        run_finetune(_cfg(tmp_path, engine="f5"), runner=runner)
    assert len(calls) == 1  # never reached the trainer


def test_trainer_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    with pytest.raises(VoiceTrainingError, match="trainer exited with code 2"):
        run_finetune(_cfg(tmp_path), skip_prepare=True, runner=lambda c: 2)


def test_missing_tool_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: False)
    with pytest.raises(VoiceTrainingError, match="not available"):
        run_finetune(_cfg(tmp_path), runner=lambda c: 0)


def test_chatterbox_missing_script_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: False)
    with pytest.raises(VoiceTrainingError, match="Chatterbox trainer script"):
        run_finetune(_cfg(tmp_path, engine="chatterbox"), runner=lambda c: 0)
