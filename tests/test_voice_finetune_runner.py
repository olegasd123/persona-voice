"""Voice fine-tune runner: prepare, train launch, dry-run, failures."""

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


def _cfg(tmp_path: Path, *, engine: str = "chatterbox") -> VoiceTrainConfig:
    return VoiceTrainConfig(
        voice_name="my_voice",
        engine=engine,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
    )


def test_prepare_chatterbox_writes_config(tmp_path: Path) -> None:
    plan = prepare(_cfg(tmp_path))
    written = Path(plan.config_filename)
    assert written.is_file()
    loaded = yaml.safe_load(written.read_text())
    assert loaded["voice_name"] == "my_voice"
    assert loaded["metadata"].endswith("metadata.csv")


def test_dry_run_launches_nothing(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    run_finetune(_cfg(tmp_path), dry_run=True, runner=lambda c: calls.append(c) or 0)
    assert calls == []


def test_run_launches_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    calls: list[list[str]] = []
    run_finetune(_cfg(tmp_path), runner=lambda c: calls.append(c) or 0)
    assert len(calls) == 1
    assert calls[0][:2] == ["python", "training/voice/chatterbox_finetune.py"]


def test_trainer_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: True)
    with pytest.raises(VoiceTrainingError, match="trainer exited with code 2"):
        run_finetune(_cfg(tmp_path), runner=lambda c: 2)


def test_missing_tool_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: False)
    with pytest.raises(VoiceTrainingError, match="trainer script"):
        run_finetune(_cfg(tmp_path), runner=lambda c: 0)


def test_chatterbox_missing_script_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ft_mod, "_tool_available", lambda plan: False)
    with pytest.raises(VoiceTrainingError, match="Chatterbox trainer script"):
        run_finetune(_cfg(tmp_path, engine="chatterbox"), runner=lambda c: 0)
