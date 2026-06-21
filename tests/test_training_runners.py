"""Train + merge runners: config writing, command building, dispatch, failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice.training import merge as merge_mod
from personavoice.training import train as train_mod
from personavoice.training.config import LoRATrainConfig
from personavoice.training.merge import (
    MergeError,
    llama_factory_export_config,
    mlx_fuse_command,
    run_merge,
)
from personavoice.training.train import TrainingError, prepare, run_training


def _cfg(tmp_path: Path, *, backend: str = "mac") -> LoRATrainConfig:
    return LoRATrainConfig(
        persona_id="hr_interviewer",
        base_model="some/base",
        backend=backend,
        data_dir=str(tmp_path / "data"),
        adapter_dir=str(tmp_path / "adapter"),
    )


# -- train -----------------------------------------------------------------------------


def test_prepare_writes_trainer_config(tmp_path: Path) -> None:
    plan = prepare(_cfg(tmp_path, backend="mac"))
    written = Path(plan.config_filename)
    assert written.is_file()
    loaded = yaml.safe_load(written.read_text())
    assert loaded["train"] is True and loaded["fine_tune_type"] == "lora"


def test_run_training_dry_run_does_not_launch(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    plan = run_training(_cfg(tmp_path), dry_run=True, runner=lambda c: calls.append(c) or 0)
    assert calls == []
    assert Path(plan.config_filename).is_file()


def test_run_training_invokes_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_mod, "_tool_available", lambda plan: True)
    calls: list[list[str]] = []
    run_training(_cfg(tmp_path), runner=lambda c: calls.append(c) or 0)
    assert calls and calls[0][:4] == ["python", "-m", "mlx_lm", "lora"]


def test_run_training_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_mod, "_tool_available", lambda plan: True)
    with pytest.raises(TrainingError, match="exited with code 2"):
        run_training(_cfg(tmp_path), runner=lambda c: 2)


def test_run_training_missing_tool_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_mod, "_tool_available", lambda plan: False)
    with pytest.raises(TrainingError, match="not available"):
        run_training(_cfg(tmp_path), runner=lambda c: 0)


# -- merge -----------------------------------------------------------------------------


def test_mlx_fuse_command_shape() -> None:
    cmd = mlx_fuse_command("base", "adapter", "out")
    assert cmd[:4] == ["python", "-m", "mlx_lm", "fuse"]
    assert "--save-path" in cmd and "out" in cmd


def test_llama_factory_export_config_has_adapter() -> None:
    cfg = llama_factory_export_config("base", "adapter", "out", template="qwen")
    assert cfg["adapter_name_or_path"] == "adapter"
    assert cfg["export_dir"] == "out"


def test_run_merge_mac_dry_run(tmp_path: Path) -> None:
    cmd = run_merge(
        backend="mac",
        base_model="base",
        adapter_dir=tmp_path / "adapter",
        out_dir=tmp_path / "merged",
        dry_run=True,
    )
    assert cmd[:4] == ["python", "-m", "mlx_lm", "fuse"]


def test_run_merge_cuda_writes_export_config(tmp_path: Path) -> None:
    out = tmp_path / "merged"
    cmd = run_merge(
        backend="cuda",
        base_model="base",
        adapter_dir=tmp_path / "adapter",
        out_dir=out,
        dry_run=True,
    )
    assert cmd[0] == "llamafactory-cli" and cmd[1] == "export"
    assert out.with_suffix(".export.yaml").is_file()


def test_run_merge_invokes_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merge_mod, "_mlx_available", lambda: True)
    calls: list[list[str]] = []
    run_merge(
        backend="mac",
        base_model="base",
        adapter_dir=tmp_path / "a",
        out_dir=tmp_path / "o",
        runner=lambda c: calls.append(c) or 0,
    )
    assert calls and calls[0][3] == "fuse"


def test_run_merge_bad_backend(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="invalid backend"):
        run_merge(backend="tpu", base_model="b", adapter_dir=tmp_path, out_dir=tmp_path)
