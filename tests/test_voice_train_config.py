"""Voice fine-tune config + plan builders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice.training.voice.config import (
    VoiceTrainConfig,
    VoiceTrainConfigError,
    build_voice_train_plan,
    chatterbox_config,
    from_voice,
    load_config,
)


def _cfg(**overrides: object) -> VoiceTrainConfig:
    base = {
        "voice_name": "my_voice",
        "data_dir": "training/voice/datasets/my_voice",
        "output_dir": "models/finetuned/my_voice",
    }
    base.update(overrides)
    return VoiceTrainConfig(**base)  # type: ignore[arg-type]


def test_chatterbox_config_maps_fields() -> None:
    out = chatterbox_config(_cfg(engine="chatterbox", base_model="base/ckpt", lora=True, epochs=30))
    assert out["base_checkpoint"] == "base/ckpt"
    assert out["metadata"].endswith("metadata.csv")
    assert out["use_lora"] is True
    assert out["epochs"] == 30


def test_build_plan_chatterbox_writes_config() -> None:
    plan = build_voice_train_plan(_cfg(engine="chatterbox"))
    assert plan.engine == "chatterbox"
    assert plan.config is not None
    assert plan.config_filename.endswith("my_voice_finetune.chatterbox.yaml")
    assert plan.command[:2] == ["python", "training/voice/chatterbox_finetune.py"]


def test_build_plan_rejects_bad_engine() -> None:
    with pytest.raises(VoiceTrainConfigError, match="invalid engine"):
        build_voice_train_plan(_cfg(engine="tortoise"))


def test_from_voice_seeds_fields() -> None:
    cfg = from_voice("v", engine="chatterbox", data_dir="d", output_dir="o", epochs=10)
    assert cfg.voice_name == "v" and cfg.engine == "chatterbox"
    assert cfg.data_dir == "d" and cfg.output_dir == "o"
    assert cfg.epochs == 10


def test_load_config_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg(engine="chatterbox", epochs=22)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    loaded = load_config(path)
    assert loaded.epochs == 22 and loaded.engine == "chatterbox"


def test_load_config_missing_file() -> None:
    with pytest.raises(VoiceTrainConfigError, match="not found"):
        load_config("/nonexistent/cfg.yaml")


def test_shipped_configs_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "training/voice/configs"
    cfg = load_config(cfg_dir / "my_voice.chatterbox.yaml")
    assert cfg.engine == "chatterbox"
    build_voice_train_plan(cfg)  # must produce a runnable plan
