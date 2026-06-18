"""Voice fine-tune config + plan builders (F5 / Chatterbox)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice.training.voice.config import (
    VoiceTrainConfig,
    VoiceTrainConfigError,
    build_voice_train_plan,
    chatterbox_config,
    f5_finetune_command,
    f5_prepare_command,
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


def test_f5_finetune_command_carries_hyperparams() -> None:
    cmd = f5_finetune_command(_cfg(engine="f5", epochs=40, learning_rate=2e-5, batch_size=6))
    assert cmd[0] == "f5-tts_finetune-cli"
    assert "--exp_name" in cmd and "F5TTS_v1_Base" in cmd
    assert cmd[cmd.index("--dataset_name") + 1] == "my_voice"
    assert cmd[cmd.index("--epochs") + 1] == "40"
    assert cmd[cmd.index("--batch_size_per_gpu") + 1] == "6"
    assert "--finetune" in cmd


def test_f5_extra_passthrough_renders_argv() -> None:
    cmd = f5_finetune_command(_cfg(engine="f5", extra={"logger": "wandb", "bnb_optimizer": True}))
    assert cmd[cmd.index("--logger") + 1] == "wandb"
    assert "--bnb_optimizer" in cmd  # bare flag for True


def test_f5_prepare_command_targets_dataset_dir() -> None:
    cmd = f5_prepare_command(_cfg(engine="f5"))
    assert cmd[0] == "f5-tts_prepare_csv_wavs"
    assert cmd[1] == "training/voice/datasets/my_voice"
    assert cmd[2].endswith("my_voice_prepared")


def test_dataset_name_defaults_to_voice_name() -> None:
    assert _cfg().resolved_dataset_name == "my_voice"
    assert _cfg(dataset_name="custom").resolved_dataset_name == "custom"


def test_chatterbox_config_maps_fields() -> None:
    out = chatterbox_config(_cfg(engine="chatterbox", base_model="base/ckpt", lora=True, epochs=30))
    assert out["base_checkpoint"] == "base/ckpt"
    assert out["metadata"].endswith("metadata.csv")
    assert out["use_lora"] is True
    assert out["epochs"] == 30


def test_build_plan_f5_has_prepare_and_no_config() -> None:
    plan = build_voice_train_plan(_cfg(engine="f5"))
    assert plan.engine == "f5"
    assert plan.tool == "f5-tts_finetune-cli"
    assert plan.prepare_command is not None
    assert plan.config is None and plan.config_filename is None


def test_build_plan_chatterbox_writes_config_no_prepare() -> None:
    plan = build_voice_train_plan(_cfg(engine="chatterbox"))
    assert plan.engine == "chatterbox"
    assert plan.prepare_command is None
    assert plan.config is not None
    assert plan.config_filename.endswith("my_voice_finetune.chatterbox.yaml")
    assert plan.command[:2] == ["python", "training/voice/chatterbox_finetune.py"]


def test_build_plan_rejects_bad_engine() -> None:
    with pytest.raises(VoiceTrainConfigError, match="invalid engine"):
        build_voice_train_plan(_cfg(engine="tortoise"))


def test_from_voice_seeds_fields() -> None:
    cfg = from_voice("v", engine="f5", data_dir="d", output_dir="o", epochs=10)
    assert cfg.voice_name == "v" and cfg.engine == "f5"
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
    for name, engine in (("my_voice.f5.yaml", "f5"), ("my_voice.chatterbox.yaml", "chatterbox")):
        cfg = load_config(cfg_dir / name)
        assert cfg.engine == engine
        build_voice_train_plan(cfg)  # must produce a runnable plan
