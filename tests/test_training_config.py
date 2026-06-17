"""M7 training config + plan builders (mlx-lm / LLaMA-Factory)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personavoice.training.config import (
    LoRATrainConfig,
    TrainConfigError,
    build_train_plan,
    from_persona,
    llama_factory_config,
    load_config,
    mlx_lm_config,
)

from .fakes import make_persona


def _cfg(**overrides: object) -> LoRATrainConfig:
    base = {
        "persona_id": "hr_interviewer",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "data_dir": "data",
        "adapter_dir": "out/adapters/hr",
    }
    base.update(overrides)
    return LoRATrainConfig(**base)  # type: ignore[arg-type]


def test_mlx_lm_config_maps_hyperparams() -> None:
    cfg = _cfg(backend="mac", rank=8, alpha=16.0, dropout=0.05, iters=300, num_layers=12)
    out = mlx_lm_config(cfg)
    assert out["train"] is True
    assert out["data"] == "data"
    assert out["adapter_path"] == "out/adapters/hr"
    assert out["iters"] == 300
    assert out["num_layers"] == 12
    assert out["lora_parameters"] == {"rank": 8, "scale": 16.0, "dropout": 0.05}


def test_llama_factory_config_qlora_and_dataset_name() -> None:
    cfg = _cfg(backend="cuda", rank=16, alpha=32.0, epochs=2.0, quantize_4bit=True)
    out = llama_factory_config(cfg)
    assert out["finetuning_type"] == "lora"
    assert out["lora_rank"] == 16
    assert out["lora_alpha"] == 32
    assert out["num_train_epochs"] == 2.0
    assert out["dataset"] == "hr_interviewer_persona"
    assert out["quantization_bit"] == 4


def test_llama_factory_config_no_quant_when_disabled() -> None:
    out = llama_factory_config(_cfg(backend="cuda", quantize_4bit=False))
    assert "quantization_bit" not in out


def test_extra_passthrough_merges() -> None:
    out = mlx_lm_config(_cfg(backend="mac", extra={"grad_checkpoint": True}))
    assert out["grad_checkpoint"] is True


def test_build_plan_mac_uses_mlx() -> None:
    plan = build_train_plan(_cfg(backend="mac"))
    assert plan.tool == "mlx_lm"
    assert plan.command[:4] == ["python", "-m", "mlx_lm", "lora"]
    assert plan.config_filename.endswith("hr_interviewer_lora.mac.yaml")
    assert plan.config_filename in plan.command


def test_build_plan_cuda_uses_llama_factory() -> None:
    plan = build_train_plan(_cfg(backend="cuda"))
    assert plan.tool == "llama_factory"
    assert plan.command[0] == "llamafactory-cli"
    assert plan.config_filename.endswith("hr_interviewer_lora.cuda.yaml")


def test_build_plan_rejects_bad_backend() -> None:
    with pytest.raises(TrainConfigError, match="invalid backend"):
        build_train_plan(_cfg(backend="tpu"))


def test_from_persona_seeds_base_model_and_ids() -> None:
    persona = make_persona("companion")
    cfg = from_persona(persona, backend="mac", data_dir="d", adapter_dir="a", iters=99)
    assert cfg.persona_id == "companion"
    assert cfg.base_model == "test-model"  # from persona.llm.base_model
    assert cfg.iters == 99


def test_from_persona_base_model_override() -> None:
    cfg = from_persona(
        make_persona(), backend="mac", data_dir="d", adapter_dir="a", base_model="other"
    )
    assert cfg.base_model == "other"


def test_load_config_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg(backend="mac", iters=42)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    loaded = load_config(path)
    assert loaded.iters == 42 and loaded.backend == "mac"


def test_load_config_missing_file() -> None:
    with pytest.raises(TrainConfigError, match="not found"):
        load_config("/nonexistent/cfg.yaml")


def test_shipped_configs_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "training/persona_lora/configs"
    for name, backend in (("hr_interviewer.mac.yaml", "mac"), ("hr_interviewer.cuda.yaml", "cuda")):
        cfg = load_config(cfg_dir / name)
        assert cfg.backend == backend
        build_train_plan(cfg)  # must produce a runnable plan
