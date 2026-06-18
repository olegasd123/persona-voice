"""Fuse a trained LoRA adapter back into base weights.

Serving is simplest with a single merged checkpoint (vLLM / mlx-lm load it like any model),
so after training we fuse the adapter:

  - **mac**  → `mlx_lm fuse --model <base> --adapter-path <adapter> --save-path <out>`.
  - **cuda** → `llamafactory-cli export <export-config>.yaml`.

(vLLM can *also* hot-load the unmerged adapter via `--lora-modules name=path`; the inference
side wires that — see `adapters/llm/_openai_compat`. Merging is the alternative when you want a
standalone model or a different server.) Command/config building is pure; `run_merge` shells out
through an injectable runner.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("personavoice.training.merge")

Runner = Callable[[list[str]], int]


class MergeError(RuntimeError):
    """The merge tool is unavailable or exited non-zero."""


def mlx_fuse_command(base_model: str, adapter_dir: str | Path, save_path: str | Path) -> list[str]:
    return [
        "python",
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        base_model,
        "--adapter-path",
        str(adapter_dir),
        "--save-path",
        str(save_path),
    ]


def llama_factory_export_config(
    base_model: str,
    adapter_dir: str | Path,
    export_dir: str | Path,
    *,
    template: str = "qwen",
) -> dict[str, Any]:
    return {
        "model_name_or_path": base_model,
        "adapter_name_or_path": str(adapter_dir),
        "template": template,
        "finetuning_type": "lora",
        "export_dir": str(export_dir),
        "export_size": 2,
        "export_legacy_format": False,
    }


def llama_factory_export_command(config_path: str | Path) -> list[str]:
    return ["llamafactory-cli", "export", str(config_path)]


def _default_runner(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def run_merge(
    *,
    backend: str,
    base_model: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    template: str = "qwen",
    dry_run: bool = False,
    runner: Runner | None = None,
) -> list[str]:
    """Fuse `adapter_dir` into `base_model`, writing the merged model to `out_dir`.

    Returns the command that was (or would be) run. Raises `MergeError` if the tool is missing
    or exits non-zero.
    """
    if backend == "mac":
        command = mlx_fuse_command(base_model, adapter_dir, out_dir)
        tool, available = "mlx_lm", _mlx_available()
    elif backend == "cuda":
        config = llama_factory_export_config(base_model, adapter_dir, out_dir, template=template)
        config_path = Path(out_dir).with_suffix(".export.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = llama_factory_export_command(config_path)
        tool, available = "llamafactory-cli", shutil.which("llamafactory-cli") is not None
    else:
        raise MergeError(f"invalid backend {backend!r}; expected 'mac' or 'cuda'")

    if dry_run:
        logger.info("dry run; not launching: %s", " ".join(command))
        return command
    if not available:
        raise MergeError(
            f"the {tool} merge tool is not available; install the training extra/image"
        )

    run = runner or _default_runner
    logger.info("launching: %s", " ".join(command))
    code = run(command)
    if code != 0:
        raise MergeError(f"{tool} exited with code {code}")
    return command


def _mlx_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None
