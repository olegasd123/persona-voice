"""Persona fine-tuning (LoRA) stack (M7).

Trains per-persona "brains" beyond prompting. The pieces mirror the cascade's discipline —
light, pure, unit-testable logic here; the heavy trainers (mlx-lm on Mac, LLaMA-Factory /
Unsloth QLoRA on CUDA) are shelled out to and lazy-checked, so this package imports and tests
on any machine without a GPU or training libs installed.

- `dataset`  — the in-character dialogue format (OpenAI `messages` JSONL) + converters to the
  mlx-lm "chat" and ShareGPT (LLaMA-Factory) shapes the trainers consume.
- `curate`   — synthesize in-character dialogues by self-chat (persona LLM vs a user simulator).
- `config`   — `LoRATrainConfig` + builders that turn it into an mlx-lm / LLaMA-Factory plan.
- `train`    — write the trainer config and run it (backend-dispatched; `--dry-run` to preview).
- `merge`    — fuse a trained adapter back into base weights for serving.
- `eval`     — heuristic persona-adherence scoring to compare prompt-only vs LoRA.
"""

from __future__ import annotations

from .config import (
    LoRATrainConfig,
    TrainPlan,
    build_train_plan,
    from_persona,
    load_config,
)
from .dataset import (
    DATASET_FORMATS,
    DatasetError,
    DialogueExample,
    read_jsonl,
    render_chatml,
    split_examples,
    to_mlx_chat,
    to_sharegpt,
    validate_example,
    write_dataset,
    write_jsonl,
)
from .eval import (
    EvalScores,
    ReplyMetrics,
    compare,
    score_replies,
    score_reply,
)

__all__ = [
    "DATASET_FORMATS",
    "DatasetError",
    "DialogueExample",
    "EvalScores",
    "LoRATrainConfig",
    "ReplyMetrics",
    "TrainPlan",
    "build_train_plan",
    "compare",
    "from_persona",
    "load_config",
    "read_jsonl",
    "render_chatml",
    "score_replies",
    "score_reply",
    "split_examples",
    "to_mlx_chat",
    "to_sharegpt",
    "validate_example",
    "write_dataset",
    "write_jsonl",
]
