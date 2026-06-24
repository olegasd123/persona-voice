"""Voice fine-tuning stack.

Trains a high-fidelity voice for a target speaker, beyond the zero-shot clone. Same
discipline as the persona-LoRA stack: light, pure, unit-testable logic here; the heavy
trainers (Chatterbox, CUDA) are shelled out to and lazy-checked, so this package
imports and tests on any machine without a GPU or training libs.

- `dataset`  — the target-speaker dataset (`metadata.csv` of `audio|text`) + validation and
  STT auto-transcription.
- `config`   — `VoiceTrainConfig` + builders that turn it into a Chatterbox plan.
- `finetune` — write any config and run the prep + trainer commands (`--dry-run` to preview).
- `evaluate` — speaker-similarity A/B of the fine-tuned voice vs the zero-shot clone.

A trained voice is folded into the registry via `voice.FinetunedVoicesStore`, which takes
precedence over a clone over a static preset in `VoiceRegistry.resolve_for_persona`.
"""

from __future__ import annotations

from .config import (
    VoiceTrainConfig,
    VoiceTrainConfigError,
    VoiceTrainPlan,
    build_voice_train_plan,
    from_voice,
    load_config,
)
from .dataset import (
    DatasetError,
    DatasetStats,
    SpeakerClip,
    metadata_line,
    parse_metadata_line,
    read_metadata,
    validate_clip,
    validate_dataset,
    write_metadata,
)
from .evaluate import (
    Embedder,
    VoiceEvalError,
    VoiceEvalScores,
    cosine_similarity,
    evaluate_voices,
    mean_similarity,
    score_ab,
)
from .finetune import VoiceTrainingError, prepare, run_finetune

__all__ = [
    "DatasetError",
    "DatasetStats",
    "Embedder",
    "SpeakerClip",
    "VoiceEvalError",
    "VoiceEvalScores",
    "VoiceTrainConfig",
    "VoiceTrainConfigError",
    "VoiceTrainPlan",
    "VoiceTrainingError",
    "build_voice_train_plan",
    "cosine_similarity",
    "evaluate_voices",
    "from_voice",
    "load_config",
    "mean_similarity",
    "metadata_line",
    "parse_metadata_line",
    "prepare",
    "read_metadata",
    "run_finetune",
    "score_ab",
    "validate_clip",
    "validate_dataset",
    "write_metadata",
]
