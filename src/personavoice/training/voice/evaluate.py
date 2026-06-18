"""Voice-quality A/B: fine-tuned voice vs zero-shot clone.

The acceptance bar is "a fine-tuned voice is clearly higher fidelity than its zero-shot clone."
The objective proxy is **speaker similarity**: synthesize the same probe lines with each voice,
embed them with a speaker encoder, and measure cosine similarity to held-out *real* clips of the
target speaker. The voice whose synthesis lands closer to the real speaker is the more faithful.

`cosine_similarity` / `mean_similarity` / `score_ab` are pure (vectors in, scores out) and fully
unit-tested with toy embeddings. `evaluate_voices` orchestrates a real A/B over an injected TTS
adapter + `Embedder`, so the CLI runs it on a GPU while tests drive it with fakes. The only heavy
piece is `ResemblyzerEmbedder`, lazy-imported behind the `voice-eval` extra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ...models import VoiceRef

Vector = Sequence[float]


class VoiceEvalError(ValueError):
    """The A/B can't run (no target clips, mismatched embeddings, …)."""


@runtime_checkable
class Embedder(Protocol):
    """Turns WAV bytes into a fixed-length speaker embedding."""

    def embed(self, wav: bytes) -> list[float]: ...


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is all-zero)."""
    if len(a) != len(b):
        raise VoiceEvalError(f"embedding length mismatch ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mean_similarity(candidates: Sequence[Vector], references: Sequence[Vector]) -> float:
    """Mean over `candidates` of each one's average cosine to all `references`."""
    if not candidates or not references:
        raise VoiceEvalError("need at least one candidate and one reference embedding")
    per_candidate = [
        sum(cosine_similarity(c, r) for r in references) / len(references) for c in candidates
    ]
    return sum(per_candidate) / len(per_candidate)


class VoiceEvalScores(BaseModel):
    """A/B result: how close each voice's synthesis sits to the real target speaker."""

    n_probes: int
    n_references: int
    finetuned_similarity: float
    clone_similarity: float
    delta: float  # finetuned - clone (positive = fine-tune is more faithful)
    margin: float  # the delta a "clear win" must clear
    passes: bool  # delta >= margin (fine-tune clearly closer to the target)


def score_ab(
    finetuned_embs: Sequence[Vector],
    clone_embs: Sequence[Vector],
    target_embs: Sequence[Vector],
    *,
    margin: float = 0.0,
) -> VoiceEvalScores:
    """Score a fine-tuned vs zero-shot-clone A/B against the target-speaker embeddings."""
    if not target_embs:
        raise VoiceEvalError("no target-speaker reference embeddings to compare against")
    ft = mean_similarity(finetuned_embs, target_embs)
    cl = mean_similarity(clone_embs, target_embs)
    delta = ft - cl
    return VoiceEvalScores(
        n_probes=len(finetuned_embs),
        n_references=len(target_embs),
        finetuned_similarity=ft,
        clone_similarity=cl,
        delta=delta,
        margin=margin,
        passes=delta >= margin,
    )


async def _synth_embeddings(
    tts: object, probes: Sequence[str], voice: VoiceRef, embedder: Embedder
) -> list[list[float]]:
    embs: list[list[float]] = []
    for probe in probes:
        wav = await tts.synthesize(probe, voice)  # type: ignore[attr-defined]
        embs.append(embedder.embed(wav))
    return embs


async def evaluate_voices(
    tts: object,
    *,
    probes: Sequence[str],
    finetuned_voice: VoiceRef,
    clone_voice: VoiceRef,
    target_clips: Sequence[bytes],
    embedder: Embedder,
    margin: float = 0.0,
) -> VoiceEvalScores:
    """Run the full A/B: synthesize each probe with both voices, embed, and score.

    `tts` is a `TTSAdapter` (typed loosely). `target_clips` are real WAV clips of the target
    speaker held out from training. Returns the `VoiceEvalScores`.
    """
    if not probes:
        raise VoiceEvalError("no probe lines to synthesize")
    if not target_clips:
        raise VoiceEvalError("no target-speaker clips to compare against")
    target_embs = [embedder.embed(clip) for clip in target_clips]
    finetuned_embs = await _synth_embeddings(tts, probes, finetuned_voice, embedder)
    clone_embs = await _synth_embeddings(tts, probes, clone_voice, embedder)
    return score_ab(finetuned_embs, clone_embs, target_embs, margin=margin)


class ResemblyzerEmbedder:
    """Speaker embeddings via Resemblyzer's `VoiceEncoder` (lazy; `voice-eval` extra)."""

    def __init__(self) -> None:
        self._encoder: object | None = None

    def _get_encoder(self) -> object:
        if self._encoder is None:
            try:
                from resemblyzer import VoiceEncoder
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise RuntimeError(
                    "resemblyzer is not installed; install the voice-eval extra: "
                    "`pip install -e '.[voice-eval]'`"
                ) from exc
            self._encoder = VoiceEncoder()
        return self._encoder

    def embed(self, wav: bytes) -> list[float]:
        import numpy as np
        from resemblyzer import preprocess_wav

        from ...audio import decode_wav

        samples, sr = decode_wav(wav)
        # Resemblyzer wants a float32 mono utterance at its own sampling rate; `preprocess_wav`
        # resamples/normalizes when given the source rate.
        processed = preprocess_wav(np.asarray(samples, dtype=np.float32), source_sr=sr)
        encoder = self._get_encoder()
        embedding = encoder.embed_utterance(processed)  # type: ignore[attr-defined]
        return [float(x) for x in np.asarray(embedding).reshape(-1)]
