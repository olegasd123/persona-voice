"""Voice-quality A/B: cosine/mean/score_ab + the evaluate_voices orchestration."""

from __future__ import annotations

from typing import ClassVar

import pytest

from personavoice.adapters.tts.base import TTSAdapter
from personavoice.models import VoiceRef
from personavoice.training.voice.evaluate import (
    VoiceEvalError,
    cosine_similarity,
    evaluate_voices,
    mean_similarity,
    score_ab,
)

# --------------------------------------------------------------------------------------
# pure scoring
# --------------------------------------------------------------------------------------


def test_cosine_identical_and_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises() -> None:
    with pytest.raises(VoiceEvalError, match="length mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_mean_similarity_averages() -> None:
    # Two candidates, one reference: mean of each candidate's cosine to the ref.
    val = mean_similarity([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])
    assert val == pytest.approx(0.5)


def test_mean_similarity_needs_both_sides() -> None:
    with pytest.raises(VoiceEvalError):
        mean_similarity([], [[1.0]])


def test_score_ab_passes_when_finetune_closer() -> None:
    target = [[1.0, 0.0]]
    finetuned = [[0.95, 0.05]]  # close to target
    clone = [[0.6, 0.8]]  # farther
    scores = score_ab(finetuned, clone, target, margin=0.05)
    assert scores.finetuned_similarity > scores.clone_similarity
    assert scores.delta > 0.05
    assert scores.passes is True
    assert scores.n_probes == 1 and scores.n_references == 1


def test_score_ab_fails_within_margin() -> None:
    target = [[1.0, 0.0]]
    scores = score_ab([[0.9, 0.1]], [[0.89, 0.11]], target, margin=0.1)
    assert scores.passes is False  # delta well under the margin


def test_score_ab_no_targets_raises() -> None:
    with pytest.raises(VoiceEvalError, match="no target"):
        score_ab([[1.0]], [[1.0]], [])


# --------------------------------------------------------------------------------------
# evaluate_voices orchestration (fake TTS + fake embedder)
# --------------------------------------------------------------------------------------


class _TagTTS(TTSAdapter):
    """A fake TTS whose output bytes encode *which voice* synthesized them."""

    name = "tag_tts"
    supports_cloning = True

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        return f"synth:{voice.id}".encode()


class _FakeEmbedder:
    """Maps the tagged bytes to fixed embeddings: the fine-tune sits closer to the target."""

    _TABLE: ClassVar[dict[bytes, list[float]]] = {
        b"real:target": [1.0, 0.0, 0.0],
        b"synth:ft": [0.95, 0.05, 0.0],
        b"synth:cl": [0.5, 0.5, 0.0],
    }

    def embed(self, wav: bytes) -> list[float]:
        return self._TABLE[wav]


async def test_evaluate_voices_end_to_end() -> None:
    scores = await evaluate_voices(
        _TagTTS(),
        probes=["one", "two"],
        finetuned_voice=VoiceRef(id="ft", model_path="/ckpt"),
        clone_voice=VoiceRef(id="cl", sample_path="/s.wav"),
        target_clips=[b"real:target", b"real:target"],
        embedder=_FakeEmbedder(),
        margin=0.05,
    )
    assert scores.passes is True
    assert scores.finetuned_similarity > scores.clone_similarity
    assert scores.n_probes == 2 and scores.n_references == 2


async def test_evaluate_voices_requires_probes_and_targets() -> None:
    tts = _TagTTS()
    with pytest.raises(VoiceEvalError, match="no probe"):
        await evaluate_voices(
            tts,
            probes=[],
            finetuned_voice=VoiceRef(id="ft"),
            clone_voice=VoiceRef(id="cl"),
            target_clips=[b"x"],
            embedder=_FakeEmbedder(),
        )
    with pytest.raises(VoiceEvalError, match="no target"):
        await evaluate_voices(
            tts,
            probes=["one"],
            finetuned_voice=VoiceRef(id="ft"),
            clone_voice=VoiceRef(id="cl"),
            target_clips=[],
            embedder=_FakeEmbedder(),
        )
