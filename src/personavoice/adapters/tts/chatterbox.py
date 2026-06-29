"""Chatterbox TTS (CUDA; MPS on Mac) — emotion-exaggeration control, zero-shot cloning.

The clean-license (MIT) cloning backend on CUDA.
Implements one-shot `synthesize` (text -> WAV bytes); `model.generate(text)` returns a
float waveform tensor at the model's native sample rate (`model.sr`, 24 kHz). Cloning adds
zero-shot cloning: when the voice carries a reference sample, we pass it as
`audio_prompt_path` so Chatterbox speaks in that voice. Fine-tuned voices add another path: when the
voice carries a `model_path`, we load that trained checkpoint (`from_local`) instead of the
base weights. `VoiceRef.emotion` drives Chatterbox's `exaggeration` knob (0.5 = neutral,
higher = more emotive). `chatterbox` is imported lazily; models are built once and cached per
checkpoint (the base under the `None` key).
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...models import VoiceRef
from .base import TTSAdapter

_DEFAULT_SAMPLE_RATE = 24000
# Chatterbox's `exaggeration` controls emotional intensity: 0.5 is neutral, higher is more
# expressive. It's the default when a voice has no `emotion`.
_DEFAULT_EXAGGERATION = 0.5
# Chatterbox accepts roughly this range; clamp so a stray value never destabilizes synthesis.
_EXAGGERATION_RANGE = (0.25, 2.0)
# Map the descriptive `emotion` strings personas/voices use to an exaggeration level. A bare
# number ("0.7") is taken literally instead; an unknown word falls back to the default. The
# canonical per-utterance emotions (`personavoice.emotion.EMOTIONS`, used by dynamic emotion
# tags) are all covered here; `tests/test_emotion.py` guards that coupling.
_EMOTION_EXAGGERATION: dict[str, float] = {
    "calm": 0.4,
    "sad": 0.4,
    "serious": 0.45,
    "neutral": 0.5,
    "neutral-warm": 0.55,
    "sympathetic": 0.55,
    "warm": 0.6,
    "friendly": 0.6,
    "kind": 0.6,
    "curious": 0.65,
    "happy": 0.7,
    "expressive": 0.75,
    "excited": 0.8,
}


def _resolve_exaggeration(emotion: str | None, default: float) -> float:
    """Turn a `VoiceRef.emotion` string into Chatterbox's `exaggeration` float.

    Accepts a literal number ("0.7"), a known descriptive word (see `_EMOTION_EXAGGERATION`),
    or falls back to `default`; the result is clamped to `_EXAGGERATION_RANGE`.
    """
    if not emotion:
        return default
    key = emotion.strip().lower()
    try:
        value = float(key)
    except ValueError:
        value = _EMOTION_EXAGGERATION.get(key, default)
    lo, hi = _EXAGGERATION_RANGE
    return max(lo, min(hi, value))


def _waveform_to_wav(waveform: Any, sample_rate: int) -> bytes:
    """Wrap a torch/numpy float waveform (shape [N] or [1, N]) as WAV bytes."""
    import numpy as np

    from ...audio import encode_wav

    if hasattr(waveform, "detach"):  # torch tensor
        waveform = waveform.detach().cpu().numpy()
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return encode_wav(samples, sample_rate)


class ChatterboxTTS(TTSAdapter):
    name = "chatterbox"
    supports_cloning = True
    implemented = True

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(model=model, options=options)
        # Cache one model per checkpoint path; `None` is the base (`from_pretrained`).
        self._models: dict[str | None, object] = {}

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        waveform, sample_rate = await asyncio.to_thread(
            self._synthesize_array, text, voice.sample_path, voice.model_path, voice.emotion
        )
        return _waveform_to_wav(waveform, sample_rate)

    def _get_model(self, model_path: str | None = None) -> object:
        """Load (and cache) the base model, or a fine-tuned checkpoint when `model_path` set."""
        if model_path not in self._models:
            try:
                from chatterbox.tts import ChatterboxTTS as _Chatterbox
            except ImportError as exc:  # pragma: no cover - only without the package installed
                raise RuntimeError(
                    "chatterbox-tts is not installed; install it in the CUDA server image: "
                    "`pip install chatterbox-tts`"
                ) from exc
            device = self.options.get("device", "cuda")
            if model_path:  # a fine-tuned voice checkpoint
                self._models[model_path] = _Chatterbox.from_local(model_path, device=device)
            else:
                self._models[model_path] = _Chatterbox.from_pretrained(device=device)
        return self._models[model_path]

    def _synthesize_array(
        self,
        text: str,
        sample_path: str | None = None,
        model_path: str | None = None,
        emotion: str | None = None,
    ) -> tuple[Any, int]:
        model = self._get_model(model_path)
        exaggeration = _resolve_exaggeration(
            emotion, self.options.get("exaggeration", _DEFAULT_EXAGGERATION)
        )
        # A reference sample → clone that voice; otherwise Chatterbox's built-in default.
        if sample_path:
            waveform = model.generate(  # type: ignore[attr-defined]
                text, audio_prompt_path=sample_path, exaggeration=exaggeration
            )
        else:
            waveform = model.generate(text, exaggeration=exaggeration)  # type: ignore[attr-defined]
        default_sr = self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE)
        sample_rate = int(getattr(model, "sr", default_sr))
        return waveform, sample_rate
