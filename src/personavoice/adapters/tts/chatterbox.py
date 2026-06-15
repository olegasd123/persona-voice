"""Chatterbox TTS (CUDA; MPS on Mac) — emotion-exaggeration control, zero-shot cloning.

The clean-license (MIT) alternative to Orpheus. M2 implements one-shot `synthesize`
(text -> WAV bytes); `model.generate(text)` returns a float waveform tensor at the model's
native sample rate (`model.sr`, 24 kHz). Zero-shot cloning via a reference sample
(`audio_prompt_path`) lands in M5. `chatterbox` is imported lazily and the model is built
once per adapter and cached.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...models import VoiceRef
from .base import TTSAdapter

_DEFAULT_SAMPLE_RATE = 24000


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
        self._model: object | None = None

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        waveform, sample_rate = await asyncio.to_thread(self._synthesize_array, text)
        return _waveform_to_wav(waveform, sample_rate)

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from chatterbox.tts import ChatterboxTTS as _Chatterbox
            except ImportError as exc:  # pragma: no cover - only without the package installed
                raise RuntimeError(
                    "chatterbox-tts is not installed; install it in the CUDA server image: "
                    "`pip install chatterbox-tts`"
                ) from exc
            self._model = _Chatterbox.from_pretrained(device=self.options.get("device", "cuda"))
        return self._model

    def _synthesize_array(self, text: str) -> tuple[Any, int]:
        model = self._get_model()
        waveform = model.generate(text)  # type: ignore[attr-defined]
        default_sr = self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE)
        sample_rate = int(getattr(model, "sr", default_sr))
        return waveform, sample_rate
