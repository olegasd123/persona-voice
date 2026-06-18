"""Chatterbox TTS (CUDA; MPS on Mac) — emotion-exaggeration control, zero-shot cloning.

The clean-license (MIT) alternative to Orpheus, and the genuine cloning backend on CUDA.
M2 implemented one-shot `synthesize` (text -> WAV bytes); `model.generate(text)` returns a
float waveform tensor at the model's native sample rate (`model.sr`, 24 kHz). M5 adds
zero-shot cloning: when the voice carries a reference sample, we pass it as
`audio_prompt_path` so Chatterbox speaks in that voice. M9 adds fine-tuned voices: when the
voice carries a `model_path`, we load that trained checkpoint (`from_local`) instead of the
base weights. `chatterbox` is imported lazily; models are built once and cached per
checkpoint (the base under the `None` key).
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
        # Cache one model per checkpoint path; `None` is the base (`from_pretrained`).
        self._models: dict[str | None, object] = {}

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        waveform, sample_rate = await asyncio.to_thread(
            self._synthesize_array, text, voice.sample_path, voice.model_path
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
            if model_path:  # a fine-tuned voice checkpoint (M9)
                self._models[model_path] = _Chatterbox.from_local(model_path, device=device)
            else:
                self._models[model_path] = _Chatterbox.from_pretrained(device=device)
        return self._models[model_path]

    def _synthesize_array(
        self, text: str, sample_path: str | None = None, model_path: str | None = None
    ) -> tuple[Any, int]:
        model = self._get_model(model_path)
        # A reference sample → clone that voice; otherwise Chatterbox's built-in default.
        if sample_path:
            waveform = model.generate(text, audio_prompt_path=sample_path)  # type: ignore[attr-defined]
        else:
            waveform = model.generate(text)  # type: ignore[attr-defined]
        default_sr = self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE)
        sample_rate = int(getattr(model, "sr", default_sr))
        return waveform, sample_rate
