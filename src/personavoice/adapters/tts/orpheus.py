"""Orpheus TTS (CUDA / RTX 4080) — expressive, emotion tags, preset voices.

Implements one-shot `synthesize` (text -> WAV bytes) via the `orpheus_tts` engine, which
runs the Orpheus LLM (vLLM-backed) and decodes its audio tokens with SNAC into a stream of
24 kHz mono 16-bit PCM chunks. We concatenate the chunks and wrap them as WAV. The engine is
imported lazily and built once per adapter (heavy) and cached.

Orpheus ships preset voices (tara, leah, jess, leo, dan, mia, zac, zoe), selected per
persona by the voice registry. Its `generate_speech(prompt, voice=...)` engine takes only a
preset name — there's no reference-sample input — so genuine zero-shot cloning goes
through **Chatterbox** (MIT, the dockerized single-GPU default) on CUDA. Hence
`supports_cloning = False` here; a persona assigned a clone keeps its Orpheus preset.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...models import VoiceRef
from .base import TTSAdapter

_DEFAULT_VOICE = "tara"
_DEFAULT_SAMPLE_RATE = 24000


def _resolve_voice(voice: VoiceRef, default: str) -> str:
    """Pick the Orpheus preset for a persona voice ref.

    Presets are bare names (`tara`, `leo`, ...). A clone-style ref (`voices/...`, or
    anything with a path separator) can't be honored without cloning, so we fall back
    to the configured default preset.
    """
    vid = voice.id
    if vid and "/" not in vid:
        return vid
    return default


def _pcm16_to_wav(raw: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian 16-bit PCM mono samples as WAV bytes."""
    import numpy as np

    from ...audio import encode_wav

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return encode_wav(samples, sample_rate)


class OrpheusTTS(TTSAdapter):
    name = "orpheus"
    supports_cloning = False  # preset voices only; clone via Chatterbox on CUDA (see module doc)
    implemented = True

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(model=model, options=options)
        self._engine: object | None = None

    @property
    def sample_rate(self) -> int:
        return int(self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE))

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        preset = _resolve_voice(voice, self.options.get("voice", _DEFAULT_VOICE))
        raw = await asyncio.to_thread(self._synthesize_pcm, text, preset)
        return _pcm16_to_wav(raw, self.sample_rate)

    def _get_engine(self) -> object:
        if self._engine is None:
            try:
                from orpheus_tts import OrpheusModel
            except ImportError as exc:  # pragma: no cover - only without the engine installed
                raise RuntimeError(
                    "orpheus_tts is not installed; install it in the CUDA server image: "
                    "`pip install orpheus-speech` (pulls vLLM). Chatterbox is the "
                    "clean-license alternative (adapter: chatterbox)."
                ) from exc
            self._engine = OrpheusModel(model_name=self.model)
        return self._engine

    def _synthesize_pcm(self, text: str, preset: str) -> bytes:
        engine = self._get_engine()
        chunks = list(engine.generate_speech(prompt=text, voice=preset))  # type: ignore[attr-defined]
        return b"".join(chunks)
