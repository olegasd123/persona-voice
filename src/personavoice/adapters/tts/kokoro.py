"""Kokoro TTS (Mac dev) — fast, no cloning. The default dev voice.

Implements one-shot `synthesize` (text → WAV bytes) for the file-based pipeline.
Streaming (`stream_tts`) is handled by the live agent. Kokoro has no zero-shot cloning, so personas whose
voice ref points at a clone (`voices/...`) fall back to the configured preset until the
cloning backend (f5_mlx / Chatterbox) is wired up separately.

`kokoro` is imported lazily and the pipeline is built once per adapter and cached.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...audio import encode_wav
from ...models import VoiceRef
from .base import TTSAdapter

_DEFAULT_VOICE = "af_heart"
_DEFAULT_SAMPLE_RATE = 24000


def _resolve_voice(voice: VoiceRef, default: str) -> str:
    """Pick the Kokoro preset for a persona voice ref.

    Kokoro presets look like `af_heart` / `am_michael`. A clone-style ref (`voices/...`,
    or anything with a path separator) can't be honored without cloning, so we use the
    configured default preset.
    """
    vid = voice.id
    if vid and "/" not in vid:
        return vid
    return default


class KokoroTTS(TTSAdapter):
    name = "kokoro"
    supports_cloning = False
    implemented = True

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(model=model, options=options)
        self._pipeline: object | None = None

    @property
    def sample_rate(self) -> int:
        return int(self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE))

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        """Synthesize `text` in the persona's voice and return WAV bytes."""
        preset = _resolve_voice(voice, self.options.get("voice", _DEFAULT_VOICE))
        samples = await asyncio.to_thread(self._synthesize_array, text, preset)
        return encode_wav(samples, self.sample_rate)

    def _get_pipeline(self) -> object:
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise RuntimeError(
                    "kokoro is not installed; install the Mac extra: `pip install -e '.[mac]'`"
                ) from exc
            # lang_code: 'a' = American English, 'b' = British, etc. (Kokoro convention).
            lang_code = self.options.get("lang_code", "a")
            self._pipeline = KPipeline(lang_code=lang_code, repo_id=self.model)
        return self._pipeline

    def _synthesize_array(self, text: str, preset: str):
        import numpy as np

        pipeline = self._get_pipeline()
        speed = float(self.options.get("speed", 1.0))

        chunks = []
        for result in pipeline(text, voice=preset, speed=speed):  # type: ignore[operator]
            # Kokoro yields a Result object (.audio) or a (graphemes, phonemes, audio) tuple.
            audio = getattr(result, "audio", None)
            if audio is None:
                audio = result[2]
            if hasattr(audio, "detach"):  # torch tensor
                audio = audio.detach().cpu().numpy()
            elif hasattr(audio, "numpy"):
                audio = audio.numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
