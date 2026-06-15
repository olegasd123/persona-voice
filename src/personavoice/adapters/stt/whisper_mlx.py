"""mlx-whisper STT (Mac / M4 Max).

M1 implements the one-shot `transcribe` used by the file-based pipeline. Streaming
(`stream`, partial transcripts) lands in M3. `mlx_whisper` is imported lazily so the
adapter constructs and `check()`s without the `mac` extra installed.
"""

from __future__ import annotations

import asyncio

from ...audio import decode_wav, resample
from ...models import Transcript
from .base import STTAdapter

# Whisper models expect 16 kHz mono audio.
_WHISPER_SR = 16000


class WhisperMLXSTT(STTAdapter):
    name = "whisper_mlx"
    implemented = True

    async def transcribe(self, audio: bytes) -> Transcript:
        """Transcribe WAV bytes with mlx-whisper.

        Decodes the wav, downmixes/resamples to 16 kHz mono, then runs the model. The
        (blocking) model call runs in a thread so the event loop stays responsive.
        """
        samples, sr = decode_wav(audio)
        samples = resample(samples, sr, _WHISPER_SR)

        result = await asyncio.to_thread(self._transcribe_array, samples)

        text = str(result.get("text", "")).strip()
        return Transcript(
            text=text,
            is_final=True,
            language=result.get("language") or self.options.get("language"),
        )

    def _transcribe_array(self, samples: object) -> dict:
        try:
            import mlx_whisper
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "mlx-whisper is not installed; install the Mac extra: `pip install -e '.[mac]'`"
            ) from exc

        kwargs: dict = {"path_or_hf_repo": self.model}
        language = self.options.get("language")
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(samples, **kwargs)
