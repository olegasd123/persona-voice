"""faster-whisper STT (CUDA / RTX 4080).

Mirrors the Mac `whisper_mlx` adapter: it implements the one-shot `transcribe` used by the
file-based pipeline; streaming partials are handled by the live agent. `faster_whisper` is imported lazily so
the adapter constructs and `check()`s without the `cuda` extra, and the (heavy) CTranslate2
model is built once on first use and cached on the instance.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...audio import decode_wav, resample
from ...models import Transcript
from .base import STTAdapter

# Whisper models expect 16 kHz mono audio.
_WHISPER_SR = 16000


def _join_segments(segments: Any) -> str:
    """Concatenate the text of faster-whisper's (lazy) segment iterator."""
    return "".join(seg.text for seg in segments)


class FasterWhisperSTT(STTAdapter):
    name = "faster_whisper"
    implemented = True

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(model=model, options=options)
        self._model: object | None = None

    async def transcribe(self, audio: bytes) -> Transcript:
        """Transcribe WAV bytes with faster-whisper (CTranslate2).

        Decodes the wav, downmixes/resamples to 16 kHz mono, then runs the model in a
        thread so the event loop stays responsive.
        """
        samples, sr = decode_wav(audio)
        samples = resample(samples, sr, _WHISPER_SR)

        text, language = await asyncio.to_thread(self._transcribe_array, samples)
        return Transcript(
            text=text.strip(),
            is_final=True,
            language=language or self.options.get("language"),
        )

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise RuntimeError(
                    "faster-whisper is not installed; install the CUDA extra: "
                    "`pip install -e '.[cuda]'`"
                ) from exc
            self._model = WhisperModel(
                self.model,
                device=self.options.get("device", "cuda"),
                compute_type=self.options.get("compute_type", "float16"),
            )
        return self._model

    def _transcribe_array(self, samples: object) -> tuple[str, str | None]:
        model = self._get_model()
        kwargs: dict[str, Any] = {"beam_size": int(self.options.get("beam_size", 5))}
        language = self.options.get("language")
        if language:
            kwargs["language"] = language

        # `transcribe` returns a (lazy segment generator, info) pair; iterating runs decode.
        segments, info = model.transcribe(samples, **kwargs)  # type: ignore[attr-defined]
        return _join_segments(segments), getattr(info, "language", None)
