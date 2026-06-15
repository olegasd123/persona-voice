"""NVIDIA Parakeet STT (CUDA / RTX 4080) via NeMo — alternative to faster-whisper.

Parakeet (e.g. `nvidia/parakeet-tdt-0.6b-v2`) is a fast, accurate English ASR model served
through NeMo. M2 implements the one-shot `transcribe`; streaming lands in M3. NeMo's
`transcribe` takes file paths, so we resample to 16 kHz mono and hand it a temp wav. `nemo`
is imported lazily and the model is built once and cached on the instance.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from ...audio import decode_wav, encode_wav, resample
from ...models import Transcript
from .base import STTAdapter

_PARAKEET_SR = 16000
_DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _first_text(outputs: Any) -> str:
    """Extract the transcript text from NeMo `transcribe()`'s return value.

    NeMo has returned several shapes across versions: a list of plain strings, a list of
    `Hypothesis` objects (with a `.text` attribute), or a `(best, all)` tuple of those.
    """
    # Unwrap a (best_hypotheses, all_hypotheses) tuple to the best list.
    if isinstance(outputs, tuple) and outputs:
        outputs = outputs[0]
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, str):
        return first
    text = getattr(first, "text", None)
    return text if isinstance(text, str) else str(first)


class ParakeetSTT(STTAdapter):
    name = "parakeet"
    implemented = True

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        super().__init__(model=model, options=options)
        self._model: object | None = None

    async def transcribe(self, audio: bytes) -> Transcript:
        samples, sr = decode_wav(audio)
        samples = resample(samples, sr, _PARAKEET_SR)
        wav16 = encode_wav(samples, _PARAKEET_SR)

        text = await asyncio.to_thread(self._transcribe_wav, wav16)
        return Transcript(
            text=text.strip(),
            is_final=True,
            language=self.options.get("language"),
        )

    def _get_model(self) -> object:
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError as exc:  # pragma: no cover - only without NeMo installed
                raise RuntimeError(
                    "NeMo (parakeet) is not installed; install the CUDA extra plus NeMo: "
                    "`pip install -e '.[cuda]'` and `pip install nemo_toolkit[asr]`"
                ) from exc
            repo = self.model or _DEFAULT_MODEL
            self._model = nemo_asr.models.ASRModel.from_pretrained(repo)
        return self._model

    def _transcribe_wav(self, wav_bytes: bytes) -> str:
        model = self._get_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)
        try:
            outputs = model.transcribe([str(tmp_path)])  # type: ignore[attr-defined]
        finally:
            tmp_path.unlink(missing_ok=True)
        return _first_text(outputs)
