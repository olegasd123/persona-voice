"""f5-tts-mlx TTS (Mac) — zero-shot voice cloning on Apple Silicon.

F5 is a *reference-text* cloning model: to speak in a voice it wants both the reference WAV
and that sample's transcript (`ref_text`). Both ride on the `VoiceRef` produced by cloning
(`sample_path` + `ref_text`, the latter filled by the cascade's STT). With no sample it falls
back to F5's bundled default reference. The heavy `f5_tts_mlx` call is lazy-imported and run
in a worker thread; it writes a WAV we decode back to float samples.

Note on licensing: the default F5 checkpoint's weights are **CC-BY-NC** (non-commercial) —
fine for Mac dev, but use Chatterbox (MIT) on CUDA for anything redistributed. See the README
"Models & licenses".
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from ...audio import decode_wav
from ...models import VoiceRef
from .base import TTSAdapter

_DEFAULT_MODEL = "lucasnewman/f5-tts-mlx"
_DEFAULT_SAMPLE_RATE = 24000
# f5_tts_mlx requires the reference clip at exactly 24 kHz; we resample any sample to match.
_F5_REF_RATE = 24000


def _f5_kwargs(text: str, voice: VoiceRef, *, model: str) -> dict[str, Any]:
    """Build the `f5_tts_mlx.generate` kwargs for one synthesis.

    Pure (no model import) so it's unit-testable: a fine-tuned voice loads its trained
    checkpoint via `model_name`; a clone passes its reference audio (and transcript, when
    known); without either, F5 uses its own built-in reference voice.
    """
    # A fine-tuned checkpoint replaces the base model the generator loads.
    model_name = voice.model_path or model
    kwargs: dict[str, Any] = {"generation_text": text, "model_name": model_name}
    if voice.sample_path:
        kwargs["ref_audio_path"] = voice.sample_path
        if voice.ref_text:
            # f5_tts_mlx.generate names the reference transcript `ref_audio_text`.
            kwargs["ref_audio_text"] = voice.ref_text
    return kwargs


class F5MLXTTS(TTSAdapter):
    name = "f5_mlx"
    supports_cloning = True
    implemented = True

    @property
    def sample_rate(self) -> int:
        return int(self.options.get("sample_rate", _DEFAULT_SAMPLE_RATE))

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        wav = await asyncio.to_thread(self._synthesize_wav, text, voice)
        # f5-tts-mlx already writes a WAV; return its bytes unchanged.
        return wav

    def _get_generate(self) -> Any:
        try:
            from f5_tts_mlx.generate import generate
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise RuntimeError(
                "f5-tts-mlx is not installed; install the Mac cloning extra: "
                "`pip install -e '.[clone-mac]'`"
            ) from exc
        return generate

    def _synthesize_wav(self, text: str, voice: VoiceRef) -> bytes:
        generate = self._get_generate()
        ref_path, tmp_ref = self._ref_at_24k(voice.sample_path)
        if tmp_ref is not None:  # point f5 at the resampled copy
            voice = voice.model_copy(update={"sample_path": ref_path})
        kwargs = _f5_kwargs(text, voice, model=self.model or _DEFAULT_MODEL)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            generate(output_path=str(out_path), **kwargs)
            data = out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)
            if tmp_ref is not None:
                Path(tmp_ref).unlink(missing_ok=True)
        # Normalize to our canonical WAV (decode → re-encode) so the rate is predictable.
        from ...audio import encode_wav

        samples, _ = decode_wav(data)
        return encode_wav(samples, self.sample_rate)

    def _ref_at_24k(self, sample_path: str | None) -> tuple[str | None, str | None]:
        """Return a 24 kHz reference path f5 will accept, resampling if needed.

        Returns `(ref_path, tmp_path)`: `tmp_path` is a temp file to delete afterwards, or
        None when the original was already 24 kHz (or there's no reference).
        """
        if not sample_path:
            return sample_path, None
        from ...audio import encode_wav, resample

        samples, sr = decode_wav(Path(sample_path).read_bytes())
        if sr == _F5_REF_RATE:
            return sample_path, None
        samples = resample(samples, sr, _F5_REF_RATE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        Path(tmp_path).write_bytes(encode_wav(samples, _F5_REF_RATE))
        return tmp_path, tmp_path
