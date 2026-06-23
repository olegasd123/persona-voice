"""f5-tts-mlx TTS (Mac) — zero-shot voice cloning on Apple Silicon.

F5 is a *reference-text* cloning model: to speak in a voice it wants both the reference WAV
and that sample's transcript (`ref_text`). Both ride on the `VoiceRef` produced by cloning
(`sample_path` + `ref_text`, the latter filled by the cascade's STT). With no sample it falls
back to F5's bundled default reference. The heavy `f5_tts_mlx` call is lazy-imported and run
in a worker thread; it writes a WAV we decode back to float samples.

**Duration:** f5_tts_mlx generates the reference clip followed by the new speech and then trims
the reference off. Left to its defaults (`duration=None, estimate_duration=False`) it feeds
`duration=None` to the sampler, which under-shoots badly — the output is a fraction of a second
("half a word"). Its `estimate_duration=True` heuristic fixes the single-sentence case but its
multi-sentence path re-scales the duration every sentence and blows up. So we compute an
explicit `duration` (seconds) ourselves and pass it: that forces f5's single-generation path
(`is_single_generation = len(sentences) <= 1 or duration is not None`) for any chunk, sized to
the text. We mirror f5's own `estimated_duration` formula, against the active clone's reference
when there is one, else f5's bundled default reference so the default voice is sized too.

Note on licensing: the default F5 checkpoint's weights are **CC-BY-NC** (non-commercial) —
fine for Mac dev, but use Chatterbox (MIT) on CUDA for anything redistributed. See the README
"Models & licenses".
"""

from __future__ import annotations

import asyncio
import functools
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

# f5_tts_mlx's bundled reference (used when no clone/sample is given — see its `generate.py`).
# We size the default-voice clip against it so it gets a correct duration too.
_F5_DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."
_F5_DEFAULT_REF_RESOURCE = "tests/test_en_1_ref_short.wav"


def _estimate_total_seconds(
    ref_seconds: float, ref_text: str | None, gen_text: str, *, speed: float
) -> float:
    """Total clip length (reference + generated speech) in seconds, mirroring f5's heuristic.

    f5 sizes a generation off the reference: it keeps the reference's chars-per-second rate and
    scales by the new text's length. Byte lengths (utf-8) match f5_tts_mlx's `estimated_duration`
    so the number we pass behaves like f5's own estimate, just computed for the whole chunk.
    """
    ref_len = max(len((ref_text or "").encode("utf-8")), 1)
    gen_len = len(gen_text.encode("utf-8"))
    return ref_seconds + (ref_seconds / ref_len) * (gen_len / speed)


def _f5_kwargs(
    text: str, voice: VoiceRef, *, model: str, duration: float | None = None
) -> dict[str, Any]:
    """Build the `f5_tts_mlx.generate` kwargs for one synthesis.

    Pure (no model import) so it's unit-testable: a fine-tuned voice loads its trained
    checkpoint via `model_name`; a clone passes its reference audio (and transcript, when
    known); without either, F5 uses its own built-in reference voice. An explicit `duration`
    (seconds) forces f5's single-generation path; without one we fall back to f5's text-length
    heuristic (correct for a single sentence).
    """
    # A fine-tuned checkpoint replaces the base model the generator loads.
    model_name = voice.model_path or model
    kwargs: dict[str, Any] = {"generation_text": text, "model_name": model_name}
    if voice.sample_path:
        kwargs["ref_audio_path"] = voice.sample_path
        if voice.ref_text:
            # f5_tts_mlx.generate names the reference transcript `ref_audio_text`.
            kwargs["ref_audio_text"] = voice.ref_text
    if duration is not None:
        kwargs["duration"] = duration
    else:
        kwargs["estimate_duration"] = True
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
        ref_path, tmp_ref, ref_seconds = self._ref_at_24k(voice.sample_path)
        if tmp_ref is not None:  # point f5 at the resampled copy
            voice = voice.model_copy(update={"sample_path": ref_path})
        duration = self._duration_seconds(voice, ref_seconds, text)
        kwargs = _f5_kwargs(text, voice, model=self.model or _DEFAULT_MODEL, duration=duration)
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

    def _duration_seconds(
        self, voice: VoiceRef, ref_seconds: float | None, gen_text: str
    ) -> float | None:
        """Seconds to ask f5 for, so the whole chunk is spoken instead of truncated.

        Sizes against the active reference (a clone) when we have one, else f5's bundled default
        reference so the default voice is sized too. Returns None only when no reference can be
        measured, leaving f5's own heuristic as a last-resort fallback.
        """
        speed = float(self.options.get("speed", 1.0)) or 1.0
        if ref_seconds is not None and voice.ref_text:
            return _estimate_total_seconds(ref_seconds, voice.ref_text, gen_text, speed=speed)
        default_seconds = self._default_ref_seconds()
        if default_seconds is None:
            return None
        return _estimate_total_seconds(default_seconds, _F5_DEFAULT_REF_TEXT, gen_text, speed=speed)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _default_ref_seconds() -> float | None:
        """Duration of f5_tts_mlx's bundled default reference clip (cached), or None if absent."""
        try:
            import pkgutil

            data = pkgutil.get_data("f5_tts_mlx", _F5_DEFAULT_REF_RESOURCE)
            if not data:
                return None
            samples, sr = decode_wav(data)
            return len(samples) / sr
        except Exception:  # pragma: no cover - defensive: fall back to f5's heuristic
            return None

    def _ref_at_24k(self, sample_path: str | None) -> tuple[str | None, str | None, float | None]:
        """Return a 24 kHz reference path f5 will accept, resampling if needed.

        Returns `(ref_path, tmp_path, ref_seconds)`: `tmp_path` is a temp file to delete
        afterwards (None when the original was already 24 kHz, or there's no reference);
        `ref_seconds` is the clip duration used to size the requested generation.
        """
        if not sample_path:
            return sample_path, None, None
        from ...audio import encode_wav, resample

        samples, sr = decode_wav(Path(sample_path).read_bytes())
        if sr == _F5_REF_RATE:
            return sample_path, None, len(samples) / sr
        samples = resample(samples, sr, _F5_REF_RATE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        Path(tmp_path).write_bytes(encode_wav(samples, _F5_REF_RATE))
        return tmp_path, tmp_path, len(samples) / _F5_REF_RATE
