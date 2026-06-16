"""TTS adapter base class.

Concrete backends (f5_mlx, orpheus, chatterbox, kokoro) subclass this and override
`stream_tts`, `synthesize`, and (where supported) `clone_voice`. `supports_cloning`
lets the orchestrator/persona layer know whether a backend can do zero-shot voices.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ...models import CheckResult, VoiceRef

# Where clone samples land when the adapter isn't told otherwise (overridden by the
# `clones_dir` option, which `VoiceCloner` points at its `ClonesStore` directory).
_DEFAULT_CLONES_DIR = "models/clones"


class TTSAdapter:
    """Text-to-speech. `stream_tts` consumes text chunks (sentence-by-sentence) and
    yields audio chunks; `clone_voice` registers a zero-shot voice from a sample."""

    name: str = "base"
    stage: str = "tts"
    supports_cloning: bool = False
    implemented: bool = False  # real backends set this True (drops the "stub" warning)

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    async def stream_tts(self, text: AsyncIterator[str], voice: VoiceRef) -> AsyncIterator[bytes]:
        """Yield one WAV chunk per incoming text chunk (M3 streaming).

        The default synthesizes each sentence-sized chunk as it arrives, so the first
        sentence can be spoken while the LLM is still generating the rest of the reply.
        Backends with a native token/audio stream may override this for finer-grained
        output; delegating to `synthesize` is correct for all current backends.
        """
        async for chunk in text:
            chunk = chunk.strip()
            if not chunk:
                continue
            yield await self.synthesize(chunk, voice)

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        """Convenience one-shot synthesis (used by the file-based M1 pipeline)."""
        raise NotImplementedError(f"{self.name}.synthesize is not implemented yet")

    async def clone_voice(self, sample_wav: bytes, name: str) -> VoiceRef:
        """Create a zero-shot voice clone from a short sample (M5).

        The cloning backends (Chatterbox, F5) synthesize directly from a reference WAV, so a
        clone is the persisted sample itself: store it and return a `VoiceRef` pointing at it.
        `synthesize` then passes `voice.sample_path` to the model as the reference. Backends
        that can't clone (Kokoro, Orpheus presets) leave `supports_cloning=False` and reject.
        """
        if not self.supports_cloning:
            raise NotImplementedError(f"{self.name} does not support voice cloning")
        path = self._write_clone_sample(sample_wav, name)
        return VoiceRef(id=name, name=name, sample_path=str(path), backend=self.name)

    def _write_clone_sample(self, sample_wav: bytes, name: str) -> Path:
        """Persist a clone's reference WAV under the adapter's clones directory."""
        clones_dir = Path(self.options.get("clones_dir", _DEFAULT_CLONES_DIR)).expanduser()
        clones_dir.mkdir(parents=True, exist_ok=True)
        path = clones_dir / f"{name}.wav"
        path.write_bytes(sample_wav)
        return path

    def check(self) -> CheckResult:
        warnings = [] if self.implemented else ["stub adapter — implemented in a later milestone"]
        if not self.supports_cloning:
            warnings.append("no voice cloning (zero-shot) support")
        return CheckResult(
            stage=self.stage,
            adapter=self.name,
            ok=True,
            detail=f"loaded (model={self.model!r}, cloning={self.supports_cloning})",
            warnings=warnings,
        )
