"""STT adapter base class.

Concrete backends (whisper_mlx, faster_whisper, parakeet) subclass this and override
`stream` (and optionally `transcribe`). Until a milestone implements a backend its
methods raise `NotImplementedError`, but the adapter still constructs and `check()`s so
`server --check` can validate config on any machine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...models import CheckResult, Transcript

# Whisper-family rate; warm-up synthesizes a short silent clip at this rate.
_WARMUP_SR = 16000


class STTAdapter:
    """Speech-to-text. `stream` consumes audio chunks and yields (partial/final) transcripts."""

    name: str = "base"
    stage: str = "stt"
    implemented: bool = False  # real backends set this True (drops the "stub" warning)

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Yield `Transcript`s as audio streams in. Real backends implement this as an
        async generator; the base raises so unimplemented adapters fail loudly."""
        raise NotImplementedError(f"{self.name}.stream is not implemented yet")

    async def transcribe(self, audio: bytes) -> Transcript:
        """Convenience one-shot transcription (used by the file-based pipeline)."""
        raise NotImplementedError(f"{self.name}.transcribe is not implemented yet")

    async def warmup(self) -> None:
        """Load the model (and run one inference) so the first real turn is fast.

        Transcribes a short silent clip so the weights and CUDA kernels are resident before a
        caller speaks — the worker calls this once at startup (see the agent's prewarm). It's a
        no-op for stub adapters; real backends may override if a cheaper warm path exists.
        """
        if not self.implemented:
            return
        import numpy as np  # local: numpy/soundfile are backend-extra deps

        from ...audio import encode_wav

        silence = np.zeros(_WARMUP_SR // 10, dtype=np.float32)  # 0.1 s
        await self.transcribe(encode_wav(silence, _WARMUP_SR))

    def check(self) -> CheckResult:
        """Lightweight validation that does NOT load model weights."""
        warnings = [] if self.implemented else ["stub adapter — implemented in a later milestone"]
        return CheckResult(
            stage=self.stage,
            adapter=self.name,
            ok=True,
            detail=f"loaded (model={self.model!r})",
            warnings=warnings,
        )
