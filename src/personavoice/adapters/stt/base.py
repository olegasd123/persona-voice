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
