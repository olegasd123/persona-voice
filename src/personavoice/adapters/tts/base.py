"""TTS adapter base class.

Concrete backends (f5_mlx, orpheus, chatterbox, kokoro) subclass this and override
`stream_tts`, `synthesize`, and (where supported) `clone_voice`. `supports_cloning`
lets the orchestrator/persona layer know whether a backend can do zero-shot voices.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...models import CheckResult, VoiceRef


class TTSAdapter:
    """Text-to-speech. `stream_tts` consumes text chunks (sentence-by-sentence) and
    yields audio chunks; `clone_voice` registers a zero-shot voice from a sample."""

    name: str = "base"
    stage: str = "tts"
    supports_cloning: bool = False

    def __init__(self, *, model: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.options = options or {}

    def stream_tts(self, text: AsyncIterator[str], voice: VoiceRef) -> AsyncIterator[bytes]:
        """Yield audio chunks as text streams in. Real backends are async generators."""
        raise NotImplementedError(f"{self.name}.stream_tts is not implemented yet")

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        """Convenience one-shot synthesis (used by the file-based M1 pipeline)."""
        raise NotImplementedError(f"{self.name}.synthesize is not implemented yet")

    async def clone_voice(self, sample_wav: bytes, name: str) -> VoiceRef:
        """Create a zero-shot voice clone from a short sample (M5)."""
        raise NotImplementedError(f"{self.name} does not support voice cloning")

    def check(self) -> CheckResult:
        warnings = ["stub adapter — inference implemented in a later milestone"]
        if not self.supports_cloning:
            warnings.append("no voice cloning (zero-shot) support")
        return CheckResult(
            stage=self.stage,
            adapter=self.name,
            ok=True,
            detail=f"loaded (model={self.model!r}, cloning={self.supports_cloning})",
            warnings=warnings,
        )
