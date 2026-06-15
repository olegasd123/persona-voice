"""Structural contracts every backend implements (the snippet in IMPLEMENTATION_PLAN.md).

These `Protocol`s are for static typing / documentation. Concrete adapters inherit from
the base classes in each stage's `base.py`, which structurally satisfy these protocols.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ..models import Msg, Persona, Transcript, VoiceRef


@runtime_checkable
class STTProtocol(Protocol):
    def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSProtocol(Protocol):
    def stream_tts(self, text: AsyncIterator[str], voice: VoiceRef) -> AsyncIterator[bytes]: ...

    async def clone_voice(self, sample_wav: bytes, name: str) -> VoiceRef: ...
