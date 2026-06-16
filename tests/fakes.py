"""In-memory fake adapters so the pipeline/demo can be tested without ML deps."""

from __future__ import annotations

from collections.abc import AsyncIterator

from personavoice.adapters.factory import Backend
from personavoice.adapters.llm.base import LLMAdapter
from personavoice.adapters.stt.base import STTAdapter
from personavoice.adapters.tts.base import TTSAdapter
from personavoice.models import Msg, Persona, Transcript, VoiceRef


class FakeSTT(STTAdapter):
    name = "fake_stt"

    def __init__(self, text: str = "hello there") -> None:
        super().__init__()
        self._text = text
        self.received: bytes | None = None

    async def transcribe(self, audio: bytes) -> Transcript:
        self.received = audio
        return Transcript(text=self._text, is_final=True, language="en")


class FakeLLM(LLMAdapter):
    name = "fake_llm"

    def __init__(self, reply: str = "a generic reply", *, tokens: list[str] | None = None) -> None:
        super().__init__()
        self._reply = reply
        # When `tokens` is given, stream exactly those (lets tests control chunk shape and
        # use a sentinel to assert mid-stream cancellation); otherwise split `reply` in two.
        self._tokens = tokens
        self.last_messages: list[Msg] | None = None
        self.completed = False  # set True only if the stream runs to exhaustion

    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        if self._tokens is not None:
            for tok in self._tokens:
                yield tok
        else:
            mid = len(self._reply) // 2  # yield in two chunks to exercise stream->collect
            yield self._reply[:mid]
            yield self._reply[mid:]
        self.completed = True


class FakeTTS(TTSAdapter):
    name = "fake_tts"
    supports_cloning = False

    def __init__(self) -> None:
        super().__init__()
        self.last_text: str | None = None
        self.last_voice: VoiceRef | None = None
        self.chunks: list[str] = []  # every text chunk synthesized (streaming uses many)

    async def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        self.last_text = text
        self.last_voice = voice
        self.chunks.append(text)
        return b"RIFF" + text.encode()  # fake "wav" bytes that carry the reply text


def make_backend(*, stt_text: str = "hello there", llm_reply: str = "a generic reply") -> Backend:
    return Backend(
        name="fake",
        stt=FakeSTT(stt_text),
        llm=FakeLLM(llm_reply),
        tts=FakeTTS(),
    )
