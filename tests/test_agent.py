"""The LiveKit streaming agent's backend-side logic (no livekit/WebRTC needed).

The WebRTC/VAD glue needs a running LiveKit server (validated live, like M2's on-4080
step); these cover the parts that are pure: barge-in dispatch and the
VAD-utterance → STT → streaming-reply routing, with a fake AudioSource and fake backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from personavoice.orchestrator import agent
from personavoice.orchestrator.turn import TurnController
from personavoice.persona import load_persona

from .fakes import make_backend


def _companion(config_dir: Path):
    return load_persona(config_dir / "personas" / "companion.yaml")


class FakeSource:
    """Stand-in for `rtc.AudioSource` — records barge-in queue flushes and frames."""

    def __init__(self) -> None:
        self.cleared = False
        self.frames: list[object] = []

    def clear_queue(self) -> None:
        self.cleared = True

    async def capture_frame(self, frame: object) -> None:
        self.frames.append(frame)


def test_require_livekit_raises_clean_error_without_extra() -> None:
    with pytest.raises(RuntimeError, match="livekit"):
        agent._require_livekit()


async def test_barge_in_interrupts_and_flushes_source(config_dir: Path) -> None:
    source = FakeSource()
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), source)

    async def slow():  # never reaches the (livekit) sink before we interrupt
        await asyncio.sleep(10)
        yield b"x"

    ag._turn.begin(slow())
    await asyncio.sleep(0)

    ag.on_user_speech_started()
    assert source.cleared is True  # queued playback dropped immediately
    await ag._turn.join()


async def test_barge_in_when_idle_is_a_noop(config_dir: Path) -> None:
    source = FakeSource()
    ag = agent.PersonaAgent(make_backend(), _companion(config_dir), source)
    ag.on_user_speech_started()
    assert source.cleared is False


async def test_utterance_transcribes_then_streams_reply(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="hello", llm_reply="Hi there. Bye.")
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)  # bypass the livekit AudioSource sink

    pcm = b"\x00\x00" * 1600  # 0.1 s of silence as 16-bit PCM
    await ag.on_user_utterance(pcm, 16000)
    await ag._turn.join()

    # STT text drove the reply, streamed sentence-by-sentence.
    assert spoken == [b"RIFF" + b"Hi there.", b"RIFF" + b"Bye."]
    assert ag._pipeline.history[0].content == "hello"


async def test_empty_utterance_is_ignored(config_dir: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    backend = make_backend(stt_text="   ", llm_reply="should not run")
    ag = agent.PersonaAgent(backend, _companion(config_dir), FakeSource())

    spoken: list[bytes] = []

    async def sink(wav: bytes) -> None:
        spoken.append(wav)

    ag._turn = TurnController(sink)

    await ag.on_user_utterance(b"\x00\x00" * 100, 16000)
    await ag._turn.join()
    assert spoken == []  # blank transcript → no reply, nothing spoken
    assert ag._pipeline.history == []
