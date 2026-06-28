"""Concurrency / admission control (Feature I) — the worker-side gate in `orchestrator/agent.py`.

These are the pure pieces: the capacity knob, the load value LiveKit gates dispatch on, and the
`request_fnc` accept/reject decision. No livekit/WebRTC or GPU needed — the live `active_jobs`
count is supplied by a fake worker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from personavoice.orchestrator import agent
from personavoice.orchestrator.busy import (
    ADMISSION_KEY,
    busy_accept_metadata,
    busy_clip_status,
    busy_clip_wav,
    busy_data_message,
    busy_retry_after,
    is_busy_metadata,
)
from personavoice.server.config import max_sessions


def test_max_sessions_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_MAX_SESSIONS", raising=False)
    assert max_sessions() == 1


def test_max_sessions_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "3")
    assert max_sessions() == 3


@pytest.mark.parametrize("raw", ["", "  ", "nope", "0", "-2"])
def test_max_sessions_floors_at_one(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    # Garbage / non-positive values fall back to a safe 1 rather than disabling the gate.
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", raw)
    assert max_sessions() == 1


@pytest.mark.parametrize(
    ("active", "capacity", "expected"),
    [(0, 1, 0.0), (1, 1, 1.0), (1, 2, 0.5), (3, 2, 1.0), (0, 0, 1.0)],
)
def test_session_load(active: int, capacity: int, expected: float) -> None:
    assert agent._session_load(active, capacity) == expected


@pytest.mark.parametrize("capacity", [1, 2, 3, 8])
def test_load_threshold_is_below_one(capacity: int) -> None:
    # The framework rejects a prod load_threshold >= 1; ours must always stay under it while still
    # leaving room to admit the first session (threshold > 0).
    t = agent._load_threshold(capacity)
    assert 0.0 < t < 1.0


def test_load_threshold_blocks_the_last_slot() -> None:
    # At one session on a capacity-1 worker, load (1.0) must exceed the threshold so the worker
    # reports unavailable.
    cap = 1
    assert agent._session_load(1, cap) >= agent._load_threshold(cap)
    assert agent._session_load(0, cap) < agent._load_threshold(cap)


def test_active_job_count_handles_missing_and_broken_worker() -> None:
    assert agent._active_job_count(None) == 0
    assert agent._active_job_count(SimpleNamespace(active_jobs=[1, 2])) == 2

    class _Broken:
        @property
        def active_jobs(self) -> list[int]:
            raise RuntimeError("api drift")

    assert agent._active_job_count(_Broken()) == 0  # fail open, never crash the worker


def test_load_fnc_publishes_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "2")
    published: list[tuple[int, int]] = []
    monkeypatch.setattr(agent, "set_sessions", lambda a, c: published.append((a, c)))

    worker = SimpleNamespace(active_jobs=[object()])
    load = agent._worker_load_fnc(worker)

    assert load == 0.5  # 1 of 2 slots
    assert published == [(1, 2)]  # mirrored into the gauge
    assert agent._LIVE_WORKER["worker"] is worker  # stashed for request_fnc


class _FakeRequest:
    """Records which terminal call request_fnc made (and any accept metadata)."""

    def __init__(self) -> None:
        self.id = "job-1"
        self.accepted = False
        self.rejected = False
        self.accept_metadata: str | None = None

    async def accept(self, *, metadata: str = "", **_kw: object) -> None:
        self.accepted = True
        self.accept_metadata = metadata

    async def reject(self, *, terminate: bool = True) -> None:
        self.rejected = True


async def test_request_fnc_accepts_under_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    monkeypatch.setitem(agent._LIVE_WORKER, "worker", SimpleNamespace(active_jobs=[]))
    rejected: list[int] = []
    monkeypatch.setattr(agent, "record_rejected_session", lambda: rejected.append(1))

    req = _FakeRequest()
    await agent._request_fnc(req)

    assert req.accepted and not req.rejected
    assert not req.accept_metadata  # a normal accept carries no busy marker
    assert rejected == []


async def test_request_fnc_admits_busy_path_at_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Over cap, the job is counted as rejected but admitted on the busy path (accept + marker) so
    # the entrypoint can play the "all lines busy" clip instead of stranding the caller in silence.
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    monkeypatch.setitem(agent._LIVE_WORKER, "worker", SimpleNamespace(active_jobs=[object()]))
    rejected: list[int] = []
    monkeypatch.setattr(agent, "record_rejected_session", lambda: rejected.append(1))

    req = _FakeRequest()
    await agent._request_fnc(req)

    assert req.accepted and not req.rejected
    assert is_busy_metadata(req.accept_metadata)  # tagged so the entrypoint takes the busy path
    assert rejected == [1]  # still counted for /metrics


# --- Busy clip (Feature I, step 3) — the pure, offline-testable pieces ------------------------


def test_busy_accept_metadata_round_trips() -> None:
    assert is_busy_metadata(busy_accept_metadata())


@pytest.mark.parametrize(
    "meta",
    [
        None,
        "",
        "   ",
        "hr_interviewer",
        '{"persona": "tutor"}',
        "{not json",
        '{"pv_admission": "x"}',
    ],
)
def test_is_busy_metadata_rejects_non_busy(meta: str | None) -> None:
    # A normal job's metadata (a bare id, a persona JSON, garbage) must not take the busy path.
    assert not is_busy_metadata(meta)
    assert ADMISSION_KEY in busy_accept_metadata()  # guards the marker key name against drift


def test_busy_clip_wav_synthesizes_a_valid_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    # No custom clip + no numpy/TTS: a recognizable busy tone is synthesized with the stdlib only.
    import io
    import wave

    monkeypatch.delenv("PERSONAVOICE_BUSY_CLIP", raising=False)
    data = busy_clip_wav()
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0  # actually carries audio


def test_busy_clip_wav_prefers_operator_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "busy.wav"
    clip.write_bytes(b"RIFFcustom-clip-bytes")
    monkeypatch.setenv("PERSONAVOICE_BUSY_CLIP", str(clip))
    assert busy_clip_wav() == b"RIFFcustom-clip-bytes"


def test_busy_clip_wav_falls_back_when_file_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misconfigured path must not raise on the reject path — fall back to the synthesized tone.
    monkeypatch.setenv("PERSONAVOICE_BUSY_CLIP", "/no/such/clip.wav")
    assert busy_clip_wav()[:4] == b"RIFF"


def test_busy_clip_status_flags_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_BUSY_CLIP", raising=False)
    assert busy_clip_status() is None  # unset → nothing to warn about
    monkeypatch.setenv("PERSONAVOICE_BUSY_CLIP", "/no/such/clip.wav")
    warning = busy_clip_status()
    assert warning is not None and "PERSONAVOICE_BUSY_CLIP" in warning


def test_busy_data_message_carries_reason_and_retry() -> None:
    import json

    msg = json.loads(busy_data_message())
    assert msg["event"] == "busy"
    assert msg["message"]
    assert msg["retry_after"] == busy_retry_after()


def test_busy_retry_after_default_env_and_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_BUSY_RETRY_AFTER", raising=False)
    assert busy_retry_after() == 10
    monkeypatch.setenv("PERSONAVOICE_BUSY_RETRY_AFTER", "30")
    assert busy_retry_after() == 30
    for raw in ("0", "-5", "nope", ""):
        monkeypatch.setenv("PERSONAVOICE_BUSY_RETRY_AFTER", raw)
        assert busy_retry_after() >= 1  # never invites an instant-retry storm


# --- Job executor selection (warm-process reuse so back-to-back calls don't re-prewarm) -------

_FAKE_AGENTS = SimpleNamespace(JobExecutorType=SimpleNamespace(THREAD="THREAD", PROCESS="PROCESS"))


def test_job_executor_defaults_thread_on_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mac default = THREAD so the warm process (and its loaded models) is reused across calls.
    monkeypatch.delenv("PERSONAVOICE_JOB_EXECUTOR", raising=False)
    assert agent._job_executor_type(_FAKE_AGENTS, "mac") == "THREAD"


def test_job_executor_defaults_process_off_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA/Linux keep LiveKit's validated PROCESS default unless explicitly overridden.
    monkeypatch.delenv("PERSONAVOICE_JOB_EXECUTOR", raising=False)
    assert agent._job_executor_type(_FAKE_AGENTS, "cuda") == "PROCESS"


def test_job_executor_env_overrides_both_ways(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_JOB_EXECUTOR", "process")
    assert agent._job_executor_type(_FAKE_AGENTS, "mac") == "PROCESS"
    monkeypatch.setenv("PERSONAVOICE_JOB_EXECUTOR", "THREAD")  # case-insensitive
    assert agent._job_executor_type(_FAKE_AGENTS, "cuda") == "THREAD"


def test_job_executor_garbage_falls_back_to_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_JOB_EXECUTOR", "nonsense")
    assert agent._job_executor_type(_FAKE_AGENTS, "mac") == "PROCESS"
