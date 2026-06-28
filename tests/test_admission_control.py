"""Concurrency / admission control (Feature I) — the worker-side gate in `orchestrator/agent.py`.

These are the pure pieces: the capacity knob, the load value LiveKit gates dispatch on, and the
`request_fnc` accept/reject decision. No livekit/WebRTC or GPU needed — the live `active_jobs`
count is supplied by a fake worker.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from personavoice.orchestrator import agent
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
    """Records which terminal call request_fnc made."""

    def __init__(self) -> None:
        self.id = "job-1"
        self.accepted = False
        self.rejected = False

    async def accept(self, **_kw: object) -> None:
        self.accepted = True

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
    assert rejected == []


async def test_request_fnc_rejects_at_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    monkeypatch.setitem(agent._LIVE_WORKER, "worker", SimpleNamespace(active_jobs=[object()]))
    rejected: list[int] = []
    monkeypatch.setattr(agent, "record_rejected_session", lambda: rejected.append(1))

    req = _FakeRequest()
    await agent._request_fnc(req)

    assert req.rejected and not req.accepted
    assert rejected == [1]  # counted for /metrics
