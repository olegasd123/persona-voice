"""Cross-process session heartbeat for admission control.

The token server's optional 503 early gate (`server/token_server.py`) needs the agent worker's
*live* session count. The worker and the token server are separate processes — and in the
self-hosted deployment the worker runs natively while the token server runs in a Docker container,
sharing state only through bind-mounted files.

The Prometheus multiprocess gauges (`obs/prometheus.py`) are one such channel, but they're written
via `mmap`, and mmap-backed writes from a native **Windows** worker aren't reflected into a Linux
container through a Windows-host bind mount — so on the CUDA/Windows box the token server reads an
empty gauge, the gate fails open, and the client never sees the "all lines busy" page (it does on
macOS, where the share propagates). Plain files *do* propagate across that same mount — it's how
`user_personas.json` and the consent store are already shared — so this module mirrors the live
count into a small JSON file the token server can read reliably on every platform.

Dependency-free (stdlib `json`/`os`) and best-effort: a write never crashes the worker's load loop
and a read never breaks `/healthz` or `/token`. A missing, stale, or malformed file reads as
"unknown", so the gate falls back to the Prometheus channel and ultimately fails open (the worker's
in-process load gate is the real capacity rail — this is only an optimization on top of it).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .prometheus import read_sessions

logger = logging.getLogger("personavoice.metrics")

# A heartbeat older than this (seconds) is ignored: the worker republishes on every load refresh
# (~0.5 s), so a file this stale means the worker stopped — better to read "unknown" and fail open
# than to pin the gate to a dead worker's last count. Generous so a worker hiccup doesn't flap it,
# and tolerant of modest host/container clock skew (a future `ts` is never treated as stale).
_MAX_AGE_S = 30.0

_FILE_ENV = "PERSONAVOICE_SESSIONS_FILE"


def sessions_file_path() -> Path | None:
    """Resolve the heartbeat file path, or None when no shared location is configured.

    `PERSONAVOICE_SESSIONS_FILE` wins; otherwise it defaults to `sessions.json` inside
    `PROMETHEUS_MULTIPROC_DIR` — already a shared, writable dir bind-mounted into the token-server
    container, so the default "just works" wherever cross-process metrics are set up. None disables
    the heartbeat (write + read become no-ops), preserving the pure-Prometheus behavior.
    """
    explicit = (os.getenv(_FILE_ENV) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    multiproc = (os.getenv("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if multiproc:
        return Path(multiproc).expanduser() / "sessions.json"
    return None


def write_sessions(active: int, capacity: int, *, path: Path | None = None) -> None:
    """Publish the worker's live `(active, capacity)` to the heartbeat file (best-effort).

    Written atomically (temp file + `os.replace`, both on the worker's own filesystem) so the token
    server never reads a torn JSON, and stamped with a wall-clock `ts` for the reader's freshness
    check. A no-op when no shared path is configured; never raises — this runs in the worker's
    LiveKit load callback, so a transient write failure just skips one heartbeat (the next refresh
    retries ~0.5 s later).
    """
    target = path or sessions_file_path()
    if target is None:
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"active": int(active), "capacity": int(capacity), "ts": time.time()})
        tmp = target.with_name(f"{target.name}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, target)
    except OSError:
        logger.debug("failed to write session heartbeat to %s", target, exc_info=True)


def read_sessions_file(
    *, path: Path | None = None, max_age_s: float = _MAX_AGE_S
) -> tuple[int, int] | None:
    """Read `(active, capacity)` from the heartbeat file, or None when unusable.

    Returns None when the file is absent, older than `max_age_s` (the worker stopped refreshing),
    or malformed — so a caller treats it as "unknown" rather than trusting a stale/garbage count.
    """
    target = path or sessions_file_path()
    if target is None:
        return None
    try:
        raw = target.read_text()
    except OSError:
        return None  # absent / unreadable — "unknown", not an error
    try:
        obj = json.loads(raw)
        active = int(obj["active"])
        capacity = int(obj["capacity"])
        ts = float(obj["ts"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.debug("malformed session heartbeat at %s", target, exc_info=True)
        return None
    if max_age_s > 0 and (time.time() - ts) > max_age_s:
        return None  # stale: the worker stopped publishing
    return active, capacity


def read_live_sessions() -> tuple[int, int] | None:
    """Live `(active, capacity)` for the admission gate and `/healthz`, most-reliable source first.

    Prefers the JSON heartbeat (propagates across the native-worker↔dockerized-token-server bind
    mount on every platform) and falls back to the Prometheus multiprocess gauges (fine on an
    all-native host / Linux, where the mmap share works). None when neither knows the count.
    """
    from_file = read_sessions_file()
    if from_file is not None:
        return from_file
    return read_sessions()
