"""The "all lines busy" reply for the admission-control over-capacity path (Feature I, step 3).

A bare `reject()` leaves an over-capacity caller alone in a silent room with no idea why. Instead
the worker briefly *accepts* the job (marking it via the agent's accept metadata), plays this
**pre-rendered** clip, sends a matching data message, and disconnects (see `orchestrator/agent.py`).

Everything here is pure and dependency-free so it's fully offline-testable (no LiveKit, no GPU, no
numpy): the clip is either an operator-supplied WAV (`PERSONAVOICE_BUSY_CLIP`) or a synthesized
two-tone telephone busy signal built with the stdlib `wave`/`array`/`math` — **zero model load**,
which is the whole point: a caller who slips past the load gate must not trigger any inference.
"""

from __future__ import annotations

import array
import io
import json
import logging
import math
import os
import wave
from pathlib import Path

logger = logging.getLogger("personavoice.agent")

# Accept-metadata marker the worker stamps on an over-capacity job so the entrypoint knows to play
# the busy clip instead of starting a full session. Carried as the agent participant's metadata.
ADMISSION_KEY = "pv_admission"
BUSY_VALUE = "busy"

# Human-readable reason, surfaced both in the data message (for a client banner) and the logs.
BUSY_MESSAGE = "All lines are busy right now — please hang up and try again in a moment."

# Sample rate of the synthesized clip. The worker resamples to its WebRTC track rate anyway, so the
# exact value only matters for the synthesized tone's fidelity; 24 kHz matches the output track.
BUSY_CLIP_RATE = 24000

# Default `Retry-After` (seconds) advertised on the token server's 503 and echoed in the busy data
# message; overridable so an operator can tune how soon a turned-away caller should retry.
_DEFAULT_RETRY_AFTER = 10

# Cache the resolved clip bytes keyed by (source path, sample rate) so repeated busy jobs don't
# re-read/re-synthesize. Keyed by the env path so a test that changes it isn't served a stale clip.
_CLIP_CACHE: dict[tuple[str, int], bytes] = {}


def busy_retry_after() -> int:
    """Seconds to advise a turned-away caller to wait, from `PERSONAVOICE_BUSY_RETRY_AFTER`.

    Floored at 1 (a `Retry-After` of 0 invites an instant retry storm); a garbage value falls back
    to the default rather than disabling the hint.
    """
    raw = (os.getenv("PERSONAVOICE_BUSY_RETRY_AFTER") or "").strip()
    if not raw:
        return _DEFAULT_RETRY_AFTER
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "PERSONAVOICE_BUSY_RETRY_AFTER=%r is not an integer; using %d",
            raw,
            _DEFAULT_RETRY_AFTER,
        )
        return _DEFAULT_RETRY_AFTER


def busy_accept_metadata() -> str:
    """The agent accept metadata that tags a job as over-capacity (read by `is_busy_metadata`)."""
    return json.dumps({ADMISSION_KEY: BUSY_VALUE})


def is_busy_metadata(meta: str | None) -> bool:
    """Whether `meta` is the over-capacity accept marker this module stamps (see `agent._request_fnc`).

    Tolerant of the non-JSON / unmarked metadata a normal job carries, returning False so only a
    job the worker itself flagged takes the busy path.
    """
    if not meta or not meta.strip().startswith("{"):
        return False
    try:
        obj = json.loads(meta)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and obj.get(ADMISSION_KEY) == BUSY_VALUE


def busy_data_message() -> bytes:
    """A `{"event":"busy",...}` data message so a client can show a banner even with audio muted."""
    return json.dumps(
        {"event": "busy", "message": BUSY_MESSAGE, "retry_after": busy_retry_after()}
    ).encode("utf-8")


def busy_clip_path() -> str:
    """The configured operator-supplied busy clip path (`PERSONAVOICE_BUSY_CLIP`), or ''."""
    return (os.getenv("PERSONAVOICE_BUSY_CLIP") or "").strip()


def busy_clip_status() -> str | None:
    """A `--check` warning if `PERSONAVOICE_BUSY_CLIP` is set but unreadable, else None.

    Misconfiguring the path isn't fatal — the worker falls back to the synthesized tone — but a
    silent fallback to a tone when the operator expected their own recording is worth flagging.
    """
    path = busy_clip_path()
    if not path:
        return None
    if not Path(path).is_file():
        return (
            f"PERSONAVOICE_BUSY_CLIP {path!r} is not a readable file — the busy reply will fall "
            "back to the synthesized tone"
        )
    return None


def busy_clip_wav(sample_rate: int = BUSY_CLIP_RATE) -> bytes:
    """The pre-rendered busy clip as WAV bytes: the operator's file, else a synthesized tone.

    `PERSONAVOICE_BUSY_CLIP` (a WAV path) wins when readable so an operator can supply a spoken
    "all lines busy" recording; otherwise a synthesized two-tone busy signal is used. Cached per
    source so a burst of over-capacity calls doesn't re-read/re-synthesize. Never raises — an
    unreadable path falls back to the tone — because this runs on the reject path and must not fail.
    """
    path = busy_clip_path()
    key = (path, sample_rate)
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    data: bytes | None = None
    if path:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            logger.warning(
                "PERSONAVOICE_BUSY_CLIP %r is unreadable (%s); using the synthesized tone",
                path,
                exc,
            )
    if data is None:
        data = _synthesize_busy_tone(sample_rate)
    _CLIP_CACHE[key] = data
    return data


def _synthesize_busy_tone(sample_rate: int) -> bytes:
    """Build a North-American telephone busy signal (480 Hz + 620 Hz, 0.5 s on/off) as WAV bytes.

    Pure stdlib (`math`/`array`/`wave`) so it needs neither numpy nor any TTS model — a recognizable
    "busy" cue with zero inference. Three on/off cycles is enough to read as busy without dragging.
    """
    on_s, off_s, cycles = 0.5, 0.5, 3
    amplitude = 0.28  # headroom so the summed tones don't clip
    on_samples = int(sample_rate * on_s)
    off_samples = int(sample_rate * off_s)
    samples = array.array("h")
    two_pi = 2.0 * math.pi
    for _ in range(cycles):
        for i in range(on_samples):
            t = i / sample_rate
            value = amplitude * (math.sin(two_pi * 480.0 * t) + math.sin(two_pi * 620.0 * t)) / 2.0
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
        samples.extend(0 for _ in range(off_samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buf.getvalue()
