"""VAD endpointing knobs, resolved from the environment (M10 latency tuning).

Endpointing — deciding the user has *finished* talking — is a direct latency lever: a long
trailing-silence window feels safe but adds dead air before the assistant replies; a short
one is snappy but risks cutting the user off (and barging in on a mid-thought pause). The
live agent uses Silero VAD (via LiveKit); these helpers turn `PERSONAVOICE_VAD_*` env vars
into the keyword arguments `silero.VAD.load(...)` accepts, so an operator can tune the
speed/safety trade-off without code changes.

The parsing is pure (no LiveKit import) so it's unit-testable offline; `agent.py` calls
`vad_load_kwargs()` and splats the result into `silero.VAD.load(**kwargs)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VadTuning:
    """Resolved endpointing knobs (None = leave the Silero default in place)."""

    min_silence_ms: int | None = None
    min_speech_ms: int | None = None
    prefix_padding_ms: int | None = None
    activation_threshold: float | None = None


def _int_env(env: dict[str, str], name: str) -> int | None:
    raw = (env.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _float_env(env: dict[str, str], name: str) -> float | None:
    raw = (env.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 <= value <= 1.0 else None


def vad_tuning_from_env(env: dict[str, str] | None = None) -> VadTuning:
    """Resolve the VAD endpointing knobs from the environment."""
    env = os.environ if env is None else env
    return VadTuning(
        min_silence_ms=_int_env(env, "PERSONAVOICE_VAD_MIN_SILENCE_MS"),
        min_speech_ms=_int_env(env, "PERSONAVOICE_VAD_MIN_SPEECH_MS"),
        prefix_padding_ms=_int_env(env, "PERSONAVOICE_VAD_PREFIX_PADDING_MS"),
        activation_threshold=_float_env(env, "PERSONAVOICE_VAD_ACTIVATION_THRESHOLD"),
    )


def vad_load_kwargs(tuning: VadTuning | None = None) -> dict[str, float]:
    """Map a `VadTuning` to `silero.VAD.load(**kwargs)` (only the values the operator set).

    Returns an empty dict when nothing is configured, so `silero.VAD.load()` keeps all of its
    own defaults — the behavior before M10.
    """
    tuning = tuning if tuning is not None else vad_tuning_from_env()
    kwargs: dict[str, float] = {}
    if tuning.min_silence_ms is not None:
        kwargs["min_silence_duration"] = tuning.min_silence_ms / 1000.0
    if tuning.min_speech_ms is not None:
        kwargs["min_speech_duration"] = tuning.min_speech_ms / 1000.0
    if tuning.prefix_padding_ms is not None:
        kwargs["prefix_padding_duration"] = tuning.prefix_padding_ms / 1000.0
    if tuning.activation_threshold is not None:
        kwargs["activation_threshold"] = tuning.activation_threshold
    return kwargs
