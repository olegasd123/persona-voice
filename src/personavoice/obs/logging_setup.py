"""Centralized logging configuration (observability).

Every entrypoint (the LiveKit agent, the token server, the CLIs) should log in one consistent
shape. `configure_logging()` installs a single root handler — plain text for humans, or
line-delimited JSON when `PERSONAVOICE_LOG_FORMAT=json` for log shippers. Level comes from
`PERSONAVOICE_LOG_LEVEL` (default INFO). It's idempotent so calling it from multiple entry
points doesn't stack duplicate handlers.

We add one custom level, `TRACE` (15), that sits between DEBUG and INFO. It turns on our own
verbose diagnostics (notably the full LLM request/response trace) on top of INFO, *without*
DEBUG's library-wide firehose — so the accepted levels are DEBUG|TRACE|INFO|WARNING|ERROR.

The `JsonFormatter` is pure (a `LogRecord` in, a JSON string out) so it's unit-testable; any
keys passed via `logger.info(..., extra={"context": {...}})` are merged into the JSON object.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping

# Custom level between DEBUG (10) and INFO (20). Setting PERSONAVOICE_LOG_LEVEL=TRACE shows our
# TRACE diagnostics + INFO and above, but hides chatty third-party DEBUG records (level 10).
TRACE = 15
logging.addLevelName(TRACE, "TRACE")

# Accepted PERSONAVOICE_LOG_LEVEL names → numeric level (loudest-first for readability).
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "TRACE": TRACE,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Attributes the stdlib sets on every LogRecord; anything else is a caller-supplied `extra`.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime", "taskName"}

# The LLM request/response dump is our only TRACE-level output, so we tint TRACE records dim-gray
# to make them easy to skim past in a busy terminal. ANSI only, and gated to a TTY (skipped when
# NO_COLOR is set), so piped/redirected logs and JSON output never pick up escape codes.
_GRAY = "\x1b[90m"
_RESET = "\x1b[0m"
_PLAIN_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class PlainFormatter(logging.Formatter):
    """Plain-text formatter that tints TRACE records (the LLM request/response dump) gray.

    Coloring is opt-in via `color=` (set from a TTY check in `configure_logging`); with it off this
    behaves exactly like a stock `Formatter(_PLAIN_FMT)`. The gray spans the whole multi-line record
    because ANSI color persists across newlines until the trailing reset.
    """

    def __init__(self, *, color: bool = False) -> None:
        super().__init__(_PLAIN_FMT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self._color and record.levelno == TRACE:
            return f"{_GRAY}{text}{_RESET}"
        return text


def _color_enabled(stream: object, env: Mapping[str, str]) -> bool:
    """Tint TRACE output only when writing to a real terminal and NO_COLOR isn't set."""
    if env.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object (level, logger, message, + extras)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any structured context attached via `extra=`.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def log_level_from_env(env: Mapping[str, str] | None = None) -> int:
    """Resolve the numeric log level from `PERSONAVOICE_LOG_LEVEL` (default INFO).

    Accepts DEBUG|TRACE|INFO|WARNING|ERROR (+ CRITICAL/NOTSET); an unknown value falls back to INFO.
    """
    env = os.environ if env is None else env
    name = (env.get("PERSONAVOICE_LOG_LEVEL") or "INFO").strip().upper()
    return _LEVELS.get(name, logging.INFO)


def configure_logging(
    *,
    level: int | None = None,
    json_format: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Install a single root logging handler. Idempotent (replaces our previous handler).

    `level`/`json_format` override the env when given. `PERSONAVOICE_LOG_FORMAT=json` selects
    the JSON formatter; anything else is the plain-text format.
    """
    env = os.environ if env is None else env
    if level is None:
        level = log_level_from_env(env)
    if json_format is None:
        json_format = (env.get("PERSONAVOICE_LOG_FORMAT") or "").strip().lower() == "json"

    root = logging.getLogger()
    root.setLevel(level)
    # Drop any handler we installed before so repeated calls don't duplicate output.
    for handler in list(root.handlers):
        if getattr(handler, "_personavoice", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler._personavoice = True  # type: ignore[attr-defined]
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(PlainFormatter(color=_color_enabled(handler.stream, env)))
    root.addHandler(handler)
