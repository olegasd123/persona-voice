"""Structured logging + per-turn metrics (observability)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from personavoice.obs import (
    TRACE,
    JsonFormatter,
    TurnMetrics,
    configure_logging,
    log_level_from_env,
    turn_metrics_from_stream,
)


def _record(msg: str, **extra: object) -> logging.LogRecord:
    rec = logging.makeLogRecord(
        {"name": "personavoice.test", "levelno": logging.INFO, "levelname": "INFO", "msg": msg}
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_emits_core_fields() -> None:
    out = JsonFormatter().format(_record("hello"))
    obj = json.loads(out)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "personavoice.test"
    assert obj["message"] == "hello"


def test_json_formatter_merges_extra_context() -> None:
    obj = json.loads(JsonFormatter().format(_record("turn metrics", turn={"persona": "pm"})))
    assert obj["turn"] == {"persona": "pm"}


def test_log_level_from_env() -> None:
    assert log_level_from_env({"PERSONAVOICE_LOG_LEVEL": "DEBUG"}) == logging.DEBUG
    assert log_level_from_env({"PERSONAVOICE_LOG_LEVEL": "nonsense"}) == logging.INFO
    assert log_level_from_env({}) == logging.INFO


def test_trace_level_sits_between_debug_and_info() -> None:
    # Custom TRACE is loud enough to show on top of INFO but quieter than DEBUG's library noise.
    assert logging.DEBUG < TRACE < logging.INFO
    assert logging.getLevelName(TRACE) == "TRACE"
    assert log_level_from_env({"PERSONAVOICE_LOG_LEVEL": "TRACE"}) == TRACE
    assert log_level_from_env({"PERSONAVOICE_LOG_LEVEL": "trace"}) == TRACE  # case-insensitive


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    configure_logging(env={})
    configure_logging(env={})
    ours = [h for h in root.handlers if getattr(h, "_personavoice", False)]
    assert len(ours) == 1
    # Clean up so we don't leave a handler around for other tests.
    for h in ours:
        root.removeHandler(h)


def test_configure_logging_json_format() -> None:
    configure_logging(json_format=True, env={})
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, "_personavoice", False)]
    assert ours and isinstance(ours[0].formatter, JsonFormatter)
    for h in ours:
        root.removeHandler(h)


@dataclass
class _FakeStream:
    first_token: float | None = None
    first_audio: float | None = None
    total: float | None = None
    reply: str = ""


def test_turn_metrics_from_stream_rounds_and_counts() -> None:
    sm = _FakeStream(first_token=0.12345, first_audio=0.6789, total=1.23456, reply="a reply here")
    tm = turn_metrics_from_stream("companion", "hello there", sm, stt_s=0.10001)
    assert tm.persona == "companion"
    assert tm.user_chars == len("hello there")
    assert tm.reply_chars == len("a reply here")
    assert tm.first_token_s == 0.1235  # rounded to 4 dp
    assert tm.stt_s == 0.1
    assert tm.interrupted is False


def test_turn_metrics_log_level_warns_on_interrupt(caplog) -> None:
    tm = TurnMetrics(persona="pm", user_chars=3, reply_chars=0, interrupted=True)
    logger = logging.getLogger("personavoice.test.metrics")
    with caplog.at_level(logging.INFO, logger=logger.name):
        tm.log(logger)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
