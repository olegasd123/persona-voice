"""Per-session overrides: SessionOptions parsing, precedence, and helpers."""

from __future__ import annotations

import json

from personavoice.models import CEFRLevel, Demeanor, SessionOptions


def test_from_metadata_full() -> None:
    meta = json.dumps({"persona": "companion", "voice": "libby", "cefr": "B1", "demeanor": "kind"})
    opts = SessionOptions.from_metadata(meta)
    assert opts.voice == "libby"
    assert opts.cefr is CEFRLevel.b1
    assert opts.demeanor is Demeanor.kind


def test_from_metadata_partial_keeps_only_present_fields() -> None:
    opts = SessionOptions.from_metadata('{"cefr": "a2"}')
    assert opts.cefr is CEFRLevel.a2
    assert opts.voice is None
    assert opts.demeanor is None


def test_from_metadata_is_case_insensitive() -> None:
    opts = SessionOptions.from_metadata('{"cefr": "b2", "demeanor": "RUDE"}')
    assert opts.cefr is CEFRLevel.b2
    assert opts.demeanor is Demeanor.rude


def test_from_metadata_drops_unknown_enum_keeps_rest() -> None:
    # One bad value must not discard the whole options object.
    opts = SessionOptions.from_metadata('{"cefr": "Z9", "demeanor": "rude"}')
    assert opts.cefr is None
    assert opts.demeanor is Demeanor.rude


def test_from_metadata_garbage_and_none() -> None:
    for bad in (None, "", "   ", "not json", "[1,2,3]", '"bare-string"'):
        assert SessionOptions.from_metadata(bad) == SessionOptions()


def test_from_metadata_ignores_extra_keys() -> None:
    opts = SessionOptions.from_metadata('{"persona": "x", "user": "u", "voice": "v"}')
    assert opts.voice == "v"


def test_merged_over_precedence() -> None:
    base = SessionOptions(voice="default", cefr=CEFRLevel.c1, demeanor=Demeanor.natural)
    override = SessionOptions(cefr=CEFRLevel.a1)
    merged = override.merged_over(base)
    # The override's set field wins; unset fields fall back to base.
    assert merged.cefr is CEFRLevel.a1
    assert merged.voice == "default"
    assert merged.demeanor is Demeanor.natural


def test_merged_over_empty_override_is_base() -> None:
    base = SessionOptions(voice="v", cefr=CEFRLevel.b1)
    assert SessionOptions().merged_over(base) == base


def test_any_set() -> None:
    assert not SessionOptions().any_set()
    assert SessionOptions(voice="v").any_set()
    assert SessionOptions(demeanor=Demeanor.natural).any_set()
