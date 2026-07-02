"""Atomic manifest writes: readers never see a torn file, failures keep the old one."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice._fsio import atomic_write_text


def test_writes_and_replaces_without_litter(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    atomic_write_text(path, "one")
    assert path.read_text() == "one"
    atomic_write_text(path, "two")
    assert path.read_text() == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]  # temp file swapped away


def test_failed_swap_keeps_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "m.json"
    atomic_write_text(path, "good")

    def boom(src: str, dst: str) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr("personavoice._fsio.os.replace", boom)
    with pytest.raises(OSError, match="disk went away"):
        atomic_write_text(path, "half-written")
    # The reader-visible file is untouched and the temp file was cleaned up.
    assert path.read_text() == "good"
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_preserves_existing_mode(tmp_path: Path) -> None:
    # mkstemp creates 0600; the rewrite must not tighten a manifest the other side of the
    # bind mount could previously read.
    path = tmp_path / "m.json"
    path.write_text("x")
    path.chmod(0o640)
    atomic_write_text(path, "y")
    assert (path.stat().st_mode & 0o777) == 0o640
