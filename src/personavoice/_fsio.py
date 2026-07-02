"""Filesystem helpers for the on-disk stores.

The JSON manifests (`user_personas.json`, `clones.json`) are shared across processes: the
Dockerized token server writes them while the host agent worker hot-reloads them mid-call
(both sides of the bind mount). A plain `Path.write_text` truncates the file before the new
bytes land, so a reader racing a writer can see a partial manifest and fail validation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so concurrent readers see the old file or the new — never a
    torn one.

    Writes to a temp file in the same directory (`path.parent` must exist), then
    `os.replace`s it into place: the swap is atomic on POSIX. An existing file keeps its
    permission bits (`mkstemp` creates 0600, which the other side of the bind mount may not
    be able to read); a fresh file gets 0644. On failure the temp file is removed and the
    previous manifest is left untouched.
    """
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(fd, mode)
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
