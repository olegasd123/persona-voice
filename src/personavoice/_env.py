"""Tiny shell-style environment interpolation for config files.

Supports `${VAR}` and `${VAR:-default}`. An unset `${VAR}` with no default expands to the
empty string. Used by the backend config loader so YAML can reference env (e.g. base URLs)
without leaking secrets into the repo.
"""

from __future__ import annotations

import os
import re

_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def expand_env_vars(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        return os.environ.get(name, default if default is not None else "")

    return _PATTERN.sub(_replace, text)
