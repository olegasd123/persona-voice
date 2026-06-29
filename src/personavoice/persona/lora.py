"""Served LoRA discovery for the custom-persona authoring picker (N3).

A persona's `llm.lora` routes to a vLLM-served adapter by basename
(`adapters/llm/_openai_compat.py`), but that only works on a LoRA-capable backend. This
surfaces the *selectable* adapters so the client can offer a dropdown (and the server can
validate a chosen name), mirroring the voice catalog's `available` / `reason` shape.

The truth of "what's served" is the `--lora-modules` vLLM was launched with — declared via
`VLLM_LORA_MODULES` (the same var `docker-compose.lora.yml` uses), as space/comma-separated
`name=path` entries. On Mac / LM Studio (`supports_lora=False`) nothing is hot-swappable, so
the list is empty (a LoRA there is merged into the base model at train time, per M7).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

# A served LoRA module is declared as `name=path`; entries are separated by whitespace or
# commas (vLLM accepts space-separated `--lora-modules name=path ...`).
_MODULE_SEP = re.compile(r"[,\s]+")


class LoraOption(BaseModel):
    """One selectable LoRA adapter for a custom persona's `llm.lora`.

    `available` is whether the active LLM backend can actually serve it (False on a
    non-LoRA backend); `reason` explains a False so the picker can grey it out.
    """

    id: str
    name: str
    available: bool = True
    reason: str | None = None


def parse_lora_modules(spec: str | None) -> list[str]:
    """Extract the served adapter *names* (left of `=`) from a `--lora-modules` spec.

    Tolerates empty/whitespace input and bare names (no `=`); de-dupes while preserving
    first-seen order so the picker is stable.
    """
    names: list[str] = []
    for token in _MODULE_SEP.split((spec or "").strip()):
        if not token:
            continue
        name = token.split("=", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def served_loras(*, supports_lora: bool, lora_modules: str | None) -> list[LoraOption]:
    """The selectable LoRA adapters for the active backend.

    Empty on a non-LoRA backend (the caller surfaces the "merged at train time" note);
    otherwise one `LoraOption` per declared `--lora-modules` name, all `available`.
    """
    if not supports_lora:
        return []
    return [LoraOption(id=name, name=name) for name in parse_lora_modules(lora_modules)]
