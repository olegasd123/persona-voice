"""Persona LoRA dataset format + converters (M7).

A training example is one in-character conversation: an optional leading `system` turn (the
persona's prompt) followed by alternating `user`/`assistant` turns, ending on `assistant` (the
target the model learns to produce). We store examples as JSONL in the OpenAI *messages* shape
— one example per line, `{"messages": [{"role", "content"}, ...]}` — which is exactly the shape
the cascade already speaks (`Msg`). So curation can dump conversations straight from the persona
LLM and the Mac trainer (mlx-lm's "chat" format) reads this file unchanged.

Two converters bridge to the CUDA trainer:
  - `to_mlx_chat`  — the native `{"messages": [...]}` line (mlx-lm `lora --data <dir>`).
  - `to_sharegpt`  — the `{"conversations": [{"from", "value"}, ...], "system": ...}` shape
    LLaMA-Factory / Unsloth consume (registered via their `dataset_info.json`).

`render_chatml` is a deterministic text rendering used for previews and by the eval harness.
Everything here is pure (no model libs), so it imports and tests on any machine.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..models import Msg, Role

DATASET_FORMATS = ("mlx_chat", "sharegpt")

# ShareGPT role names (LLaMA-Factory's default mapping).
_SHAREGPT_FROM = {Role.user: "human", Role.assistant: "gpt"}


class DatasetError(ValueError):
    """A dialogue example or dataset file is malformed."""


class DialogueExample(BaseModel):
    """One in-character training conversation."""

    model_config = ConfigDict(extra="ignore")

    messages: list[Msg]


def validate_example(example: DialogueExample) -> None:
    """Raise `DatasetError` unless the example is a well-formed training conversation.

    Rules: a single optional `system` first; then strictly alternating `user`/`assistant`
    starting with `user`; at least one full user→assistant exchange; ends on `assistant`
    (the supervised target); no blank content.
    """
    messages = example.messages
    if not messages:
        raise DatasetError("example has no messages")

    body = messages
    if messages[0].role is Role.system:
        body = messages[1:]
    if any(m.role is Role.system for m in body):
        raise DatasetError("a 'system' turn may only appear first")
    if not body:
        raise DatasetError("example has no user/assistant turns")

    for m in messages:
        if not m.content.strip():
            raise DatasetError(f"empty content in a {m.role.value!r} turn")

    expected = (Role.user, Role.assistant)
    for i, m in enumerate(body):
        if m.role is not expected[i % 2]:
            raise DatasetError(
                "turns must alternate user/assistant starting with 'user'; "
                f"turn {i} is {m.role.value!r}"
            )
    if body[-1].role is not Role.assistant:
        raise DatasetError("a training example must end on an 'assistant' turn")
    if len(body) < 2:
        raise DatasetError("need at least one user->assistant exchange")


# --------------------------------------------------------------------------------------
# Read / write the native messages JSONL
# --------------------------------------------------------------------------------------


def _parse_line(line: str, *, lineno: int) -> DialogueExample:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"line {lineno}: invalid JSON: {exc}") from exc
    try:
        example = DialogueExample.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError → uniform DatasetError
        raise DatasetError(f"line {lineno}: invalid example: {exc}") from exc
    try:
        validate_example(example)
    except DatasetError as exc:
        raise DatasetError(f"line {lineno}: {exc}") from exc
    return example


def read_jsonl(path: str | Path) -> list[DialogueExample]:
    """Read + validate a messages-JSONL dataset, skipping blank lines."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    out: list[DialogueExample] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            out.append(_parse_line(line, lineno=lineno))
    if not out:
        raise DatasetError(f"{path}: no examples found")
    return out


def write_jsonl(path: str | Path, examples: list[DialogueExample]) -> None:
    """Write examples as native messages-JSONL (one example per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(to_mlx_chat(ex), ensure_ascii=False) for ex in examples]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Converters
# --------------------------------------------------------------------------------------


def to_mlx_chat(example: DialogueExample) -> dict[str, Any]:
    """The native mlx-lm "chat" line: `{"messages": [{"role", "content"}, ...]}`."""
    return {"messages": [{"role": m.role.value, "content": m.content} for m in example.messages]}


def to_sharegpt(example: DialogueExample) -> dict[str, Any]:
    """LLaMA-Factory / Unsloth ShareGPT shape.

    The leading `system` (if any) becomes the top-level `system` field; user/assistant turns
    become `conversations` entries keyed `human`/`gpt`.
    """
    messages = example.messages
    system = ""
    body = messages
    if messages and messages[0].role is Role.system:
        system = messages[0].content
        body = messages[1:]
    conversations = [{"from": _SHAREGPT_FROM[m.role], "value": m.content} for m in body]
    out: dict[str, Any] = {"conversations": conversations}
    if system:
        out["system"] = system
    return out


_BOLD_ITALIC_RE = re.compile(r"(\*\*|__)(.*?)\1", re.DOTALL)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+")


def to_spoken(text: str) -> str:
    """Strip markdown so the text reads cleanly through TTS (a *voice* assistant).

    Removes the exact formatting `eval.is_spoken_clean` flags — bold/italic markers, code spans,
    headers, and list bullets — while keeping the words. Training a persona LoRA on spoken-clean
    targets is what lets it beat prompting alone on the spoken-clean metric.
    """
    text = _BOLD_ITALIC_RE.sub(r"\2", text)
    text = re.sub(r"`+", "", text)  # inline code + fences
    lines = []
    for line in text.splitlines():
        line = _HEADER_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        lines.append(_NUMBERED_RE.sub("", line))
    text = "\n".join(lines).replace("**", "").replace("__", "")  # any unpaired markers
    return text.strip()


def clean_example(example: DialogueExample) -> DialogueExample:
    """Return a copy with every user/assistant turn run through `to_spoken` (system left as-is)."""
    cleaned = [
        m if m.role is Role.system else Msg(role=m.role, content=to_spoken(m.content))
        for m in example.messages
    ]
    return DialogueExample(messages=cleaned)


def render_chatml(example: DialogueExample) -> str:
    """A deterministic ChatML-style rendering (previews / eval), not a trainer format."""
    parts = [f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>" for m in example.messages]
    return "\n".join(parts)


def write_dataset(
    path: str | Path, examples: list[DialogueExample], *, fmt: str = "mlx_chat"
) -> None:
    """Write `examples` in a trainer format (`mlx_chat` JSONL or `sharegpt` JSON array)."""
    if fmt not in DATASET_FORMATS:
        raise DatasetError(f"unknown dataset format {fmt!r}; expected one of {DATASET_FORMATS}")
    for ex in examples:
        validate_example(ex)
    if fmt == "mlx_chat":
        write_jsonl(path, examples)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [to_sharegpt(ex) for ex in examples]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_examples(
    examples: list[DialogueExample], *, valid_fraction: float = 0.1, seed: int = 0
) -> tuple[list[DialogueExample], list[DialogueExample]]:
    """Shuffle and split into (train, valid). Always keeps >=1 train example."""
    if not 0.0 <= valid_fraction < 1.0:
        raise DatasetError("valid_fraction must be in [0.0, 1.0)")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_valid = int(len(shuffled) * valid_fraction)
    n_valid = min(n_valid, len(shuffled) - 1) if shuffled else 0
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid
