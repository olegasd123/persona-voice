"""Served-LoRA parsing + the catalog gated by backend capability."""

from __future__ import annotations

import pytest

from personavoice.persona.lora import parse_lora_modules, served_loras


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("hr=/m/hr", ["hr"]),
        ("hr=/m/hr pm=/m/pm", ["hr", "pm"]),  # space separated
        ("hr=/m/hr,pm=/m/pm", ["hr", "pm"]),  # comma separated
        ("  hr=/m/hr   pm=/m/pm  ", ["hr", "pm"]),  # extra whitespace
        ("bareName", ["bareName"]),  # no '=' → the token is the name
        ("hr=/m/hr hr=/m/hr2", ["hr"]),  # de-duped, first-seen order
        ("", []),
        (None, []),
    ],
)
def test_parse_lora_modules(spec: str | None, expected: list[str]) -> None:
    assert parse_lora_modules(spec) == expected


def test_served_loras_empty_on_non_lora_backend() -> None:
    # Even with modules declared, a non-LoRA backend serves none (they aren't hot-swappable).
    assert served_loras(supports_lora=False, lora_modules="hr=/m/hr") == []


def test_served_loras_on_lora_backend() -> None:
    opts = served_loras(supports_lora=True, lora_modules="hr=/m/hr pm=/m/pm")
    assert [(o.id, o.name, o.available) for o in opts] == [
        ("hr", "hr", True),
        ("pm", "pm", True),
    ]
    assert all(o.reason is None for o in opts)
