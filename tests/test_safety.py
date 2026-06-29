"""The moderation seam: the no-op default and the rule/keyword guard."""

from __future__ import annotations

from personavoice.models import Demeanor
from personavoice.safety import (
    KeywordModerator,
    ModerationCategory,
    NoopModerator,
    build_moderator,
    moderator_from_env,
)


async def test_noop_never_flags() -> None:
    mod = NoopModerator()
    assert not (await mod.check_input("I want to die")).flagged
    assert not (await mod.check_output("you are an idiot")).flagged


async def test_keyword_crisis_input_short_circuits() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_input("sometimes I just want to die")
    assert verdict.flagged
    assert verdict.crisis
    assert ModerationCategory.crisis in verdict.categories
    assert verdict.replacement  # a calm, resource-pointing reply


async def test_keyword_ordinary_input_is_clean() -> None:
    mod = KeywordModerator()
    assert not (await mod.check_input("this homework is killing me, ha")).flagged
    assert not (await mod.check_input("tell me about the weather")).flagged


async def test_keyword_output_bounds_harassment() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_output("Honestly, you're an idiot.", demeanor=Demeanor.rude)
    assert verdict.flagged
    assert ModerationCategory.harassment in verdict.categories
    assert verdict.replacement


async def test_keyword_output_bounds_threats() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_output("I will hurt you if you don't.")
    assert verdict.flagged
    assert ModerationCategory.threat in verdict.categories


async def test_keyword_brusque_output_within_bounds_is_allowed() -> None:
    mod = KeywordModerator()
    # Curt/blunt but not abusive — the `rude` demeanor stays allowed.
    assert not (await mod.check_output("No. That's wrong. Try again.")).flagged


async def test_keyword_extra_patterns() -> None:
    mod = KeywordModerator(extra_output_patterns=[r"\bbanned-phrase\b"])
    verdict = await mod.check_output("here is a banned-phrase in the reply")
    assert verdict.flagged
    assert ModerationCategory.hate in verdict.categories


def test_build_moderator() -> None:
    assert isinstance(build_moderator("keyword"), KeywordModerator)
    assert isinstance(build_moderator("rule"), KeywordModerator)
    assert isinstance(build_moderator(None), NoopModerator)
    assert isinstance(build_moderator("none"), NoopModerator)
    assert isinstance(build_moderator("bogus"), NoopModerator)


def test_moderator_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("PERSONAVOICE_MODERATION", raising=False)
    assert isinstance(moderator_from_env(), NoopModerator)
    monkeypatch.setenv("PERSONAVOICE_MODERATION", "keyword")
    assert isinstance(moderator_from_env(), KeywordModerator)
