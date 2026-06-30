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
    # The broadened set still must not trip on ordinary hyperbole / topic mentions.
    assert not (await mod.check_input("can we end this meeting early")).flagged
    assert not (await mod.check_input("the cold is killing me out here")).flagged


async def test_keyword_crisis_recall_broadened() -> None:
    mod = KeywordModerator()
    # Common passive-ideation phrasings the original 8-pattern set missed.
    for phrase in [
        "honestly everyone would be better off without me",
        "I don't want to be here anymore",
        "there's nothing to live for",
        "I wish I were dead",
        "I keep thinking about ending it all",
        "lately I just feel like killing myself",
    ]:
        verdict = await mod.check_input(phrase)
        assert verdict.flagged and verdict.crisis, phrase
        assert ModerationCategory.crisis in verdict.categories


async def test_keyword_output_bounds_harassment() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_output("Honestly, you're an idiot.", demeanor=Demeanor.rude)
    assert verdict.flagged
    assert ModerationCategory.harassment in verdict.categories
    assert verdict.replacement
    # Broadened abuse phrasings are caught too.
    for reply in ["You're a worthless failure.", "Nobody likes you.", "I hate you."]:
        assert (await mod.check_output(reply, demeanor=Demeanor.rude)).flagged, reply


async def test_keyword_output_bounds_threats() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_output("I will hurt you if you don't.")
    assert verdict.flagged
    assert ModerationCategory.threat in verdict.categories


async def test_keyword_output_blocks_self_harm_encouragement() -> None:
    mod = KeywordModerator()
    verdict = await mod.check_output("Honestly, you should just kill yourself.")
    assert verdict.flagged
    assert ModerationCategory.self_harm in verdict.categories
    # The graver breach gets the caring, resource-pointing line — not the curt de-escalation.
    assert "988" in (verdict.replacement or "")


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
    # Default-on (fail-safe): unset / empty / unrecognized values still yield the rule guard.
    assert isinstance(build_moderator(None), KeywordModerator)
    assert isinstance(build_moderator(""), KeywordModerator)
    assert isinstance(build_moderator("bogus"), KeywordModerator)
    # Only an explicit opt-out turns it off.
    assert isinstance(build_moderator("none"), NoopModerator)
    assert isinstance(build_moderator("off"), NoopModerator)


def test_moderator_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("PERSONAVOICE_MODERATION", raising=False)
    assert isinstance(moderator_from_env(), KeywordModerator)  # on by default
    monkeypatch.setenv("PERSONAVOICE_MODERATION", "none")
    assert isinstance(moderator_from_env(), NoopModerator)
    monkeypatch.setenv("PERSONAVOICE_MODERATION", "keyword")
    assert isinstance(moderator_from_env(), KeywordModerator)
