"""Rolling profile (M8): fact merge/dedup, prompt building, response parsing, builder."""

from __future__ import annotations

from personavoice.memory.profile import (
    ProfileBuilder,
    ProfileFact,
    UserProfile,
    apply_update,
    build_profile_prompt,
    merge_facts,
    parse_profile_response,
)
from personavoice.memory.store import MemoryTurn
from personavoice.models import Role

from .fakes import make_persona


def _turn(content: str, role: Role = Role.user) -> MemoryTurn:
    return MemoryTurn(session_id="s1", persona_id="companion", role=role, content=content)


# --------------------------------------------------------------------------------------
# merge_facts
# --------------------------------------------------------------------------------------


def test_merge_facts_dedups_case_and_space_insensitively() -> None:
    existing = [ProfileFact(text="Likes jazz")]
    merged = merge_facts(existing, ["likes   jazz", "Has a dog", "has a dog"])
    assert [f.text for f in merged] == ["Likes jazz", "Has a dog"]


def test_merge_facts_drops_blanks() -> None:
    assert merge_facts([], ["  ", "real fact", ""]) == merge_facts([], ["real fact"])


def test_merge_facts_caps_total() -> None:
    existing = [ProfileFact(text=f"fact {i}") for i in range(45)]
    merged = merge_facts(existing, ["brand new fact"])
    assert len(merged) == 40
    assert merged[-1].text == "brand new fact"  # newest kept


# --------------------------------------------------------------------------------------
# build_profile_prompt
# --------------------------------------------------------------------------------------


def test_build_profile_prompt_includes_known_and_turns() -> None:
    profile = UserProfile(user_id="alice", summary="A musician.", facts=[ProfileFact(text="plays piano")])
    msgs = build_profile_prompt(profile, [_turn("I just adopted a cat")])
    assert msgs[0].role is Role.system
    body = msgs[1].content
    assert "plays piano" in body
    assert "A musician." in body
    assert "I just adopted a cat" in body


def test_build_profile_prompt_handles_empty_profile() -> None:
    msgs = build_profile_prompt(UserProfile(user_id="alice"), [_turn("hello")])
    assert "(none yet)" in msgs[1].content


# --------------------------------------------------------------------------------------
# parse_profile_response
# --------------------------------------------------------------------------------------


def test_parse_clean_json() -> None:
    facts, summary = parse_profile_response('{"facts": ["a", "b"], "summary": "S"}')
    assert facts == ["a", "b"]
    assert summary == "S"


def test_parse_fenced_json() -> None:
    text = '```json\n{"facts": ["x"], "summary": "Y"}\n```'
    assert parse_profile_response(text) == (["x"], "Y")


def test_parse_json_embedded_in_prose() -> None:
    text = 'Sure! Here is the memory:\n{"facts": ["likes tea"], "summary": "tea drinker"}\nHope that helps.'
    assert parse_profile_response(text) == (["likes tea"], "tea drinker")


def test_parse_junk_returns_nothing() -> None:
    assert parse_profile_response("I could not find anything.") == ([], None)


def test_parse_filters_nonstring_facts_and_blank_summary() -> None:
    facts, summary = parse_profile_response('{"facts": ["ok", 3, "", null], "summary": "   "}')
    assert facts == ["ok"]
    assert summary is None


# --------------------------------------------------------------------------------------
# apply_update
# --------------------------------------------------------------------------------------


def test_apply_update_merges_and_replaces_summary() -> None:
    profile = UserProfile(user_id="alice", summary="old", facts=[ProfileFact(text="a")])
    updated = apply_update(profile, ["b"], "new summary")
    assert updated.fact_texts() == ["a", "b"]
    assert updated.summary == "new summary"
    assert profile.summary == "old"  # original untouched (pure)


def test_apply_update_keeps_summary_when_none() -> None:
    profile = UserProfile(user_id="alice", summary="keep me")
    assert apply_update(profile, ["x"], None).summary == "keep me"


# --------------------------------------------------------------------------------------
# ProfileBuilder (with a fake LLM)
# --------------------------------------------------------------------------------------


class _ReplyLLM:
    """Minimal LLM stub: returns a fixed reply from `.chat` (the slice the builder uses)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages: object, persona: object) -> str:
        self.calls += 1
        return self.reply


async def test_builder_extracts_facts() -> None:
    llm = _ReplyLLM('{"facts": ["name is Sam", "loves hiking"], "summary": "An outdoorsy person."}')
    builder = ProfileBuilder(llm, make_persona())
    updated = await builder.update(UserProfile(user_id="alice"), [_turn("Hi, I'm Sam and I love hiking")])
    assert updated.fact_texts() == ["name is Sam", "loves hiking"]
    assert updated.summary == "An outdoorsy person."


async def test_builder_noop_without_turns() -> None:
    llm = _ReplyLLM('{"facts": ["x"], "summary": "y"}')
    profile = UserProfile(user_id="alice")
    assert await ProfileBuilder(llm).update(profile, []) is profile
    assert llm.calls == 0  # didn't even call the LLM


async def test_builder_unchanged_on_junk_reply() -> None:
    llm = _ReplyLLM("no json here, sorry")
    profile = UserProfile(user_id="alice", summary="prior")
    result = await ProfileBuilder(llm).update(profile, [_turn("something")])
    assert result is profile  # nothing parseable -> profile preserved


async def test_builder_blank_turns_skipped() -> None:
    llm = _ReplyLLM('{"facts": ["a"], "summary": "s"}')
    profile = UserProfile(user_id="alice")
    assert await ProfileBuilder(llm).update(profile, [_turn("   ")]) is profile
    assert llm.calls == 0
