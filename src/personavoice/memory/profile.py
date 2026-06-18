"""Rolling user profile: durable facts + a running summary.

A profile is what lets the assistant *recall prior-session facts*: instead of replaying a
whole transcript (token-expensive, and gone once context rolls over), we distill it into a
compact, persistent profile that's injected every turn.

    recent turns ──▶ ProfileBuilder.update ──▶ UserProfile { summary, facts[] }

The distillation is the only impure part — it asks an LLM to extract durable facts and a
short summary. Everything around it (prompt building, JSON parsing, fact dedup/merge) is
pure and unit-tested with a fake LLM, mirroring `training/curate.py`. The LLM is asked for
strict JSON; parsing is defensive (strips code fences, tolerates surrounding prose) so a
chatty local model doesn't break the update.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..models import Msg, Role
from .store import MemoryTurn

# Cap how much we carry so injection stays cheap and the summary doesn't grow unbounded.
_MAX_FACTS = 40
_MAX_SUMMARY_CHARS = 1200


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize(text: str) -> str:
    """Lowercased, whitespace-collapsed key for near-duplicate fact detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


class ProfileFact(BaseModel):
    """One durable thing learned about the user (e.g. "prefers to be called Sam")."""

    model_config = ConfigDict(extra="ignore")

    text: str
    ts: str = Field(default_factory=_utcnow)


class UserProfile(BaseModel):
    """A compact, persistent memory of a user, injected into the prompt each turn."""

    model_config = ConfigDict(extra="ignore")

    user_id: str
    summary: str = ""
    facts: list[ProfileFact] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utcnow)

    def fact_texts(self) -> list[str]:
        return [f.text for f in self.facts]

    def is_empty(self) -> bool:
        return not self.summary.strip() and not self.facts


def merge_facts(existing: list[ProfileFact], new_texts: Iterable[str]) -> list[ProfileFact]:
    """Append genuinely new facts (case/space-insensitive dedup), newest-trimmed to the cap."""
    seen = {_normalize(f.text) for f in existing}
    merged = list(existing)
    for raw in new_texts:
        text = raw.strip()
        key = _normalize(text)
        if not text or key in seen:
            continue
        seen.add(key)
        merged.append(ProfileFact(text=text))
    # Keep the most recent facts if we've blown past the cap.
    return merged[-_MAX_FACTS:]


def _turns_block(turns: list[MemoryTurn]) -> str:
    """Render turns as a readable `Role: content` block for the distillation prompt."""
    lines = [f"{t.role.value.capitalize()}: {t.content.strip()}" for t in turns if t.content.strip()]
    return "\n".join(lines)


PROFILE_SYSTEM_PROMPT = (
    "You maintain a long-term memory profile of a user for a voice assistant. "
    "From the conversation excerpt, extract durable facts worth remembering across future "
    "sessions: the user's name, preferences, goals, important life details, and how they "
    "like to be addressed or talked to. Ignore one-off small talk and anything ephemeral. "
    "Also write a brief running summary of who this user is. "
    'Reply with ONLY a JSON object: {"facts": ["...", "..."], "summary": "..."}. '
    "Each fact must be a short, self-contained sentence. If there is nothing durable to add, "
    'return {"facts": [], "summary": "<keep the prior summary>"}.'
)


def build_profile_prompt(profile: UserProfile, turns: list[MemoryTurn]) -> list[Msg]:
    """The message list that asks the LLM to update `profile` from `turns` (pure)."""
    known = "\n".join(f"- {t}" for t in profile.fact_texts()) or "(none yet)"
    prior_summary = profile.summary.strip() or "(none yet)"
    user_content = (
        f"Known facts so far:\n{known}\n\n"
        f"Prior summary:\n{prior_summary}\n\n"
        f"New conversation excerpt:\n{_turns_block(turns)}\n\n"
        "Update the memory. Return only the JSON object."
    )
    return [
        Msg(role=Role.system, content=PROFILE_SYSTEM_PROMPT),
        Msg(role=Role.user, content=user_content),
    ]


def parse_profile_response(text: str) -> tuple[list[str], str | None]:
    """Parse the LLM's JSON reply into `(new_facts, summary)`; tolerant of fences/prose.

    Returns `([], None)` when nothing parseable is found, so a malformed reply leaves the
    existing profile untouched rather than wiping it.
    """
    obj = _extract_json_object(text)
    if obj is None:
        return [], None
    raw_facts = obj.get("facts", [])
    facts: list[str] = []
    if isinstance(raw_facts, list):
        facts = [f.strip() for f in raw_facts if isinstance(f, str) and f.strip()]
    raw_summary = obj.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) and raw_summary.strip() else None
    return facts, summary


def _extract_json_object(text: str) -> dict | None:
    """Best-effort: parse the first JSON object in `text` (handles ```json fences/prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop a leading ```json / ``` fence and any trailing fence.
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} span anywhere in the text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def apply_update(profile: UserProfile, new_facts: list[str], summary: str | None) -> UserProfile:
    """Return a new profile with `new_facts` merged in and `summary` replaced if given (pure)."""
    facts = merge_facts(profile.facts, new_facts)
    new_summary = (summary if summary is not None else profile.summary)[:_MAX_SUMMARY_CHARS]
    return profile.model_copy(
        update={"facts": facts, "summary": new_summary, "updated_at": _utcnow()}
    )


class _ChatLLM(Protocol):
    """The slice of an LLM adapter the builder needs (full reply, non-streamed)."""

    async def chat(self, messages: list[Msg], persona: object) -> str: ...


class ProfileBuilder:
    """Distills conversation turns into an updated `UserProfile` via an injected LLM."""

    def __init__(self, llm: _ChatLLM, persona: object | None = None) -> None:
        self._llm = llm
        # Some adapters' `chat` take a persona for sampling params; pass a lightweight stand-in.
        self._persona = persona

    async def update(self, profile: UserProfile, turns: list[MemoryTurn]) -> UserProfile:
        """Ask the LLM to extract new facts/summary from `turns` and fold them into `profile`.

        No turns, or a reply with nothing parseable, leaves the profile unchanged.
        """
        usable = [t for t in turns if t.content.strip()]
        if not usable:
            return profile
        messages = build_profile_prompt(profile, usable)
        reply = await self._llm.chat(messages, self._persona)
        new_facts, summary = parse_profile_response(reply)
        if not new_facts and summary is None:
            return profile
        return apply_update(profile, new_facts, summary)
