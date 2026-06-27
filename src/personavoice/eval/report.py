"""Post-session feedback report — score the *user*, not the system.

The rest of `eval/` (`wer.py`, `mos.py`, `dashboard.py`) scores the *system*; nothing scores
the *user*. For a mock-interview persona (`pm_interviewer` / `hr_interviewer`) or a language
tutor (`language_teacher`) the obvious product feature is an end-of-call report on how the
*user* did. This module is that report.

It's a distill-style pass, deliberately mirroring `memory/profile.py`: a pure rubric prompt +
defensive JSON parsing wrapped around a single LLM call, so everything except the model itself
is unit-tested with a fake LLM. The rubric is persona-shaped — an interview persona is scored
on STAR structure / clarity / conciseness / relevance; a tutor on grammar / vocabulary /
fluency with concrete corrections; anything else gets a light general rubric. Only the
*transcript text* is available here, so audio-only signals (true speaking pace, pronunciation)
are out of scope — the report sticks to what the words show.

Generation is gated by consent at the call site (the `ConversationMemory` facade) exactly like
the rest of memory; `PERSONAVOICE_SESSION_REPORTS` chooses *which* personas get a report
(default: interview + language only).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..models import Msg, Persona, Role

# Score range for a rubric dimension. 0 is reserved for "couldn't judge this from the text".
_MIN_SCORE = 0
_MAX_SCORE = 5
# Bound how much we keep so a persisted report (and the prompt) stay small.
_MAX_ITEMS = 12
_MAX_SUMMARY_CHARS = 800


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReportKind(StrEnum):
    """Which rubric to score a session against (chosen from the persona)."""

    interview = "interview"
    language = "language"
    general = "general"


class RubricScore(BaseModel):
    """One scored rubric dimension (e.g. "STAR structure": 3/5 with a comment)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    score: int = Field(ge=_MIN_SCORE, le=_MAX_SCORE)
    comment: str = ""


class SessionReport(BaseModel):
    """The end-of-session feedback report on the user's performance."""

    model_config = ConfigDict(extra="ignore")

    persona_id: str
    session_id: str
    kind: ReportKind
    summary: str = ""
    scores: list[RubricScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=_utcnow)

    def is_empty(self) -> bool:
        """True when the model returned nothing usable (so the caller can skip persisting)."""
        return not (self.summary.strip() or self.scores or self.strengths or self.improvements)


# --------------------------------------------------------------------------------------
# Rubric definitions (drive the prompt; persona-shaped)
# --------------------------------------------------------------------------------------

# The dimensions the model is asked to score, per kind. Kept to things a *transcript* can
# actually show — pace/pronunciation need audio and are out of scope here.
_RUBRIC_DIMENSIONS: dict[ReportKind, tuple[str, ...]] = {
    ReportKind.interview: ("STAR structure", "Clarity", "Conciseness", "Relevance"),
    ReportKind.language: ("Grammar", "Vocabulary", "Fluency"),
    ReportKind.general: ("Clarity", "Engagement"),
}

# Per-kind framing for the system prompt: who is being judged and how.
_RUBRIC_GUIDANCE: dict[ReportKind, str] = {
    ReportKind.interview: (
        "This was a mock interview. Evaluate the CANDIDATE — the user, not the interviewer. "
        "Judge how well their answers used the STAR structure (Situation, Task, Action, "
        "Result), stayed clear and to the point, and actually answered the question asked. "
        'Note any filler-word habit ("um", "like", "you know") in the relevant comment.'
    ),
    ReportKind.language: (
        "This was a language-learning conversation. Evaluate the LEARNER — the user, not the "
        "tutor — at their apparent proficiency level. Assess grammar accuracy, vocabulary "
        "range, and fluency/coherence. Put concrete corrections (what they said → a better "
        "phrasing) in the improvements list."
    ),
    ReportKind.general: (
        "Evaluate the USER's side of this conversation (not the assistant's) for how clearly "
        "they expressed themselves and how engaged they were."
    ),
}


def report_kind_for(persona_id: str) -> ReportKind:
    """Pick the rubric for a persona id (robust to custom personas via name cues)."""
    pid = (persona_id or "").lower()
    if "interview" in pid:
        return ReportKind.interview
    if "teacher" in pid or "tutor" in pid or "language" in pid:
        return ReportKind.language
    return ReportKind.general


# --------------------------------------------------------------------------------------
# Generation policy (which personas get a report)
# --------------------------------------------------------------------------------------


class ReportPolicy(StrEnum):
    off = "off"  # never generate
    auto = "auto"  # interview + language personas only (the default)
    all = "all"  # every persona


def report_policy(env: Mapping[str, str] | None = None) -> ReportPolicy:
    """Resolve the report-generation policy from ``PERSONAVOICE_SESSION_REPORTS``.

    Unset/empty (or any unrecognized value) → ``auto`` (interview + language only, the sensible
    product default); ``off``/``0``/``false`` disables it; ``all``/``1``/``true`` reports on
    every persona. Consent still gates everything downstream.
    """
    env = os.environ if env is None else env
    val = (env.get("PERSONAVOICE_SESSION_REPORTS") or "").strip().lower()
    if val in ("0", "false", "no", "off", "none"):
        return ReportPolicy.off
    if val in ("1", "true", "yes", "on", "all", "every"):
        return ReportPolicy.all
    return ReportPolicy.auto


def report_enabled_for(persona_id: str, *, policy: ReportPolicy | None = None) -> bool:
    """Whether a session with this persona should get an end-of-call report."""
    policy = policy if policy is not None else report_policy()
    if policy is ReportPolicy.off:
        return False
    if policy is ReportPolicy.all:
        return True
    return report_kind_for(persona_id) in (ReportKind.interview, ReportKind.language)


# --------------------------------------------------------------------------------------
# Prompt building (pure)
# --------------------------------------------------------------------------------------


def _transcript_block(transcript: list[Msg]) -> str:
    """Render the conversation as a readable `Role: content` block for the rubric prompt."""
    lines = [
        f"{m.role.value.capitalize()}: {m.content.strip()}"
        for m in transcript
        if m.role in (Role.user, Role.assistant) and m.content.strip()
    ]
    return "\n".join(lines)


def _report_system_prompt(kind: ReportKind) -> str:
    dims = ", ".join(_RUBRIC_DIMENSIONS[kind])
    return (
        "You are an expert coach writing a short, constructive feedback report on a user after "
        "a spoken practice session. " + _RUBRIC_GUIDANCE[kind] + " Be specific and reference "
        "what they actually said; be encouraging but honest.\n"
        f"Score each of these dimensions from 1 (poor) to 5 (excellent): {dims}. "
        "Use 0 only for a dimension the transcript can't show.\n"
        'Reply with ONLY a JSON object: {"summary": "...", "scores": [{"name": "...", '
        '"score": 0-5, "comment": "..."}], "strengths": ["..."], "improvements": ["..."]}. '
        "`summary` is one or two sentences; `strengths` and `improvements` are short, "
        "actionable bullet strings. If there is too little to assess, return a JSON object with "
        'an empty "scores" list and say so in "summary".'
    )


def build_report_prompt(
    persona: Persona, transcript: list[Msg], *, kind: ReportKind | None = None
) -> list[Msg]:
    """The message list that asks the LLM to score `transcript` for `persona` (pure)."""
    kind = kind if kind is not None else report_kind_for(persona.id)
    dims = "\n".join(f"- {d}" for d in _RUBRIC_DIMENSIONS[kind])
    user_content = (
        f"Dimensions to score:\n{dims}\n\n"
        f"Conversation transcript:\n{_transcript_block(transcript)}\n\n"
        "Write the feedback report. Return only the JSON object."
    )
    return [
        Msg(role=Role.system, content=_report_system_prompt(kind)),
        Msg(role=Role.user, content=user_content),
    ]


# --------------------------------------------------------------------------------------
# Response parsing (pure, defensive — a chatty local model must not break the report)
# --------------------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: parse the first JSON object in `text` (handles ```json fences/prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _str_list(value: Any) -> list[str]:
    """Coerce a JSON value into a list of non-empty strings (tolerating junk entries)."""
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()][:_MAX_ITEMS]


def _coerce_score(value: Any) -> int:
    """Coerce a model-supplied score (int/float/"4"/"4/5") into a clamped 0-5 int."""
    if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
        return 0
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            return 0
        num = float(match.group())
    else:
        return 0
    return max(_MIN_SCORE, min(_MAX_SCORE, round(num)))


def _parse_scores(value: Any) -> list[RubricScore]:
    """Parse the `scores` array into validated `RubricScore`s (skipping malformed entries)."""
    if not isinstance(value, list):
        return []
    out: list[RubricScore] = []
    for item in value[:_MAX_ITEMS]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        comment = item.get("comment")
        out.append(
            RubricScore(
                name=name.strip(),
                score=_coerce_score(item.get("score")),
                comment=comment.strip() if isinstance(comment, str) else "",
            )
        )
    return out


def parse_report_response(
    text: str, *, persona_id: str, session_id: str, kind: ReportKind
) -> SessionReport | None:
    """Parse the LLM's JSON reply into a `SessionReport`, or None when nothing is parseable.

    Tolerant of code fences and surrounding prose (like `memory/profile.py`). Returns None for
    an unparseable reply *or* a parseable-but-empty one, so the caller never persists a blank
    report.
    """
    obj = _extract_json_object(text)
    if obj is None:
        return None
    raw_summary = obj.get("summary")
    summary = raw_summary.strip()[:_MAX_SUMMARY_CHARS] if isinstance(raw_summary, str) else ""
    report = SessionReport(
        persona_id=persona_id,
        session_id=session_id,
        kind=kind,
        summary=summary,
        scores=_parse_scores(obj.get("scores")),
        strengths=_str_list(obj.get("strengths")),
        improvements=_str_list(obj.get("improvements")),
    )
    return None if report.is_empty() else report


# --------------------------------------------------------------------------------------
# Builder (the one impure piece — a single LLM call)
# --------------------------------------------------------------------------------------


class _ChatLLM(Protocol):
    """The slice of an LLM adapter the builder needs (full reply, non-streamed)."""

    async def chat(self, messages: list[Msg], persona: object) -> str: ...


class ReportBuilder:
    """Turns a session transcript into a `SessionReport` via an injected LLM."""

    def __init__(self, llm: _ChatLLM) -> None:
        self._llm = llm

    async def build(
        self,
        persona: Persona,
        transcript: list[Msg],
        *,
        session_id: str,
        min_user_turns: int = 2,
    ) -> SessionReport | None:
        """Score `transcript` for `persona`, or None when there's too little to assess.

        Skips trivial sessions (fewer than `min_user_turns` non-empty user turns) without
        calling the LLM, and returns None when the reply has nothing parseable.
        """
        user_turns = sum(1 for m in transcript if m.role is Role.user and m.content.strip())
        if user_turns < max(1, min_user_turns):
            return None
        kind = report_kind_for(persona.id)
        messages = build_report_prompt(persona, transcript, kind=kind)
        reply = await self._llm.chat(messages, persona)
        return parse_report_response(reply, persona_id=persona.id, session_id=session_id, kind=kind)
