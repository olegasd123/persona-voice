"""Post-session feedback report: rubric selection, policy, prompt, parsing, builder."""

from __future__ import annotations

from personavoice.eval.report import (
    ReportBuilder,
    ReportKind,
    ReportPolicy,
    SessionReport,
    build_report_prompt,
    parse_report_response,
    report_enabled_for,
    report_kind_for,
    report_policy,
)
from personavoice.models import Msg, Role

from .fakes import make_persona

# A clean rubric reply the fake LLM can return.
_GOOD_JSON = (
    '{"summary": "Solid answers, a bit long-winded.", '
    '"scores": [{"name": "STAR structure", "score": 4, "comment": "Clear situation + result"}, '
    '{"name": "Conciseness", "score": 2, "comment": "Rambled on the second answer"}], '
    '"strengths": ["Concrete examples"], "improvements": ["Tighten the middle"]}'
)


def _transcript() -> list[Msg]:
    return [
        Msg(role=Role.assistant, content="Tell me about a time you led a project."),
        Msg(role=Role.user, content="Sure. At my last job I led a migration..."),
        Msg(role=Role.assistant, content="What was the outcome?"),
        Msg(role=Role.user, content="We cut latency by 40 percent."),
    ]


class _ReplyLLM:
    """Minimal LLM stub: returns a fixed reply from `.chat` (the slice the builder uses)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages: object, persona: object) -> str:
        self.calls += 1
        return self.reply


# --------------------------------------------------------------------------------------
# report_kind_for
# --------------------------------------------------------------------------------------


def test_report_kind_for_interview() -> None:
    assert report_kind_for("hr_interviewer") is ReportKind.interview
    assert report_kind_for("pm_interviewer") is ReportKind.interview


def test_report_kind_for_language() -> None:
    assert report_kind_for("language_teacher") is ReportKind.language
    assert report_kind_for("french-tutor") is ReportKind.language


def test_report_kind_for_general_fallback() -> None:
    assert report_kind_for("companion") is ReportKind.general
    assert report_kind_for("") is ReportKind.general


# --------------------------------------------------------------------------------------
# report_policy / report_enabled_for
# --------------------------------------------------------------------------------------


def test_report_policy_default_is_auto() -> None:
    assert report_policy({}) is ReportPolicy.auto
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": ""}) is ReportPolicy.auto
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": "garbage"}) is ReportPolicy.auto


def test_report_policy_off_and_all() -> None:
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": "off"}) is ReportPolicy.off
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": "0"}) is ReportPolicy.off
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": "all"}) is ReportPolicy.all
    assert report_policy({"PERSONAVOICE_SESSION_REPORTS": "true"}) is ReportPolicy.all


def test_report_enabled_for_auto_targets_interview_and_language() -> None:
    assert report_enabled_for("hr_interviewer", policy=ReportPolicy.auto) is True
    assert report_enabled_for("language_teacher", policy=ReportPolicy.auto) is True
    assert report_enabled_for("companion", policy=ReportPolicy.auto) is False


def test_report_enabled_for_off_and_all() -> None:
    assert report_enabled_for("hr_interviewer", policy=ReportPolicy.off) is False
    assert report_enabled_for("companion", policy=ReportPolicy.all) is True


# --------------------------------------------------------------------------------------
# build_report_prompt
# --------------------------------------------------------------------------------------


def test_build_report_prompt_includes_dimensions_and_transcript() -> None:
    msgs = build_report_prompt(make_persona("hr_interviewer"), _transcript())
    assert msgs[0].role is Role.system
    assert "STAR structure" in msgs[0].content  # interview rubric in the system prompt
    body = msgs[1].content
    assert "cut latency by 40 percent" in body  # user turn rendered
    assert "Conciseness" in body  # dimension list rendered


def test_build_report_prompt_language_rubric() -> None:
    msgs = build_report_prompt(make_persona("language_teacher"), _transcript())
    assert "Grammar" in msgs[0].content
    assert "Vocabulary" in msgs[1].content


# --------------------------------------------------------------------------------------
# parse_report_response
# --------------------------------------------------------------------------------------


def _parse(text: str) -> SessionReport | None:
    return parse_report_response(
        text, persona_id="hr_interviewer", session_id="s1", kind=ReportKind.interview
    )


def test_parse_clean_json() -> None:
    report = _parse(_GOOD_JSON)
    assert report is not None
    assert report.persona_id == "hr_interviewer"
    assert report.session_id == "s1"
    assert report.kind is ReportKind.interview
    assert [s.name for s in report.scores] == ["STAR structure", "Conciseness"]
    assert report.scores[0].score == 4
    assert report.strengths == ["Concrete examples"]
    assert report.improvements == ["Tighten the middle"]


def test_parse_fenced_json() -> None:
    report = _parse(f"```json\n{_GOOD_JSON}\n```")
    assert report is not None
    assert report.summary.startswith("Solid answers")


def test_parse_junk_returns_none() -> None:
    assert _parse("I'm not sure how to grade this, sorry.") is None


def test_parse_empty_object_returns_none() -> None:
    # Parseable but nothing usable -> None, so the caller never persists a blank report.
    assert _parse('{"scores": [], "summary": "  "}') is None


def test_parse_score_coercion_and_clamping() -> None:
    text = (
        '{"summary": "ok", "scores": ['
        '{"name": "a", "score": "4/5"}, '
        '{"name": "b", "score": 9}, '
        '{"name": "c", "score": 3.6}, '
        '{"name": "d", "score": true}]}'
    )
    report = _parse(text)
    assert report is not None
    by_name = {s.name: s.score for s in report.scores}
    assert by_name == {"a": 4, "b": 5, "c": 4, "d": 0}  # parsed / clamped / rounded / bool->0


def test_parse_skips_malformed_score_entries() -> None:
    text = '{"summary": "ok", "scores": ["nope", {"comment": "no name"}, {"name": "good", "score": 5}]}'
    report = _parse(text)
    assert report is not None
    assert [s.name for s in report.scores] == ["good"]


def test_parse_filters_nonstring_bullets() -> None:
    text = '{"summary": "ok", "strengths": ["real", 3, "", null], "improvements": "not a list"}'
    report = _parse(text)
    assert report is not None
    assert report.strengths == ["real"]
    assert report.improvements == []


# --------------------------------------------------------------------------------------
# ReportBuilder (with a fake LLM)
# --------------------------------------------------------------------------------------


async def test_builder_produces_report() -> None:
    llm = _ReplyLLM(_GOOD_JSON)
    report = await ReportBuilder(llm).build(
        make_persona("hr_interviewer"), _transcript(), session_id="s1"
    )
    assert report is not None
    assert report.kind is ReportKind.interview
    assert report.session_id == "s1"
    assert llm.calls == 1


async def test_builder_skips_trivial_session_without_calling_llm() -> None:
    llm = _ReplyLLM(_GOOD_JSON)
    short = [Msg(role=Role.user, content="hi")]  # one user turn < min_user_turns
    report = await ReportBuilder(llm).build(make_persona("hr_interviewer"), short, session_id="s1")
    assert report is None
    assert llm.calls == 0


async def test_builder_none_on_junk_reply() -> None:
    llm = _ReplyLLM("no json here")
    report = await ReportBuilder(llm).build(
        make_persona("hr_interviewer"), _transcript(), session_id="s1"
    )
    assert report is None
    assert llm.calls == 1
