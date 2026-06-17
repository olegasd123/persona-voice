"""M7 eval harness: scorers, aggregation, compare, and the probe runner."""

from __future__ import annotations

import pytest

from personavoice.models import TurnStyle
from personavoice.training.eval import (
    compare,
    ends_with_question,
    evaluate,
    is_spoken_clean,
    score_replies,
    score_reply,
    turn_style_fit,
    word_count,
)

from .fakes import FakeLLM, make_persona

# -- pure scorers ----------------------------------------------------------------------


def test_word_count() -> None:
    assert word_count("one two three") == 3


@pytest.mark.parametrize(
    "text,expected",
    [("How are you?", True), ("I'm fine.", False), ('He asked "really?"', True)],
)
def test_ends_with_question(text: str, expected: bool) -> None:
    assert ends_with_question(text) is expected


@pytest.mark.parametrize(
    "text,clean",
    [
        ("Hello there, nice to meet you.", True),
        ("Here are options:\n- one\n- two", False),
        ("That is **very** important", False),
        ("Use the `run` command", False),
        ("Great work 🎉", False),
        ("# Heading", False),
    ],
)
def test_is_spoken_clean(text: str, clean: bool) -> None:
    assert is_spoken_clean(text) is clean


def test_turn_style_fit_within_window_is_one() -> None:
    text = " ".join(["word"] * 10)
    assert turn_style_fit(text, TurnStyle.concise) == 1.0


def test_turn_style_fit_penalizes_overlong_concise() -> None:
    short = turn_style_fit(" ".join(["w"] * 10), TurnStyle.concise)
    long = turn_style_fit(" ".join(["w"] * 90), TurnStyle.concise)
    assert long < short


def test_turn_style_fit_penalizes_too_short() -> None:
    assert turn_style_fit("hi", TurnStyle.concise) < 1.0


def test_score_reply_fields() -> None:
    persona = make_persona(turn_style=TurnStyle.concise)
    m = score_reply("Can you tell me more?", persona, keywords=["tell"])
    assert m.ends_with_question is True
    assert m.spoken_clean is True
    assert m.keyword_hits == 1


# -- aggregation -----------------------------------------------------------------------


def test_score_replies_aggregates() -> None:
    persona = make_persona(turn_style=TurnStyle.concise, follow_up_probability=1.0)
    replies = ["Tell me more about that?", "What happened next?"]
    scores = score_replies(replies, persona)
    assert scores.n == 2
    assert scores.question_rate == 1.0
    assert scores.spoken_clean_rate == 1.0
    assert scores.follow_up_alignment == 1.0  # target 1.0 matches question_rate 1.0
    assert 0.0 <= scores.persona_adherence <= 1.0


def test_follow_up_alignment_reflects_mismatch() -> None:
    # persona that should rarely ask, but every reply is a question -> low alignment
    persona = make_persona(follow_up_probability=0.0)
    scores = score_replies(["Why?", "How?"], persona)
    assert scores.question_rate == 1.0
    assert scores.follow_up_alignment == 0.0


def test_keyword_coverage_neutral_without_keywords() -> None:
    scores = score_replies(["anything at all"], make_persona())
    assert scores.keyword_coverage == 1.0


def test_keyword_coverage_counts_hits() -> None:
    scores = score_replies(
        ["the STAR framework helps"], make_persona(), keywords=["star", "result"]
    )
    assert scores.keyword_coverage == 0.5  # 1 of 2 keywords present


def test_score_replies_empty() -> None:
    scores = score_replies([], make_persona())
    assert scores.n == 0 and scores.persona_adherence == 0.0


def test_compare_reports_deltas() -> None:
    persona = make_persona(turn_style=TurnStyle.concise)
    base = score_replies([" ".join(["w"] * 200)], persona)  # too long -> low style fit
    lora = score_replies(["A short, clean reply."], persona)
    deltas = compare(base, lora)
    assert "n" not in deltas
    _b, _l, d = deltas["turn_style_fit"]
    assert d > 0  # the lora-side reply fits the concise style better


# -- runner ----------------------------------------------------------------------------


async def test_evaluate_runs_probes_through_llm() -> None:
    persona = make_persona(follow_up_probability=1.0)
    scores = await evaluate(FakeLLM("Tell me more about that?"), persona, probes=["a", "b", "c"])
    assert scores.n == 3
    assert scores.question_rate == 1.0
