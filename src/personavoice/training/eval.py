"""Persona-adherence eval harness.

To answer "did the LoRA actually help?" we score replies on cheap, deterministic proxies for
in-character behavior — no second LLM judge needed for the first pass:

  - **turn-style fit** — does reply length match the persona's `turn_style` (concise/balanced/
    verbose)? Interviewers/teachers that ramble are off-character.
  - **spoken-clean** — free of markdown / lists / emoji (this is a *voice* assistant).
  - **question rate** — fraction of replies that ask something, compared against the persona's
    `follow_up_probability` (interviewers should mostly ask; a companion less so).
  - **keyword coverage** — optional persona vocabulary that should show up.

`score_reply` is pure (text + persona → metrics); `score_replies` aggregates into `EvalScores`
with a single `persona_adherence` composite. `collect_replies`/`evaluate` run an injected
`LLMAdapter` over a probe set, so the CLI can A/B a prompt-only persona vs a LoRA one (same
persona, `llm.lora` toggled). Scoring is unit-tested with fakes; the live A/B needs a served
model (vLLM with the adapter, or a merged checkpoint).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from ..adapters.llm.base import LLMAdapter
from ..models import Msg, Persona, Role, TurnStyle
from ..persona.prompt import build_messages

# Generic conversational probes (the *user* side). Persona-neutral on purpose so the same set
# exercises every persona; override with a probes file for sharper, in-domain prompts.
DEFAULT_PROBES = (
    "Hi, thanks for making the time today.",
    "I'm a little nervous, to be honest.",
    "Can you tell me a bit about what we'll be doing?",
    "Sorry, could you say that another way?",
    "I had a tough week at work.",
    "What do you think I should focus on?",
    "Let me tell you about a project I worked on recently.",
    "I'm not sure I agree with that.",
)

# Reply-length windows (in words) per turn style; fit decays linearly outside the window.
_TURN_STYLE_WINDOW: dict[TurnStyle, tuple[int, int]] = {
    TurnStyle.concise: (4, 45),
    TurnStyle.balanced: (10, 80),
    TurnStyle.verbose: (40, 200),
}

# Markdown / formatting that reads badly through TTS.
_MARKDOWN_RE = re.compile(r"(\*\*|__|```|`|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s)", re.MULTILINE)
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f0ff]", re.UNICODE
)


def word_count(text: str) -> int:
    return len(text.split())


def ends_with_question(text: str) -> bool:
    stripped = text.rstrip().rstrip("\"')")
    return stripped.endswith("?")


def is_spoken_clean(text: str) -> bool:
    """True when the reply has no markdown/list/emoji formatting."""
    return _MARKDOWN_RE.search(text) is None and _EMOJI_RE.search(text) is None


def turn_style_fit(text: str, turn_style: TurnStyle) -> float:
    """1.0 inside the style's word window, decaying to 0.0 outside it."""
    low, high = _TURN_STYLE_WINDOW[turn_style]
    n = word_count(text)
    if low <= n <= high:
        return 1.0
    if n < low:
        return max(0.0, n / low) if low else 1.0
    # Above the window: full window-width of grace before hitting 0.
    overflow = n - high
    return max(0.0, 1.0 - overflow / high)


def keyword_hits(text: str, keywords: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lowered)


class ReplyMetrics(BaseModel):
    """Per-reply measurements."""

    words: int
    ends_with_question: bool
    spoken_clean: bool
    turn_style_fit: float
    keyword_hits: int


def score_reply(
    text: str, persona: Persona, *, keywords: Sequence[str] | None = None
) -> ReplyMetrics:
    return ReplyMetrics(
        words=word_count(text),
        ends_with_question=ends_with_question(text),
        spoken_clean=is_spoken_clean(text),
        turn_style_fit=turn_style_fit(text, persona.behavior.turn_style),
        keyword_hits=keyword_hits(text, keywords or ()),
    )


class EvalScores(BaseModel):
    """Aggregate persona-adherence scores over a set of replies (all in [0, 1] except means)."""

    n: int
    mean_words: float
    question_rate: float
    spoken_clean_rate: float
    turn_style_fit: float
    follow_up_alignment: float  # how close question_rate is to the persona's target
    keyword_coverage: float
    persona_adherence: float  # composite headline number


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_replies(
    replies: Sequence[str], persona: Persona, *, keywords: Sequence[str] | None = None
) -> EvalScores:
    """Aggregate per-reply metrics into the headline `EvalScores`."""
    if not replies:
        return EvalScores(
            n=0,
            mean_words=0.0,
            question_rate=0.0,
            spoken_clean_rate=0.0,
            turn_style_fit=0.0,
            follow_up_alignment=0.0,
            keyword_coverage=0.0,
            persona_adherence=0.0,
        )
    metrics = [score_reply(r, persona, keywords=keywords) for r in replies]
    question_rate = _mean([float(m.ends_with_question) for m in metrics])
    spoken_clean_rate = _mean([float(m.spoken_clean) for m in metrics])
    style_fit = _mean([m.turn_style_fit for m in metrics])
    target = persona.behavior.follow_up_probability
    follow_up_alignment = 1.0 - abs(question_rate - target)
    if keywords:
        keyword_coverage = _mean([min(1.0, m.keyword_hits / len(keywords)) for m in metrics])
    else:
        keyword_coverage = 1.0  # neutral when no vocabulary is specified

    # Composite: spoken-clean and turn-style fit are the non-negotiables for a voice persona;
    # follow-up alignment and keyword coverage shape the in-character flavor.
    persona_adherence = _mean([spoken_clean_rate, style_fit, follow_up_alignment, keyword_coverage])
    return EvalScores(
        n=len(replies),
        mean_words=_mean([float(m.words) for m in metrics]),
        question_rate=question_rate,
        spoken_clean_rate=spoken_clean_rate,
        turn_style_fit=style_fit,
        follow_up_alignment=follow_up_alignment,
        keyword_coverage=keyword_coverage,
        persona_adherence=persona_adherence,
    )


def compare(base: EvalScores, lora: EvalScores) -> dict[str, tuple[float, float, float]]:
    """Field-by-field (base, lora, delta) for the numeric scores."""
    out: dict[str, tuple[float, float, float]] = {}
    for field_name in EvalScores.model_fields:
        if field_name == "n":
            continue
        b = float(getattr(base, field_name))
        l_ = float(getattr(lora, field_name))
        out[field_name] = (b, l_, l_ - b)
    return out


async def collect_replies(
    llm: LLMAdapter, persona: Persona, probes: Sequence[str], *, bare_prompt: bool = False
) -> list[str]:
    """Run the persona over each probe (one independent turn) and collect the replies.

    `bare_prompt` sends only the authored `system_prompt`, dropping the derived directives
    (turn-style + the "speak without markdown/lists" nudge) that `render_system_prompt` adds.
    A persona LoRA's whole point is internalizing those behaviors into the weights, so a
    bare-prompt A/B is where the LoRA should beat prompting alone ("brains beyond prompting").
    """
    replies: list[str] = []
    for probe in probes:
        if bare_prompt:
            messages = [
                Msg(role=Role.system, content=persona.system_prompt.strip()),
                Msg(role=Role.user, content=probe),
            ]
        else:
            messages = build_messages(persona, user_input=probe)
        replies.append((await llm.chat(messages, persona)).strip())
    return replies


async def evaluate(
    llm: LLMAdapter,
    persona: Persona,
    *,
    probes: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    bare_prompt: bool = False,
) -> EvalScores:
    """Convenience: collect replies for `probes` and score them."""
    replies = await collect_replies(llm, persona, probes or DEFAULT_PROBES, bare_prompt=bare_prompt)
    return score_replies(replies, persona, keywords=keywords)
