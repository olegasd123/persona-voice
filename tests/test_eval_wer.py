"""Word Error Rate scoring for STT eval."""

from __future__ import annotations

import pytest

from personavoice.eval.wer import corpus_wer, normalize_text, tokenize, wer, word_errors


def test_identical_text_is_zero_wer() -> None:
    assert wer("the quick brown fox", "the quick brown fox") == 0.0


def test_normalization_ignores_case_and_punctuation() -> None:
    assert normalize_text("Hello, World!") == "hello world"
    assert wer("Hello, world.", "hello world") == 0.0


def test_keeps_word_internal_apostrophes_and_hyphens() -> None:
    assert tokenize("don't state-of-the-art") == ["don't", "state-of-the-art"]


def test_single_substitution() -> None:
    e = word_errors("the cat sat", "the dog sat")
    assert (e.substitutions, e.deletions, e.insertions) == (1, 0, 0)
    assert e.reference_words == 3
    assert e.rate == pytest.approx(1 / 3)


def test_deletion_and_insertion() -> None:
    # hypothesis drops a word -> one deletion.
    assert word_errors("a b c", "a c").deletions == 1
    # hypothesis adds a word -> one insertion.
    assert word_errors("a c", "a b c").insertions == 1


def test_empty_reference_with_hypothesis_is_full_error() -> None:
    assert wer("", "") == 0.0
    assert wer("", "spurious words") == 1.0


def test_corpus_wer_is_pooled() -> None:
    # 1 error over (3 + 2) = 5 reference words -> 0.2, not the mean of per-utt rates.
    pairs = [("the cat sat", "the dog sat"), ("good morning", "good morning")]
    assert corpus_wer(pairs) == pytest.approx(1 / 5)


def test_corpus_wer_empty() -> None:
    assert corpus_wer([]) == 0.0
