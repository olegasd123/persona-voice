"""Dataset format: validation, JSONL round-trip, and the trainer converters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice.models import Msg, Role
from personavoice.training.dataset import (
    DatasetError,
    DialogueExample,
    clean_example,
    read_jsonl,
    render_chatml,
    split_examples,
    to_mlx_chat,
    to_sharegpt,
    to_spoken,
    validate_example,
    write_dataset,
    write_jsonl,
)
from personavoice.training.eval import is_spoken_clean


def _ex(*roles_contents: tuple[Role, str]) -> DialogueExample:
    return DialogueExample(messages=[Msg(role=r, content=c) for r, c in roles_contents])


def _good() -> DialogueExample:
    return _ex(
        (Role.system, "You are a tester."),
        (Role.user, "hi"),
        (Role.assistant, "hello, how are you?"),
        (Role.user, "good"),
        (Role.assistant, "glad to hear it"),
    )


# -- validation ------------------------------------------------------------------------


def test_validate_accepts_well_formed() -> None:
    validate_example(_good())  # no raise


def test_validate_accepts_no_system() -> None:
    validate_example(_ex((Role.user, "hi"), (Role.assistant, "hello")))


def test_validate_rejects_empty() -> None:
    with pytest.raises(DatasetError, match="no messages"):
        validate_example(DialogueExample(messages=[]))


def test_validate_rejects_non_alternating() -> None:
    with pytest.raises(DatasetError, match="alternate"):
        validate_example(_ex((Role.user, "a"), (Role.user, "b")))


def test_validate_rejects_not_ending_on_assistant() -> None:
    with pytest.raises(DatasetError, match="end on an 'assistant'"):
        validate_example(_ex((Role.user, "a"), (Role.assistant, "b"), (Role.user, "c")))


def test_validate_rejects_blank_content() -> None:
    with pytest.raises(DatasetError, match="empty content"):
        validate_example(_ex((Role.user, "  "), (Role.assistant, "b")))


def test_validate_rejects_system_not_first() -> None:
    with pytest.raises(DatasetError, match="only appear first"):
        validate_example(_ex((Role.user, "a"), (Role.system, "sys"), (Role.assistant, "b")))


def test_validate_rejects_single_turn() -> None:
    # a lone user turn has no assistant target to learn
    with pytest.raises(DatasetError, match="assistant"):
        validate_example(_ex((Role.user, "a")))


# -- converters ------------------------------------------------------------------------


def test_to_mlx_chat_shape() -> None:
    out = to_mlx_chat(_good())
    assert list(out) == ["messages"]
    assert out["messages"][0] == {"role": "system", "content": "You are a tester."}
    assert out["messages"][-1]["role"] == "assistant"


def test_to_sharegpt_splits_system_and_maps_roles() -> None:
    out = to_sharegpt(_good())
    assert out["system"] == "You are a tester."
    froms = [c["from"] for c in out["conversations"]]
    assert froms == ["human", "gpt", "human", "gpt"]
    assert out["conversations"][0]["value"] == "hi"


def test_to_sharegpt_without_system_omits_field() -> None:
    out = to_sharegpt(_ex((Role.user, "a"), (Role.assistant, "b")))
    assert "system" not in out


def test_render_chatml_is_deterministic() -> None:
    text = render_chatml(_ex((Role.user, "a"), (Role.assistant, "b")))
    assert text == "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>"


# -- io --------------------------------------------------------------------------------


def test_write_then_read_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_good(), _good()])
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].messages[1].content == "hi"


def test_read_jsonl_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(to_mlx_chat(_good())) + "\nnot json\n")
    with pytest.raises(DatasetError, match="line 2"):
        read_jsonl(path)


def test_read_jsonl_empty_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n  \n")
    with pytest.raises(DatasetError, match="no examples"):
        read_jsonl(path)


def test_write_dataset_sharegpt_is_json_array(tmp_path: Path) -> None:
    path = tmp_path / "data.sharegpt.json"
    write_dataset(path, [_good()], fmt="sharegpt")
    data = json.loads(path.read_text())
    assert isinstance(data, list) and data[0]["system"] == "You are a tester."


def test_write_dataset_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unknown dataset format"):
        write_dataset(tmp_path / "x", [_good()], fmt="bogus")


def test_write_dataset_validates_before_writing(tmp_path: Path) -> None:
    bad = _ex((Role.user, "a"), (Role.user, "b"))
    with pytest.raises(DatasetError):
        write_dataset(tmp_path / "x.jsonl", [bad], fmt="mlx_chat")


def test_shipped_sample_dataset_is_valid() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample = repo_root / "training/persona_lora/datasets/hr_interviewer.sample.jsonl"
    examples = read_jsonl(sample)
    assert len(examples) >= 3


# -- split -----------------------------------------------------------------------------


def test_split_is_deterministic_and_keeps_a_train_example() -> None:
    examples = [_good() for _ in range(10)]
    train, valid = split_examples(examples, valid_fraction=0.2, seed=1)
    assert len(valid) == 2 and len(train) == 8
    # same seed -> same split
    train2, valid2 = split_examples(examples, valid_fraction=0.2, seed=1)
    assert len(train2) == 8 and len(valid2) == 2


def test_split_never_empties_train() -> None:
    train, valid = split_examples([_good()], valid_fraction=0.9, seed=0)
    assert len(train) == 1 and len(valid) == 0


# -- spoken-clean normalization --------------------------------------------------------


def test_to_spoken_strips_markdown_and_passes_is_spoken_clean() -> None:
    raw = "**Situation:** we had `incidents`.\n- one\n- two\n# Heading\n1. first"
    out = to_spoken(raw)
    assert "**" not in out and "`" not in out
    assert "Situation: we had incidents." in out
    assert "one" in out and "two" in out and "Heading" in out and "first" in out
    assert is_spoken_clean(out)  # the eval's own spoken-clean check now passes


def test_to_spoken_keeps_plain_text() -> None:
    assert (
        to_spoken("Tell me about a time you led a team.") == "Tell me about a time you led a team."
    )


def test_clean_example_cleans_turns_but_not_system() -> None:
    ex = _ex(
        (Role.system, "You are a tester."),
        (Role.user, "hi"),
        (Role.assistant, "Sure! **Step 1:** breathe. Then answer."),
    )
    cleaned = clean_example(ex)
    assert cleaned.messages[0].content == "You are a tester."  # system untouched
    assert "**" not in cleaned.messages[2].content
    assert is_spoken_clean(cleaned.messages[2].content)
    validate_example(cleaned)  # still well-formed
