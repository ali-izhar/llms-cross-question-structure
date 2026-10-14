"""Tests for src/mmlu_utils.py: a bare wrong letter is never scored by text, and colliding question
prefixes resolve to the saved row (prefix + subject + letter, then the saved index)."""

from __future__ import annotations

import json
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mmlu_utils  # noqa: E402
from mmlu_utils import check_answer, load_questions_from_saved, normalize_answer  # noqa: E402

LETTERS = "ABCD"


def _caller_score(answer: str, correct_letter: str, correct_text: str) -> bool:
    """Letter prefix OR text match, as the original runners scored."""
    return answer.upper().startswith(correct_letter) or check_answer(answer, correct_text)


@pytest.mark.parametrize(
    "reply, gold",
    [
        ("C", "Carbon dioxide"),
        ("B", "Both of the above"),
        ("D", "Decreased demand"),
        ("c.", "Carbon dioxide"),
        ("(D)", "Hydrogen and deuterium"),
    ],
)
def test_bare_letter_never_matches_gold_text(reply, gold):
    assert check_answer(reply, gold) is False


def test_reported_case_scored_wrong_by_caller():
    # Gold is A ("Carbon dioxide"); the model replied C.
    assert _caller_score("C", "A", "Carbon dioxide") is False


@pytest.mark.parametrize(
    "reply, gold",
    [
        ("Carbon dioxide", "Carbon dioxide"),
        ("carbon dioxide.", "Carbon dioxide"),
        ("The answer is carbon dioxide", "Carbon dioxide"),
        ("42", "42"),
        ("7", "7"),
    ],
)
def test_text_reply_still_matches(reply, gold):
    assert check_answer(reply, gold) is True


@pytest.mark.parametrize(
    "reply, gold",
    [
        ("carbon", "Carbon dioxide"),  # partial reply contained in gold
        ("14", "4"),  # gold is a substring of a different token
        ("", "Carbon dioxide"),
        ("Carbon dioxide", ""),
    ],
)
def test_partial_or_empty_does_not_match(reply, gold):
    assert check_answer(reply, gold) is False


_letter_reply = st.tuples(
    st.sampled_from(["", "(", "[", "**", " "]),
    st.sampled_from(list("BCDbcd")),
    st.sampled_from(["", ".", ")", ":", "**", " ", "\n"]),
).map("".join)


@settings(max_examples=500)
@given(reply=_letter_reply, gold=st.text(min_size=1, max_size=60))
def test_hypothesis_bare_letter_is_letter_scored_only(reply, gold):
    assert check_answer(reply, gold) is False


@settings(max_examples=500)
@given(
    wrong=st.sampled_from(list(LETTERS)),
    right=st.sampled_from(list(LETTERS)),
    gold=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCD", min_size=1, max_size=40),
)
def test_hypothesis_wrong_letter_never_correct(wrong, right, gold):
    if wrong != right:
        assert _caller_score(wrong, right, gold) is False


@settings(max_examples=500)
@given(gold=st.text(min_size=1, max_size=60))
def test_hypothesis_exact_text_reply_matches(gold):
    n = normalize_answer(gold)
    if n and n not in ("a", "b", "c", "d"):
        assert check_answer(gold, gold) is True


class _FakeDataset:
    """Minimal stand-in for a HF Dataset: row iteration, row and column access."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]


def _row(question, subject, answer, tag):
    return {
        "question": question,
        "subject": subject,
        "answer": answer,
        "choices": [f"{tag}-{c}" for c in LETTERS],
    }


def _install_fake(monkeypatch, rows):
    import datasets

    fake = _FakeDataset(rows)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)
    return fake


def _write_meta(tmp_path, entries):
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(entries))
    return p


PREFIX = "Q" * 100  # every question below shares this 100-character prefix


def test_collision_resolved_by_subject_and_letter(monkeypatch, tmp_path):
    rows = [
        _row(PREFIX + " first", "physics", 0, "r0"),
        _row(PREFIX + " second", "physics", 2, "r1"),  # a last-row pick would be wrong
    ]
    _install_fake(monkeypatch, rows)
    meta = _write_meta(tmp_path, [
        {"index": 0, "question": PREFIX, "subject": "physics", "correct_letter": "A"},
    ])
    (q,) = load_questions_from_saved(meta)
    assert q["question"] == PREFIX + " first"
    assert q["correct_letter"] == "A"
    assert q["correct_text"] == "r0-A"


def test_collision_resolved_by_saved_sample_index(monkeypatch, tmp_path):
    # Two rows equal in prefix, subject AND letter: only the index can decide.
    rows = [
        _row(PREFIX + " first", "physics", 1, "r0"),
        _row(PREFIX + " second", "physics", 1, "r1"),
    ]
    _install_fake(monkeypatch, rows)
    sampled = mmlu_utils._sampled_mmlu_rows(_FakeDataset(rows))
    meta = _write_meta(tmp_path, [
        {"index": i, "question": PREFIX, "subject": "physics", "correct_letter": "B"}
        for i in range(len(sampled))
    ])
    qs = load_questions_from_saved(meta)
    assert [q["choices"][0] for q in qs] == [f"r{r}-A" for r in sampled]


def test_unresolvable_collision_raises(monkeypatch, tmp_path):
    rows = [
        _row(PREFIX + " first", "physics", 1, "r0"),
        _row(PREFIX + " second", "physics", 1, "r1"),
    ]
    _install_fake(monkeypatch, rows)
    meta = _write_meta(tmp_path, [
        {"index": 99, "question": PREFIX, "subject": "physics", "correct_letter": "B"},
    ])
    with pytest.raises(ValueError, match="Ambiguous MMLU match"):
        load_questions_from_saved(meta)


@settings(max_examples=200, deadline=None)
@given(
    answers=st.lists(st.integers(0, 3), min_size=1, max_size=8),
    pick=st.integers(0, 7),
)
def test_hypothesis_saved_letter_is_preserved(answers, pick):
    import datasets
    import tempfile

    pick %= len(answers)
    rows = [_row(PREFIX + f" v{i}", "physics", a, f"r{i}") for i, a in enumerate(answers)]
    fake = _FakeDataset(rows)
    sampled = mmlu_utils._sampled_mmlu_rows(fake)
    idx = sampled.index(pick) if pick in sampled else 10_000
    letter = LETTERS[answers[pick]]
    unique = answers.count(answers[pick]) == 1
    orig = datasets.load_dataset
    datasets.load_dataset = lambda *a, **k: fake
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "q.json")
            with open(p, "w") as f:
                json.dump([{"index": idx, "question": PREFIX, "subject": "physics",
                            "correct_letter": letter}], f)
            if unique or idx != 10_000:
                (q,) = load_questions_from_saved(p)
                assert q["correct_letter"] == letter
                assert q["question"] == PREFIX + f" v{pick}"
            else:
                with pytest.raises(ValueError):
                    load_questions_from_saved(p)
    finally:
        datasets.load_dataset = orig


SAVED = os.path.join(HERE, os.pardir, "results", "provenance", "questions_Qwen_Qwen2.5-7B-Instruct.json")


def _mmlu_cached() -> bool:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset

        load_dataset("cais/mmlu", "all", split="test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not os.path.exists(SAVED), reason="saved question metadata absent")
def test_saved_500_reconstruct_exactly():
    if not _mmlu_cached():
        pytest.skip("cais/mmlu not in the local HF cache")
    with open(SAVED) as f:
        meta = json.load(f)
    qs = load_questions_from_saved(SAVED)
    assert len(qs) == len(meta) == 500
    assert [q["correct_letter"] for q in qs] == [m["correct_letter"] for m in meta]
    assert [q["question"][:100] for q in qs] == [m["question"] for m in meta]
    assert [q["subject"] for q in qs] == [m["subject"] for m in meta]
    assert all(q["correct_text"] for q in qs)
    assert len({(q["question"], tuple(q["choices"])) for q in qs}) == 500
