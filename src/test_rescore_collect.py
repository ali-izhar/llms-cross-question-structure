"""Tests for src/rescore.py (rotation runs) and the run guards of src/collect.py (no GPU, no vLLM)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect import is_v3_output, load_pins  # noqa: E402
from plan import rotate  # noqa: E402
from rescore import chosen_option, rescore_run  # noqa: E402

LETTERS = "ABCD"
QS = [
    {"row": 10, "question": "q0", "choices": ["man walks", "two", "three", "four"], "gold_letter": "B", "gold_idx": 1},
    {"row": 11, "question": "q1", "choices": ["red", "green", "blue", "Apriori"], "gold_letter": "D", "gold_idx": 3},
]


def _rot_run(tmp: Path, texts: np.ndarray, gold_rot=None) -> Path:
    (tmp / "questions_mmlu.json").write_text(json.dumps(QS))
    K, N = texts.shape
    gold = np.array([q["gold_letter"] for q in QS])
    if gold_rot is None:
        gold_rot = np.array([[LETTERS[(q["gold_idx"] + k) % 4] for q in QS] for k in range(K)])
    rid = "mmlu_greedyrot_test_model"
    np.savez_compressed(tmp / f"{rid}.npz", texts=texts.astype(str), gold=gold, gold_rot=gold_rot,
                        question_rows=np.array([q["row"] for q in QS]), correct=np.zeros((K, N), np.int8),
                        parsed=np.full((K, N), "", dtype="<U1"), chosen_option=np.full((K, N), -1, np.int8))
    (tmp / f"{rid}.json").write_text(json.dumps({"run_id": rid, "task": "mmlu", "condition": "greedyrot",
                                                 "unparseable_rate": 1.0, "accuracy": 0.0}))
    return tmp / f"{rid}.npz"


def test_rescore_rotation_run(tmp_path):
    # q0 gold = option 1 ("two"); q1 gold = option 3 ("Apriori")
    texts = np.array([
        ["B", "A man walks"],          # r0: q0 B -> option 1 (correct); q1 article_ambiguous
        ["C", "A or B"],               # r1: q0 C -> (2-1)%4 = 1 (correct); q1 hedge
        ["D. three", "B Apriori"],     # r2: q0 D -> 1 (correct); q1 B -> (1-2)%4 = 3 (correct)
        ["A man walks", "c"],          # r3: q0 "A man walks": A shows option (0-3)%4 = 1 = "two": no text match
    ])
    f = _rot_run(tmp_path, texts)
    rep = rescore_run(f, tmp_path, dry=False)
    z = np.load(f)
    meta = json.loads(f.with_suffix(".json").read_text())
    assert z["parsed"].tolist() == [["B", ""], ["C", ""], ["D", "B"], ["", "C"]]
    assert z["chosen_option"].tolist() == [[1, -1], [1, -1], [1, 3], [-1, (2 - 3) % 4]]
    assert z["correct"].tolist() == [[1, 0], [1, 0], [1, 1], [0, 1]]
    assert z["parse_reason"].tolist() == [["lead", "article_ambiguous"], ["lead", "hedge"], ["lead", "lead"],
                                          ["article_ambiguous", "lowercase"]]
    assert meta["accuracy_by_rotation"] == [0.5, 0.5, 1.0, 0.5]
    assert meta["unparseable_by_rotation"] == [0.5, 0.5, 0.0, 0.5]
    assert meta["letter_share_by_rotation"][2] == {"A": 0.0, "B": 0.5, "C": 0.0, "D": 0.5}
    assert meta["parser_version"] == "v3.1"
    assert rep["n_article_ambiguous"] == 2
    assert not list(tmp_path.glob("*.tmp*"))  # atomic writes leave no temp file


def test_rescore_option_text_uses_rotation(tmp_path):
    # r3 shows option 1 ("two") at A, option 0 ("man walks") at B: "B man walks" is option 0 by text
    texts = np.array([["x", "x"], ["x", "x"], ["x", "x"], ["A two", "x"]])
    f = _rot_run(tmp_path, texts)
    rescore_run(f, tmp_path, dry=False)
    z = np.load(f)
    assert z["parsed"][3, 0] == "A" and z["chosen_option"][3, 0] == 1 and z["correct"][3, 0] == 1


def test_rescore_rejects_broken_gold_rot(tmp_path):
    texts = np.array([["B", "D"]] * 4)
    bad = np.array([["B", "D"]] * 4)   # not rotated
    f = _rot_run(tmp_path, texts, gold_rot=bad)
    with pytest.raises(ValueError, match="rotation rule"):
        rescore_run(f, tmp_path, dry=True)


def test_rescore_dry_run_writes_nothing(tmp_path):
    f = _rot_run(tmp_path, np.array([["B", "D"]] * 4))
    before = (f.read_bytes(), f.with_suffix(".json").read_text())
    rep = rescore_run(f, tmp_path, dry=True)
    assert rep["correct_changes"] > 0
    assert (f.read_bytes(), f.with_suffix(".json").read_text()) == before


def test_rescore_requires_backup(tmp_path):
    r = subprocess.run([sys.executable, str(HERE / "rescore.py"), "--dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "--backup" in r.stderr


def test_chosen_option_inverse_of_rotation():
    for r in range(4):
        for j in range(4):
            shown_letter = LETTERS[rotate(list(range(4)), r).index(j)]
            P = np.full((4, 1), "", dtype="<U1")
            P[r, 0] = shown_letter
            assert chosen_option(P, 4)[r, 0] == j


def test_is_v3_output(tmp_path):
    js = tmp_path / "x.json"
    assert not is_v3_output(js)
    js.write_text(json.dumps({"accuracy": 1}))
    assert not is_v3_output(js)           # v2 file: no tokenization record
    js.write_text(json.dumps({"tokenization": {}}))
    assert not is_v3_output(js)           # record without the other conditions (see test_collect_guards.py)


def test_load_pins(tmp_path):
    m = "org/model"
    (tmp_path / "prompt_check.json").write_text(json.dumps({"models": {m: {
        "tokenizer_json_sha256": "abc", "revision": "rev1", "allowed_decode_kinds": ["exact"]}}}))
    (tmp_path / "reference_hashes.json").write_text(json.dumps({"models": {m: {"revision": "rev1", "hashes": {}}}}))
    p = load_pins(m, tmp_path)
    assert p["allowed_decode_kinds"] == {"exact"} and p["revision"] == "rev1"
    (tmp_path / "reference_hashes.json").write_text(json.dumps({"models": {m: {"revision": "rev2", "hashes": {}}}}))
    with pytest.raises(AssertionError):
        load_pins(m, tmp_path)
    with pytest.raises(KeyError):
        load_pins("other/model", tmp_path)


@pytest.mark.parametrize("spec", ["org/model", ":all", "org/model:"])
def test_run_v3_refuses_bad_spec(tmp_path, spec):
    r = subprocess.run(["bash", str(HERE / "run_v3.sh"), str(tmp_path / "o"), "20", spec],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "bad spec" in r.stdout
    assert not (tmp_path / "o").exists()


def test_collect_refuses_run_not_in_plan(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, str(HERE / "collect.py"), "--model", "microsoft/phi-2", "--plan", "v3",
                        "--runs", "mmlu_greedy_microsoft_phi-2", "--n-questions", "2", "--out-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "not in the v3 plan" in (r.stderr + r.stdout)
    assert not (tmp_path / "o").exists()  # refused before anything is written


def test_rescore_skip_condition(tmp_path):
    f = _rot_run(tmp_path, np.array([["B", "D"]] * 4))
    assert rescore_run(f, tmp_path, dry=True, skip_conditions=("greedyrot",)) is None
    assert rescore_run(f, tmp_path, dry=True, skip_conditions=("greedy",)) is not None
