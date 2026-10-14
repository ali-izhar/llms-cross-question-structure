"""Tests for the v3 tokenizer fix (src/canonical_tok.py) and the option rotation (src/plan.py).

The tokenizer tests read the local HF cache; a model that is not cached is skipped.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_tok import _equal_up_to_space_after_special  # noqa: E402
from plan import positions_to_options, rotate  # noqa: E402
from rescore import chosen_option  # noqa: E402

LETTERS = "ABCD"
USER = ("Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
        "Question: What is 6 • 24 ÷ (8 − 5)?\nA. 48\nB. 54\nC. 16\nD. 32\nAnswer:")


def _prompter(model):
    from huggingface_hub.errors import LocalEntryNotFoundError

    from canonical_tok import CanonicalPrompter

    try:
        return CanonicalPrompter(model)
    except (FileNotFoundError, LocalEntryNotFoundError) as e:  # not cached: skip; any other error fails
        pytest.skip(f"{model} not available: {e}")


@pytest.mark.parametrize("model", ["deepseek-ai/deepseek-llm-7b-chat", "01-ai/Yi-1.5-6B-Chat",
                                   "allenai/OLMo-2-1124-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                                   "microsoft/phi-2", "meta-llama/Llama-3.2-1B-Instruct"])
def test_prompt_is_canonical(model):
    cp = _prompter(model)
    text = cp.render(USER)
    ids = cp.ids(USER)
    chk = cp.check_prompt(text, ids)
    assert chk["ok"], chk["problems"]
    assert USER in text


def test_deepseek_keeps_spaces():
    cp = _prompter("deepseek-ai/deepseek-llm-7b-chat")
    ids = cp.ids(USER)
    dec = cp.decode_piecewise(ids)
    assert "User: Answer the following multiple choice question." in dec
    assert "\nA. 48\n" in dec


def test_yi_im_start_is_one_token():
    cp = _prompter("01-ai/Yi-1.5-6B-Chat")
    ids = cp.ids(USER)
    tid = cp.special_added["<|im_start|>"]
    assert ids.count(tid) == cp.render(USER).count("<|im_start|>") == 2


def test_check_rejects_wrong_ids():
    cp = _prompter("Qwen/Qwen2.5-0.5B-Instruct")
    text = cp.render(USER)
    ids = cp.ids(USER)
    bad = cp.tok.encode(text.replace("<|im_start|>", ""), add_special_tokens=False).ids
    assert not cp.check_prompt(text, bad)["ok"]
    split = list(ids)
    j = split.index(cp.special_added["<|im_start|>"])
    # the Yi defect: <|im_start|> sent as several pieces
    split[j:j + 1] = [t for piece in ("<|", "im_start", "|>") for t in cp.tok.encode(piece, add_special_tokens=False).ids]
    chk = cp.check_prompt(text, split)
    assert not chk["ok"]
    assert any("special token" in p for p in chk["problems"])


def test_pinned_date_in_template():
    cp = _prompter("meta-llama/Llama-3.2-1B-Instruct")
    assert "Today Date: 10 Oct 2026" in cp.render(USER)


# rotation and the option index of a parsed letter

def test_rotate_examples():
    assert rotate(["a", "b", "c", "d"], 0) == ["a", "b", "c", "d"]
    assert rotate(["a", "b", "c", "d"], 1) == ["d", "a", "b", "c"]
    assert rotate(["a", "b", "c", "d"], 2) == ["c", "d", "a", "b"]


@given(st.integers(0, 3), st.integers(0, 3))
def test_rotation_maps_option_to_position(r, j):
    opts = ["o0", "o1", "o2", "o3"]
    shown = rotate(opts, r)
    assert shown[(j + r) % 4] == opts[j]
    # the letter at position p maps back to option (p - r) mod 4
    P = np.full((4, 1), "", dtype="<U1")
    P[r, 0] = LETTERS[(j + r) % 4]
    assert chosen_option(P, 4)[r, 0] == j


@given(st.integers(0, 3), st.lists(st.floats(-20, 0), min_size=4, max_size=4))
def test_logprob_positions_to_options(r, lp):
    # the logprob of the letter shown at position p belongs to option (p - r) mod 4
    by_opt = positions_to_options(lp, r)
    for p in range(4):
        assert by_opt[(p - r) % 4] == lp[p]
    shown = rotate(["o0", "o1", "o2", "o3"], r)
    for j in range(4):
        assert shown[(j + r) % 4] == f"o{j}"


# exact decode allowances

def test_equal_up_to_space_after_special():
    added = {"<s>": 1, "</s>": 2, "<|user|>": 3}
    assert _equal_up_to_space_after_special("<s> [INST] hi", "<s>[INST] hi", added)
    assert _equal_up_to_space_after_special("</s> \n<|user|>", "</s>\n<|user|>", added)
    assert not _equal_up_to_space_after_special("<s>  [INST]", "<s>[INST]", added)   # two spaces: no
    assert not _equal_up_to_space_after_special("a b", "ab", added)                   # not after a special
    assert _equal_up_to_space_after_special("<s> x", "<s> x", added)                  # text has the space


def test_apply_strip_flags_phi3():
    cp = _prompter("microsoft/Phi-3-mini-4k-instruct")
    assert cp.strip_flags["<|user|>"] == (False, True)
    t = "<|user|>\nHi\n<|end|>\n<|assistant|>\n"
    assert cp.apply_strip_flags(t) == "<|user|>Hi\n<|end|><|assistant|>"
    text = cp.render("Hello world.\nA. x")
    chk = cp.check_prompt(text, cp.encode_text(text))
    assert chk["ok"] and chk["decode"] == "added_token_strip"


def test_apply_strip_flags_noop_without_flags():
    cp = _prompter("Qwen/Qwen2.5-0.5B-Instruct")
    assert cp.strip_flags == {}
    assert cp.apply_strip_flags("<|im_start|>user\nx") == "<|im_start|>user\nx"


def test_added_token_literal_deepseek():
    cp = _prompter("deepseek-ai/deepseek-llm-7b-chat")
    text = cp.render("What is 6 ÷ 3?")
    ids = cp.encode_text(text)
    assert cp.decode(ids, skip_special_tokens=False) != text      # the ByteLevel decoder prints U+FFFD
    assert cp.decode_piecewise(ids) == text
    assert cp.check_prompt(text, ids)["decode"] == "added_token_literal"
