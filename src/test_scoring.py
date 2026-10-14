"""Tests for src/scoring.py."""

from __future__ import annotations

import os
import re
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (ACCEPTS, extract_final_number, option_text_matches, parse_choice,  # noqa: E402
                     parse_choice_letter, score_choice, score_gsm8k, starts_with_A_word)

LETTERS = "ABCD"


OPTS = ["man walks to the store", "The Bees win their first game", "isn't happy but he feeds", "Apriori"]


@pytest.mark.parametrize("reply,expected", [
    ("A", "A"), ("C.", "C"), ("(B)", "B"), ("**D**", "D"), ("Answer: A", "A"),
    ("answer:B", "B"), ("  C\n", "C"), ("B) The mitochondria", "B"), ("D. 42", "D"),
    ("**Answer: C**", "C"), ("Answer: (D)", "D"), ("A.", "A"), ("A: yes", "A"),
    ("C, because", "C"), ("B<|im_end|>", "B"), ("A\n\nExplanation", "A"),
    ("A. A man walks", "A"), ("A is the correct answer", "A"), ("B is the most plausible", "B"),
    ("A would be best", "A"), ("A (The man takes his jacket off)", "A"), ("A (Al) is the correct answer", "A"),
    ("B - because", "B"), ("A (A. Make sure to massage", "A"), ("C. A DNA sequence", "C"),
    ("D. (B) (C)", "D"), ("C. True, False", "C"), ("C. isn't happy but", "C"),
    # v3.1 rule f: B/C/D + word are accepted (no article reading)
    ("C describes the fallacy", "C"), ("B Adaptation to change", "B"), ("D Apr", "D"), ("B the bees", "B"),
    # v3.1 rule f: A + connective word
    ("A because the", "A"), ("A since it", "A"), ("A as the", "A"), ("A for Mineral-based foundations", "A"),
    ("A seems right", "A"), ("A appears to be", "A"), ("A was the", "A"), ("A would", "A"), ("A is", "A"),
    # stage 1b: standalone lowercase letter
    ("b", "B"), ("b)", "B"), ("(b)", "B"), ("b.", "B"), ("d:", "D"), (" a \n", "A"), ("Answer: c", "C"),
    ("**c**", None),
    # stage 2: answer phrase in the first line; Option/Choice at line start
    ("The answer is B", "B"), ("The correct answer is C", "C"), ("The correct answer is: D. Ph", "D"),
    ("Based on the text, the answer is C.", "C"), ("Correct answer: D. To begin", "D"),
    ("The correct answer is (A)", "A"), ("The answer would be **B**", "B"),
    ("Option C", "C"), ("option B.", "B"), ("Choice D", "D"), ("**Option A**", "A"), ("Option: (B)", "B"),
    ("Choice B is correct", "B"), ("- Option C", "C"),
])
def test_parsed(reply, expected):
    assert parse_choice_letter(reply) == expected


@pytest.mark.parametrize("reply,reason", [
    # hedges and lists
    ("A or D", "hedge"), ("A or D:\n\nA.", "hedge"), (" C or D (depending", "hedge"), ("A and C are both", "hedge"),
    ("A, C", "hedge"), ("A,C", "hedge"), ("A, B, or C are all", "hedge"), ("A/B", "hedge"), ("A D", "hedge"),
    ("B C D", "hedge"), ("A & B", "hedge"), ("The answer is A or B", "hedge"), ("Option A or B", "hedge"),
    # negations (v3.1 list)
    ("D is not an option", "negation"), ("A is not correct because", "negation"), ("C is incorrect.", "negation"),
    ("B is wrong", "negation"), ("A are not", "negation"), ("A isn't right", "negation"), ("A (y+3) is not", "negation"),
    ("B is false", "negation"), ("B was not", "negation"), ("C wasn't the", "negation"), ("D were not", "negation"),
    ("C does not apply", "negation"), ("B doesn't", "negation"), ("A do not", "negation"), ("D did not", "negation"),
    ("B would not be", "negation"), ("C wouldn't", "negation"), ("D should not", "negation"), ("A shouldn't", "negation"),
    ("B can not", "negation"), ("C cannot be", "negation"), ("D can't be", "negation"), ("A could not", "negation"),
    ("B couldn't", "negation"), ("C will not", "negation"), ("D won't", "negation"), ("A aren't", "negation"),
    ("The answer is B is not", "negation"),
    # article-led A
    ("A man walks", "article_ambiguous"), ("A patient with this set", "article_ambiguous"),
    ("A resistor in a circuit", "article_ambiguous"), ("A DNA sequence", "article_ambiguous"),
    ("A best describes", "article_ambiguous"), ("A statement that best", "article_ambiguous"),
    ("A wave transfers energy", "article_ambiguous"), ("A Chris", "article_ambiguous"),
    # letter + letter/digit
    ("Cats", "letter_then_alnum"), ("B2", "letter_then_alnum"), ("D2. False", "letter_then_alnum"),
    ("Based on", "letter_then_alnum"), ("Correct: B", "letter_then_alnum"), ("Answer", "letter_then_alnum"),
    ("Dogs bark", "letter_then_alnum"), ("Choices are hard", "letter_then_alnum"),
    # nothing readable
    ("a man", "no_letter"), ("E", "no_letter"), ("", "no_letter"), ("   ", "no_letter"), ("b c", "no_letter"),
    ("bc", "no_letter"), ("b) foo", "no_letter"), ("e", "no_letter"), ("I think A", "no_letter"),
    ("The answer is not A", "no_letter"), ("Answer:", "no_letter"), ("(E)", "no_letter"), ("**", "no_letter"),
    ("1", "no_letter"), ("None of these", "no_letter"), ("The answer is a dog", "no_letter"),
    ("I cannot choose an answer that promotes", "no_letter"), ("The option B is", "no_letter"),
    ("Optional B", "no_letter"), ("Option Bob", "no_letter"),
])
def test_unparseable_with_reason(reply, reason):
    assert parse_choice(reply) == (None, reason)


def test_reason_codes():
    assert parse_choice("C")[1] == "lead"
    assert parse_choice("c")[1] == "lowercase"
    assert parse_choice("Option C")[1] == "answer_phrase"
    assert parse_choice("The answer is C")[1] == "answer_phrase"


def test_answer_phrase_only_first_line():
    assert parse_choice_letter("Let me think.\nThe answer is B") is None
    assert parse_choice_letter("Let me think.\nOption B") is None
    assert parse_choice("The answer is B\nmore")[1] == "answer_phrase"


@pytest.mark.parametrize("reply,options,expected", [
    ("A man walks to the store", OPTS, "A"),          # option A text: "man walks ..."
    ("A man walks", OPTS, "A"),                       # truncated reply, prefix of option
    ("A woman walks", OPTS, None),                    # article-led, not option A's text
    ("A ma", OPTS, None),                             # 2 chars < min(4, len(option))
    ("B The Bees win their first", OPTS, "B"),
    ("B The Bulls win", OPTS, "B"),                   # B/C/D: accepted without a text match
    ("C isn't happy but he feeds", OPTS, "C"),        # option text, not a negation
    ("C isn't correct", OPTS, None),
    ("D Apriori.", OPTS, "D"),
    ("The answer is A man walks", OPTS, "A"),
    ("The answer is A woman", OPTS, None),
])
def test_option_text_rule(reply, options, expected):
    assert parse_choice_letter(reply, options) == expected


def test_short_option_match():
    opts = ["No", "2", "Nope", "Yes"]
    assert parse_choice_letter("A No", opts) == "A"
    assert parse_choice_letter("A Nope", opts) is None   # "No" does not match "Nope" (word boundary)
    assert parse_choice_letter("A no.", opts) == "A"
    assert option_text_matches("no", "No.")
    assert not option_text_matches("nope", "No")


def test_options_must_have_four():
    with pytest.raises(ValueError):
        parse_choice("A", ["x", "y"])


def test_score_counts_unparseable_as_wrong():
    assert score_choice("Based on the text, B", "B") == (0, "")
    assert score_choice("B", "B") == (1, "B")
    assert score_choice("C", "B") == (0, "C")
    assert score_choice("A or B", "A") == (0, "")
    assert score_choice("A man walks", "A") == (0, "")
    assert score_choice("A man walks", "A", OPTS) == (1, "A")
    assert score_choice("b", "B") == (1, "B")


def test_old_bug_cases():
    # Old caller: answer.upper().startswith(letter) -> "Based on" counted as B.
    assert score_choice("Based on", "B") == (0, "")
    # Old hellaswag fallback read the article "a" as option A.
    assert score_choice("a man is", "A") == (0, "")


PUNCT = st.characters(categories=["P", "S"])


@given(st.sampled_from(LETTERS), PUNCT, st.text(max_size=8))
def test_letter_then_punctuation_parses(letter, p, rest):
    # Punctuation or a symbol right after the letter always accepts.
    assert parse_choice_letter(letter + p + rest) == letter


@given(st.sampled_from(LETTERS), st.sampled_from(["", "Answer: ", "(", "**", "** Answer: (", " \n"]))
def test_tolerated_prefixes(letter, prefix):
    assert parse_choice_letter(prefix + letter) == letter
    assert parse_choice_letter(prefix + letter.lower()) == letter


@given(st.text(max_size=40), st.one_of(st.none(), st.lists(st.text(max_size=12), min_size=4, max_size=4)))
def test_result_is_letter_or_none_and_whitespace_invariant(s, opts):
    r, why = parse_choice(s, opts)
    assert r is None or (r in LETTERS and r in s.upper())
    assert (r is None) == (why not in ACCEPTS)
    assert parse_choice_letter("  " + s + "\n", opts) == r


@given(st.sampled_from(LETTERS), st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=6))
def test_letter_followed_by_alnum_is_unparseable(letter, tail):
    assert parse_choice_letter(letter + tail) is None
    assert parse_choice_letter(letter.lower() + tail) is None


@given(st.text(alphabet=st.characters(categories=["Ll"]), min_size=1, max_size=20))
def test_lowercase_start_is_unparseable_unless_standalone(s):
    r = parse_choice_letter(s)
    if re.fullmatch(r"[a-d]", s):
        assert r == s.upper()
    else:
        assert r is None


@given(st.sampled_from("abcd"), st.sampled_from(["", ")", ".", ":"]), st.sampled_from(["", "(", "Answer: ", " "]))
def test_lowercase_standalone(x, suffix, prefix):
    assert parse_choice(prefix + x + suffix) == (x.upper(), "lowercase")
    assert parse_choice_letter(prefix + x + suffix + " word") is None


@given(st.sampled_from(LETTERS), st.sampled_from(LETTERS), st.sampled_from([" or ", " and ", ", ", ",", "/", " ", " & "]),
       st.text(max_size=10))
def test_hedges_rejected(x, y, joiner, rest):
    reply = x + joiner + y + (" " + rest if rest else "")
    if x != y:
        assert parse_choice(reply) == (None, "hedge")


NEGS = ["is not", "are not", "is incorrect", "is wrong", "isn't", "IS NOT", "was not", "wasn't", "were not",
        "does not", "doesn't", "do not", "did not", "would not", "wouldn't", "should not", "can not", "cannot",
        "can't", "could not", "couldn't", "will not", "won't", "aren't", "is false"]


@given(st.sampled_from(LETTERS), st.sampled_from(NEGS), st.text(max_size=10))
def test_negations_rejected(x, neg, rest):
    assert parse_choice(f"{x} {neg} {rest}") == (None, "negation")


@given(st.sampled_from(LETTERS), st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
       st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", max_size=12))
def test_letter_space_word(x, word, more):
    reply = f"{x} {word} {more}"
    r, why = parse_choice(reply)
    if why in ("hedge", "negation"):
        return
    if x != "A" or re.match(r"(?:is|was|would|because|since|as|for|seems|appears)\b", word):
        assert r == x
    else:
        assert (r, why) == (None, "article_ambiguous")
        opts = ["zzzz"] * 4
        opts[0] = f"{word} {more}"
        assert parse_choice_letter(reply, opts) == "A"


@given(st.sampled_from(LETTERS), st.sampled_from(["The answer is ", "The correct answer is: ", "Answer: ",
                                                 "So, the answer would be ", "Final answer: (", "Option ",
                                                 "Choice ", "**Option ", "Option: "]))
def test_answer_phrase(x, phrase):
    assert parse_choice_letter(phrase + x) == x
    assert parse_choice_letter(phrase + x + ".") == x


# GSM8K

@pytest.mark.parametrize("text,expected", [
    ("so 3+4 = 7\n#### 7", 7.0), ("#### 1,234", 1234.0), ("The total is $18.", 18.0),
    ("#### 5\nwait, #### 6", 6.0), ("answer -3", -3.0), ("no number here", None),
    ("costs $1,000.50 total", 1000.5), ("#### $42", 42.0), ("We get 0.5", 0.5),
])
def test_extract(text, expected):
    assert extract_final_number(text) == expected


@given(st.integers(-10**6, 10**6))
def test_extract_roundtrip(n):
    assert extract_final_number(f"Step 1: 2 + 2 = 4.\n#### {n:,}") == float(n)
    assert extract_final_number(f"blah 17 then {n}") == float(n)
    assert score_gsm8k(f"#### {n}", n) == (1, float(n))
    assert score_gsm8k(f"#### {n + 1}", n)[0] == 0


def test_A_word_flag():
    assert starts_with_A_word("A man walks")
    assert starts_with_A_word(" A is the correct answer")
    assert parse_choice_letter("A man walks") is None  # flagged and, since v3, unparseable (article_ambiguous)
    assert not starts_with_A_word("A.")
    assert not starts_with_A_word("B is right")
