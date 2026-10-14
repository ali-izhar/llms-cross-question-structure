"""Strict scorers for the data collection (MMLU, HellaSwag, GSM8K).

Multiple choice (MMLU, HellaSwag) -- `parse_choice` (v3.1 rule, frozen before R2)
-------------------------------------------------------------------------------
A reply is read in up to three stages. Stage 1 reads an uppercase letter at the
START of the reply; stage 1b reads a standalone lowercase letter; stage 2 reads an
explicit answer phrase in the FIRST LINE.

Stage 1, leading letter.
  1. Strip surrounding whitespace.
  2. Repeatedly remove these leading wrappers (and whitespace after them):
     "Answer:" (any case), "(", "**".
  3. The next character must be an UPPERCASE letter X in A-D. Then the text after
     X (the "tail") decides, in this order:
       a. REJECT "hedge": the tail starts with ANOTHER option letter joined by "or",
          "and", "&", ",", "/" or plain spaces: "A or D", "A and C", "A, C", "A/B",
          "A D", "B C D" ("A (A. ..." repeats X itself and is not a hedge).
       b. REJECT "negation": the tail starts (with no punctuation before it) with an
          optional bracketed text and then one of
            is|are|was|were + not|incorrect|wrong|false, or the "n't" forms;
            does|do|did|would|should|can|could|will + not, or the "n't" forms;
            cannot, can't, won't.
          "D is not an option", "A (y+3) is not", "B wasn't", "C does not apply",
          "A cannot be". Exception: the tail is the text of option X itself (a
          HellaSwag ending "isn't happy ..."). "C. isn't happy" has punctuation
          first, so it is never a negation.
       c. ACCEPT: the tail is empty ("C").
       d. ACCEPT: the tail starts with a punctuation or symbol character (Unicode
          category P* or S*): "C.", "B) foo", "C, because", "A. A man", "B<|im_end|>".
       e. REJECT "letter_then_alnum": the tail starts with a letter or a digit
          ("Cats", "B2").
       f. The tail starts with whitespace; REST = the text after it.
          ACCEPT if the whitespace contains a newline ("A\n\nExplanation"),
          if REST starts with punctuation/symbol ("A (The man ...)", "B - x"),
          if REST starts with "is" or "would be",
          or if the option texts are known and REST matches the text of option X
          (case-folded, whitespace-collapsed, trailing punctuation removed; one is a
          prefix of the other over at least min(4, len(option)) characters; a reply
          that runs past the option text must break at a word boundary).
          Otherwise:
            X in B, C, D: ACCEPT ("C describes the fallacy", "B Adaptation ...").
              These letters have no article reading.
            X == A: ACCEPT only if REST starts with one of the words
              is|was|would|because|since|as|for|seems|appears ("A because ...",
              "A for Mineral-based ..."); else REJECT "article_ambiguous"
              ("A patient with ...", "A man walks", "A best describes ...").
Stage 1b, lowercase letter (only if stage 1 did not accept or reject by a/b):
  after the wrappers of step 2, the WHOLE remaining reply is one lowercase letter
  a-d with at most one ")", "." or ":" after it: "b", "b)", "(b)", "b.", "Answer: c".
  Reason "lowercase". A lowercase letter followed by anything else is not read.
Stage 2, answer phrase (only if stages 1 and 1b did not accept; a hedge or negation
  found in stage 1 is final). In the first line, the first match of
    "answer is", "answer would be", "answer:" (anywhere in the line, any case), or
    "Option" / "Choice" at the start of the line (after non-word characters),
  followed by optional ":", "(" / "**" and an uppercase letter X in A-D that is not
  followed by another letter; the tail of X is judged by rules a-f. Examples:
  "The correct answer is B", "Based on ..., the answer is C", "Option C",
  "**Choice B**".

Anything else is unparseable (None). An unparseable reply is scored WRONG and is
counted separately (see `score_choice`). `parse_choice` also returns a reason code.

GSM8K -- `extract_final_number`:
  The number after the LAST "####" marker if one exists, otherwise the last
  number in the text. Commas and a leading "$" are ignored. Compared to the
  gold value numerically (|x - gold| < 1e-6).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

LETTERS = "ABCD"
PARSER_VERSION = "v3.1"
_PREFIX = re.compile(r"^(?:answer\s*:|\(|\*\*)\s*", re.IGNORECASE)
_NUMBER = r"-?\$?\s?(?:\d[\d,]*)?\.?\d+"

# a. hedge / list: another option letter right after X.
_HEDGE = re.compile(r"^(?:[ \t]*[,/&][ \t]*|[ \t]+(?:or|and|&)[ \t]+|[ \t]+)\(?([A-D])\b")
# b. negation, optionally after a bracketed option text: "A (y+3) is not".
_NEGATION = re.compile(
    r"^(?:[ \t]*\([^)\n]*\))?[ \t]*(?:"
    r"(?:is|are|was|were)(?:n't|[ \t]+(?:not|incorrect|wrong|false)\b)"
    r"|(?:does|do|did|would|should|could|will)(?:n't|[ \t]+not\b)"
    r"|can[ \t]+not\b|cannot\b|can't|won't)",
    re.IGNORECASE)
_AFFIRM = re.compile(r"^(?:is|would\s+be)\b")
_A_CONNECT = re.compile(r"^(?:is|was|would|because|since|as|for|seems|appears)\b")
_LOWER = re.compile(r"^([a-d])[).:]?$")
_ANSWER_PHRASE = re.compile(
    r"(?:answer\s*(?:is|would\s+be|:)|^\W*(?:option|choice))\s*:?\s*(?:\(|\*\*)*\s*(?=(?-i:[A-D])(?![a-zA-Z]))",
    re.IGNORECASE)

# Reason codes.
ACCEPT_LEAD = "lead"
ACCEPT_LOWER = "lowercase"
ACCEPT_PHRASE = "answer_phrase"
ACCEPTS = (ACCEPT_LEAD, ACCEPT_LOWER, ACCEPT_PHRASE)
REJECT_HEDGE = "hedge"
REJECT_NEGATION = "negation"
REJECT_ARTICLE = "article_ambiguous"     # "A" + word that is not option A's text
REJECT_ALNUM = "letter_then_alnum"      # "Cats", "B2"
REJECT_NONE = "no_letter"


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _norm(s: str) -> str:
    s = " ".join(s.casefold().split())
    return s.rstrip(".,;:!?)]}*\"' ")


def option_text_matches(rest: str, option: str) -> bool:
    """True if `rest` (the reply after "X ") continues with the text of the option."""
    r, o = _norm(rest), _norm(option)
    if not r or not o:
        return False
    n = min(len(r), len(o))
    if n < min(4, len(o)):
        return False
    if r.startswith(o):  # the reply goes on after the full option text: needs a word boundary
        return len(r) == len(o) or not (r[len(o)].isalnum() and o[-1].isalnum())
    return o.startswith(r)


def _judge_tail(letter: str, tail: str, options: Sequence[str] | None) -> str:
    """Return "ok" or a reject reason code for the text that follows the letter."""
    h = _HEDGE.match(tail)
    if h and h.group(1) != letter:  # "A (A. ..." repeats the letter: not a hedge
        return REJECT_HEDGE
    if _NEGATION.match(tail):
        # "C isn't happy ..." can be the option text itself (HellaSwag endings).
        if options is not None and option_text_matches(tail, options[LETTERS.index(letter)]):
            return "ok"
        return REJECT_NEGATION
    if not tail:
        return "ok"
    c = tail[0]
    if _is_punct(c):
        return "ok"
    if not c.isspace():
        return REJECT_ALNUM
    m = re.match(r"\s+", tail)
    ws, rest = m.group(0), tail[m.end():]
    if "\n" in ws or not rest:
        return "ok"
    if _is_punct(rest[0]) or _AFFIRM.match(rest):
        return "ok"
    if options is not None and option_text_matches(rest, options[LETTERS.index(letter)]):
        return "ok"
    if letter != "A":
        return "ok"
    if _A_CONNECT.match(rest):
        return "ok"
    return REJECT_ARTICLE


def parse_choice(reply: str, options: Sequence[str] | None = None) -> tuple[str | None, str]:
    """Return (letter or None, reason code) under the v3.1 rule (module docstring)."""
    if not isinstance(reply, str):
        return None, REJECT_NONE
    if options is not None and len(options) != 4:
        raise ValueError("options must have 4 entries")
    s = reply.strip()
    while True:
        m = _PREFIX.match(s)
        if not m or m.end() == 0:
            break
        s = s[m.end():]
    lead_reason = REJECT_NONE
    if s and s[0] in LETTERS:
        r = _judge_tail(s[0], s[1:], options)
        if r == "ok":
            return s[0], ACCEPT_LEAD
        if r in (REJECT_HEDGE, REJECT_NEGATION):
            return None, r
        lead_reason = r
    lm = _LOWER.match(s)
    if lm:
        return lm.group(1).upper(), ACCEPT_LOWER
    first_line = reply.strip().split("\n", 1)[0]
    m = _ANSWER_PHRASE.search(first_line)
    if m:
        x = first_line[m.end()]
        r = _judge_tail(x, first_line[m.end() + 1:], options)
        if r == "ok":
            return x, ACCEPT_PHRASE
        return None, r
    return None, lead_reason


def parse_choice_letter(reply: str, options: Sequence[str] | None = None) -> str | None:
    """Return the answer letter A-D under the v3.1 rule, or None."""
    return parse_choice(reply, options)[0]


def starts_with_A_word(reply: str) -> bool:
    """Diagnostic (v2): reply starts with "A" + space + lowercase word ("A man", "A is")."""
    return bool(re.match(r"^\s*(?:(?i:answer)\s*:|\(|\*\*|\s)*A[ \t]+[a-z]", reply))


def score_choice(reply: str, gold_letter: str, options: Sequence[str] | None = None) -> tuple[int, str]:
    """Return (correct 0/1, parsed letter or '' if unparseable)."""
    letter = parse_choice_letter(reply, options)
    if letter is None:
        return 0, ""
    return int(letter == gold_letter), letter


def _to_float(tok: str) -> float | None:
    tok = tok.replace(",", "").replace("$", "").replace(" ", "")
    try:
        return float(tok)
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    """Final numeric answer: after the last '####', else the last number."""
    if not isinstance(text, str):
        return None
    marks = list(re.finditer(r"####\s*(" + _NUMBER + ")", text))
    if marks:
        return _to_float(marks[-1].group(1))
    nums = re.findall(_NUMBER, text)
    for tok in reversed(nums):
        v = _to_float(tok)
        if v is not None:
            return v
    return None


def score_gsm8k(text: str, gold: float) -> tuple[int, float | None]:
    """Return (correct 0/1, extracted number or None)."""
    v = extract_final_number(text)
    if v is None:
        return 0, None
    return int(abs(v - float(gold)) < 1e-6), v
