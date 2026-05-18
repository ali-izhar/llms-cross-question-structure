"""
Shared HellaSwag utilities: loader, prompt builder, answer matcher.

Mirrors the pattern of `mmlu_utils.py` so that within-batch / ensemble code
can swap MMLU for HellaSwag with minimal changes.

HellaSwag (Zellers et al., 2019) is a 4-way multiple-choice commonsense
reasoning benchmark. Each example has a context sentence + 4 candidate
continuations; the model must pick the one a human finds plausible.

Dataset path: `Rowan/hellaswag`, validation split (~10k examples).
We sample a stratified subset (200 by default) by `activity_label` and
freeze it to disk so every model sees the same questions.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CHOICE_LETTERS = ["A", "B", "C", "D"]


# ─── Answer matching ──────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def check_hellaswag_answer(completion: str, correct_idx: int) -> bool:
    """Return True iff the model's free-form completion matches the
    correct multiple-choice option.

    Strategy (mirrors the MMLU runner):
    1. Strip whitespace; look at the first non-space character. If it's
       one of A/B/C/D (case-insensitive), compare to the correct letter.
    2. Otherwise look for a stand-alone letter token within the first
       few tokens.
    3. Returns False if nothing parseable is found.
    """
    if not isinstance(completion, str):
        return False
    if not (0 <= correct_idx < 4):
        return False
    correct_letter = CHOICE_LETTERS[correct_idx]

    text = completion.strip()
    if not text:
        return False

    # Common formats: "A", "A.", "A)", "A:", "(A)", "**A**", "Answer: A"
    # Strip a leading "answer:" or "the answer is" if present.
    cleaned = re.sub(r"^(?:the\s+answer\s+is|answer)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.lstrip(" *(`'\"[")
    if cleaned and cleaned[0].upper() == correct_letter:
        return True

    # Fallback: search for a stand-alone letter A-D in the first 30 chars.
    head = text[:30].upper()
    m = re.search(r"\b([ABCD])\b", head)
    if m and m.group(1) == correct_letter:
        return True

    return False


# ─── Prompt builder ───────────────────────────────────────────────────────


def build_hellaswag_prompt(context: str, endings: list[str]) -> str:
    """Format a HellaSwag example as an A/B/C/D multiple-choice prompt.

    The phrasing tracks `build_mmlu_prompt`: explicit "letter only"
    instruction, four lettered options, trailing `Answer:` cue.
    """
    if len(endings) != 4:
        raise ValueError(f"HellaSwag expects 4 endings, got {len(endings)}")

    prompt = (
        "Choose the most plausible continuation of the scenario below. "
        "Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Scenario: {context.strip()}\n"
    )
    for letter, ending in zip(CHOICE_LETTERS, endings):
        prompt += f"{letter}. {ending.strip()}\n"
    prompt += "Answer:"
    return prompt


# ─── Stratified loader ────────────────────────────────────────────────────


def _stratified_sample(
    items: list[dict], n: int, key: str, seed: int
) -> list[dict]:
    """Stratified sample of `n` items from `items`, balanced over `key`.

    Allocates per-stratum counts proportional to the stratum's share of
    the population (with at least 1 sample per stratum that has any
    examples and remains under the budget). Within each stratum the
    selection is deterministic given `seed`.
    """
    import random

    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_stratum[item[key]].append(item)

    total = len(items)
    # Proportional allocation, rounded down.
    alloc: dict[str, int] = {}
    for stratum, group in by_stratum.items():
        share = len(group) * n / total
        alloc[stratum] = int(share)

    # Fill remainder using the largest fractional parts first.
    remainder = n - sum(alloc.values())
    if remainder > 0:
        fracs = sorted(
            by_stratum.keys(),
            key=lambda s: (len(by_stratum[s]) * n / total) - alloc[s],
            reverse=True,
        )
        for stratum in fracs:
            if remainder <= 0:
                break
            if alloc[stratum] < len(by_stratum[stratum]):
                alloc[stratum] += 1
                remainder -= 1

    chosen: list[dict] = []
    for stratum, group in by_stratum.items():
        k = min(alloc.get(stratum, 0), len(group))
        if k <= 0:
            continue
        idxs = list(range(len(group)))
        rng.shuffle(idxs)
        for j in idxs[:k]:
            chosen.append(group[j])

    # Sort for stable order across runs (by the dataset index `ind`).
    chosen.sort(key=lambda x: x.get("ind", 0))
    return chosen


def load_hellaswag(
    n: int = 200,
    seed: int = 42,
    cache_path: str | Path | None = None,
) -> list[dict]:
    """Load `n` HellaSwag validation examples, stratified by activity_label.

    Returns a list of dicts with the keys consumed by the within-batch
    runner:
      - "ind": dataset index (HellaSwag's own field)
      - "activity_label": str
      - "context": str   (== `ctx`)
      - "endings": list[str]   (length 4)
      - "correct_idx": int   (0..3)
      - "correct_letter": str   ("A".."D")
      - "correct_text": str     (the chosen ending text)

    If `cache_path` is provided and exists, the saved selection is
    loaded verbatim (so every model sees the same questions). Otherwise
    a fresh stratified sample is drawn and saved to `cache_path`.
    """
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            saved = json.load(f)
        return [_inflate_record(r) for r in saved]

    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    pool: list[dict] = []
    for item in ds:
        # `label` is stored as a string in HellaSwag; coerce to int.
        try:
            correct_idx = int(item["label"])
        except (TypeError, ValueError):
            continue
        if not (0 <= correct_idx < 4):
            continue
        endings = list(item["endings"])
        if len(endings) != 4:
            continue
        pool.append({
            "ind": int(item["ind"]),
            "activity_label": item["activity_label"],
            "context": item["ctx"],
            "endings": endings,
            "correct_idx": correct_idx,
            "correct_letter": CHOICE_LETTERS[correct_idx],
            "correct_text": endings[correct_idx],
        })

    if len(pool) < n:
        raise ValueError(
            f"HellaSwag validation has {len(pool)} usable examples, asked for {n}"
        )

    chosen = _stratified_sample(pool, n=n, key="activity_label", seed=seed)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(chosen, f, indent=2)

    return chosen


def _inflate_record(r: dict) -> dict:
    """Re-derive convenience fields from a saved record (cache compatibility)."""
    correct_idx = int(r["correct_idx"])
    endings = list(r["endings"])
    return {
        "ind": int(r["ind"]),
        "activity_label": r.get("activity_label", ""),
        "context": r["context"],
        "endings": endings,
        "correct_idx": correct_idx,
        "correct_letter": CHOICE_LETTERS[correct_idx],
        "correct_text": endings[correct_idx],
    }


def stratification_summary(items: Iterable[dict]) -> dict:
    """Quick diagnostic: returns label histogram for a saved selection."""
    counts = Counter(item["activity_label"] for item in items)
    return {
        "n_examples": sum(counts.values()),
        "n_strata": len(counts),
        "top_strata": counts.most_common(10),
    }
