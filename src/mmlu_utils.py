"""MMLU prompt builder, text-answer matcher, and the loader that recovers the original 500 questions."""

import json
import re
import string


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def check_answer(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    gold_n = normalize_answer(gold)
    if not pred or not gold_n:
        return False
    # A bare option letter ("B", "c.", "(D)") is scored by the caller's letter
    # check only; it must not match the gold text as a substring.
    if pred in ("a", "b", "c", "d"):
        return False
    # Whole-string equality, or the gold text as whole tokens inside a longer
    # reply. Never `pred in gold_n`: a short reply is a substring of many texts.
    return pred == gold_n or f" {gold_n} " in f" {pred} "


def build_mmlu_prompt(question: str, choices: list[str]) -> str:
    prompt = "Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def _sampled_mmlu_rows(ds, n_per_subject: int = 10, max_total: int = 500) -> list[int]:
    """Rows drawn by the original sampler (seed 42 per subject, sorted subjects); position = saved `index`."""
    import numpy as np

    by_subject: dict[str, list[int]] = {}
    for row, subj in enumerate(ds["subject"]):
        by_subject.setdefault(subj, []).append(row)
    rows = []
    for subj in sorted(by_subject):
        items = by_subject[subj]
        np.random.seed(42)
        selected = np.random.choice(len(items), size=min(n_per_subject, len(items)), replace=False)
        for idx in selected:
            rows.append(items[idx])
            if len(rows) >= max_total:
                return rows
    return rows


def load_questions_from_saved(metadata_path) -> list[dict]:
    """Match saved questions to cais/mmlu by 100-char prefix + subject + letter, then by saved `index`.

    An ambiguity the index does not resolve raises ValueError.
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    choices_map = ["A", "B", "C", "D"]
    by_key: dict[tuple, list[int]] = {}
    for row, item in enumerate(ds):
        key = (item["question"][:100], item["subject"], choices_map[item["answer"]])
        by_key.setdefault(key, []).append(row)

    def to_question(row: int) -> dict:
        item = ds[row]
        correct_idx = item["answer"]
        return {
            "question": item["question"],
            "choices": item["choices"],
            "correct_letter": choices_map[correct_idx],
            "correct_text": item["choices"][correct_idx],
            "subject": item["subject"],
        }

    sampled_rows = None
    questions = []
    for m in meta:
        candidates = by_key.get((m["question"][:100], m["subject"], m["correct_letter"]), [])
        if len(candidates) == 1:
            questions.append(to_question(candidates[0]))
        elif len(candidates) > 1:
            if sampled_rows is None:
                sampled_rows = _sampled_mmlu_rows(ds)
            idx = m.get("index")
            if isinstance(idx, int) and 0 <= idx < len(sampled_rows) and sampled_rows[idx] in candidates:
                questions.append(to_question(sampled_rows[idx]))
            else:
                raise ValueError(
                    f"Ambiguous MMLU match for saved index {idx!r} "
                    f"({m['subject']}, {m['correct_letter']}): dataset rows {candidates} "
                    f"share the prefix {m['question'][:100]!r}"
                )
        else:
            questions.append({
                "question": m["question"],
                "choices": ["A", "B", "C", "D"],
                "correct_letter": m["correct_letter"],
                "correct_text": "",
                "subject": m["subject"],
            })
    return questions
