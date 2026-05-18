"""
Shared MMLU utilities: answer matching, prompt building, question loading.
Used by run_experiment.py, run_ensemble.py, run_mistral.py,
run_temperature_sweep.py, run_semantic_pilot.py, run_semantic_full.py.
"""

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
    return gold_n in pred or pred in gold_n


def build_mmlu_prompt(question: str, choices: list[str]) -> str:
    prompt = "Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def load_questions_from_saved(metadata_path) -> list[dict]:
    """Load the same questions used in the primary experiment.

    Reads saved question metadata, reloads the MMLU dataset, and matches
    questions by their first 100 characters to reconstruct full question
    objects (with choices).
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    choices_map = ["A", "B", "C", "D"]
    by_prefix = {}
    for item in ds:
        prefix = item["question"][:100]
        correct_idx = item["answer"]
        by_prefix[prefix] = {
            "question": item["question"],
            "choices": item["choices"],
            "correct_letter": choices_map[correct_idx],
            "correct_text": item["choices"][correct_idx],
            "subject": item["subject"],
        }

    questions = []
    for m in meta:
        prefix = m["question"][:100]
        if prefix in by_prefix:
            questions.append(by_prefix[prefix])
        else:
            questions.append({
                "question": m["question"],
                "choices": ["A", "B", "C", "D"],
                "correct_letter": m["correct_letter"],
                "correct_text": "",
                "subject": m["subject"],
            })
    return questions
