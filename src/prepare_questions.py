"""Freeze the question sets to results/questions_*.json (CPU only, offline HF cache).

MMLU: the 500 original questions mapped to cais/mmlu test rows. HellaSwag: the 200 provenance items, resolved
by ind + exact text. GSM8K: the first 100 test problems. MMLU dev: the exemplar pool of the S2 condition.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results"
PROVENANCE = ROOT / "results" / "provenance"   # question lists of the original runs
LETTERS = "ABCD"


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def prepare_mmlu(ds) -> list[dict]:
    from mmlu_utils import _sampled_mmlu_rows, load_questions_from_saved

    saved_path = PROVENANCE / "questions_Qwen_Qwen2.5-7B-Instruct.json"
    meta = json.loads(saved_path.read_text())
    resolved = load_questions_from_saved(saved_path)
    assert len(resolved) == 500 == len(meta)

    sampled = _sampled_mmlu_rows(ds)
    by_exact: dict[tuple, list[int]] = {}
    for row, it in enumerate(ds):
        key = (it["question"], it["subject"], tuple(it["choices"]), int(it["answer"]))
        by_exact.setdefault(key, []).append(row)

    out = []
    n_sampled_agree = 0
    for i, (m, q) in enumerate(zip(meta, resolved)):
        if q["correct_text"] == "" and q["choices"] == ["A", "B", "C", "D"]:
            raise RuntimeError(f"position {i}: loader fell back to a placeholder (no dataset match)")
        key = (q["question"], q["subject"], tuple(q["choices"]), LETTERS.index(q["correct_letter"]))
        rows = by_exact[key]
        # Exact duplicates exist in MMLU; prefer the row the original sampler drew.
        row = sampled[i] if sampled[i] in rows else rows[0]
        n_sampled_agree += int(sampled[i] == row)
        assert m["question"] == q["question"][:100], i
        assert m["subject"] == q["subject"] and m["correct_letter"] == q["correct_letter"], i
        out.append({
            "pos": i,
            "dataset": "cais/mmlu",
            "config": "all",
            "split": "test",
            "row": int(row),
            "n_exact_duplicate_rows": len(rows),
            "subject": q["subject"],
            "question": q["question"],
            "choices": list(q["choices"]),
            "gold_letter": q["correct_letter"],
            "gold_idx": LETTERS.index(q["correct_letter"]),
        })
    print(f"MMLU: 500 resolved; row == original sampler row at {n_sampled_agree}/500 positions")
    return out


def prepare_mmlu_dev() -> list[dict]:
    from datasets import load_dataset

    dev = load_dataset("cais/mmlu", "all", split="dev")
    return [{
        "row": r, "subject": it["subject"], "question": it["question"],
        "choices": list(it["choices"]), "gold_letter": LETTERS[int(it["answer"])],
    } for r, it in enumerate(dev)]


def prepare_hellaswag() -> list[dict]:
    from datasets import load_dataset

    saved = json.loads((PROVENANCE / "questions_hellaswag.json").read_text())
    ds = load_dataset("Rowan/hellaswag", split="validation")
    # `ind` is not unique in the validation split: also match exact context and endings
    by_ind: dict[int, list[int]] = {}
    for r, v in enumerate(ds["ind"]):
        by_ind.setdefault(int(v), []).append(r)
    out = []
    for i, s in enumerate(saved):
        rows = [r for r in by_ind[int(s["ind"])]
                if ds[r]["ctx"] == s["context"] and list(ds[r]["endings"]) == list(s["endings"])]
        assert len(rows) == 1, f"item {i}: {len(rows)} matching rows"
        row = rows[0]
        it = ds[row]
        assert int(it["label"]) == int(s["correct_idx"]), f"label mismatch at {i}"
        out.append({
            "pos": i, "dataset": "Rowan/hellaswag", "split": "validation", "row": int(row),
            "ind": int(it["ind"]), "source_id": it["source_id"], "activity_label": it["activity_label"],
            "context": it["ctx"], "endings": list(it["endings"]),
            "gold_letter": LETTERS[int(it["label"])], "gold_idx": int(it["label"]),
        })
    print(f"HellaSwag: {len(out)} resolved by ind + exact ctx/endings/label match")
    return out


def prepare_gsm8k() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(100):
        it = ds[i]
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", it["answer"])
        gold = float(m.group(1).replace(",", ""))
        out.append({"pos": i, "dataset": "openai/gsm8k", "config": "main", "split": "test",
                    "row": i, "question": it["question"], "gold_answer": gold,
                    "gold_solution": it["answer"]})
    print("GSM8K: first 100 test problems")
    return out


def main() -> None:
    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    sets = {
        "questions_mmlu.json": prepare_mmlu(ds),
        "questions_mmlu_dev.json": prepare_mmlu_dev(),
        "questions_hellaswag.json": prepare_hellaswag(),
        "questions_gsm8k.json": prepare_gsm8k(),
    }
    for name, items in sets.items():
        (OUT / name).write_text(json.dumps(items, indent=1))
        print(f"wrote {name}: {len(items)} items, sha256[:16]={sha(items)}")


if __name__ == "__main__":
    main()
