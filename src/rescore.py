"""Re-score finished runs from their saved texts with the current src/scoring.py (atomic in-place rewrite).

The parser gets each question's option texts (rotated for greedyrot runs). --backup DIR is required unless --dry-run.
    python src/rescore.py [--dir results] [--dry-run] [--backup DIR] [--log FILE]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan import rotate  # noqa: E402
from scoring import LETTERS, PARSER_VERSION, parse_choice, score_gsm8k, starts_with_A_word  # noqa: E402

SCORER_SHA = hashlib.sha256((HERE / "scoring.py").read_bytes()).hexdigest()[:16]


def options_for(meta: dict, qdir: Path, rows: np.ndarray) -> tuple[list[list[str]], str]:
    task = meta["task"]
    if task == "mmlu":
        qs = json.loads((qdir / "questions_mmlu.json").read_text())
        opts = [q["choices"] for q in qs]
    else:
        qs = json.loads((qdir / "questions_hellaswag.json").read_text())
        opts = [q["endings"] for q in qs]
    qrows = [q["row"] for q in qs][: len(rows)]
    if list(map(int, rows)) != qrows:
        raise ValueError(f"{meta['run_id']}: question_rows differ from the question file")
    return opts, task


def rescore_run(npz: Path, qdir: Path, dry: bool, skip_conditions: tuple = ()) -> dict | None:
    js = npz.with_suffix(".json")
    if (not js.exists() or npz.name.endswith(".partial.npz") or npz.name.startswith("mmlu_ensemble")
            or ".tmp." in npz.name):
        return None
    meta = json.loads(js.read_text())
    if meta.get("condition") in skip_conditions:
        return None
    z = dict(np.load(npz))
    texts, gold = z["texts"], z["gold"]
    K, N = texts.shape
    mc = meta["task"] in ("mmlu", "hellaswag")
    rot = meta.get("condition") == "greedyrot"
    C = np.zeros((K, N), np.int8)
    rep: dict = {"run_id": meta["run_id"], "K": K, "N": N}
    if mc:
        opts, _ = options_for(meta, qdir, z["question_rows"])
        P = np.full((K, N), "", dtype="<U1")
        R = np.full((K, N), "", dtype="<U20")
        reasons: collections.Counter = collections.Counter()
        aw = 0
        trans: collections.Counter = collections.Counter()
        examples: dict = {}
        old_P = z["parsed"]
        for k in range(K):
            for i in range(N):
                o = rotate(opts[i], k) if rot else opts[i]
                t = str(texts[k, i])
                letter, why = parse_choice(t, o)
                reasons[why] += 1
                R[k, i] = why
                L = letter or ""
                P[k, i] = L
                g = str(z["gold_rot"][k, i]) if rot else str(gold[i])
                if rot and g != LETTERS[(LETTERS.index(str(gold[i])) + k) % 4]:
                    raise ValueError(f"{meta['run_id']}: gold_rot[{k},{i}] breaks the rotation rule")
                C[k, i] = int(L == g and L != "")
                aw += starts_with_A_word(t)
                if L != str(old_P[k, i]):
                    key = f"{old_P[k, i] or '-'}->{L or '-'} ({why})"
                    trans[key] += 1
                    examples.setdefault(key, t[:60])
        unparse = float((P == "").mean())
        rep["parsed_changes"] = int((P != old_P).sum())
        rep["transitions"] = dict(trans.most_common())
        rep["examples"] = examples
        rep["parse_reason_counts"] = dict(reasons)
        rep["n_article_ambiguous"] = int(reasons.get("article_ambiguous", 0))
        meta["n_A_word_replies"] = int(aw)
        meta["parse_reason_counts"] = dict(reasons)
        meta["parsed_letter_counts"] = {L: int((P == L).sum()) for L in ["A", "B", "C", "D", ""]}
    else:
        P = np.full((K, N), np.nan)
        for k in range(K):
            for i in range(N):
                c, v = score_gsm8k(str(texts[k, i]), float(gold[i]))
                C[k, i] = c
                P[k, i] = np.nan if v is None else v
        unparse = float(np.isnan(P).mean())
    old_C = z["correct"]
    rep.update({
        "correct_changes": int((C != old_C).sum()),
        "correct_0_to_1": int(((old_C == 0) & (C == 1)).sum()),
        "correct_1_to_0": int(((old_C == 1) & (C == 0)).sum()),
        "acc_old": float(old_C.mean()), "acc_new": float(C.mean()),
        "acc_change_points": round(100 * (float(C.mean()) - float(old_C.mean())), 3),
        "unparse_old": float(meta["unparseable_rate"]), "unparse_new": unparse,
        "scorer_old": meta.get("scorer_sha256"), "scorer_new": SCORER_SHA,
    })
    if dry:
        return rep
    p = C.mean(axis=0)
    if rot:
        meta["accuracy_by_rotation"] = [float(C[k].mean()) for k in range(K)]
        meta["unparseable_by_rotation"] = [float((P[k] == "").mean()) for k in range(K)]
        meta["letter_share_by_rotation"] = [{L: float((P[k] == L).mean()) for L in LETTERS} for k in range(K)]
    meta.update({
        "accuracy": float(C.mean()), "unparseable_rate": unparse,
        "N_b_eps0.05": int(((p > 0.05) & (p < 0.95)).sum()) if K > 1 else None,
        "n_always_correct": int((p == 1).sum()), "n_always_wrong": int((p == 0).sum()),
        "scorer_sha256": SCORER_SHA,
    })
    if mc:
        meta["parser_version"] = PARSER_VERSION
        z["parse_reason"] = R
    z["correct"], z["parsed"] = C, P
    if rot:
        z["chosen_option"] = chosen_option(P, K)
    tmp = npz.with_name(npz.name + ".tmp.npz")
    np.savez_compressed(tmp, **z)
    os.replace(tmp, npz)
    write_json_atomic(js, meta)
    return rep


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def chosen_option(P: np.ndarray, K: int) -> np.ndarray:
    """Original option index of the parsed letter at rotation k (-1 if unparseable)."""
    out = np.full(P.shape, -1, np.int8)
    for k in range(K):
        for i in range(P.shape[1]):
            if P[k, i]:
                out[k, i] = (LETTERS.index(P[k, i]) - k) % 4
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "results"))
    ap.add_argument("--qdir", default=None, help="question files (default: --dir)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--skip-condition", action="append", default=[],
                    help="do not rescore runs of this condition (e.g. greedy: the v2 greedy runs, subsumed by R2)")
    a = ap.parse_args()
    d = Path(a.dir)
    qdir = Path(a.qdir) if a.qdir else d
    files = sorted(f for f in d.glob("*.npz") if not f.name.endswith(".partial.npz"))
    if not a.dry_run and not a.backup:
        sys.exit("--backup DIR is required unless --dry-run")
    if not a.dry_run:
        b = Path(a.backup)
        if b.exists():
            sys.exit(f"backup dir {b} exists; refusing to overwrite")
        b.mkdir(parents=True)
        for f in files:
            if json.loads(f.with_suffix(".json").read_text()).get("condition") in a.skip_condition \
                    if f.with_suffix(".json").exists() else False:
                continue
            for g in (f, f.with_suffix(".json")):
                if g.exists():
                    shutil.copy2(g, b / g.name)
    reports = []
    for f in files:
        r = rescore_run(f, qdir, a.dry_run, tuple(a.skip_condition))
        if r:
            reports.append(r)
            print(f"{r['run_id']}: cells changed {r['correct_changes']} (0->1 {r['correct_0_to_1']}, "
                  f"1->0 {r['correct_1_to_0']}); parsed changed {r.get('parsed_changes', '-')}; "
                  f"acc {r['acc_old']:.4f}->{r['acc_new']:.4f} ({r['acc_change_points']:+.2f} pts); "
                  f"unparse {r['unparse_old']:.4f}->{r['unparse_new']:.4f}")
    if a.log:
        write_json_atomic(Path(a.log), {"scorer_sha256": SCORER_SHA, "parser_version": PARSER_VERSION,
                                        "dry_run": a.dry_run, "backup": a.backup, "runs": reports})


if __name__ == "__main__":
    main()
