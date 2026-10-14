"""Rotation-run analyses: near ties, replicate A vs B agreement, and v2 greedy vs rotation 0.

    .venv/bin/python src/r2_analysis.py [--r2 results/r2] [--v2 results] [--out results/r2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan import ENSEMBLE_MODELS, GREEDYROT, run_id, slug  # noqa: E402
from scoring import parse_choice  # noqa: E402

EPS = 0.13  # nats: the largest first-token shift seen between engine starts before deterministic kernels


def near_ties(z) -> dict:
    """Cells whose top-two letter logprobs differ by < EPS; "undetermined" if only one letter is in the top 20
    and its lower-bound gap (vs the 20th token) is < EPS."""
    L = z["letter_logprob_first"]                       # [4, N, 4]
    K, _, _ = L.shape
    srt = np.sort(np.where(np.isfinite(L), L, -np.inf), axis=2)[:, :, ::-1]
    top1, top2 = srt[:, :, 0], srt[:, :, 1]
    two = np.isfinite(top2) & np.isfinite(top1)
    with np.errstate(invalid="ignore"):
        gap = np.where(two, top1 - top2, np.nan)
    last = z["top_lp_first"][:, :, -1]                  # lowest of the saved top tokens
    lower = np.where(~two & np.isfinite(top1), top1 - last, np.nan)
    return {
        "near_tie_by_rotation": [int(((gap[k] < EPS) & two[k]).sum()) for k in range(K)],
        "undetermined_by_rotation": [int((lower[k] < EPS).sum()) for k in range(K)],
        "no_letter_in_top20_by_rotation": [int((~np.isfinite(top1[k])).sum()) for k in range(K)],
        "near_tie_any_rotation_questions": int((((gap < EPS) & two).any(axis=0)).sum()),
        "gap_median_by_rotation": [float(np.nanmedian(gap[k])) for k in range(K)],
    }


def gaps(z) -> np.ndarray:
    L = z["letter_logprob_first"]
    srt = np.sort(np.where(np.isfinite(L), L, -np.inf), axis=2)[:, :, ::-1]
    with np.errstate(invalid="ignore"):
        g = srt[:, :, 0] - srt[:, :, 1]
    return np.where(np.isfinite(g), g, np.inf)


def a_vs_b(za, zb) -> dict:
    K = za["parsed"].shape[0]
    flip = za["parsed"] != zb["parsed"]
    cflip = za["correct"] != zb["correct"]
    near = (gaps(za) < EPS) | (gaps(zb) < EPS)
    out = {"letter_agreement_by_rotation": [float(1 - flip[k].mean()) for k in range(K)],
           "correct_agreement_by_rotation": [float(1 - cflip[k].mean()) for k in range(K)],
           "n_letter_flips_by_rotation": [int(flip[k].sum()) for k in range(K)],
           "n_correct_flips_by_rotation": [int(cflip[k].sum()) for k in range(K)],
           "n_texts_differ": int((za["texts"] != zb["texts"]).sum()),
           "max_abs_diff_letter_logprob": float(np.nanmax(np.abs(za["letter_logprob_first"] - zb["letter_logprob_first"])))
           if np.isfinite(za["letter_logprob_first"]).any() else None,
           "n_flips": int(flip.sum()), "n_flips_on_near_tie": int((flip & near).sum()),
           "near_tie_rate_all_cells": float(near.mean())}
    out["flip_near_tie_share"] = out["n_flips_on_near_tie"] / out["n_flips"] if out["n_flips"] else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", default=str(HERE.parent / "results" / "r2"))
    ap.add_argument("--v2", default=str(HERE.parent / "results"))
    ap.add_argument("--archive", default=str(HERE.parent / "results" / "provenance" / "v2_broken_tokenization"))
    ap.add_argument("--rep", default=str(HERE.parent / "results" / "r2_rep"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r2, v2 = Path(a.r2), Path(a.v2)
    out = Path(a.out) if a.out else r2
    qs = json.loads((HERE.parent / "results" / "questions_mmlu.json").read_text())
    nt, ag, ab = {}, {}, {}
    for m in ENSEMBLE_MODELS:
        f = r2 / f"{run_id(m, GREEDYROT)}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        nt[m] = near_ties(z)
        fb = Path(a.rep) / f.name
        if fb.exists():
            ab[m] = a_vs_b(z, np.load(fb))
        g = v2 / f"mmlu_greedy_{slug(m)}.npz"
        src = "results"  # v2_source labels keep the paths the files had when first written
        if not g.exists() and (Path(a.archive) / g.name).exists():
            g = Path(a.archive) / g.name
            src = "archive (non-canonical v2 prompt ids)"
        elif not g.exists():
            g = HERE.parent / "results" / "provenance" / "v2_greedy" / g.name
            src = "archive/results/v2_greedy"
        if g.exists():
            y = np.load(g)
            N = z["parsed"].shape[1]
            p2 = np.array([parse_choice(str(y["texts"][0, i]), qs[i]["choices"])[0] or "" for i in range(N)])
            c2 = np.array([int(p2[i] != "" and p2[i] == qs[i]["gold_letter"]) for i in range(N)])
            p0, c0 = z["parsed"][0], z["correct"][0]
            ag[m] = {"v2_source": src, "letter_agreement": float((p2 == p0).mean()),
                     "n_letter_differ": int((p2 != p0).sum()), "correct_agreement": float((c2 == c0).mean()),
                     "acc_v2_greedy_v31": float(c2.mean()), "acc_r2_r0": float(c0.mean()),
                     "texts_identical": int(sum(str(y["texts"][0, i]) == str(z["texts"][0, i]) for i in range(N)))}
    (out / "near_ties.json").write_text(json.dumps({"eps": EPS, "models": nt}, indent=1))
    (out / "v2_vs_r2_rotation0.json").write_text(json.dumps(ag, indent=1))
    if ab:
        tot = sum(v["n_flips"] for v in ab.values())
        tot_near = sum(v["n_flips_on_near_tie"] for v in ab.values())
        (out / "a_vs_b.json").write_text(json.dumps({"eps": EPS, "total_flips": tot, "total_flips_on_near_tie": tot_near,
                                                     "models": ab}, indent=1))
    for m in nt:
        print(m, nt[m]["near_tie_by_rotation"], ag.get(m, {}).get("letter_agreement"))


if __name__ == "__main__":
    main()
