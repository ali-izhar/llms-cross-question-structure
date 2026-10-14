"""I. Family-wise test (v3_G_familywise G1) without Llama-3.2-1B, and joint removals.

I1: 23 models, rotation 0 and all-4, every step-1 null plus the soft-preference null, 2000 draws each.
I2: all-4, Llama-3.2-1B removed with one model of H.P1 and one of H.P2 (21 models), 2PL and Rasch nulls.
Run: python v3_I_llama.py {run,assemble,all} [--nproc N]; specification quoted in the output JSON meta.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import OUT, write_json

import v3_G_familywise as G
import v3_H_followup as H
import v3_softpref as V

SEED_I = 3602610143             # far (> 1e8) from every other seed literal; tested
STREAM_I = {"i1": 1, "i1_sp": 2, "i2": 3}
LLAMA_1B = V.LLAMA_1B
I1_SCORES = ("rot0", "ge4")
N_I1 = 2000
N_I2 = 1000
I2_NULLS = ("2pl_post", "rasch_jml")
LOG_DIR = OUT / "logs_I"
CKPT = LOG_DIR / "ckpt"
SPEC = Path(__file__).resolve().parents[1] / "results/analysis/specs/I_llama.md"
# Earlier p_T on the stored 1000-draw step-1 nulls, for comparison only.
HOSTILE_REF = {"rot0": {"3pl_post": 0.013, "letter_post": 0.010}, "ge4": {"2pl_post": 0.065, "rasch_jml": 0.116}}


def seed_i(*key: int) -> list[int]:
    if not key or not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k > 0 for k in key):
        raise ValueError(f"stream key components must be positive integers: {key}")
    return [SEED_I, *[int(k) for k in key]]


def data_23() -> dict:
    return V.variant_data("no_llama1b")


def i1_task(args) -> dict:
    """One (score, null) on the 23-model data; same draws as one run_suite call (stream keyed by null index)."""
    score, null = args
    d = data_23()
    ctx = G.score_context(d, score, V.family_nulls(score))
    E = G.run_nulls(ctx, [null], seed_i(STREAM_I["i1"], G.score_code(score)), N_I1, f"I1_{score}_{null}")[null]
    return {"score": score, "null": null, "obs": ctx.obs, "E": E}


def i1_sp_task(_: int = 0) -> dict:
    data, _d = V.sp_data_of("no_llama1b")
    keeps, obs = V.observed_keeps(data)
    keeps = {s: keeps[s] for s in I1_SCORES}
    E, fail = V.sp_null(V.load_base("no_llama1b"), data.gold, data.chosen < 0, keeps, N_I1,
                        np.random.default_rng(seed_i(STREAM_I["i1_sp"], 1)))
    return {"obs": {s: obs[s] for s in I1_SCORES}, "E": E, "n_failed": fail}


def i2_subsets(names: list[str]) -> list[dict]:
    if names.count(LLAMA_1B) != 1:
        raise ValueError("Llama-3.2-1B not in the data exactly once")
    out = []
    for a in H.P1:
        for b in H.P2:
            removed = [LLAMA_1B, a, b]
            out.append({"removed": removed, "keep": [n for n in names if n not in removed]})
    return out


def i2_task(args) -> dict:
    idx, keep = args
    d = G.subset_rows(V.variant_data("full"), keep)
    ctx = G.score_context(d, "ge4", I2_NULLS)
    E = G.run_nulls(ctx, list(I2_NULLS), seed_i(STREAM_I["i2"], idx + 1), N_I2, f"I2i{idx}")
    return {"idx": idx, "keep": keep, "M": len(d["names"]),
            "res": {nm: {**G.familywise(ctx.obs, e), "E_top4": e[:, :4].copy()} for nm, e in E.items()}}


V.KIND_FN.update({"I1": i1_task, "I1sp": i1_sp_task, "I2": i2_task})


def i1_items() -> list:
    return [("I1", (sc, nm)) for sc in I1_SCORES for nm in V.family_nulls(sc)]


def i2_items() -> list:
    return [("I2", (i, s["keep"])) for i, s in enumerate(i2_subsets(V.variant_data("full")["names"]))]


def all_items() -> list:
    longest = {"letter_post": 0, "3pl_post": 1, "2pl_post": 2, "fixed_margin": 3}
    i1 = sorted(i1_items(), key=lambda it: longest.get(it[1][1], 9))
    return i1 + i2_items() + [("I1sp", 0)]


def accuracy_rot0(names: list[str]) -> dict:
    """Rotation-0 accuracy per model (unparsed = wrong)."""
    d = V.variant_data("full")
    acc = d["CR"][:, :, 0].mean(axis=1)
    return {n: float(acc[d["names"].index(n)]) for n in names}


def accuracy_all4(names: list[str]) -> dict:
    d = V.variant_data("full")
    acc = V.SCORES["ge4"](d["CR"]).mean(axis=1)
    return {n: float(acc[d["names"].index(n)]) for n in names}


def i2_from_h2() -> dict:
    """H2 reference removals that include Llama-3.2-1B, and accuracy-matched ones (from raw_v3_H_h2.json)."""
    subs = json.loads((OUT / "raw_v3_H_h2.json").read_text())["subsets"]
    ref = [s for s in subs if s["kind"] == "reference"]
    pairs = [s for s in subs if s["kind"] == "pair"]
    names = V.variant_data("full")["names"]

    def row(s):
        return {"removed": [m.split("/")[-1] for m in s["removed"]],
                **{nm: {"p_T": s[nm]["p_T"], "valid": s[nm]["valid"], "reject": bool(s[nm]["valid"] and s[nm]["p_T"] < G.ALPHA)}
                   for nm in I2_NULLS}}

    def summ(rows):
        out = {"n": len(rows)}
        for nm in I2_NULLS:
            p = np.array([r[nm]["p_T"] for r in rows])
            out[nm] = {"n_reject": int(sum(r[nm]["reject"] for r in rows)), "p_T_min_median_max":
                       [float(p.min()), float(np.median(p)), float(p.max())] if p.size else None}
        return out

    with_l = [row(s) for s in ref if LLAMA_1B in s["removed"]]
    out = {"with_llama1b": {"rows": sorted(with_l, key=lambda r: -r["2pl_post"]["p_T"]), "summary": summ(with_l)},
           "pair_removals": [row(s) for s in pairs]}
    for label, acc_fn in (("rot0_accuracy", accuracy_rot0), ("all4_accuracy", accuracy_all4)):
        acc = acc_fn(names)
        lo, hi = min(acc[n] for n in H.P1 + H.P2), max(acc[n] for n in H.P1 + H.P2)
        strict = [row(s) for s in ref if all(lo <= acc[m] <= hi for m in s["removed"])]
        rec = {"pair_model_accuracy": {n.split("/")[-1]: acc[n] for n in H.P1 + H.P2},
               "range_as_specified": [lo, hi], "as_specified": {"rows": strict, "summary": summ(strict)}}
        if label == "rot0_accuracy":
            band = (0.62, 0.71)        # the band behind the specification's count of 15
            hb = [row(s) for s in ref if all(band[0] <= acc[m] <= band[1] for m in s["removed"])]
            rec["hostile_band"] = {"band": list(band), "rows": sorted(hb, key=lambda r: -r["2pl_post"]["p_T"]),
                                   "summary": summ(hb),
                                   "removed_accuracy_range": ([min(acc[m] for s in ref if all(band[0] <= acc[m] <= band[1]
                                                                                              for m in s["removed"])
                                                                   for m in s["removed"]),
                                                               max(acc[m] for s in ref if all(band[0] <= acc[m] <= band[1]
                                                                                              for m in s["removed"])
                                                                   for m in s["removed"])] if hb else None)}
        out[f"accuracy_matched_{label}"] = rec
    return out


def stored_reference() -> dict:
    """familywise on the stored 1000-draw step-1 nulls of no_llama1b."""
    z = np.load(OUT / "raw_v3_F_family_eigs.npz")
    T = json.loads((OUT / "v3_F_softpref.json").read_text())["step1_construction_family"]["table"]["no_llama1b"]
    out = {}
    for sc in I1_SCORES:
        obs = np.array(T[sc]["top_eigs"])
        out[sc] = {nm: {k: G.familywise(obs, z[f"no_llama1b_{sc}__{nm}"])[k] for k in ("valid", "p_T", "p_S", "T_pct")}
                   for nm in V.family_nulls(sc)}
    return out


DEVIATIONS = [
    "I1 runs the whole step-1 suite of each score (v3_softpref.family_nulls): rotation 0 adds curveball, BA and KG-l "
    "to the listed nulls, all-4 adds curveball; nothing listed is left out. Each null is one job; its draws equal "
    "those of one run_suite call per score (child stream by the null's ALL_NULLS index).",
    "I1 SP null: 'refit' = the stored 23-model base fit (raw_v3_F_base_fits.npz, no_llama1b; the same fit_sp call "
    "as F, which F checked equal to its sp fit), 2000 new draws conditional on the observed unparsed mask.",
    "I2 accuracy-matched control: 'within the accuracy range of the pair models' (rotation-0 accuracy, as "
    "item07; also all-4 accuracy) is reported as specified; the specification's count of 15 comes from the hostile "
    "round-4 band 0.62-0.71 (acc_match.py), which is reported too. The counts differ (see the output).",
]


def part_assemble() -> None:
    items = all_items()
    if any(not V._ckpt_file(CKPT, *it).exists() for it in items):
        raise RuntimeError("not all I jobs are done")
    res = V.run_checkpointed(items, CKPT, n_proc=1)
    i1, store = {sc: {} for sc in I1_SCORES}, {}
    for r in res["I1"]:
        fw = G.familywise(r["obs"], r["E"])
        fw["reject_G1"] = G.g1_reject(fw)
        i1[r["score"]][r["null"]] = fw
        store[f"{r['score']}__{r['null']}"] = r["E"]
    sp = res["I1sp"][0]
    for sc in I1_SCORES:
        fw = G.familywise(sp["obs"][sc], sp["E"][sc])
        fw["reject_G1"] = G.g1_reject(fw)
        fw["n_failed_draws"] = sp["n_failed"][sc]
        i1[sc]["sp_common_tau"] = fw
        store[f"{sc}__sp_common_tau"] = sp["E"][sc]
    g1_full = json.loads((OUT / "v3_G_familywise.json").read_text())["G1_familywise"]["by_score_and_null"]
    table = {sc: {nm: {"valid": fw["valid"], "pct_rank1to4": fw["pct_rank1to4"], "ranks_above_p99": fw["ranks_above_p99"],
                       "ranks_beyond_ability_per_rank": fw["ranks_beyond_ability"], "T_pct": fw["T_pct"],
                       "T_argmax_rank": fw["T_argmax_rank"], "p_T": fw["p_T"], "p_S": fw["p_S"],
                       "G1_rejects_valid_and_p_T_lt_0p05": fw["reject_G1"],
                       "full_24_models_G1": {k: g1_full[sc][nm][k] for k in ("valid", "p_T", "ranks_beyond_ability")}
                       if nm in g1_full[sc] else None,
                       "hostile_stored_1000_draw_p_T": HOSTILE_REF[sc].get(nm)}
                  for nm, fw in i1[sc].items()} for sc in I1_SCORES}
    subs = i2_subsets(V.variant_data("full")["names"])
    i2 = []
    store2 = {}
    for r in sorted(res["I2"], key=lambda r: r["idx"]):
        s = subs[r["idx"]]
        assert s["keep"] == r["keep"] and r["M"] == 21
        rec = {"removed": [m.split("/")[-1] for m in s["removed"]], "M": r["M"]}
        for nm in I2_NULLS:
            fw = G._strip(r["res"][nm])
            fw["reject_G1"] = G.g1_reject(fw)
            rec[nm] = fw
            store2[f"i2_{r['idx']}__{nm}"] = r["res"][nm]["E_top4"]
        i2.append(rec)
    np.savez_compressed(OUT / "raw_v3_I_i1_null_eigs.npz", **store)
    np.savez_compressed(OUT / "raw_v3_I_i2_null_top4.npz", **store2)
    write_json("raw_v3_I_i1", {"by_score_and_null": i1})
    h2x = i2_from_h2()
    write_json("v3_I_llama", {
        "meta": {"specification": SPEC.read_text() if SPEC.exists() else None,
                 "seed_namespace": {"SEED_I": SEED_I, "streams": STREAM_I},
                 "n_draws": {"I1": N_I1, "I2": N_I2}, "statistic": "v3_G_familywise.familywise (G1)",
                 "data_I1": "variant_data('no_llama1b'): 23 models, all 500 questions"},
        "I1_without_llama1b": {"table": table, "stored_1000_draw_step1_reference": stored_reference()},
        "I2_joint_removal": {"new_runs": i2, "from_H2_no_new_run": h2x},
        "Deviations": DEVIATIONS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["run", "assemble", "all"])
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    if a.part in ("run", "all"):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        V.run_checkpointed(all_items(), CKPT, n_proc=a.nproc)
    if a.part in ("assemble", "all"):
        part_assemble()
    print(f"[I] done ({time.time() - t0:.0f}s)", flush=True)
