"""
H follow-ups on the all-4 score (24 models), using the v3_G_familywise G1 test; specification quoted in the
output JSON meta. H1: size of G1 under 2PL / Rasch one-ability truth; H2: removing one model of each sibling pair
P1, P2 vs the 190 removals of 2 other models; H3: letter shares and 2PL residual model-pair correlations.
Run: python v3_H_followup.py {run | assemble | all} [--nproc N]  -> results/analysis/v3_H_followup.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from contextlib import contextmanager
from pathlib import Path

import v2lib  # noqa: F401  (threads = 1 per process, sys.path)
import numpy as np
from scipy.special import expit
from v2lib import OUT, write_json

import v3_across as V3
import v3_G_familywise as G
import v3_softpref as V

SEED_H = 2402610141             # > 1e8 from every earlier seed literal (tested)
STREAM_H = {"h1_truth": 1, "h1_null": 2, "h2_null": 3}
SCORE = "ge4"
NULLS = ("2pl_post", "rasch_jml")
TRUTHS = ("2pl_post", "rasch_jml")      # truth A, truth B
N_H1 = 300
N_H1_NULL = 500
N_H2 = 1000
FP_LIMIT = 0.075
ALPHA = G.ALPHA
P1 = ("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct")
P2 = ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")
SIBLING_PAIRS = [p for c in G.CLUSTERS for p in itertools.combinations(c, 2)]
LOG_DIR = OUT / "logs_H"
CKPT = LOG_DIR / "ckpt"
TRUTH_FILE = OUT / "raw_v3_H_h1_truth_datasets.npz"
SPEC = Path(__file__).resolve().parents[1] / "results/analysis/specs/H_followup.md"


def seed_h(*key: int) -> list[int]:
    if not key or not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k > 0 for k in key):
        raise ValueError(f"stream key components must be positive integers: {key}")
    return [SEED_H, *[int(k) for k in key]]


@contextmanager
def capture_null_matrices(store: list):
    """Inside the block, append every null draw (kept columns) that v3_across passes to corr_eigs to store."""
    orig = V3.corr_eigs

    def f(Y):
        store.append(np.array(Y, dtype=np.int8, copy=True))
        return orig(Y)
    V3.corr_eigs = f
    try:
        yield
    finally:
        V3.corr_eigs = orig


def null_matrices(ctx: V3.Context, null: str, seed, n: int, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(n, M, n_keep) int8 null draws of `null` and their (n, M) eigenvalues, via v3_across.run_suite."""
    mats: list = []
    with capture_null_matrices(mats):
        E = G.run_nulls(ctx, [null], seed, n, tag)[null]
    if len(mats) != n:
        raise RuntimeError(f"captured {len(mats)} matrices for {n} draws")
    return np.stack(mats), E


def real_context() -> V3.Context:
    return V3.Context(V.SCORES[SCORE](V.variant_data("full")["CR"]))


def h1_truth_task(t: int) -> dict:
    """N_H1 datasets from truth TRUTHS[t] fitted to the real all-4 data."""
    ctx = real_context()
    X, E = null_matrices(ctx, TRUTHS[t], seed_h(STREAM_H["h1_truth"], t + 1), N_H1, f"H1truth{t}")
    return {"t": t, "X": X, "E": E}


def h1_task(args) -> dict:
    t, rep = args
    X = np.load(TRUTH_FILE)[f"truth{t}"][rep]
    ctx = V3.Context(X)
    E = G.run_nulls(ctx, list(NULLS), seed_h(STREAM_H["h1_null"], t + 1, rep + 1), N_H1_NULL, f"H1t{t}r{rep}")
    return {"t": t, "rep": rep, "n_cols": int(ctx.keep.sum()),
            "res": {nm: {**G.familywise(ctx.obs, e), "E_top4": e[:, :4].copy()} for nm, e in E.items()}}


def h2_subsets(names: list[str]) -> list[dict]:
    """4 pair removals (one of P1, one of P2), then the 190 reference removals of 2 models outside P1 u P2."""
    for n in P1 + P2:
        if names.count(n) != 1:
            raise ValueError(f"model not in the data exactly once: {n}")
    out = [{"kind": "pair", "removed": [a, b]} for a in P1 for b in P2]
    others = [n for n in names if n not in P1 + P2]
    out += [{"kind": "reference", "removed": list(c)} for c in itertools.combinations(others, 2)]
    for s in out:
        s["keep"] = [n for n in names if n not in s["removed"]]
    return out


def h2_task(args) -> dict:
    idx, keep = args
    d = G.subset_rows(V.variant_data("full"), keep)
    ctx = G.score_context(d, SCORE, NULLS)
    E = G.run_nulls(ctx, list(NULLS), seed_h(STREAM_H["h2_null"], idx + 1), N_H2, f"H2i{idx}")
    return {"idx": idx, "keep": keep,
            "res": {nm: {**G.familywise(ctx.obs, e), "E_top4": e[:, :4].copy()} for nm, e in E.items()}}


V.KIND_FN.update({"H1truth": h1_truth_task, "H1": h1_task, "H2": h2_task})


def expected_2pl(fit, keep: np.ndarray) -> np.ndarray:
    """(M, n_keep) E[sigma(a_i (theta_m - b_i))] over each kept column's grid posterior of (a_i, b_i)."""
    a, b = fit.node_params()                                         # (Q,)
    W = np.exp(fit.log_post[keep])                                  # (n_keep, Q)
    W = W / W.sum(axis=1, keepdims=True)
    P = expit(a[:, None] * (fit.theta[None, :] - b[:, None]))        # (Q, M)
    return (W @ P).T


def h3() -> dict:
    d = V.variant_data("full")
    names = d["names"]
    short = [n.split("/")[-1] for n in names]
    shares = V3.shares_of(d["shown_choice"][:, :, 0])               # (M, 4), unparsed in the denominator
    letters = {short[m]: {"pi_A_B_C_D": shares[m].round(4).tolist(), "max_share": float(shares[m].max()),
                          "max_letter": "ABCD"[int(shares[m].argmax())], "flag_max_gt_0p5": bool(shares[m].max() > 0.5)}
               for m in range(len(names))}
    ctx = real_context()
    fit = ctx.fit_2pl()
    Xk = ctx.X[:, ctx.keep].astype(float)
    R = Xk - expected_2pl(fit, ctx.keep)
    C = np.corrcoef(R)
    iu = np.triu_indices(len(names), 1)
    order = np.argsort(-C[iu])
    rank = np.empty(order.size, dtype=int)
    rank[order] = np.arange(1, order.size + 1)
    pairs = [{"rank": int(rank[k]), "r": float(C[i, j]), "a": short[i], "b": short[j]}
             for k, (i, j) in enumerate(zip(*iu, strict=True))]
    pairs.sort(key=lambda p: p["rank"])
    sib = []
    for a, b in SIBLING_PAIRS:
        i, j = sorted((names.index(a), names.index(b)))
        k = int(np.flatnonzero((iu[0] == i) & (iu[1] == j))[0])
        sib.append({"pair": [short[i], short[j]], "r": float(C[i, j]), "rank": int(rank[k])})
    off = C[iu]
    return {"letter_shares_rotation0": letters, "n_models_max_share_gt_0p5": int(sum(v["flag_max_gt_0p5"]
                                                                                  for v in letters.values())),
            "residual_corr": {"definition": "Pearson r across the 391 split all-4 columns of observed minus "
                                            "2PL-expected correctness (expected = posterior mean of sigma(a_i "
                                            "(theta_m - b_i)) over each column's grid posterior, the MML fit of "
                                            "the paper's 2PL null)",
                              "n_pairs": int(off.size), "offdiag_mean": float(off.mean()), "offdiag_sd": float(off.std()),
                              "top10": pairs[:10], "sibling_pairs": sib}}


def h1_items() -> list:
    return [("H1", (t, r)) for r in range(N_H1) for t in range(len(TRUTHS))]


def h2_items() -> list:
    names = V.variant_data("full")["names"]
    return [("H2", (i, s["keep"])) for i, s in enumerate(h2_subsets(names))]


def part_run(n_proc: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tr = V.run_checkpointed([("H1truth", t) for t in range(len(TRUTHS))], CKPT, n_proc=2)["H1truth"]
    if not TRUTH_FILE.exists():
        np.savez_compressed(TRUTH_FILE, **{f"truth{r['t']}": r["X"] for r in tr},
                            **{f"truth{r['t']}_eigs": r["E"] for r in tr})
    items = []
    h1, h2 = h1_items(), h2_items()
    for k in range(len(h2)):                        # interleave (H2 jobs are longer)
        items += h2[k:k + 1] + h1[3 * k:3 * k + 3]
    items += h1[3 * len(h2):]
    V.run_checkpointed(items, CKPT, n_proc=n_proc)


def _load(items) -> dict:
    todo = [it for it in items if not V._ckpt_file(CKPT, *it).exists()]
    if todo:
        raise RuntimeError(f"{len(todo)} jobs not done")
    return V.run_checkpointed(items, CKPT, n_proc=1)


DEVIATIONS = [
    "H1 truth = the paper's null simulator itself (v3_across.run_null via run_suite): the 2PL truth draws each "
    "question's (a_i, b_i) from its grid posterior (posterior draws, not the posterior mode) and every dataset has "
    "the 391 observed split columns, each conditioned on being split (the paper's column-split handling); the "
    "paper's 2PL/Rasch nulls have no unparsed mask, so neither has the truth. The hostile check (scratchpad/"
    "hostile3/size2pl.py) instead drew all 500 columns unconditioned and kept the split ones.",
    "H1 calibrated p = fraction of truth datasets with p_T <= the real all-4 p_T of G1 (2000 draws); the truth p_T "
    "use 500 draws (grid 1/501). Reported for every (truth, null); the decision uses it only if FP > 0.075.",
    "H2 percentile rank of a pair-removal p_T = 100 (#{ref < p} + #{ref = p} / 2) / 190; 'exceeds the 95th "
    "percentile' = p_T > np.percentile(reference p_T, 95). A pair subset counts as 'rejecting' only with a valid null.",
    "Run detail (no effect on results): the part_run interleave listed the last 18 H1 jobs twice (812 jobs); seeds are "
    "deterministic, so each duplicate rewrote the same checkpoint; assemble reads the 600 unique H1 jobs (loop fixed "
    "after the run).",
    "H3 residual correlations use the MML 2PL of the paper's null (posterior-mean expected correctness), not the "
    "hostile reviewer's JML fit; ranks among the 276 pairs.",
]


def part_assemble() -> None:
    names = V.variant_data("full")["names"]
    g1 = json.loads((OUT / "v3_G_familywise.json").read_text())["G1_familywise"]["by_score_and_null"][SCORE]
    real_p = {nm: g1[nm]["p_T"] for nm in NULLS}
    # H1
    h1r = _load(h1_items())["H1"]
    h1, store = {}, {}
    for t, tr in enumerate(TRUTHS):
        h1[tr] = {}
        for nm in NULLS:
            recs = [r["res"][nm] for r in h1r if r["t"] == t]
            pts = np.array([x["p_T"] for x in recs])
            val = np.array([x["valid"] for x in recs])
            k = int(np.sum(val & (pts < ALPHA)))
            n = len(recs)
            h1[tr][nm] = {"n": n, "fp_valid_and_p_T_lt_alpha": k / n, "fp_ci95": V.clopper_pearson(k, n),
                          "k": k, "frac_valid": float(val.mean()), "median_p_T": float(np.median(pts)),
                          "p_T_quantiles_5_25_50": np.percentile(pts, [5, 25, 50]).tolist(),
                          "per_rank_rule_beyond_ge1_and_valid": float(np.mean([x["valid"] and x["ranks_beyond_ability"] >= 1
                                                                               for x in recs])),
                          "p_S_lt_alpha_and_valid": float(np.mean(val & (np.array([x["p_S"] for x in recs]) < ALPHA))),
                          "real_p_T": real_p[nm], "calibrated_p_of_real": float(np.mean(pts <= real_p[nm]))}
            store[f"truth_{tr}__null_{nm}__p_T"] = pts
            store[f"truth_{tr}__null_{nm}__valid"] = val
    h1_dec = {}
    for nm in NULLS:
        ok = all(h1[tr][nm]["fp_valid_and_p_T_lt_alpha"] <= FP_LIMIT for tr in TRUTHS)
        h1_dec[nm] = {"size_controlled": ok, "fp_by_truth": {tr: h1[tr][nm]["fp_valid_and_p_T_lt_alpha"] for tr in TRUTHS},
                      "real_p_T": real_p[nm],
                      "calibrated_p_by_truth": {tr: h1[tr][nm]["calibrated_p_of_real"] for tr in TRUTHS},
                      "statement": (f"the all-4 {nm} rejection (p_T = {real_p[nm]:.3f}) is size-controlled (FP <= "
                                    f"{FP_LIMIT} under both truths)" if ok else
                                    f"the all-4 {nm} rejection (p_T = {real_p[nm]:.3f}) is NOT size-controlled; "
                                    f"calibrated p = " + ", ".join(f"{h1[tr][nm]['calibrated_p_of_real']:.3f} ({tr} truth)"
                                                                   for tr in TRUTHS))}
    np.savez_compressed(OUT / "raw_v3_H_h1.npz", **store)
    write_json("raw_v3_H_h1", {"per_dataset": [{"t": TRUTHS[r["t"]], "rep": r["rep"], "n_cols": r["n_cols"],
                                                **{nm: G._strip(r["res"][nm]) for nm in NULLS}}
                                               for r in sorted(h1r, key=lambda r: (r["t"], r["rep"]))]})
    # H2
    subs = h2_subsets(names)
    h2r = sorted(_load(h2_items())["H2"], key=lambda r: r["idx"])
    for s, r in zip(subs, h2r, strict=True):
        assert s["keep"] == r["keep"]
    pair_idx = [i for i, s in enumerate(subs) if s["kind"] == "pair"]
    ref_idx = [i for i, s in enumerate(subs) if s["kind"] == "reference"]
    h2, store2 = {"by_null": {}}, {}
    for nm in NULLS:
        ref = np.array([h2r[i]["res"][nm]["p_T"] for i in ref_idx])
        ref_rej = [h2r[i]["res"][nm]["valid"] and h2r[i]["res"][nm]["p_T"] < ALPHA for i in ref_idx]
        q95 = float(np.percentile(ref, 95))
        pairs = []
        for i in pair_idx:
            fw = h2r[i]["res"][nm]
            p = fw["p_T"]
            pairs.append({"removed": [m.split("/")[-1] for m in subs[i]["removed"]], "p_T": p, "valid": fw["valid"],
                          "reject": G.g1_reject(fw), "pct_rank_in_reference": 100 * (np.mean(ref < p) + 0.5 * np.mean(ref == p)),
                          "exceeds_ref_p95": bool(p > q95), "T_pct": fw["T_pct"], "pct_rank1to4": fw["pct_rank1to4"]})
        top_ref = sorted(((h2r[i]["res"][nm]["p_T"], [m.split("/")[-1] for m in subs[i]["removed"]]) for i in ref_idx),
                         reverse=True)[:10]
        h2["by_null"][nm] = {"pair_removals": pairs, "reference_n": len(ref), "reference_p_T_quantiles_5_25_50_75_95":
                             np.percentile(ref, [5, 25, 50, 75, 95]).tolist(), "reference_p95": q95,
                             "reference_max": float(ref.max()), "reference_frac_reject": float(np.mean(ref_rej)),
                             "reference_frac_valid": float(np.mean([h2r[i]["res"][nm]["valid"] for i in ref_idx])),
                             "reference_top10_p_T": top_ref, "n_pairs_exceeding_p95": int(sum(p["exceeds_ref_p95"] for p in pairs))}
        store2[f"{nm}__reference_p_T"] = ref
        store2[f"{nm}__pair_p_T"] = np.array([p["p_T"] for p in pairs])
    # mean reference p_T by removed model
    for nm in NULLS:
        by = {}
        for i in ref_idx:
            for m in subs[i]["removed"]:
                by.setdefault(m.split("/")[-1], []).append(h2r[i]["res"][nm]["p_T"])
        h2["by_null"][nm]["reference_mean_p_T_by_removed_model"] = {m: float(np.mean(v)) for m, v in sorted(by.items())}
    n_ex = {nm: h2["by_null"][nm]["n_pairs_exceeding_p95"] for nm in NULLS}
    if all(v == 4 for v in n_ex.values()):
        dec = "the excess is carried by the two near-duplicate pairs (all 4 pair-removal p_T above the reference 95th percentile under both nulls)"
    elif all(v == 0 for v in n_ex.values()):
        dec = "none of the pair removals exceeds the reference 95th percentile: drop the pair attribution; the excess is not attributable"
    else:
        dec = f"mixed: pair removals above the reference 95th percentile: {n_ex} of 4 (reported as is)"
    h2["decision"] = {"n_pairs_exceeding_ref_p95": n_ex, "statement": dec}
    np.savez_compressed(OUT / "raw_v3_H_h2.npz", **store2)
    write_json("raw_v3_H_h2", {"subsets": [{"idx": i, "kind": s["kind"], "removed": s["removed"],
                                            **{nm: G._strip(h2r[i]["res"][nm]) for nm in NULLS}}
                                           for i, s in enumerate(subs)]})
    h3r = h3()
    write_json("raw_v3_H_h3", h3r)
    write_json("v3_H_followup", {
        "meta": {"specification": SPEC.read_text() if SPEC.exists() else None, "seed_namespace": {"SEED_H": SEED_H,
                 "streams": STREAM_H}, "n": {"H1_datasets_per_truth": N_H1, "H1_null_draws": N_H1_NULL,
                                             "H2_null_draws": N_H2}, "statistic": "v3_G_familywise.familywise (G1)",
                 "real_all4_G1_p_T": real_p, "P1": P1, "P2": P2},
        "H1_size_under_one_ability_truth": {"by_truth_and_null": h1, "decision": h1_dec},
        "H2_pair_removal_vs_reference": h2, "H3_baselines": h3r, "Deviations": DEVIATIONS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["run", "assemble", "all"])
    ap.add_argument("--nproc", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    if a.part in ("run", "all"):
        part_run(a.nproc)
    if a.part in ("assemble", "all"):
        part_assemble()
    print(f"[H] done ({time.time() - t0:.0f}s)", flush=True)
