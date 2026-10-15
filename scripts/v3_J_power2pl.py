"""
J. Power of the all-4-correct G1 test from the valid 2PL fit (J1), and what the second eigenvalue of a
one-ability null follows (J2). Specification: results/analysis/specs/J_power_2pl.md (written before the run).
Run: python v3_J_power2pl.py [--nproc N] [--n N_J1]  -> results/analysis/v3_J_power2pl.json (+ raw_v3_J_per_dataset.json).
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import v2lib  # noqa: F401  (threads = 1 per process, sys.path)
import numpy as np
from scipy.special import expit
from v2lib import write_json

import v3_across as V3
import v3_G_familywise as G
import v3_H_followup as H
import v3_softpref as V
from null_models import split_mask

SEED_J = 3502610143             # new namespace, > 1e8 from SEED_G and SEED_H
STREAM_J = {"j1_data": 1, "j1_null": 2, "j2_null": 3}
S_J1 = (0.0, 0.5, 1.0, 2.0)
N_J1 = 200
N_J1_NULL = 500
N_J2 = 200
NULLS = ("2pl_post", "rasch_jml")
SPEC = Path(__file__).resolve().parents[1] / "results/analysis/specs/J_power_2pl.md"


def seed_j(*key: int) -> list[int]:
    return [SEED_J, *[int(k) for k in key]]


def truth():
    """(fit, keep, stem on kept columns, a_med, sd_theta) of the 2PL fit to the real all-4 matrix."""
    d = V.variant_data("full")
    ctx = V3.Context(V.SCORES["ge4"](d["CR"]))
    fit = ctx.fit_2pl()
    a_eap, _ = fit.eap()
    keep = ctx.keep
    return fit, keep, V.stem_mask(d["subj"])[keep], float(np.median(a_eap[keep])), float(np.std(fit.theta))


def planted(fit, keep, stem, a_med, sd_th, s: float, rng: np.random.Generator) -> np.ndarray:
    """(M, n_keep) 2PL draw (posterior (a_i, b_i)) plus s a_med sd(theta) z_m on STEM columns; split columns."""
    a_nodes, b_nodes = fit.node_params()
    cdf = fit.posterior_cdf()[keep]
    z = rng.standard_normal(fit.theta.size)
    extra = s * a_med * sd_th * z[:, None] * stem[None, :].astype(float)

    def draw(idx: np.ndarray) -> np.ndarray:
        u = rng.random(idx.size) * cdf[idx, -1]
        q = (cdf[idx] < u[:, None]).sum(axis=1)
        a, b = a_nodes[q], b_nodes[q]
        P = expit(a[None, :] * (fit.theta[:, None] - b[None, :]) + extra[:, idx])
        return (rng.random(P.shape) < P).astype(np.int8)

    S = draw(np.arange(stem.size))
    for _ in range(100000):
        bad = np.flatnonzero(~split_mask(S))
        if bad.size == 0:
            return S
        S[:, bad] = draw(bad)
    raise RuntimeError("split sampling did not terminate")


def j1_task(args) -> dict:
    s, rep = args
    si = S_J1.index(s) + 1
    fit, keep, stem, a_med, sd_th = truth()
    X = planted(fit, keep, stem, a_med, sd_th, s, np.random.default_rng(seed_j(STREAM_J["j1_data"], si, rep + 1)))
    ctx = V3.Context(X)
    E = G.run_nulls(ctx, list(NULLS), seed_j(STREAM_J["j1_null"], si, rep + 1), N_J1_NULL, f"J1s{s:g}r{rep}")
    return {"s": s, "rep": rep, "res": {nm: G._strip(G.familywise(ctx.obs, e)) for nm, e in E.items()}}


def loading_pass_corr(X: np.ndarray) -> list[float]:
    """|r| between the eigenvector-2 and -3 loadings and the column pass rate (kept columns)."""
    _, _, Vl, keep = V3.eig_vectors(X)
    f = X[:, keep].mean(axis=0)
    return [float(abs(np.corrcoef(Vl[:, j], f)[0, 1])) for j in (1, 2)]


def j2_task(args) -> dict:
    score, null, k = args
    X = V.SCORES[score](V.variant_data("full")["CR"])
    ctx = V3.Context(X)
    mats, _ = H.null_matrices(ctx, null, seed_j(STREAM_J["j2_null"], k), N_J2, f"J2{score}{null}")
    R = np.array([loading_pass_corr(m) for m in mats])
    q = lambda c: np.percentile(R[:, c], [5, 50, 95]).round(3).tolist()  # noqa: E731
    return {"score": score, "null": null, "observed_abs_r_eigvec2_eigvec3": loading_pass_corr(X),
            "null_abs_r_eigvec2_p05_p50_p95": q(0), "null_abs_r_eigvec3_p05_p50_p95": q(1)}


def summarize_j1(rows: list[dict]) -> dict:
    out = {}
    for s in S_J1:
        rs = [r for r in rows if r["s"] == s]
        out[f"{s:g}"] = {}
        for nm in NULLS:
            recs = [r["res"][nm] for r in rs]
            n = len(recs)
            k_fw = sum(G.g1_reject(x) for x in recs)
            k_r2 = sum(x["valid"] and 2 in x["ranks_above_p99"] for x in recs)
            out[f"{s:g}"][nm] = {"n": n, "frac_valid": float(np.mean([x["valid"] for x in recs])),
                                 "detect_familywise": k_fw / n, "detect_familywise_ci95": V.clopper_pearson(k_fw, n),
                                 "detect_rank2_per_rank": k_r2 / n, "detect_rank2_ci95": V.clopper_pearson(k_r2, n),
                                 "median_lambda2": float(np.median([x["observed_top_eigs"][1] for x in recs]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--n", type=int, default=N_J1)
    a = ap.parse_args()
    t0 = time.time()
    _, keep, stem, a_med, sd_th = truth()
    j2_items = [("ge4", "2pl_post", 1), ("ge4", "rasch_jml", 2), ("rot0", "3pl_post", 3)]
    j1_items = [(s, r) for s in S_J1 for r in range(a.n)]
    with Pool(a.nproc) as pool:
        j2 = pool.map(j2_task, j2_items)
        j1 = pool.map(j1_task, j1_items, chunksize=1)
    write_json("v3_J_power2pl", {
        "meta": {"spec": SPEC.read_text(), "post_hoc": True, "seed": SEED_J, "n_datasets_per_s": a.n,
                 "n_null_draws": N_J1_NULL, "n_j2_draws": N_J2, "n_kept_cols": int(keep.sum()),
                 "n_stem_kept": int(stem.sum()), "a_med": a_med, "sd_theta": sd_th, "seconds": time.time() - t0},
        "J1_power_2pl_truth": summarize_j1(j1),
        "J2_eigvec_vs_pass_rate": j2})
    write_json("raw_v3_J_per_dataset", {"meta": {"note": "per-dataset J1 records of v3_J_power2pl.json"},
                                         "J1_per_dataset": sorted(j1, key=lambda r: (r["s"], r["rep"]))})


if __name__ == "__main__":
    main()
