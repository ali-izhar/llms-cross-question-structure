"""Per-pass overdispersion and pass-index trend on borderline columns; null = independent column shuffles.

T_disp = var(ddof=1) of per-pass totals s_k (null mean K/(K-1) sum f_i(1-f_i)), T_trend = rho(k, s_k),
T_pc = |rho(k, top PC score)|; Holm over all p-values. Called by item02_shared.py, item04_dispersion.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Single-threaded BLAS: small matrices, many parallel processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from null_models import borderline_mask, column_shuffle, perm_pvalue_upper, standardize_columns  # noqa: E402

SEED = 20260420
EPS = 0.05
N_PERM = 5000


def statistics(X: np.ndarray) -> tuple[float, float, float]:
    """(T_disp, T_trend, T_pc) for a (K, N_b) 0/1 matrix."""
    K = X.shape[0]
    s = X.sum(axis=1)
    k = np.arange(K)
    Z = standardize_columns(X.astype(np.float64))
    # Top left singular vector of Z = top PC score per pass (sign irrelevant under |rho|).
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    return (float(s.var(ddof=1)), float(spearmanr(k, s)[0]), float(abs(spearmanr(k, U[:, 0])[0])))


def main(models: list[tuple[str, str]],
         out_path: Path,
         row_order: str = "row k = k-th pass in generation order (src/run_experiment.py)") -> dict:
    """Write the per-model statistics and p-values to out_path; models: (name, matrix path)."""
    rng_all = np.random.default_rng(SEED).spawn(len(models))
    out = {"meta": {"seed": SEED, "eps": EPS, "n_perm": N_PERM,
                    "null": "independent column shuffles across passes",
                    "row_order": row_order},
           "models": []}
    for (name, fname), rng in zip(models, rng_all):
        C = np.load(fname).astype(np.int8)
        K = C.shape[0]
        X = C[:, borderline_mask(C, EPS)]
        f = X.mean(axis=0)
        disp, trend, pc = statistics(X)
        null = np.array([statistics(column_shuffle(X, rng)) for _ in range(N_PERM)])
        expected_var = K / (K - 1) * float(np.sum(f * (1 - f)))
        rec = {
            "model": name, "K": int(K), "N_b": int(X.shape[1]),
            "per_pass_total_mean": float(X.sum(1).mean()),
            "var_per_pass_total": disp,
            "var_expected_under_null_analytic": expected_var,
            "var_null_mean_mc": float(null[:, 0].mean()),
            "var_null_p95": float(np.percentile(null[:, 0], 95)),
            "dispersion_ratio_obs_over_expected": disp / expected_var,
            "p_overdispersion": perm_pvalue_upper(disp, null[:, 0]),
            "p_underdispersion": perm_pvalue_upper(-disp, -null[:, 0]),
            "spearman_pass_index_vs_total": trend,
            "p_trend_two_sided": perm_pvalue_upper(abs(trend), np.abs(null[:, 1])),
            "abs_spearman_pass_index_vs_top_pc": pc,
            "p_trend_top_pc_two_sided": perm_pvalue_upper(pc, null[:, 2]),
        }
        out["models"].append(rec)
        print(f"{name:16s} Nb={rec['N_b']:3d} var={disp:6.2f} E0={expected_var:6.2f} "
              f"ratio={rec['dispersion_ratio_obs_over_expected']:.3f} p_disp={rec['p_overdispersion']:.3f} "
              f"rho_k={trend:+.3f} p={rec['p_trend_two_sided']:.3f} |rho_pc|={pc:.3f} p={rec['p_trend_top_pc_two_sided']:.3f}")
    # Holm step-down over every p-value in the table.
    keys = ["p_overdispersion", "p_underdispersion", "p_trend_two_sided", "p_trend_top_pc_two_sided"]
    flat = [(i, k, rec[k]) for i, rec in enumerate(out["models"]) for k in keys]
    order = sorted(range(len(flat)), key=lambda j: flat[j][2])
    running = 0.0
    for rank, j in enumerate(order):
        i, k, pv = flat[j]
        running = max(running, min(1.0, (len(flat) - rank) * pv))
        out["models"][i].setdefault("holm_adjusted", {})[k] = running
    out["meta"]["multiplicity"] = f"Holm over {len(flat)} p-values ({len(models)} models x {len(keys)} tests)"
    Path(out_path).write_text(json.dumps(out, indent=2))
    for rec in out["models"]:
        print(rec["model"], {k: round(v, 3) for k, v in rec["holm_adjusted"].items()})
    return out
