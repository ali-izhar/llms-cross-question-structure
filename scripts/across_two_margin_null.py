"""Across-model correlation eigenvalues (split columns) against one-ability nulls: Rasch (JML, CML),
fixed margins (curveball), 2PL sigma(a_i (theta_m - b_i)) by marginal ML (posterior / prior draws), 2PL
slope stress tests. Parametric draws are conditional on every observed split column staying split.
main() is called by item07_across.py and item13_unparseable.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from null_models import (  # noqa: E402
    MP_BUFFER, TwoPLFit, buffered_count, corr_eigs, curveball, fit_twopl_mml, mp_edge,
    perm_pvalue_upper, rasch_cml_rows, rasch_column_ml, rasch_jml, rasch_prob,
    sample_bernoulli_split, split_mask, split_spectrum, twopl_marginal_loglik, twopl_sample,
)

SEED = 20260417
N_DRAWS = 1000
N_RANKS = 4
CURVEBALL_BURNIN = 20000      # trades before the first recorded draw
CURVEBALL_THIN = 2000         # trades between recorded draws (~83 per row)
CURVEBALL_RESTARTS = 200      # independent restarts for the mixing check
CURVEBALL_RESTART_TRADES = 5000
TWOPL_GRID = dict(n_a=41, n_b=401, hw_a=5.0, hw_b=5.0)        # log-slope x difficulty nodes
TWOPL_GRID_CHECK = dict(n_a=61, n_b=801, hw_a=6.0, hw_b=6.0)
TWOPL_N_RESTARTS = 3          # random-start check for the MML optimum
STRESS_SLOPE_FACTORS = (1.25, 1.5, 2.0)   # unfitted stress tests: every slope multiplied


def summarize_null(name: str, eigs: np.ndarray, edges: np.ndarray, obs_eigs: np.ndarray,
                   obs_edge: float, notes: str) -> dict:
    """eigs: (B, M) null eigenvalues (desc); edges: (B,) lambda_+ per draw."""
    counts = (eigs > MP_BUFFER * edges[:, None]).sum(axis=1)
    obs_count = buffered_count(obs_eigs, obs_edge)
    top = eigs[:, :N_RANKS]
    ratio = top / edges[:, None]
    obs_top = obs_eigs[:N_RANKS]
    obs_ratio = obs_top / obs_edge
    p99 = np.percentile(top, 99, axis=0)
    p95 = np.percentile(top, 95, axis=0)
    p99_ratio = np.percentile(ratio, 99, axis=0)
    exceed_raw = obs_top > p99
    exceed_ratio = obs_ratio > p99_ratio

    def contiguous(mask: np.ndarray) -> int:
        n = 0
        for v in mask:
            if not v:
                break
            n += 1
        return n

    return {
        "notes": notes,
        "n_draws": int(eigs.shape[0]),
        "n_split_per_draw": {"mean": float(np.mean(edges_to_nsplit(edges))),
                             "min": int(np.min(edges_to_nsplit(edges))),
                             "max": int(np.max(edges_to_nsplit(edges)))},
        "count_above_1p05_edge": {
            "distribution": {str(k): int(v) for k, v in enumerate(np.bincount(counts)) if v},
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "frac_ge_observed": float(np.mean(counts >= obs_count)),
        },
        "rank_eigs_mean": top.mean(axis=0).tolist(),
        "rank_eigs_p95": p95.tolist(),
        "rank_eigs_p99": p99.tolist(),
        "rank_eig_over_edge_p99": p99_ratio.tolist(),
        "observed_minus_p99": (obs_top - p99).tolist(),
        "mc_pvalue_per_rank": [perm_pvalue_upper(obs_top[j], top[:, j]) for j in range(N_RANKS)],
        "n_observed_ranks_1to4_above_p99": int(exceed_raw.sum()),
        "observed_ranks_above_p99": [int(j + 1) for j in np.flatnonzero(exceed_raw)],
        "contiguous_ranks_above_p99": contiguous(exceed_raw),
        "n_observed_ranks_above_p99_edge_normalized": int(exceed_ratio.sum()),
    }


_M_ROWS = 24  # set in main; used to invert lambda_+ -> N_split


def edges_to_nsplit(edges: np.ndarray) -> np.ndarray:
    """N_split from lambda_+ = (1 + sqrt(N_split / M))^2."""
    return np.rint(_M_ROWS * (np.sqrt(edges) - 1.0) ** 2).astype(int)


def run_parametric(P: np.ndarray, rng: np.random.Generator, conditional: bool) -> tuple[np.ndarray, np.ndarray]:
    """N_DRAWS Bernoulli(P) matrices -> (eigs (N_DRAWS, M), lambda_+ per draw)."""
    M = P.shape[0]
    eigs = np.empty((N_DRAWS, M))
    edges = np.empty(N_DRAWS)
    for d in range(N_DRAWS):
        if conditional:
            S = sample_bernoulli_split(P, rng)
            eigs[d] = corr_eigs(S)
            edges[d] = mp_edge(S.shape[1], M)
        else:
            S = (rng.random(P.shape) < P).astype(np.int8)
            e, lp, _ = split_spectrum(S)
            eigs[d, :e.size] = e
            eigs[d, e.size:] = 0.0
            edges[d] = lp
    return eigs, edges


def main(matrix_path: Path,
         names_path: Path,
         out_path: Path,
         stress_factors: tuple = STRESS_SLOPE_FACTORS,
         eig_store: dict | None = None) -> dict:
    """Run every null on matrix_path and write out_path.

    eig_store: optional dict that receives the raw (n_draws, M) null eigenvalues per null key and "observed".
    """
    global _M_ROWS
    t0 = time.time()
    C = np.load(matrix_path).astype(np.int8)                                     # (24, 500)
    names = json.loads(Path(names_path).read_text())
    M, N = C.shape
    _M_ROWS = M
    smask = split_mask(C)
    X = C[:, smask]                                                              # (24, 460)
    n_split = X.shape[1]
    obs_eigs = corr_eigs(X)
    obs_edge = mp_edge(n_split, M)
    obs_count = buffered_count(obs_eigs, obs_edge)
    print(f"observed: N_split={n_split} lambda_+={obs_edge:.3f} top={np.round(obs_eigs[:6], 2)} "
          f"count>1.05={obs_count}")

    rngs = dict(zip(["a_jml", "a_cml", "b", "b_restart", "c_post", "c_prior",
                     "a_jml_uncond", "c_post_uncond", "c_stress"],
                    np.random.default_rng(SEED).spawn(9)))
    nulls: dict[str, dict] = {}
    fits: dict[str, dict] = {}

    # (a) Rasch, JML on split columns.
    theta_j, b_j, it_j = rasch_jml(X)
    P_j = rasch_prob(theta_j, b_j)                                               # (24, 460)
    fits["rasch_jml"] = {
        "iterations": it_j,
        "max_abs_row_sum_residual": float(np.abs(P_j.sum(1) - X.sum(1)).max()),
        "max_abs_col_sum_residual": float(np.abs(P_j.sum(0) - X.sum(0)).max()),
        "theta": dict(zip(names, theta_j.round(4).tolist())),
    }
    e, ed = run_parametric(P_j, rngs["a_jml"], conditional=True)
    nulls["a_rasch_jml"] = summarize_null("a_rasch_jml", e, ed, obs_eigs, obs_edge,
                                          "Rasch, JML; conditional on split")
    if eig_store is not None:
        eig_store["observed"] = obs_eigs
        eig_store["a_rasch_jml"] = e
    print("a_jml done", time.time() - t0)

    # (a') Rasch, CML theta (all 500 columns; column sums condition), ML b on split columns.
    theta_c = rasch_cml_rows(C)
    b_c = rasch_column_ml(theta_c, X.sum(axis=0).astype(float))
    P_c = rasch_prob(theta_c, b_c)
    fits["rasch_cml"] = {
        "corr_theta_cml_vs_jml": float(np.corrcoef(theta_c, theta_j)[0, 1]),
        "sd_theta_cml": float(theta_c.std()), "sd_theta_jml": float(theta_j.std()),
        "max_abs_row_sum_residual": float(np.abs(P_c.sum(1) - X.sum(1)).max()),
    }
    e, ed = run_parametric(P_c, rngs["a_cml"], conditional=True)
    nulls["a_rasch_cml"] = summarize_null("a_rasch_cml", e, ed, obs_eigs, obs_edge,
                                          "Rasch, CML theta + ML b; conditional on split")
    if eig_store is not None:
        eig_store["a_rasch_cml"] = e
    print("a_cml done", time.time() - t0)

    # (b) Fixed margins (curveball), one chain with burn-in and thinning.
    rng = rngs["b"]
    S = curveball(X, CURVEBALL_BURNIN, rng)
    e = np.empty((N_DRAWS, M))
    for d in range(N_DRAWS):
        S = curveball(S, CURVEBALL_THIN, rng)
        e[d] = corr_eigs(S)
    assert np.array_equal(S.sum(0), X.sum(0)) and np.array_equal(S.sum(1), X.sum(1))
    ed = np.full(N_DRAWS, obs_edge)
    nulls["b_fixed_margin"] = summarize_null("b_fixed_margin", e, ed, obs_eigs, obs_edge,
                                             "curveball chain, exact row+column sums")
    if eig_store is not None:
        eig_store["b_fixed_margin"] = e.copy()
    lag1 = [float(np.corrcoef(e[:-1, j], e[1:, j])[0, 1]) for j in range(N_RANKS)]
    # Mixing check: independent restarts from the observed matrix.
    rng_r = rngs["b_restart"]
    e_r = np.array([corr_eigs(curveball(X, CURVEBALL_RESTART_TRADES, rng_r))
                    for _ in range(CURVEBALL_RESTARTS)])
    nulls["b_fixed_margin"]["mixing"] = {
        "burnin_trades": CURVEBALL_BURNIN, "thin_trades": CURVEBALL_THIN,
        "lag1_autocorr_ranks_1to4": lag1,
        "restart_draws": CURVEBALL_RESTARTS, "restart_trades": CURVEBALL_RESTART_TRADES,
        "restart_rank_means": e_r[:, :N_RANKS].mean(0).tolist(),
        "chain_rank_means": e[:, :N_RANKS].mean(0).tolist(),
        "restart_rank_p99": np.percentile(e_r[:, :N_RANKS], 99, axis=0).tolist(),
    }
    print("b done", time.time() - t0)

    # (c) 2PL, marginal ML over question parameters.
    fit = fit_twopl_mml(C, **TWOPL_GRID)
    fit_fine = fit_twopl_mml(C, **TWOPL_GRID_CHECK)
    rng_s = np.random.default_rng(SEED + 1)
    restarts = [fit_twopl_mml(C, **TWOPL_GRID, theta0=rng_s.standard_normal(M)) for _ in range(TWOPL_N_RESTARTS)]
    a_eap, b_eap = fit.eap()
    W = np.exp(fit.log_post)
    on_edge = (np.isclose(np.abs(fit.z_grid[:, 0]), TWOPL_GRID["hw_a"])
               | np.isclose(np.abs(fit.z_grid[:, 1]), TWOPL_GRID["hw_b"]))
    acc = C.mean(axis=1)
    fits["twopl_mml"] = {
        "converged": fit.converged, "iterations": fit.n_iter,
        "marginal_loglik": fit.loglik,
        "hyper_mu_log_a": float(fit.hyper[0]), "hyper_sd_log_a": float(fit.hyper[1]),
        "hyper_mu_b": float(fit.hyper[2]), "hyper_sd_b": float(fit.hyper[3]),
        "theta_scale": "mean 0, SD 1 over the 24 models",
        "corr_theta_vs_accuracy": float(np.corrcoef(fit.theta, acc)[0, 1]),
        "a_eap_quantiles_0_10_50_90_100_split_cols": np.quantile(a_eap[smask], [0, .1, .5, .9, 1]).tolist(),
        "max_posterior_mass_on_grid_boundary": float((W[:, on_edge]).sum(axis=1).max()),
        "grid": TWOPL_GRID,
        "grid_check": {
            "grid": TWOPL_GRID_CHECK,
            "marginal_loglik": fit_fine.loglik,
            "hyper": fit_fine.hyper.tolist(),
            "max_abs_theta_diff": float(np.abs(fit_fine.theta - fit.theta).max()),
        },
        "random_restarts": {
            "n": TWOPL_N_RESTARTS,
            "marginal_loglik": [r.loglik for r in restarts],
            "converged": [r.converged for r in restarts],
            "max_abs_theta_diff": [float(np.abs(r.theta - fit.theta).max()) for r in restarts],
        },
        "theta": dict(zip(names, fit.theta.round(4).tolist())),
    }
    for key, mode in (("c_post", "posterior"), ("c_prior", "prior")):
        rng = rngs[key]
        e = np.empty((N_DRAWS, M))
        for d in range(N_DRAWS):
            S = twopl_sample(fit, rng, mode=mode, condition_on_split=True)[:, smask]
            # Conditional sampling keeps every observed split column split, so N_split = n_split.
            e[d] = corr_eigs(S)
        nulls[f"{key}_2pl"] = summarize_null(key, e, np.full(N_DRAWS, obs_edge), obs_eigs, obs_edge,
                                             f"2PL MML, (a_i, b_i) from {mode}; conditional on split")
        if eig_store is not None:
            eig_store[f"{key}_2pl"] = e
        print(key, "done", time.time() - t0)

    # Stress tests (not fitted): every slope multiplied by a factor; the loglik change records how
    # strongly the data reject the scaled model.
    ll_fit = twopl_marginal_loglik(C, fit.theta, fit.hyper, fit.z_grid)
    stress_rngs = rngs["c_stress"].spawn(len(stress_factors))
    for factor, rng in zip(stress_factors, stress_rngs):
        stress = TwoPLFit(fit.theta, fit.hyper.copy(), fit.z_grid, fit.log_post, fit.loglik,
                          fit.converged, fit.n_iter)
        stress.hyper[0] += np.log(factor)
        ll_s = twopl_marginal_loglik(C, fit.theta, stress.hyper, fit.z_grid)
        e = np.empty((N_DRAWS, M))
        for d in range(N_DRAWS):
            e[d] = corr_eigs(twopl_sample(stress, rng, mode="posterior", condition_on_split=True)[:, smask])
        key = f"c_post_2pl_stress_slopes_x{factor:g}"
        nulls[key] = summarize_null(
            key, e, np.full(N_DRAWS, obs_edge), obs_eigs, obs_edge,
            f"STRESS TEST, not a fitted null: 2PL posterior draws with every slope multiplied by {factor:g}")
        nulls[key]["marginal_loglik_of_scaled_model_minus_fit"] = float(ll_s - ll_fit)
        if eig_store is not None:
            eig_store[key] = e
        nulls[key]["loglik_note"] = (
            "theta, sd_log_a, mu_b, sd_b held at the fit; the eigenvalue law depends only on "
            "a_i (theta_m - b_i), so this is the invariant 'steeper' direction. Holding the other "
            "parameters fixed makes |delta loglik| an upper bound on the profile penalty.")
        print(key, "done", time.time() - t0)

    # Sensitivity: unconditional versions (re-filter split columns per draw).
    e, ed = run_parametric(rasch_prob(theta_j, b_j), rngs["a_jml_uncond"], conditional=False)
    nulls["a_rasch_jml_unconditional"] = summarize_null(
        "a_uncond", e, ed, obs_eigs, obs_edge,
        "Rasch JML, unconditional; N_split varies; compare edge-normalized field")
    rng = rngs["c_post_uncond"]
    e = np.zeros((N_DRAWS, M))
    ed = np.empty(N_DRAWS)
    for d in range(N_DRAWS):
        S = twopl_sample(fit, rng, mode="posterior", condition_on_split=False)
        ev, lp, _ = split_spectrum(S)
        e[d, :ev.size] = ev
        ed[d] = lp
    nulls["c_post_2pl_unconditional"] = summarize_null(
        "c_uncond", e, ed, obs_eigs, obs_edge,
        "2PL posterior, all 500 columns, re-filter split; compare edge-normalized field")
    print("sensitivity done", time.time() - t0)

    # Tetrachoric feasibility.
    A11 = X.T @ X
    A10 = X.T @ (1 - X)
    A01 = (1 - X).T @ X
    A00 = (1 - X).T @ (1 - X)
    iu = np.triu_indices(n_split, 1)
    zero_cell = (np.minimum(np.minimum(A11, A10), np.minimum(A01, A00)) == 0)[iu]
    tetrachoric = {
        "feasible": False,
        "n_pairs": int(zero_cell.size),
        "frac_pairs_with_a_zero_cell": float(zero_cell.mean()),
        "reason": (
            "With 24 models per 2x2 table, {:.1%} of the {} question pairs have an empty cell, "
            "where the tetrachoric ML estimate sits on the boundary |r|=1. The resulting matrix is "
            "dominated by boundary values and is not positive semidefinite, and the MP edge does "
            "not apply to it. The 2PL null already handles the binary-data artifact that motivates "
            "tetrachoric correlations: null and observed go through the same Pearson pipeline."
        ).format(zero_cell.mean(), zero_cell.size),
    }

    primary = nulls["c_post_2pl"]
    statement = {
        "primary_null": "c_post_2pl (question-matched 2PL, conditional on split)",
        "ranks_above_p99_by_null": {k: v["observed_ranks_above_p99"] for k, v in nulls.items()},
        "supported_sentence": None,
    }

    out = {
        "meta": {
            "seed": SEED, "n_draws_per_null": N_DRAWS, "mp_buffer": MP_BUFFER,
            "matrix": f"{matrix_path} ({M} models x {N} MMLU, tau=0)",
            "columns": "split columns (0 < column sum < 24)",
            "correlation": "columns centered, divided by population SD; R = Z^T Z / M; lambda_+ = (1+sqrt(N_split/M))^2",
            "p99_rule": "observed lambda_j > 99th percentile of null lambda_j at the same rank j",
            "runtime_s": None,
        },
        "observed": {
            "n_split": n_split, "lambda_plus": obs_edge,
            "lambda_plus_buffered": MP_BUFFER * obs_edge,
            "top_eigs": obs_eigs[:8].tolist(), "count_above_1p05_edge": obs_count,
        },
        "fits": fits,
        "nulls": nulls,
        "tetrachoric": tetrachoric,
        "statement": statement,
    }
    # The primary null has one ability direction: count ranks 2..4 above p99, contiguous from rank 2.
    above = [j + 1 in primary["observed_ranks_above_p99"] for j in range(N_RANKS)]
    n_beyond_ability = 0
    for v in above[1:]:
        if not v:
            break
        n_beyond_ability += 1
    statement["rank1_above_primary_p99"] = bool(above[0])
    statement["n_ranks_2to4_contiguous_above_primary_p99"] = n_beyond_ability
    sent = (f"Against a one-factor 2PL null with heterogeneous slopes (question-matched, "
            f"{N_DRAWS} draws), eigenvalues {', '.join(str(j + 1) for j in range(N_RANKS) if above[j]) or 'none'} "
            f"exceed their rank-matched 99th percentiles: one ability direction plus {n_beyond_ability}.")
    statement["supported_sentence"] = sent
    fitted = [k for k in nulls if "stress" not in k]
    common = sorted(set.intersection(*(set(nulls[k]["observed_ranks_above_p99"]) for k in fitted)))
    statement["ranks_above_p99_under_every_fitted_null"] = common
    statement["robustness"] = (
        f"Ranks above p99 under every fitted null ({', '.join(fitted)}): {common or 'none'}. "
        "Under the unfitted stress tests that make every "
        "question steeper than fitted, the ranks above p99 are: "
        + "; ".join(f"x{f:g}: {nulls[f'c_post_2pl_stress_slopes_x{f:g}']['observed_ranks_above_p99']}"
                    for f in stress_factors) + ".")
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v["rank_eigs_p99"], v["observed_ranks_above_p99"],
                          v["count_above_1p05_edge"]["distribution"]) for k, v in nulls.items()}, indent=1))
    print(sent)
    return out
