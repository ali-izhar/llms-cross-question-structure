"""Power of the within-model test (MP strict/buffered, Horn, MC level-5%, within-subject r) against a planted
pass-level factor c_ki = 1[sqrt(rho) u_k + sqrt(1 - rho) e_ki < Phi^-1(f_i)] on a (100, 500) matrix.
Scenarios: null, diffuse, random 30/9-question clusters, subject cluster. main() is called by item03_power.py.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Single-threaded BLAS: small matrices, many parallel processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from null_models import (  # noqa: E402
    MP_BUFFER, borderline_mask, column_shuffle, corr_eigs, mp_edge,
    perm_pvalue_upper, probit_factor_sample,
)

SEED = 20260419
EPS = 0.05
K = 100
N_SIM = 500             # simulated matrices per scenario
N_SIM_NULL = 2000       # rho = 0 draws (calibration of the MC and subject tests)
N_SURR = 100            # Horn surrogates per simulated matrix
N_PERM_SUBJECT = 2000   # column-shuffle permutations, observed within-subject test
CLUSTER_SIZES = {"cluster30": 30, "cluster9": 9}

SCENARIOS = (
    [("null", 0.0, None, "random")]
    + [("diffuse", r, None, "random") for r in (0.01, 0.02, 0.05)]
    + [(name, r, n, "random") for name, n in CLUSTER_SIZES.items() for r in (0.1, 0.2)]
    + [("cluster9_subject", r, 9, "subject") for r in (0.1, 0.2)]
)


def within_subject_mean_r(X: np.ndarray, same: np.ndarray, iu: tuple) -> float:
    """Mean Pearson r over same-subject pairs; X (K, n) fixed columns."""
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X.T)
    return float(np.nanmean(R[iu][same]))


def horn_p95_rank1(X: np.ndarray, rng: np.random.Generator) -> float:
    """95th percentile of lambda_1 over N_SURR column-shuffle surrogates."""
    return float(np.percentile([corr_eigs(column_shuffle(X, rng))[0] for _ in range(N_SURR)], 95))


def run_scenario(args) -> dict:
    name, rho, n_load, placement, f, b_idx, subj_cluster, same, seed, n_sim = args
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(b_idx.size, 1)
    rows = []
    for i_sim in range(n_sim):
        loading = np.zeros(f.size)
        if rho > 0:
            if n_load is None:
                sel = b_idx
            elif placement == "subject":
                sel = subj_cluster
            else:
                sel = rng.choice(b_idx, n_load, replace=False)
            loading[sel] = np.sqrt(rho)
        S = probit_factor_sample(f, loading, K, rng)                # (K, 500)
        bm = borderline_mask(S, EPS)
        X = S[:, bm]
        e = corr_eigs(X)
        lp = mp_edge(X.shape[1], K)
        # Horn surrogates are the costly part: computed on the first N_SIM draws only.
        horn = horn_p95_rank1(X, rng) if i_sim < N_SIM else np.nan
        subj_stat = within_subject_mean_r(S[:, b_idx], same, iu)
        rows.append((e[0], lp, horn, subj_stat, X.shape[1]))
    arr = np.array(rows)
    return {"name": name, "rho": rho, "n_loaded": n_load, "placement": placement, "draws": arr}


def horn_rate(lam: np.ndarray, horn: np.ndarray) -> float:
    ok = np.isfinite(horn)
    return float(np.mean(lam[ok] > horn[ok]))


def implied_phi(f_b: np.ndarray, rho: float) -> float:
    """Mean phi over borderline pairs implied by latent correlation rho.

    phi_ij = (Phi2(t_i, t_j; rho) - p_i p_j) / sqrt(p_i q_i p_j q_j).
    """
    t = norm.ppf(f_b)
    iu = np.triu_indices(f_b.size, 1)
    ti, tj = t[iu[0]], t[iu[1]]
    pi, pj = f_b[iu[0]], f_b[iu[1]]
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    p11 = mvn.cdf(np.column_stack([ti, tj]))
    return float(np.mean((p11 - pi * pj) / np.sqrt(pi * (1 - pi) * pj * (1 - pj))))


def main(matrix_path: Path,
         questions_path: Path,
         out_path: Path,
         max_workers: int | None = None) -> dict:
    """Run all scenarios and write the power table to out_path; questions_path: JSON list with "subject"."""
    C = np.load(matrix_path).astype(np.int8)                         # (100, 500)
    qs = json.loads(Path(questions_path).read_text())
    assert C.shape == (K, 500)
    f = C.mean(axis=0)
    bmask = borderline_mask(C, EPS)
    b_idx = np.flatnonzero(bmask)                                     # (121,)
    subj = np.array([q["subject"] for q in qs])
    subj_b = subj[b_idx]
    iu = np.triu_indices(b_idx.size, 1)
    same = subj_b[iu[0]] == subj_b[iu[1]]

    # Subject-concentrated 9-question cluster: fill from the largest subjects.
    names, counts = np.unique(subj_b, return_counts=True)
    order = np.lexsort((names, -counts))
    subj_cluster = []
    for k in order:
        members = b_idx[subj_b == names[k]]
        subj_cluster.extend(members[: 9 - len(subj_cluster)].tolist())
        if len(subj_cluster) == 9:
            break
    subj_cluster = np.array(subj_cluster)
    cluster_subjects = sorted(set(subj[subj_cluster].tolist()))

    # Observed statistics.
    X = C[:, bmask]
    e_obs = corr_eigs(X)
    lp_obs = mp_edge(X.shape[1], K)
    rng_obs = np.random.default_rng(SEED)
    horn_obs = horn_p95_rank1(X, rng_obs)
    obs_subj = within_subject_mean_r(X, same, iu)
    obs_cross = float(np.corrcoef(X.T)[iu][~same].mean())
    perm = np.array([within_subject_mean_r(column_shuffle(X, rng_obs), same, iu)
                     for _ in range(N_PERM_SUBJECT)])

    # Split-half reliability.
    half = K // 2
    r_border = float(np.corrcoef(C[:half, bmask].mean(0), C[half:, bmask].mean(0))[0, 1])
    r_all = float(np.corrcoef(C[:half].mean(0), C[half:].mean(0))[0, 1])

    # Simulations.
    seeds = np.random.SeedSequence(SEED + 1).spawn(len(SCENARIOS))
    jobs = []
    for (name, rho, n_load, placement), ss in zip(SCENARIOS, seeds):
        n = N_SIM_NULL if name == "null" else N_SIM
        jobs.append((name, rho, n_load, placement, f, b_idx, subj_cluster, same,
                     int(ss.generate_state(1)[0]), n))
    with ProcessPoolExecutor(max_workers=max_workers or len(jobs)) as ex:
        results = list(ex.map(run_scenario, jobs))

    null = next(r for r in results if r["name"] == "null")["draws"]
    ratio_null = null[:, 0] / null[:, 1]
    mc_thr = float(np.percentile(ratio_null, 95))
    subj_thr = float(np.percentile(null[:, 3], 95))

    scen_out = []
    for r in results:
        d = r["draws"]
        lam, lp, horn, subj_stat, nb = d.T
        scen_out.append({
            "scenario": r["name"], "rho_latent": r["rho"], "n_loaded": r["n_loaded"] or int(b_idx.size),
            "placement": r["placement"], "n_sim": int(d.shape[0]),
            "power_mp_strict": float(np.mean(lam > lp)),
            "power_mp_buffered_1p05": float(np.mean(lam > MP_BUFFER * lp)),
            "power_horn_contiguous": horn_rate(lam, horn),
            "power_mc_level_0p05": float(np.mean(lam / lp > mc_thr)),
            "power_within_subject_test_0p05": float(np.mean(subj_stat > subj_thr)),
            "mean_lambda1_over_edge": float(np.mean(lam / lp)),
            "mean_n_borderline": float(nb.mean()),
            "implied_mean_phi_loaded_pairs": implied_phi(f[b_idx], r["rho"]) if r["rho"] > 0 else 0.0,
        })

    out = {
        "meta": {
            "seed": SEED, "K": K, "eps": EPS, "n_sim_per_scenario": N_SIM, "n_sim_null": N_SIM_NULL,
            "n_horn_surrogates_per_sim": N_SURR, "n_perm_subject_test": N_PERM_SUBJECT,
            "matrix": str(matrix_path),
            "generative_model": "c_ki = 1[sqrt(rho) u_k + sqrt(1-rho) e_ki < Phi^-1(f_i)] for loaded i, f_i observed pass rates (all 500 columns simulated, borderline filter re-applied)",
            "rules": {
                "mp_strict": "lambda_1 > lambda_+ (MP count >= 1)",
                "mp_buffered_1p05": "lambda_1 > 1.05 lambda_+",
                "horn_contiguous": f"lambda_1 > p95 of lambda_1 over {N_SURR} column shuffles of the simulated borderline matrix",
                "mc_level_0p05": "lambda_1/lambda_+ > 95th pct of the rho=0 scenario",
                "within_subject_test": f"mean same-subject r over the {b_idx.size} observed borderline columns > 95th pct of the rho=0 scenario",
            },
            "subject_cluster": {"questions": subj_cluster.tolist(), "subjects": cluster_subjects,
                                "n_same_subject_pairs_in_cluster": int(sum(
                                    subj[a] == subj[b] for i, a in enumerate(subj_cluster) for b in subj_cluster[i + 1:]))},
        },
        "observed": {
            "n_borderline": int(b_idx.size), "lambda1": float(e_obs[0]), "lambda_plus": lp_obs,
            "mp_strict_count": int((e_obs > lp_obs).sum()),
            "mp_buffered_count": int((e_obs > MP_BUFFER * lp_obs).sum()),
            "horn_p95_rank1": horn_obs, "horn_contiguous_count_ge1": bool(e_obs[0] > horn_obs),
        },
        "calibration_from_null_scenario": {
            "size_mp_strict": float(np.mean(null[:, 0] > null[:, 1])),
            "size_mp_buffered": float(np.mean(null[:, 0] > MP_BUFFER * null[:, 1])),
            "size_horn_contiguous": horn_rate(null[:, 0], null[:, 2]),
            "n_null_draws_with_horn": int(np.isfinite(null[:, 2]).sum()),
            "mc_threshold_lambda1_over_edge_p95": mc_thr,
            "within_subject_threshold_p95": subj_thr,
        },
        "within_subject_test_observed": {
            "n_borderline": int(b_idx.size), "n_subjects": int(names.size),
            "n_same_subject_pairs": int(same.sum()), "n_cross_subject_pairs": int((~same).sum()),
            "max_borderline_per_subject": int(counts.max()),
            "observed_mean_r_within": obs_subj, "observed_mean_r_across": obs_cross,
            "perm_null_mean": float(perm.mean()), "perm_null_sd": float(perm.std(ddof=1)),
            "perm_null_p95": float(np.percentile(perm, 95)),
            "perm_pvalue_upper": perm_pvalue_upper(obs_subj, perm),
        },
        "split_half": {
            "passes": "1-50 vs 51-100",
            "r_borderline_columns": r_border, "n_borderline": int(b_idx.size),
            "r_all_columns": r_all,
        },
        "scenarios": scen_out,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["observed"]), json.dumps(out["calibration_from_null_scenario"]))
    print(json.dumps(out["within_subject_test_observed"]), json.dumps(out["split_half"]))
    for s in scen_out:
        print(f"{s['scenario']:17s} rho={s['rho_latent']:.2f} n={s['n_loaded']:3d} "
              f"MP={s['power_mp_strict']:.3f} MPbuf={s['power_mp_buffered_1p05']:.3f} "
              f"Horn={s['power_horn_contiguous']:.3f} MC={s['power_mc_level_0p05']:.3f} "
              f"subj={s['power_within_subject_test_0p05']:.3f} phi={s['implied_mean_phi_loaded_pairs']:.4f}")
    return out
