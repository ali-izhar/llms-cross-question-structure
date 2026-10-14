"""Single-family sub-ensembles and eigenvector-family alignment (called by item08_09_sub_ensembles_family.py).

With M rows, lambda_1 >= N_split / rank (rank <= M - 1), so lambda_1 / lambda_+ is compared with finite-M
nulls: A per-question rate, B ability only, F fixed margins (curveball). eta^2 = between-family SS / total SS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from null_models import (  # noqa: E402
    MP_BUFFER, corr_eigh_rows, curveball, eta_squared, family_of, perm_pvalue_upper,
    split_mask, split_spectrum,
)

SEED = 20260418
N_DRAWS = 2000
N_DRAWS_FIXED_MARGIN = 1000
CURVEBALL_BURNIN = 5000
CURVEBALL_THIN = 500
N_PERM = 20000
N_EIGVEC = 4
RANK_TOL = 1e-9


def lam1_over_edge(S: np.ndarray) -> tuple[float, float]:
    """(lambda_1 / lambda_+, (N_split / rank) / lambda_+) on split columns."""
    e, lp, n_split = split_spectrum(S)
    rank = int((e > RANK_TOL * e[0]).sum())
    return float(e[0] / lp), float(n_split / rank / lp)


def null_summary(vals: np.ndarray, bounds: np.ndarray, obs: float) -> dict:
    """Summary of null lambda_1 / lambda_+ draws and the MC p-value of obs."""
    return {
        "lam1_over_edge_mean": float(vals.mean()),
        "lam1_over_edge_p99": float(np.percentile(vals, 99)),
        "lam1_over_edge_max": float(vals.max()),
        "trace_bound_over_edge_mean": float(bounds.mean()),
        "frac_draws_lam1_above_edge": float(np.mean(vals > 1.0)),
        "mc_pvalue_observed": perm_pvalue_upper(obs, vals),
    }


def sub_ensemble(C: np.ndarray, names: list[str], family: str, rng_seq) -> dict:
    """Observed lambda_1 / lambda_+ of one family's rows against nulls A, B, F; rng_seq = (rA, rB, rF)."""
    idx = [i for i, n in enumerate(names) if family_of(n) == family]
    S = C[idx]
    M = S.shape[0]
    e, lp, n_split = split_spectrum(S)
    rank = int((e > RANK_TOL * e[0]).sum())
    trace_bound = n_split / rank
    obs = float(e[0] / lp)

    p_q = S.mean(axis=0)                     # (500,) per-question rate
    a_m = S.mean(axis=1)                     # (M,) per-model accuracy
    rA, rB, rF = rng_seq
    A = np.array([lam1_over_edge((rA.random(S.shape) < p_q).astype(np.int8)) for _ in range(N_DRAWS)])
    B = np.array([lam1_over_edge((rB.random(S.shape) < a_m[:, None]).astype(np.int8)) for _ in range(N_DRAWS)])
    X = S[:, split_mask(S)]
    chain = curveball(X, CURVEBALL_BURNIN, rF)
    F = []
    for _ in range(N_DRAWS_FIXED_MARGIN):
        chain = curveball(chain, CURVEBALL_THIN, rF)
        F.append(lam1_over_edge(chain))
    F = np.array(F)

    return {
        "models": [names[i] for i in idx],
        "M": M,
        "n_split": n_split,
        "rank": rank,
        "rank_max_possible": M - 1,
        "trace": float(e.sum()),
        "lambda_plus": lp,
        "trace_bound_lambda1_ge": trace_bound,
        "trace_bound_over_edge": trace_bound / lp,
        "eigs": e[:rank].tolist(),
        "lambda1": float(e[0]),
        "lambda1_over_edge": obs,
        "n_above_edge": int((e > lp).sum()),
        "n_above_1p05_edge": int((e > MP_BUFFER * lp).sum()),
        "accuracies": dict(zip([names[i] for i in idx], a_m.round(4).tolist())),
        "null_A_per_question_rate": null_summary(A[:, 0], A[:, 1], obs),
        "null_B_ability_only": null_summary(B[:, 0], B[:, 1], obs),
        "null_F_fixed_margin": null_summary(F[:, 0], F[:, 1], obs),
    }


def eigvec_family(C: np.ndarray, names: list[str], rng: np.random.Generator) -> dict:
    """Family eta^2 of the top N_EIGVEC eigenvectors, with label-permutation p-values."""
    X = C[:, split_mask(C)]
    w, V = corr_eigh_rows(X)                                    # (24,), (24, 24)
    fam = np.array([family_of(n) for n in names])
    acc = C.mean(axis=1)
    acc_c = acc - acc.mean()
    acc_sq = acc_c ** 2 - np.mean(acc_c ** 2)
    out = {"family_labels": dict(zip(names, fam.tolist())), "eigenvectors": []}
    perms = [rng.permutation(fam) for _ in range(N_PERM)]
    # Robustness: only families with >= 2 members (a singleton puts its whole
    # deviation into the between-family sum of squares).
    multi = np.array([np.sum(fam == f) >= 2 for f in fam])
    fam_m = fam[multi]
    perms_m = [rng.permutation(fam_m) for _ in range(N_PERM)]
    for j in range(N_EIGVEC):
        v = V[:, j]
        v = v * np.sign(np.corrcoef(v, acc)[0, 1] or 1.0)       # sign convention: + with accuracy
        eta = eta_squared(v, fam)
        null = np.array([eta_squared(v, pf) for pf in perms])
        eta_m = eta_squared(v[multi], fam_m)
        null_m = np.array([eta_squared(v[multi], pf) for pf in perms_m])
        out["eigenvectors"].append({
            "rank": j + 1,
            "eigenvalue": float(w[j]),
            "eta_sq_family": eta,
            "perm_pvalue": perm_pvalue_upper(eta, null),
            "perm_null_eta_sq_p95": float(np.percentile(null, 95)),
            "eta_sq_multi_member_families_only": eta_m,
            "perm_pvalue_multi_member_families_only": perm_pvalue_upper(eta_m, null_m),
            "corr_with_accuracy": float(np.corrcoef(v, acc)[0, 1]),
            "corr_with_centered_accuracy_squared": float(np.corrcoef(v, acc_sq)[0, 1]),
            "family_mean_loading": {f: float(v[fam == f].mean()) for f in np.unique(fam)},
            "loadings": dict(zip(names, v.round(4).tolist())),
        })
    # Joint test on eigenvectors 2..4: sum of eta^2.
    joint = sum(e["eta_sq_family"] for e in out["eigenvectors"][1:])
    Vs = V[:, 1:N_EIGVEC]
    null_joint = np.array([sum(eta_squared(Vs[:, k], pf) for k in range(Vs.shape[1])) for pf in perms])
    out["joint_eigvec_2to4"] = {"sum_eta_sq": float(joint),
                                "perm_pvalue": perm_pvalue_upper(joint, null_joint)}
    joint_m = sum(e["eta_sq_multi_member_families_only"] for e in out["eigenvectors"][1:])
    null_joint_m = np.array([sum(eta_squared(Vs[multi, k], pf) for k in range(Vs.shape[1])) for pf in perms_m])
    out["joint_eigvec_2to4"].update({"sum_eta_sq_multi_member_only": float(joint_m),
                                     "perm_pvalue_multi_member_only": perm_pvalue_upper(joint_m, null_joint_m)})
    out["multi_member_families"] = sorted(set(fam_m.tolist()))
    out["n_perm"] = N_PERM
    out["note"] = ("eta^2 uses all ten families (five singletons); label permutation keeps group "
                   "sizes, so singletons are accounted for in the null.")
    return out
