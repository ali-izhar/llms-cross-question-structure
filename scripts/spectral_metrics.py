"""TW z and MP margin per borderline threshold with a Bernoulli Monte Carlo null (item01_within.py), and the
effective-rank share exp(H) / (K - 1) within models and exp(H) / (M - 1) across (item11_effective_rank.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from null_models import (  # noqa: E402
    MP_BUFFER, borderline_mask, corr_eigs, mp_edge, shannon_effective_rank_rows,
    split_mask, tw_z,
)

SEED = 20260421
N_DRAWS = 2000
EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def z_at(C: np.ndarray, eps: float) -> tuple[float, int, int, int]:
    """(z, N_b, MP strict count, MP buffered count) at threshold eps."""
    X = C[:, borderline_mask(C, eps)]
    K, Nb = X.shape
    if Nb < 2:  # no spectrum to test (only reached on tiny matrices)
        return float("nan"), Nb, 0, 0
    e = corr_eigs(X)
    lp = mp_edge(Nb, K)
    return tw_z(e[0], K, Nb), Nb, int((e > lp).sum()), int((e > MP_BUFFER * lp).sum())


def margin_and_tw(C: np.ndarray, eps: float):
    """(lambda_+ - lambda_max, sigma_TW, N_b) at cut eps; R = Cn^T Cn / K, Cn = columns z-scored (ddof=0)."""
    C = C.astype(np.float64)
    K = C.shape[0]
    rate = C.mean(axis=0)
    keep = (rate > eps) & (rate < 1 - eps)
    n_b = int(keep.sum())
    if n_b < 2:
        return np.nan, np.nan, n_b
    Cb = C[:, keep]
    sd = Cb.std(axis=0)
    Cn = (Cb - Cb.mean(axis=0, keepdims=True)) / np.where(sd == 0, 1.0, sd)
    ev = np.sort(np.linalg.eigvalsh((Cn.T @ Cn) / K))[::-1]
    gamma = n_b / K
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    sigma_tw = ((1 + np.sqrt(gamma))
                * (1 + 1 / np.sqrt(gamma)) ** (1 / 3) * K ** (-2 / 3))
    return lam_plus - ev[0], sigma_tw, n_b


def tw_z_eps_sweep(C: np.ndarray, name: str, n_draws: int = N_DRAWS, seed: int = SEED,
                   epsilons: list[float] = EPSILONS) -> dict:
    """TW z, N_b, MP counts per eps for C (K, N) 0/1; null: independent Bernoulli(f_i), filter re-applied per draw."""
    K = C.shape[0]
    f = C.mean(axis=0)
    rng = np.random.default_rng(seed)
    null = np.empty((n_draws, len(epsilons)))
    for d in range(n_draws):
        S = (rng.random(C.shape) < f[None, :]).astype(np.int8)
        null[d] = [z_at(S, eps)[0] for eps in epsilons]
    rows = []
    for j, eps in enumerate(epsilons):
        z, Nb, n_mp, n_buf = z_at(C, eps)
        rows.append({
            "eps": eps, "N_b": Nb, "z_obs": z, "mp_count": n_mp, "mp_buffered_count": n_buf,
            "null_mean": float(null[:, j].mean()), "null_p90": float(np.percentile(null[:, j], 90)),
            "null_p95": float(np.percentile(null[:, j], 95)),
            "null_p99": float(np.percentile(null[:, j], 99)),
            "percentile_of_obs_in_null": float(100 * np.mean(null[:, j] < z)),
            "mc_pvalue_upper": float((1 + np.sum(null[:, j] >= z)) / (1 + n_draws)),
        })
    return {"model": name, "K": int(K), "n_draws": n_draws, "seed": seed,
            "null": f"independent Bernoulli(f_i), f_i observed over all {C.shape[1]} questions; borderline filter re-applied per draw",
            "by_eps": rows}


def effective_rank_share(models: list[tuple[str, str]], ensemble_path: Path) -> dict:
    """exp(H) share for split and borderline columns; models: (name, matrix path)."""
    rows = []
    for name, fname in models:
        C = np.load(fname).astype(np.float64)
        K = C.shape[0]
        rec = {"model": name, "K": K, "max": K - 1}
        for label, mask in (("split", split_mask(C)), ("borderline_eps0.05", borderline_mask(C, 0.05))):
            er = shannon_effective_rank_rows(C[:, mask])
            rec[label] = {"n_cols": int(mask.sum()), "exp_H": er, "share_of_K_minus_1": er / (K - 1)}
        rows.append(rec)
    A = np.load(ensemble_path).astype(np.float64)
    er = shannon_effective_rank_rows(A[:, split_mask(A)])
    M = A.shape[0]
    return {
        "definition": "exp(H), H = -sum p_j log p_j, p_j = normalized eigenvalues of the column-centered covariance (unscaled columns)",
        "within_model": rows,
        "range_split": [min(r["split"]["share_of_K_minus_1"] for r in rows),
                        max(r["split"]["share_of_K_minus_1"] for r in rows)],
        "range_borderline": [min(r["borderline_eps0.05"]["share_of_K_minus_1"] for r in rows),
                             max(r["borderline_eps0.05"]["share_of_K_minus_1"] for r in rows)],
        "across_24_models_split": {"exp_H": er, "share_of_M_minus_1": er / (M - 1)},
    }
