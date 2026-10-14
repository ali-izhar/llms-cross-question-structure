"""TW z = (lambda_max - lambda_+) / sigma_TW and its Monte Carlo null H1 (used by item01_within.py).

H1: independent Bernoulli(p_i) columns at the observed rates, borderline filter re-applied per draw.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

N_MC = 1000                # MC samples per model
EPS = 0.05                 # borderline threshold


def top_eig_corr(C: np.ndarray) -> float:
    """Largest eigenvalue of R = Cn^T Cn / K, Cn = columns z-scored with population SD (zero SD -> 1)."""
    K, _ = C.shape                                         # (K, N)
    C_centered = C - C.mean(axis=0, keepdims=True)         # (K, N)
    sds = C.std(axis=0)                                    # (N,) population SD (ddof=0)
    sds = np.where(sds == 0.0, 1.0, sds)
    C_norm = C_centered / sds                              # (K, N)
    # The (K, K) Gram has the same non-zero spectrum as R; use the smaller one.
    if K <= C_norm.shape[1]:
        G = (C_norm @ C_norm.T) / K                        # (K, K)
    else:
        G = (C_norm.T @ C_norm) / K
    return float(sla.eigvalsh(G, subset_by_index=[G.shape[0] - 1, G.shape[0] - 1])[0])


def tw_z(lam1: float, K: int, Nb: int) -> float:
    """(lambda_max - lambda_+) / sigma_TW."""
    gamma = Nb / K
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    sigma_tw = (1.0 + np.sqrt(gamma)) * (1.0 + 1.0 / np.sqrt(gamma)) ** (1.0 / 3.0) * K ** (-2.0 / 3.0)
    return (lam1 - lam_plus) / sigma_tw


def sample_bernoulli(K: int, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(K, N) matrix of independent Bernoulli(p_i) cells."""
    U = rng.random((K, p.size))
    return (U < p[None, :]).astype(np.float64)


def load_model(name: str, fname: str) -> dict:
    """Flip rates and observed z of one (K, N) correctness matrix."""
    C = np.load(fname).astype(np.float64)                  # (K, N)
    K, N = C.shape
    flip = C.mean(axis=0)
    borderline = (flip > EPS) & (flip < 1.0 - EPS)
    CB = C[:, borderline]
    Nb = CB.shape[1]
    lam1 = top_eig_corr(CB)
    return {
        "name": name,
        "K": K,
        "N": N,
        "Nb": Nb,
        "p_full": flip,            # all N columns' empirical p
        "p_borderline": flip[borderline],
        "z_obs": tw_z(lam1, K, Nb),
        "lam1_obs": lam1,
    }


def mc_h1_filter(model: dict, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """H1 null of z: simulate all N columns at the observed p_i, re-apply the borderline filter per draw."""
    K = model["K"]
    p = model["p_full"]
    zs = np.empty(n_mc)
    for m in range(n_mc):
        C = sample_bernoulli(K, p, rng)                    # (K, N)
        flip = C.mean(axis=0)
        b = (flip > EPS) & (flip < 1.0 - EPS)
        if b.sum() < 2:
            zs[m] = np.nan
            continue
        Nb_m = int(b.sum())
        lam1 = top_eig_corr(C[:, b])
        zs[m] = tw_z(lam1, K, Nb_m)
    return zs


def summary_stats(zs: np.ndarray) -> dict:
    zs = zs[np.isfinite(zs)]
    return {
        "mean": float(np.mean(zs)),
        "std":  float(np.std(zs, ddof=1)),
        "median": float(np.median(zs)),
        "q05":  float(np.quantile(zs, 0.05)),
        "q95":  float(np.quantile(zs, 0.95)),
        "n":    int(zs.size),
    }


def percentile_of(z_obs: float, zs: np.ndarray) -> float:
    """Fraction of MC samples at or below z_obs."""
    zs = zs[np.isfinite(zs)]
    return float(np.mean(zs <= z_obs))
