"""Ensemble-size sweep: d_across (Shannon effective rank) of N_SAMPLES random k-model subsets.

Used by item11_effective_rank.py; the module RNG (seed 42) is the sweep seed.
"""

import numpy as np

RNG = np.random.default_rng(42)
N_SAMPLES = 500


def d_across_of_submatrix(C_sub: np.ndarray) -> float | None:
    """Effective rank from Shannon entropy of disagreement-column covariance eigenvalues."""
    M, N = C_sub.shape
    if M < 3:
        return None
    col = C_sub.sum(axis=0)
    disagree = (col > 0) & (col < M)
    if disagree.sum() < M:
        return None
    C_d = C_sub[:, disagree].astype(np.float64)
    C_c = C_d - C_d.mean(axis=0, keepdims=True)
    gram = C_c @ C_c.T  # M × M, shares nonzero eigenvalues with column covariance
    eig = np.sort(np.linalg.eigvalsh(gram))[::-1]
    eig = eig[eig > 1e-10]
    if len(eig) == 0:
        return None
    p = eig / eig.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def sweep(C: np.ndarray, k_values: list[int]) -> dict:
    M = C.shape[0]
    results = {}
    for k in k_values:
        if k > M or k < 3:
            continue
        n_samples = 1 if k == M else N_SAMPLES
        ds = []
        for _ in range(n_samples):
            idx = RNG.choice(M, size=k, replace=False)
            d = d_across_of_submatrix(C[idx])
            if d is not None:
                ds.append(d)
        if not ds:
            continue
        ds = np.array(ds)
        results[k] = {
            "mean": float(ds.mean()),
            "median": float(np.median(ds)),
            "q25": float(np.quantile(ds, 0.25)),
            "q75": float(np.quantile(ds, 0.75)),
            "min": float(ds.min()),
            "max": float(ds.max()),
            "n": len(ds),
        }
    return results
