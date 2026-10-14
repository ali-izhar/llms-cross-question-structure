"""Max over question pairs of Cramer's V and mutual information (nats) in a (K, N_b) categorical matrix,
with an independent column-shuffle null. Used by item06_answer_identity.py.
"""

from __future__ import annotations

import numpy as np

SEED = 20260414

def _contingency(x: np.ndarray, y: np.ndarray, n_cats: int) -> np.ndarray:
    """(n_cats, n_cats) contingency table of x, y in [0, n_cats)."""
    idx = x * n_cats + y
    counts = np.bincount(idx, minlength=n_cats * n_cats)
    return counts.reshape(n_cats, n_cats)


def cramers_v_pair(x: np.ndarray, y: np.ndarray, n_cats: int) -> float:
    """Cramer's V of two length-K categorical vectors; 0.0 if either is constant or K < 5."""
    K = len(x)
    if K < 5:
        return 0.0
    C = _contingency(x, y, n_cats)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    keep_r = row_sums > 0
    keep_c = col_sums > 0
    if keep_r.sum() < 2 or keep_c.sum() < 2:
        return 0.0
    C = C[np.ix_(keep_r, keep_c)]
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    n = C.sum()
    expected = np.outer(row_sums, col_sums) / n
    # expected > 0 everywhere because we dropped empty rows/cols
    chi2 = ((C - expected) ** 2 / expected).sum()
    k_min = min(C.shape) - 1
    if k_min == 0 or n <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * k_min)))


def mutual_information_pair(x: np.ndarray, y: np.ndarray, n_cats: int) -> float:
    """Mutual information in nats; 0.0 if K < 5."""
    K = len(x)
    if K < 5:
        return 0.0
    C = _contingency(x, y, n_cats).astype(np.float64)
    n = C.sum()
    if n <= 0:
        return 0.0
    joint = C / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((joint > 0) & (denom > 0),
                             np.log(joint / np.where(denom > 0, denom, 1.0)),
                             0.0)
    mi = float((joint * log_ratio).sum())
    return max(mi, 0.0)  # clip tiny negatives from fp rounding


def max_pairwise_stats_binary_vectorized(A: np.ndarray) -> tuple[float, float]:
    """Binary case of max_pairwise_stats from the 2x2 cell counts, all pairs at once (n11 = X^T X).

    For 2x2 tables Cramer's V = |phi| = |n11 n00 - n10 n01| / sqrt(n1x n0x n1y n0y).
    """
    K, Nb = A.shape
    X = A.astype(np.float64)
    col_sum = X.sum(axis=0)              # (Nb,) == n1x
    n11 = X.T @ X                        # (Nb, Nb) joint counts
    n10 = col_sum[:, None] - n11
    n01 = col_sum[None, :] - n11
    n00 = K - n11 - n10 - n01

    # Constant columns have zero denominator -> V = 0.
    p1x = col_sum
    p0x = K - col_sum
    denom = np.sqrt(np.outer(p1x * p0x, p1x * p0x))
    num = n11 * n00 - n10 * n01
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, num / denom, 0.0)
    V = np.abs(phi)
    np.fill_diagonal(V, 0.0)
    max_v = float(V.max()) if Nb >= 2 else 0.0

    # MI = sum_ab p_ab log(p_ab / (p_a p_b)), nats.
    def _cell(n):
        p = n / K
        return p
    p11 = _cell(n11)
    p10 = _cell(n10)
    p01 = _cell(n01)
    p00 = _cell(n00)
    p_x1 = (col_sum / K)[:, None]
    p_x0 = 1.0 - p_x1
    p_y1 = (col_sum / K)[None, :]
    p_y0 = 1.0 - p_y1

    def _term(p_joint, p_a, p_b):
        # p log(p / (p_a p_b)), 0 if p == 0 or marginals == 0.
        out = np.zeros_like(p_joint)
        denom_ab = p_a * p_b
        mask = (p_joint > 0) & (denom_ab > 0)
        out[mask] = p_joint[mask] * np.log(p_joint[mask] / denom_ab[mask])
        return out

    MI = (
        _term(p11, p_x1, p_y1)
        + _term(p10, p_x1, p_y0)
        + _term(p01, p_x0, p_y1)
        + _term(p00, p_x0, p_y0)
    )
    np.fill_diagonal(MI, 0.0)
    MI = np.maximum(MI, 0.0)  # clip fp-rounding negatives
    max_mi = float(MI.max()) if Nb >= 2 else 0.0
    return max_v, max_mi


def max_pairwise_stats(A: np.ndarray, n_cats: int) -> tuple[float, float]:
    """Max Cramer's V and max MI over all question pairs of A (K, N_b), entries in [0, n_cats)."""
    if n_cats == 2:
        return max_pairwise_stats_binary_vectorized(A)
    K, Nb = A.shape
    max_v = 0.0
    max_mi = 0.0
    for i in range(Nb):
        xi = A[:, i]
        for j in range(i + 1, Nb):
            xj = A[:, j]
            v = cramers_v_pair(xi, xj, n_cats)
            if v > max_v:
                max_v = v
            mi = mutual_information_pair(xi, xj, n_cats)
            if mi > max_mi:
                max_mi = mi
    return max_v, max_mi


def permutation_null(
    A: np.ndarray, n_cats: int, n_perms: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(max_V, max_MI) arrays over n_perms independent column shuffles."""
    K, Nb = A.shape
    vs = np.empty(n_perms)
    mis = np.empty(n_perms)
    for p in range(n_perms):
        B = A.copy()
        for j in range(Nb):
            rng.shuffle(B[:, j])
        vs[p], mis[p] = max_pairwise_stats(B, n_cats)
    return vs, mis
