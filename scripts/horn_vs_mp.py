"""Horn parallel analysis and Marchenko-Pastur signal counts (used by item01_within.py, item13_unparseable.py).

Horn: column-shuffle surrogates, observed eigenvalue vs the rank-matched 95th percentile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

EPS = 0.05


def borderline_filter(C: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Columns with flip rate strictly between eps and 1 - eps."""
    flip_rates = C.mean(axis=0)  # (N,)
    mask = (flip_rates > eps) & (flip_rates < 1 - eps)
    return C[:, mask]


def corr_eigs_desc(X: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the column correlation matrix of X (K, N); NaN -> 0."""
    R = np.corrcoef(X.T)
    R = np.nan_to_num(R, nan=0.0)
    w = eigh(R, eigvals_only=True)
    return np.sort(w)[::-1]


def mp_signal(eig_obs: np.ndarray, lambda_plus: float, buffer: float = 1.0) -> int:
    """Count eigenvalues above buffer * lambda_plus (1 = strict, 1.05 = buffered)."""
    return int((eig_obs > buffer * lambda_plus).sum())


def horn_pa(C_B: np.ndarray, n_iter: int, seed: int) -> tuple[np.ndarray, int, int, np.ndarray]:
    """(p95 per rank, d_contiguous, d_any_above, eig_obs) under column-shuffle surrogates.

    d_contiguous: classical Horn (leading ranks above p95); d_any_above over-counts (~5% of bulk ranks by chance).
    """
    rng = np.random.default_rng(seed)
    K, Nb = C_B.shape
    eig_rand = np.empty((n_iter, Nb), dtype=np.float64)
    for it in range(n_iter):
        C_rand = C_B.copy()
        for j in range(Nb):
            rng.shuffle(C_rand[:, j])
        eig_rand[it] = corr_eigs_desc(C_rand)
    p95 = np.percentile(eig_rand, 95, axis=0)  # (Nb,)
    eig_obs = corr_eigs_desc(C_B)

    d_contig = 0
    for r in range(Nb):
        if eig_obs[r] > p95[r]:
            d_contig = r + 1
        else:
            break

    d_any = int((eig_obs > p95).sum())

    return p95, d_contig, d_any, eig_obs
