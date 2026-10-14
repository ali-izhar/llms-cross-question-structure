"""Borderline-threshold sweep helpers (used by item01_within.py): MP margin and sigma_TW per eps."""

from __future__ import annotations

import numpy as np

EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


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
