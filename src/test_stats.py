"""Regression tests for src/stats.py.

Covers Fix 4 (classical Horn contiguous counting). Exercises:
  - IID Bernoulli               -> Horn contiguous count = 0 (not ~5)
  - Planted rank-1 (global bias) -> count = 1
  - Planted rank-3              -> count = 3

Run:
    uv run python src/test_stats.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stats import parallel_analysis  # noqa: E402


SEED = 20260417


def _make_iid_bernoulli(K: int, N: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((K, N)) < p).astype(np.float64)


def _make_rank_k(K: int, N: int, k: int, rng: np.random.Generator,
                 noise: float = 0.25) -> np.ndarray:
    """Planted rank-k binary matrix: C_ij = 1 if sum over k latent factors
    exceeds a threshold, else 0. Latent factors are i.i.d. N(0,1) per
    (model, factor) and (question, factor); cross-term is the "signal".
    """
    U = rng.standard_normal((K, k)) / np.sqrt(k)      # (K, k)
    V = rng.standard_normal((N, k)) / np.sqrt(k)      # (N, k)
    signal = U @ V.T                                  # (K, N)
    noise_mat = noise * rng.standard_normal((K, N))
    logits = signal + noise_mat
    # Per-column threshold so each question has ~50% base rate (borderline).
    thresh = np.median(logits, axis=0, keepdims=True)
    return (logits > thresh).astype(np.float64)


def test_iid_bernoulli_gives_zero() -> None:
    """Pure noise -> contiguous Horn count should be 0."""
    rng = np.random.default_rng(SEED)
    K, N = 100, 150
    C = _make_iid_bernoulli(K, N, p=0.5, rng=rng)
    # Seed numpy legacy RNG used inside parallel_analysis (np.random.shuffle).
    np.random.seed(SEED)
    result = parallel_analysis(C, label="iid_bernoulli", n_iter=100)
    d = result["d"]
    assert d == 0, f"expected 0 contiguous signals on IID Bernoulli, got {d}"
    print(f"  [OK] IID Bernoulli -> d = {d} (expected 0)")


def test_planted_rank_1_gives_one() -> None:
    """Rank-1 planted signal -> contiguous Horn count should be 1."""
    rng = np.random.default_rng(SEED + 1)
    K, N = 80, 150
    C = _make_rank_k(K, N, k=1, rng=rng, noise=0.15)
    np.random.seed(SEED + 1)
    result = parallel_analysis(C, label="rank_1", n_iter=100)
    d = result["d"]
    assert d == 1, f"expected 1 contiguous signal on rank-1 planted, got {d}"
    print(f"  [OK] planted rank-1 -> d = {d} (expected 1)")


def test_planted_rank_3_gives_three() -> None:
    """Rank-3 planted signal -> contiguous Horn count should be 3."""
    rng = np.random.default_rng(SEED + 2)
    K, N = 80, 150
    C = _make_rank_k(K, N, k=3, rng=rng, noise=0.10)
    np.random.seed(SEED + 2)
    result = parallel_analysis(C, label="rank_3", n_iter=100)
    d = result["d"]
    assert d == 3, f"expected 3 contiguous signals on rank-3 planted, got {d}"
    print(f"  [OK] planted rank-3 -> d = {d} (expected 3)")


def main() -> None:
    print("Running Horn contiguous-counting regression tests ...")
    test_iid_bernoulli_gives_zero()
    test_planted_rank_1_gives_one()
    test_planted_rank_3_gives_three()
    print("All tests passed.")


if __name__ == "__main__":
    main()
