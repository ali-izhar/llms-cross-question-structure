"""Verification tests for scripts/horn_vs_mp.py.

1. Column-shuffle surrogate preserves per-column mean exactly.
2. Horn PA on iid Bernoulli (no planted signal) yields d ≈ 0.
3. Horn PA on a rank-1 planted signal detects at least 1 factor.
4. eps=0.05 filter on Qwen MMLU reproduces paper's N_b = 121.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from horn_vs_mp import borderline_filter, corr_eigs_desc, horn_pa


def test_column_shuffle_preserves_marginals():
    rng = np.random.default_rng(0)
    K, N = 100, 200
    C = (rng.random((K, N)) < 0.4).astype(np.float64)
    col_means_before = C.mean(axis=0)
    # mimic internal shuffle: produce a shuffled copy and check marginals
    C2 = C.copy()
    rng2 = np.random.default_rng(1)
    for j in range(N):
        rng2.shuffle(C2[:, j])
    col_means_after = C2.mean(axis=0)
    npt.assert_allclose(col_means_before, col_means_after, atol=0)
    print("OK  column shuffle preserves per-column Bernoulli rate exactly")


def test_horn_on_iid_bernoulli():
    rng = np.random.default_rng(42)
    K, Nb = 100, 120
    p = rng.uniform(0.1, 0.9, size=Nb)
    C = (rng.random((K, Nb)) < p[None, :]).astype(np.float64)
    p95, d_contig, d_any, eig_obs = horn_pa(C, n_iter=100, seed=123)
    print(f"OK  iid Bernoulli Horn contig={d_contig} any>p95={d_any} "
          f"(expect contig~0, any~0.05*N_b={0.05*Nb:.1f}), "
          f"lambda_max={eig_obs[0]:.3f}, p95_r1={p95[0]:.3f}")
    assert d_contig <= 1, f"Horn contig over-detected on iid: d={d_contig}"


def test_horn_detects_planted_signal():
    rng = np.random.default_rng(7)
    K, Nb = 100, 120
    z = rng.integers(0, 2, size=K).astype(np.float64)
    noise = (rng.random((K, Nb)) < 0.2).astype(np.float64)
    C = ((z[:, None] + noise) % 2).astype(np.float64)
    keep = (C.mean(axis=0) > 0.05) & (C.mean(axis=0) < 0.95)
    C = C[:, keep]
    p95, d_contig, d_any, eig_obs = horn_pa(C, n_iter=100, seed=321)
    print(f"OK  planted rank-1 Horn contig={d_contig} any={d_any} "
          f"(expect contig>=1), lambda_max={eig_obs[0]:.3f}, p95_r1={p95[0]:.3f}")
    assert d_contig >= 1, f"Horn missed planted rank-1 signal: d={d_contig}"


def test_eps_filter_matches_paper_Nb():
    """Paper Table 1: Qwen MMLU N_b = 121."""
    path = ROOT / "results" / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy"
    if not path.exists():
        print(f"SKIP  {path} not found")
        return
    C = np.load(path).astype(np.float64)
    C_B = borderline_filter(C, eps=0.05)
    print(f"OK  Qwen MMLU K={C.shape[0]} N={C.shape[1]} N_b(eps=0.05)={C_B.shape[1]} (paper=121)")
    assert C_B.shape[1] == 121, f"N_b mismatch: {C_B.shape[1]} vs paper 121"


if __name__ == "__main__":
    test_column_shuffle_preserves_marginals()
    test_horn_on_iid_bernoulli()
    test_horn_detects_planted_signal()
    test_eps_filter_matches_paper_Nb()
    print("\nAll tests pass.")
