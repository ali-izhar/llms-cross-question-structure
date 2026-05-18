"""
Shared statistical analysis functions: Marchenko-Pastur test and parallel analysis.
Used by analysis.py, run_semantic_pilot.py, run_semantic_full.py, run_mistral.py,
run_temperature_sweep.py, run_gsm8k.py.
"""

import numpy as np


def marchenko_pastur_test(C, label=""):
    """Test whether eigenvalue spectrum has signal above MP noise."""
    K, N = C.shape
    gamma = N / K

    # Compute correlation matrix
    C_centered = C - C.mean(axis=0, keepdims=True)
    stds = C.std(axis=0)
    stds[stds == 0] = 1
    C_norm = C_centered / stds
    R = (C_norm.T @ C_norm) / K

    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # MP upper edge
    lambda_plus = (1 + np.sqrt(gamma)) ** 2

    # Count signal eigenvalues (5% buffer above MP edge)
    n_signal = int((eigenvalues > lambda_plus * 1.05).sum())

    total_var = eigenvalues.sum()
    signal_var = eigenvalues[:n_signal].sum() if n_signal > 0 else 0
    signal_frac = signal_var / total_var if total_var > 0 else 0

    result = {
        "label": label,
        "K": K,
        "N": N,
        "gamma": float(gamma),
        "lambda_plus": float(lambda_plus),
        "n_signal": n_signal,
        "signal_variance_fraction": float(signal_frac),
        "top_10_eigenvalues": eigenvalues[:10].tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }

    print(f"\n{'='*60}")
    print(f"MARCHENKO-PASTUR TEST: {label}")
    print(f"{'='*60}")
    print(f"  Matrix shape:      {K} x {N}")
    print(f"  Aspect ratio γ:    {gamma:.3f}")
    print(f"  MP upper edge λ+:  {lambda_plus:.3f}")
    print(f"  Signal eigenvalues: {n_signal}")
    print(f"  Signal variance:    {signal_frac:.1%}")
    print(f"  Top 5 eigenvalues:  {[f'{v:.3f}' for v in eigenvalues[:5]]}")
    if n_signal == 0:
        print(f"  >>> NO structure above MP noise.")
    else:
        print(f"  >>> {n_signal} signal dimensions detected!")
    print(f"{'='*60}\n")

    return result


def parallel_analysis(C, label="", n_iter=200):
    """Horn's parallel analysis for determining number of factors.

    Uses the CLASSICAL Horn (1965) contiguous leading-factor rule: retain
    factors in rank order until the first rank where the observed eigenvalue
    falls at or below the 95th percentile of the surrogate distribution, then
    stop. The older rule ``(eig_obs > threshold).sum()`` over all ranks
    over-counts by ~5% of ranks (the false-positive rate per rank under the
    null), which produced ~5 "signals" on IID Bernoulli data.

    Reference: Horn, J. L. "A rationale and test for the number of factors in
    factor analysis." Psychometrika 30(2):179-185, 1965.
    """
    K, N = C.shape

    R_obs = np.corrcoef(C.T)
    R_obs = np.nan_to_num(R_obs, nan=0.0)
    eig_obs = np.sort(np.linalg.eigvalsh(R_obs))[::-1]

    # Generate random data preserving marginal distributions
    eig_random = np.zeros((n_iter, N))
    for it in range(n_iter):
        # Permute each column independently (destroys inter-column correlations)
        C_rand = C.copy()
        for j in range(N):
            np.random.shuffle(C_rand[:, j])
        R_rand = np.corrcoef(C_rand.T)
        R_rand = np.nan_to_num(R_rand, nan=0.0)
        eig_random[it] = np.sort(np.linalg.eigvalsh(R_rand))[::-1]

    eig_threshold = np.percentile(eig_random, 95, axis=0)

    # Classical Horn (1965): contiguous leading-factor rule.
    n_check = min(N // 2, len(eig_obs))
    d = 0
    for i in range(n_check):
        if eig_obs[i] > eig_threshold[i]:
            d += 1
        else:
            break

    result = {
        "label": label,
        "d": d,
        "observed_eigenvalues": eig_obs[:20].tolist(),
        "threshold_eigenvalues": eig_threshold[:20].tolist(),
    }

    print(f"\n{'='*60}")
    print(f"PARALLEL ANALYSIS: {label}")
    print(f"{'='*60}")
    print(f"  Dimensionality d = {d}")
    print(f"  Top 10 eigenvalues (observed vs 95th pctl threshold):")
    print(f"  {'Factor':>8} {'Observed':>10} {'Threshold':>10} {'Retain?':>8}")
    for i in range(min(10, N)):
        retain = "YES" if eig_obs[i] > eig_threshold[i] else "no"
        print(f"  {i+1:>8} {eig_obs[i]:>10.3f} {eig_threshold[i]:>10.3f} {retain:>8}")
    print(f"{'='*60}\n")

    return result
