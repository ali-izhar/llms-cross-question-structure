"""
Backfill MMLU analysis JSONs for models whose correctness matrix is on disk
but whose analysis JSON is missing (or lacks TW / margin / Horn fields).

Produces per-model JSON at results/analysis_{slug}.json with:
  N_b, gamma, K
  lambda_plus (MP upper edge)
  lambda_max (top eigenvalue of correlation matrix, + top-5)
  margin = lambda_plus - lambda_max
  tw_sigma, tw_z
  horn: observed top-5 eigenvalues, 95th and 99th percentile thresholds,
        signal_count at 95% and 99%

This script reuses stats.marchenko_pastur_test for the MP eigenvalue
computation, re-runs parallel_analysis with percentile=95 and 99, and
computes Tracy-Widom statistics in closed form.

Naming follows the existing analysis.py convention:
  results/analysis_{model_slug}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))

from stats import marchenko_pastur_test  # noqa: E402


# Canonical repo-wide seed (Fix H1). Earlier revisions used seed=0 by default
# in horn_percentiles; pinning to the repo seed ensures bit-for-bit reproducibility
# of Horn surrogates across the audit run.
SEED = 20260417


TARGETS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

# Paper Table 1 references for verification (MMLU block)
PAPER = {
    "mistralai/Mistral-7B-Instruct-v0.3":       dict(N_b=283, lambda_plus=7.19, lambda_max=6.85, margin=+0.34, tw_z=-2.37),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":      dict(N_b=403, lambda_plus=9.04, lambda_max=9.05, margin=-0.01, tw_z=+0.05),
    "microsoft/Phi-3-mini-4k-instruct":         dict(N_b=269, lambda_plus=6.97, lambda_max=6.78, margin=+0.19, tw_z=-1.36),
    "meta-llama/Meta-Llama-3-8B-Instruct":      dict(N_b=173, lambda_plus=5.36, lambda_max=5.05, margin=+0.31, tw_z=-2.37),
    "Qwen/Qwen2.5-7B-Instruct":                 dict(N_b=121, lambda_plus=4.41, lambda_max=4.17, margin=+0.24, tw_z=-2.01),
}


def load(slug):
    C = np.load(RESULTS_DIR / f"correctness_matrix_{slug}.npy")
    with open(RESULTS_DIR / f"questions_{slug}.json") as f:
        meta = json.load(f)
    return C, meta


def borderline(C, meta, eps=0.05):
    flip = C.mean(axis=0)
    mask = (flip > eps) & (flip < 1 - eps)
    return C[:, mask], [m for m, b in zip(meta, mask) if b], mask


def tw_stats(K: int, N_b: int, lambda_max: float, lambda_plus: float) -> dict:
    """Tracy-Widom statistics.

    sigma_TW = (1+sqrt(gamma))(1 + 1/sqrt(gamma))^(1/3) * K^(-2/3)
    z = (lambda_max - lambda_plus) / sigma_TW
    Under MP null, z ~ TW1 (mean ~ -1.77, std ~ 1.27).
    """
    gamma = N_b / K
    sg = np.sqrt(gamma)
    sigma = (1 + sg) * (1 + 1.0 / sg) ** (1.0 / 3.0) * K ** (-2.0 / 3.0)
    z = (lambda_max - lambda_plus) / sigma
    return {"sigma": float(sigma), "z": float(z)}


def horn_percentiles(C_b: np.ndarray, n_iter: int = 200, seed: int = SEED) -> dict:
    """Run Horn parallel analysis, return 95th and 99th percentile thresholds
    and signal counts at both levels."""
    K, N = C_b.shape
    rng = np.random.default_rng(seed)

    R_obs = np.corrcoef(C_b.T)
    R_obs = np.nan_to_num(R_obs, nan=0.0)
    eig_obs = np.sort(np.linalg.eigvalsh(R_obs))[::-1]

    eig_random = np.zeros((n_iter, N))
    for it in range(n_iter):
        C_rand = C_b.copy()
        for j in range(N):
            rng.shuffle(C_rand[:, j])
        R_rand = np.corrcoef(C_rand.T)
        R_rand = np.nan_to_num(R_rand, nan=0.0)
        eig_random[it] = np.sort(np.linalg.eigvalsh(R_rand))[::-1]

    p95 = np.percentile(eig_random, 95, axis=0)
    p99 = np.percentile(eig_random, 99, axis=0)

    n_check = min(N // 2, len(eig_obs))
    d95 = int((eig_obs[:n_check] > p95[:n_check]).sum())
    d99 = int((eig_obs[:n_check] > p99[:n_check]).sum())

    return {
        "observed_top10": eig_obs[:10].tolist(),
        "p95_top10": p95[:10].tolist(),
        "p99_top10": p99[:10].tolist(),
        "signal_count_p95": d95,
        "signal_count_p99": d99,
        "n_iter": n_iter,
    }


def analyze(model_name: str) -> dict:
    slug = model_name.replace("/", "_")
    C, meta = load(slug)
    K, N = C.shape
    C_b, meta_b, mask = borderline(C, meta, eps=0.05)
    _, N_b = C_b.shape
    gamma = N_b / K

    mp = marchenko_pastur_test(C_b, label=slug)
    top5 = mp["top_10_eigenvalues"][:5]
    lambda_max = top5[0]
    lambda_plus = mp["lambda_plus"]
    margin = lambda_plus - lambda_max

    tw = tw_stats(K, N_b, lambda_max, lambda_plus)
    horn = horn_percentiles(C_b, seed=SEED)

    # Fix 5 (convention harmonization): report BOTH strict-edge and buffered
    # MP signal counts. The paper's primary convention is strict (buffer=1.0,
    # lambda > lambda_+); buffered (1.05) is the 5%-buffer robustness check.
    # Mirrors scripts/horn_vs_mp.py to prevent cross-script name drift.
    eigs_all = np.asarray(mp["eigenvalues"], dtype=np.float64)
    n_signal_strict = int((eigs_all > lambda_plus).sum())
    n_signal_buffered = int(mp["n_signal"])  # already uses 1.05 buffer

    out = {
        "model": model_name,
        "benchmark": "mmlu",
        "epsilon": 0.05,
        "K": int(K),
        "N_total": int(N),
        "N_borderline": int(N_b),
        "gamma": float(gamma),
        "lambda_plus": float(lambda_plus),
        "lambda_max": float(lambda_max),
        "top5_eigenvalues": [float(v) for v in top5],
        "margin": float(margin),  # lambda_plus - lambda_max (positive = below edge)
        "tw_sigma": tw["sigma"],
        "tw_z": tw["z"],
        "n_signal_mp_strict": n_signal_strict,
        "n_signal_mp_buffered_1p05": n_signal_buffered,
        # kept for backward compatibility with any existing consumer
        "n_signal_mp_5pct_buffer": n_signal_buffered,
        "horn": horn,
    }

    # Write per-model JSON (consistent with existing analysis_{slug}.json naming)
    out_path = RESULTS_DIR / f"analysis_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    return out


def compare_with_paper(rows):
    print("\n" + "=" * 96)
    print(f"{'model':40} {'N_b':>5} {'λ+':>7} {'λmax':>7} {'margin':>8} {'TW z':>7}  vs paper (flag)")
    print("=" * 96)
    for r in rows:
        p = PAPER.get(r["model"])
        flags = []
        if p:
            if r["N_borderline"] != p["N_b"]:
                flags.append(f"N_b {r['N_borderline']} vs {p['N_b']}")
            if abs(r["lambda_plus"] - p["lambda_plus"]) > 0.02:
                flags.append(f"λ+ {r['lambda_plus']:.2f} vs {p['lambda_plus']:.2f}")
            # margin in paper is (lambda_plus - lambda_max); sign-convention-matched
            computed_margin = r["lambda_plus"] - r["lambda_max"]
            if abs(computed_margin - p["margin"]) > 0.02:
                flags.append(f"margin {computed_margin:+.2f} vs {p['margin']:+.2f}")
            if abs(r["tw_z"] - p["tw_z"]) > 0.1:
                flags.append(f"TW z {r['tw_z']:+.2f} vs {p['tw_z']:+.2f}")
        tag = "MATCH" if not flags else "MISMATCH: " + "; ".join(flags)
        print(f"{r['model']:40} {r['N_borderline']:>5} {r['lambda_plus']:>7.2f} {r['lambda_max']:>7.2f} "
              f"{r['lambda_plus'] - r['lambda_max']:>+8.2f} {r['tw_z']:>+7.2f}  {tag}")


def main():
    rows = []
    for m in TARGETS:
        rows.append(analyze(m))

    # Also re-compute for Qwen and Mistral to sanity-check against paper
    for m in ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        slug = m.replace("/", "_")
        if (RESULTS_DIR / f"correctness_matrix_{slug}.npy").exists():
            C, meta = load(slug)
            C_b, _, _ = borderline(C, meta, eps=0.05)
            K, N_b = C_b.shape
            mp = marchenko_pastur_test(C_b, label=slug)
            lam = mp["top_10_eigenvalues"][0]
            lp = mp["lambda_plus"]
            tw = tw_stats(K, N_b, lam, lp)
            rows.append({
                "model": m, "N_borderline": N_b, "lambda_plus": lp,
                "lambda_max": lam, "tw_z": tw["z"],
            })

    compare_with_paper(rows)


if __name__ == "__main__":
    main()
