"""
Matched-Bernoulli null for d_across.
====================================

Paper claim: on 500 MMLU questions, 24 models at tau=0, d_across ~= 18.7
(Shannon effective rank of centered-column covariance, restricted to
disagreement columns; paper Sec. 2.4 and Sec. 3.3). Also 4 under MP
counting.

Reviewer worry: per-question heterogeneity in difficulty alone can
inflate the effective rank even under FULL independence across models.
If an independent-Bernoulli null already gives d ~ 18+, the "rich
cross-model epistemic structure" story collapses: the observed number
would just be what heterogeneity of p_i gives mechanically.

This script builds three matched-Bernoulli nulls and compares:
  A) per-question rate p_i (preserves question difficulty)
  B) per-model rate a_m    (preserves model capability)
  C) global mean rate      (most conservative)

For each null, 500 Monte Carlo draws. For each draw compute both
  (a) Shannon effective rank exp(H) (paper convention)
  (b) MP signal count with 1.05 buffer (paper's "MP counting")
and report mean/std/95%-CI/99th-percentile, plus z-score vs observed.

Canonical reference for effective rank (exp of entropy of normalized
eigenvalues): Roy & Vetterli, "The effective rank: A measure of
effective dimensionality," EUSIPCO 2007.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DRAWS = 500
SEED = 20260417
MP_BUFFER = 1.05  # paper's 5% buffer above lambda_+
EIG_TOL = 1e-10   # positive-eigenvalue cutoff


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def shannon_effective_rank(C: np.ndarray) -> float | None:
    """exp(H) of normalized eigenvalues of the centered covariance
    restricted to disagreement columns. Returns None if fewer than M
    disagreement columns (ill-posed).

    Matches scripts/ensemble_size_sweep.py::d_across_of_submatrix and
    paper Sec. 2.4.

    Shape contract: C is (M, N) with 0/1 entries. Output scalar.
    """
    M, N = C.shape
    col = C.sum(axis=0)
    disagree = (col > 0) & (col < M)
    # NOTE: matching the paper's implementation, we require enough disagreement
    # columns to make the M x M Gram full-rank in expectation. If fewer than M,
    # declare undefined. On observed data, 459 >> 24 so this never trips;
    # on null C at global rate ~0.65 it also holds with high probability.
    if int(disagree.sum()) < M:
        return None
    Cd = C[:, disagree].astype(np.float64)
    Cc = Cd - Cd.mean(axis=0, keepdims=True)
    gram = Cc @ Cc.T
    # scipy.linalg.eigh preferred: LAPACK symmetric path, stable.
    eig = sla.eigvalsh(gram)
    eig = np.sort(eig)[::-1]
    eig = eig[eig > EIG_TOL]
    if eig.size == 0:
        return None
    p = eig / eig.sum()
    # NOTE: + 1e-12 inside log to avoid -inf when p is exactly zero after tol
    # cut. Not a semantic change; only triggers on degenerate draws.
    H = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(H))


def mp_signal_count(C: np.ndarray) -> int | None:
    """Marchenko-Pastur signal count on the correlation matrix of
    disagreement columns (K models as "samples", N_b questions as
    "variables"). Uses 5% buffer above lambda_+ = (1 + sqrt(gamma))^2
    with gamma = N/K, matching src/stats.py::marchenko_pastur_test.

    NOTE: This is the convention used in the paper's Sec. 3.3 "MP
    counting" remark that gives 4 on observed data. It differs from
    the effective rank in both the normalization (correlation vs
    covariance) and the counting rule (hard threshold vs entropy).

    NOTE: We compute eigenvalues of the K x K Gram (Cn @ Cn.T)/K instead
    of the N_b x N_b correlation (Cn.T @ Cn)/K. The two share identical
    nonzero spectra (min(K, N_b)=K in this regime), so the MP count is
    identical while being ~ (N_b/K)^3 times cheaper. Verified to machine
    precision against the full-size R on observed data.
    """
    M, N = C.shape
    col = C.sum(axis=0)
    disagree = (col > 0) & (col < M)
    if int(disagree.sum()) < 2:
        return None
    Cd = C[:, disagree].astype(np.float64)
    K_, Nb = Cd.shape
    mu = Cd.mean(axis=0, keepdims=True)
    sd = Cd.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)  # avoid divide-by-zero on deg columns
    Cn = (Cd - mu) / sd
    # K x K gram, same nonzero eigvals as the N_b x N_b correlation / K.
    R_small = (Cn @ Cn.T) / K_
    eig = sla.eigvalsh(R_small)
    gamma = Nb / K_
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    return int((eig > lam_plus * MP_BUFFER).sum())


# ---------------------------------------------------------------------------
# Null generators
# ---------------------------------------------------------------------------

def draw_null_A(p_per_q: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """Null A: per-question rate. Row m, column i ~ Bernoulli(p_i).
    Preserves column means in expectation; destroys model identity."""
    # shape: (M, N); broadcast p across rows
    u = rng.random((M, p_per_q.size))
    return (u < p_per_q[None, :]).astype(np.int8)


def draw_null_B(a_per_m: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Null B: per-model rate. Row m, column i ~ Bernoulli(a_m).
    Preserves row means in expectation; destroys question identity."""
    u = rng.random((a_per_m.size, N))
    return (u < a_per_m[:, None]).astype(np.int8)


def draw_null_C(p_global: float, M: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Null C: global rate. All entries ~ Bernoulli(p_global).
    Preserves grand mean only; most conservative (no heterogeneity)."""
    u = rng.random((M, N))
    return (u < p_global).astype(np.int8)


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------

@dataclass
class NullStats:
    d_eff_rank: list[float]
    d_mp: list[int]
    n_undefined_eff: int
    n_undefined_mp: int


def run_null(
    name: str,
    drawer: Callable[[np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
) -> NullStats:
    d_eff = []
    d_mp = []
    n_und_eff = 0
    n_und_mp = 0
    for _ in range(n_draws):
        C = drawer(rng)
        e = shannon_effective_rank(C)
        if e is None:
            n_und_eff += 1
        else:
            d_eff.append(e)
        m = mp_signal_count(C)
        if m is None:
            n_und_mp += 1
        else:
            d_mp.append(m)
    return NullStats(
        d_eff_rank=d_eff,
        d_mp=d_mp,
        n_undefined_eff=n_und_eff,
        n_undefined_mp=n_und_mp,
    )


def summarize(values: list[float]) -> dict:
    if len(values) == 0:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci95_lo": float(np.quantile(arr, 0.025)),
        "ci95_hi": float(np.quantile(arr, 0.975)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def z_score(observed: float, values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=np.float64)
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    return float((observed - arr.mean()) / sd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    C_obs = np.load(RESULTS / "ensemble_expanded_correctness.npy")
    assert C_obs.shape == (24, 500), f"unexpected shape {C_obs.shape}"
    assert set(np.unique(C_obs).tolist()).issubset({0, 1})
    M, N = C_obs.shape

    # Observed marginals
    p_per_q = C_obs.mean(axis=0)       # (N,)
    a_per_m = C_obs.mean(axis=1)       # (M,)
    p_global = float(C_obs.mean())

    # Observed metrics
    obs_eff = shannon_effective_rank(C_obs.astype(np.int8))
    obs_mp = mp_signal_count(C_obs.astype(np.int8))
    assert obs_eff is not None and obs_mp is not None

    print(f"Observed: shape={C_obs.shape}, global mean={p_global:.4f}")
    print(f"Observed d_across (Shannon effective rank) = {obs_eff:.3f}  "
          f"[paper reports ~18.7]")
    print(f"Observed d_across (MP signal count)        = {obs_mp}         "
          f"[paper reports 4]")

    # NOTE: Use independent child RNGs per null (Fix C1). The three nulls are
    # not exchangeable; threading a single Generator through them in series
    # makes every draw dependent on null ordering. Spawning independent
    # children (PCG64 stream-splitting) decorrelates them and keeps each
    # null's draw sequence stable under reordering.
    rng_master = np.random.default_rng(SEED)
    rng_A, rng_B, rng_C = rng_master.spawn(3)

    nulls: dict[str, NullStats] = {}

    # Null A: per-question
    nulls["A_per_question"] = run_null(
        "A",
        lambda r: draw_null_A(p_per_q, M, r),
        rng_A,
    )
    # Null B: per-model
    nulls["B_per_model"] = run_null(
        "B",
        lambda r: draw_null_B(a_per_m, N, r),
        rng_B,
    )
    # Null C: global
    nulls["C_global"] = run_null(
        "C",
        lambda r: draw_null_C(p_global, M, N, r),
        rng_C,
    )

    # Report
    report = {
        "seed": SEED,
        "n_draws": N_DRAWS,
        "matrix_shape": list(C_obs.shape),
        "global_mean": p_global,
        "per_question_rate_stats": {
            "mean": float(p_per_q.mean()),
            "std": float(p_per_q.std(ddof=1)),
            "min": float(p_per_q.min()),
            "max": float(p_per_q.max()),
            "n_deterministic_correct": int((p_per_q == 1.0).sum()),
            "n_deterministic_wrong": int((p_per_q == 0.0).sum()),
        },
        "per_model_accuracy_stats": {
            "mean": float(a_per_m.mean()),
            "std": float(a_per_m.std(ddof=1)),
            "min": float(a_per_m.min()),
            "max": float(a_per_m.max()),
        },
        "observed": {
            "d_effective_rank": obs_eff,
            "d_mp_signal": obs_mp,
        },
        "nulls": {},
    }

    for key, ns in nulls.items():
        eff_sum = summarize(ns.d_eff_rank)
        mp_sum = summarize([float(x) for x in ns.d_mp])
        z_eff = z_score(obs_eff, ns.d_eff_rank)
        z_mp = z_score(float(obs_mp), [float(x) for x in ns.d_mp])
        # fraction of null draws where null >= observed (upper tail)
        def tail_frac(vals, obs):
            if len(vals) == 0:
                return None
            return float(np.mean(np.asarray(vals) >= obs))
        report["nulls"][key] = {
            "effective_rank": eff_sum,
            "mp_signal": mp_sum,
            "z_effective_rank": z_eff,
            "z_mp_signal": z_mp,
            "frac_null_ge_obs_eff": tail_frac(ns.d_eff_rank, obs_eff),
            "frac_null_ge_obs_mp": tail_frac(ns.d_mp, obs_mp),
            "n_undefined_eff": ns.n_undefined_eff,
            "n_undefined_mp": ns.n_undefined_mp,
        }

    out_json = RESULTS / "d_across_null.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")

    # Console summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Observed effective rank: {obs_eff:.3f}")
    print(f"Observed MP signal:      {obs_mp}")
    for key, ns in nulls.items():
        eff = np.asarray(ns.d_eff_rank)
        mp = np.asarray(ns.d_mp)
        print(f"\nNull {key}:")
        print(f"  eff rank: mean={eff.mean():.3f} std={eff.std(ddof=1):.3f}  "
              f"95% CI=[{np.quantile(eff,0.025):.3f}, {np.quantile(eff,0.975):.3f}]  "
              f"p99={np.quantile(eff,0.99):.3f}  "
              f"z(obs)={(obs_eff - eff.mean())/eff.std(ddof=1):+.2f}")
        print(f"  MP sig:   mean={mp.mean():.3f} std={mp.std(ddof=1):.3f}  "
              f"95% CI=[{np.quantile(mp,0.025):.3f}, {np.quantile(mp,0.975):.3f}]  "
              f"p99={np.quantile(mp,0.99):.3f}  "
              f"z(obs)={(obs_mp - mp.mean())/mp.std(ddof=1):+.2f}")

    # Figure
    make_figure(obs_eff, obs_mp, nulls, FIG_DIR / "d_across_null.pdf")
    print(f"\nWrote {FIG_DIR / 'd_across_null.pdf'}")


def make_figure(
    obs_eff: float,
    obs_mp: int,
    nulls: dict[str, NullStats],
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    colors = {
        "A_per_question": "#d62728",
        "B_per_model": "#2ca02c",
        "C_global": "#1f77b4",
    }
    labels = {
        "A_per_question": "Null A: per-question rate",
        "B_per_model": "Null B: per-model rate",
        "C_global": "Null C: global rate",
    }

    # Effective rank
    ax = axes[0]
    for key, ns in nulls.items():
        arr = np.asarray(ns.d_eff_rank)
        ax.hist(arr, bins=30, alpha=0.45, label=labels[key], color=colors[key])
    ax.axvline(obs_eff, color="black", lw=1.8, ls="--",
               label=f"observed = {obs_eff:.2f}")
    ax.axvline(23, color="grey", lw=1.0, ls=":", label="ceiling M-1 = 23")
    ax.set_xlabel(r"$d_{\rm across}$  (Shannon effective rank $\exp H$)")
    ax.set_ylabel("draws")
    ax.set_title("Paper metric (Sec. 2.4)")
    ax.legend(fontsize=7, loc="upper left")

    # MP signal count
    ax = axes[1]
    all_mp = []
    for ns in nulls.values():
        all_mp.extend(ns.d_mp)
    max_mp = max(max(all_mp) if all_mp else 0, obs_mp) + 1
    bins = np.arange(-0.5, max_mp + 1.5, 1.0)
    for key, ns in nulls.items():
        arr = np.asarray(ns.d_mp)
        ax.hist(arr, bins=bins, alpha=0.45, label=labels[key], color=colors[key])
    ax.axvline(obs_mp, color="black", lw=1.8, ls="--",
               label=f"observed = {obs_mp}")
    ax.set_xlabel(r"MP signal count (5% buffer)")
    ax.set_ylabel("draws")
    ax.set_title("Paper's MP counting")
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Matched-Bernoulli null for cross-model dimensionality "
        f"(24 x 500 MMLU, tau=0; {N_DRAWS} draws per null, seed={SEED})",
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
