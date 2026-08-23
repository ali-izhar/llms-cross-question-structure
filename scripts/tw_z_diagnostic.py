"""
TW z-score diagnostic: explain the 4-of-5-below-mean deficit.

Phenomenon:
    Under the paper's null (Prop. 1), z = (lambda_max - lambda_+) / sigma_TW
    converges to Tracy-Widom F1 (mean -1.21, std 1.27).  Observed MMLU:
        Qwen2.5-7B       z = -2.01
        Mistral-7B-v0.3  z = -2.37
        Phi-3-mini       z = -1.36
        Meta-Llama-3-8B  z = -2.37
        SmolLM2-1.7B     z = +0.05
    Four of five sit at or below the TW mean; unlikely under the null.

Hypotheses (all tested by Monte Carlo on independent Bernoulli matrices
matching the empirical (K, N_b, {p_i}) of each model):
    H1 borderline filter biases the edge (filter post-simulation as we
       do on data).
    H2 column standardization at finite K under-estimates SD, shrinking
       the top eigenvalue.  Tested by simulating under the null WITHOUT
       re-filtering and computing z with the same pipeline.
    H3 non-universality at moderate gamma: simulate IID Gaussian columns
       at matching (K, N_b) to see if TW F1 holds at these sizes.
    H4 finite-K correction: compute empirical mean/std of z under
       independent Bernoulli at K=100 and compare to the asymptotic
       -1.21 / 1.27.

Output: results/tw_z_diagnostic.md plus figures/tw_z_diagnostic.pdf.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_MC = 1000                # MC samples per hypothesis / model
EPS = 0.05                 # Borderline threshold (paper)
SEED = 20260417            # Fixed seed for reproducibility

MODELS = [
    # (display name, correctness matrix filename)
    ("Qwen2.5-7B",       "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy"),
    ("Mistral-7B-v0.3",  "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy"),
    ("Phi-3-mini",       "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy"),
    ("Meta-Llama-3-8B",  "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy"),
    ("SmolLM2-1.7B",     "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy"),
]

TW_MEAN_ASYMP = -1.2065335745         # F1 (GOE) mean
TW_STD_ASYMP  =  1.2680224200         # F1 (GOE) std


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def top_eig_corr(C: np.ndarray) -> float:
    """Largest eigenvalue of the column-standardized correlation matrix.

    Mirrors src/stats.py :: marchenko_pastur_test exactly: divide by K
    (not K-1), use sample SD, replace zero SD by 1.
    """
    K, _ = C.shape                                         # (K, N)
    C_centered = C - C.mean(axis=0, keepdims=True)         # (K, N)
    sds = C.std(axis=0)                                    # (N,) population SD (ddof=0)
    sds = np.where(sds == 0.0, 1.0, sds)
    C_norm = C_centered / sds                              # (K, N)
    # R = C_norm.T @ C_norm / K is N x N; for K <= N use (K x K) Gram
    # with identical non-zero spectrum, then take largest eigenvalue.
    if K <= C_norm.shape[1]:
        G = (C_norm @ C_norm.T) / K                        # (K, K)
    else:
        G = (C_norm.T @ C_norm) / K
    return float(sla.eigvalsh(G, subset_by_index=[G.shape[0] - 1, G.shape[0] - 1])[0])


def tw_z(lam1: float, K: int, Nb: int) -> float:
    """(lambda_max - lambda_+) / sigma_TW (paper, Prop. 1)."""
    gamma = Nb / K
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    sigma_tw = (1.0 + np.sqrt(gamma)) * (1.0 + 1.0 / np.sqrt(gamma)) ** (1.0 / 3.0) * K ** (-2.0 / 3.0)
    return (lam1 - lam_plus) / sigma_tw


def sample_bernoulli(K: int, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """K x N Bernoulli(p_i) matrix (independent across rows and columns)."""
    # (K, N) <- broadcast p over rows
    U = rng.random((K, p.size))
    return (U < p[None, :]).astype(np.float64)


# ---------------------------------------------------------------------------
# Load empirical data
# ---------------------------------------------------------------------------

def load_model(name: str, fname: str) -> dict:
    C = np.load(RESULTS / fname).astype(np.float64)        # (K, N)
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


# ---------------------------------------------------------------------------
# Monte Carlo simulators (one per hypothesis)
# ---------------------------------------------------------------------------

def mc_h1_filter(model: dict, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """H1: simulate N columns with empirical p_i, re-filter by borderline,
    compute z exactly as the pipeline does.  Uses the full flip-rate
    vector so each MC run experiences borderline-selection noise."""
    K, N = model["K"], model["N"]
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


def mc_h2_no_filter(model: dict, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """H2: simulate exactly Nb Bernoulli columns with the *observed*
    borderline p_i (no re-filter).  Isolates the standardization artifact
    from the filter: we compare its z distribution to TW F1."""
    K, Nb = model["K"], model["Nb"]
    p = model["p_borderline"]
    zs = np.empty(n_mc)
    for m in range(n_mc):
        C = sample_bernoulli(K, p, rng)                    # (K, Nb)
        lam1 = top_eig_corr(C)
        zs[m] = tw_z(lam1, K, Nb)
    return zs


def mc_h3_gaussian(model: dict, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """H3: simulate K x Nb IID Gaussian columns, column-standardize,
    compute z.  If TW F1 holds at this (K, Nb), the distribution should
    be centered at -1.21 with std 1.27."""
    K, Nb = model["K"], model["Nb"]
    zs = np.empty(n_mc)
    for m in range(n_mc):
        X = rng.standard_normal((K, Nb))
        lam1 = top_eig_corr(X)
        zs[m] = tw_z(lam1, K, Nb)
    return zs


def mc_h4_bernoulli_fixed(model: dict, rng: np.random.Generator, n_mc: int = N_MC) -> np.ndarray:
    """H4: same as H2 but with p_i = 0.5 (maximum Bernoulli variance,
    farthest from degeneracy) at the same (K, Nb).  Isolates
    finite-K shift from the specific empirical p distribution."""
    K, Nb = model["K"], model["Nb"]
    p = np.full(Nb, 0.5)
    zs = np.empty(n_mc)
    for m in range(n_mc):
        C = sample_bernoulli(K, p, rng)
        lam1 = top_eig_corr(C)
        zs[m] = tw_z(lam1, K, Nb)
    return zs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results: list[dict], outpath: Path) -> None:
    """One panel per model: KDE-style histogram of H1 simulated z, with
    vertical lines at observed z and at TW F1 mean; plus a 5th panel
    comparing simulated means across models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(3.2 * (n_models + 1), 3.2), sharey=False)

    for ax, r in zip(axes[:-1], results):
        zs = r["mc_h1"]
        zs = zs[np.isfinite(zs)]
        ax.hist(zs, bins=40, density=True, color="#bdd7e7", edgecolor="#2171b5", linewidth=0.6, alpha=0.9)
        # TW F1 mean reference line
        ax.axvline(TW_MEAN_ASYMP, color="#888", linestyle="--", linewidth=1.0, label=r"TW$_1$ mean $-1.21$")
        # Simulated mean
        ax.axvline(np.mean(zs), color="#2171b5", linestyle="-", linewidth=1.2, label=f"sim mean {np.mean(zs):.2f}")
        # Observed z
        ax.axvline(r["z_obs"], color="#d62728", linestyle="-", linewidth=1.6, label=f"observed {r['z_obs']:+.2f}")
        ax.set_title(f"{r['name']}\n$K$={r['K']}, $N_b$={r['Nb']}, $\\gamma$={r['Nb']/r['K']:.2f}", fontsize=9)
        ax.set_xlabel("TW $z$")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left", frameon=False)

    # Summary panel: simulated-mean vs observed z for each model
    ax = axes[-1]
    xs = np.arange(len(results))
    sim_means = np.array([np.nanmean(r["mc_h1"]) for r in results])
    sim_q05  = np.array([np.nanquantile(r["mc_h1"], 0.05) for r in results])
    sim_q95  = np.array([np.nanquantile(r["mc_h1"], 0.95) for r in results])
    obs = np.array([r["z_obs"] for r in results])
    ax.errorbar(xs - 0.12, sim_means,
                yerr=[sim_means - sim_q05, sim_q95 - sim_means],
                fmt="o", color="#2171b5", capsize=3, label="H1 sim mean (5-95%)")
    ax.scatter(xs + 0.12, obs, color="#d62728", marker="D", s=40, zorder=5, label="observed")
    ax.axhline(TW_MEAN_ASYMP, color="#888", linestyle="--", linewidth=1.0, label=r"TW$_1$ mean $-1.21$")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"].split("-")[0] for r in results], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TW $z$")
    ax.set_title("H1 simulated vs. observed")
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"[fig] -> {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[load] empirical correctness matrices")
    models = [load_model(name, fname) for (name, fname) in MODELS]
    for m in models:
        print(f"  {m['name']:20s} K={m['K']} Nb={m['Nb']} z_obs={m['z_obs']:+.3f}")

    # Run the four Monte-Carlo suites
    results = []
    for m in models:
        print(f"[mc] {m['name']} ...")
        entry = dict(m)
        entry["mc_h1"] = mc_h1_filter(m, rng)
        entry["mc_h2"] = mc_h2_no_filter(m, rng)
        entry["mc_h3"] = mc_h3_gaussian(m, rng)
        entry["mc_h4"] = mc_h4_bernoulli_fixed(m, rng)
        for key in ("mc_h1", "mc_h2", "mc_h3", "mc_h4"):
            s = summary_stats(entry[key])
            print(f"  {key}: mean={s['mean']:+.3f} std={s['std']:.3f} pct(z_obs)={percentile_of(m['z_obs'], entry[key]):.3f}")
        results.append(entry)

    # Emit JSON of everything numerical
    out = {
        "config": {"N_MC": N_MC, "EPS": EPS, "SEED": SEED,
                   "TW_MEAN_ASYMP": TW_MEAN_ASYMP, "TW_STD_ASYMP": TW_STD_ASYMP},
        "per_model": {},
    }
    for r in results:
        out["per_model"][r["name"]] = {
            "K": r["K"], "N": r["N"], "Nb": r["Nb"],
            "gamma": r["Nb"] / r["K"],
            "z_obs": r["z_obs"],
            "lam1_obs": r["lam1_obs"],
            "h1_filter":        summary_stats(r["mc_h1"]),
            "h2_no_filter":     summary_stats(r["mc_h2"]),
            "h3_gaussian":      summary_stats(r["mc_h3"]),
            "h4_fixed_p_0_5":   summary_stats(r["mc_h4"]),
            "pct_z_obs_h1": percentile_of(r["z_obs"], r["mc_h1"]),
            "pct_z_obs_h2": percentile_of(r["z_obs"], r["mc_h2"]),
        }
    with open(RESULTS / "tw_z_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] -> {RESULTS / 'tw_z_diagnostic.json'}")

    make_figure(results, FIG_DIR / "tw_z_diagnostic.pdf")


if __name__ == "__main__":
    main()
