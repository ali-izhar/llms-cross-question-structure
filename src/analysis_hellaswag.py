"""
Analysis: HellaSwag Within-Model Epistemic Dimensionality
==========================================================

Sister script to `analysis.py`. Reads the per-model HellaSwag correctness
matrices written by `run_within_batch_hellaswag.py` and runs:

  1. Marchenko-Pastur null test (signal above noise?)
  2. Horn's parallel analysis (how many factors?)
  3. Within-stratum vs between-stratum correlation (topic confound check;
     uses HellaSwag's `activity_label` instead of MMLU's `subject`).
  4. Cross-model summary table.

Outputs:
  - results/analysis_hellaswag_{model_slug}.json (per model)
  - results/analysis_hellaswag_summary.json (one row per model)
  - results/figures/eigenvalue_spectrum_hellaswag_{model_slug}.pdf
  - results/figures/flip_rates_hellaswag_{model_slug}.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from stats import marchenko_pastur_test, parallel_analysis  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 12, "figure.dpi": 150,
})

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def load_data(model_slug: str):
    C = np.load(RESULTS_DIR / f"correctness_matrix_hellaswag_{model_slug}.npy")
    with open(RESULTS_DIR / f"questions_hellaswag_{model_slug}.json") as f:
        meta = json.load(f)
    return C, meta


def get_borderline(C, meta, eps=0.05):
    flip_rates = C.mean(axis=0)
    mask = (flip_rates > eps) & (flip_rates < 1 - eps)
    return C[:, mask], [m for m, b in zip(meta, mask) if b], mask


def stratum_confound(C_b, meta_b):
    """Within- vs between-`activity_label` correlation."""
    labels = [m.get("activity_label", "") for m in meta_b]
    R = np.corrcoef(C_b.T)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, np.nan)
    within, between = [], []
    for i in range(len(meta_b)):
        for j in range(i + 1, len(meta_b)):
            if np.isnan(R[i, j]):
                continue
            (within if labels[i] == labels[j] else between).append(R[i, j])
    return {
        "within_mean_corr": float(np.mean(within)) if within else 0.0,
        "between_mean_corr": float(np.mean(between)) if between else 0.0,
        "ratio": float(np.mean(within) / np.mean(between)) if between and np.mean(between) != 0 else float("inf"),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
    }


def plot_spectrum(mp, pa, model_name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    eig = np.array(mp["eigenvalues"])
    a1.hist(eig, bins=50, density=True, alpha=0.7, color="#1E88E5", label="Observed")
    a1.axvline(mp["lambda_plus"], color="red", linestyle="--", linewidth=2,
               label=f"MP edge (λ+={mp['lambda_plus']:.2f})")
    a1.set_xlabel("Eigenvalue"); a1.set_ylabel("Density")
    a1.set_title("HellaSwag: Eigenvalue vs MP null"); a1.legend()

    n = min(30, len(pa["observed_eigenvalues"]))
    x = np.arange(1, n + 1)
    a2.plot(x, pa["observed_eigenvalues"][:n], "o-", color="#E53935", linewidth=2, label="Observed")
    a2.plot(x, pa["threshold_eigenvalues"][:n], "s--", color="#43A047", linewidth=1.5, label="95th pctl random")
    a2.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Kaiser")
    if pa["d"] > 0 and pa["d"] <= n:
        a2.axvline(pa["d"] + 0.5, color="orange", linewidth=2, alpha=0.7, label=f"d={pa['d']}")
    a2.set_xlabel("Factor"); a2.set_ylabel("Eigenvalue")
    a2.set_title("Scree plot + parallel analysis"); a2.legend(); a2.set_xlim(0.5, n + 0.5)

    plt.suptitle(f"HellaSwag within-model: {model_name.split('/')[-1]}", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / f"eigenvalue_spectrum_hellaswag_{model_name.replace('/', '_')}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_fliprates(meta, model_name):
    fr = [m["flip_rate"] for m in meta]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fr, bins=50, color="#1E88E5", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(0.05, color="red", linestyle="--", label="Borderline threshold")
    ax.axvline(0.95, color="red", linestyle="--")
    ax.set_xlabel("Flip rate"); ax.set_ylabel("# questions")
    ax.set_title(f"HellaSwag flip rates: {model_name.split('/')[-1]}")
    n_ac = sum(1 for f in fr if f == 1.0)
    n_aw = sum(1 for f in fr if f == 0.0)
    n_b = sum(1 for f in fr if 0.05 < f < 0.95)
    ax.text(0.95, 0.95, f"Always correct: {n_ac}\nAlways wrong: {n_aw}\nBorderline: {n_b}",
            transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / f"flip_rates_hellaswag_{model_name.replace('/', '_')}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def analyze_one(model_name):
    slug = model_name.replace("/", "_")
    C, meta = load_data(slug)
    K, N = C.shape
    print(f"\n{'#'*70}\n# {model_name}  (K={K}, N={N})\n{'#'*70}")
    plot_fliprates(meta, model_name)

    C_b, meta_b, _ = get_borderline(C, meta)
    K, N_b = C_b.shape
    print(f"Borderline: {N_b}/{N}")

    if N_b < 15:
        print("  -> Too few borderline questions for MP test.")
        return {
            "model": model_name, "benchmark": "hellaswag",
            "K": int(K), "N": int(N), "N_borderline": int(N_b),
            "d_within": None, "note": "insufficient_borderline",
        }

    mp = marchenko_pastur_test(C_b, label=f"hellaswag/{slug}")
    pa = parallel_analysis(C_b, label=f"hellaswag/{slug}")
    plot_spectrum(mp, pa, model_name)

    confound = stratum_confound(C_b, meta_b)
    print(f"\nStratum confound: within={confound['within_mean_corr']:.4f} "
          f"between={confound['between_mean_corr']:.4f} "
          f"ratio={confound['ratio']:.2f}")

    out = {
        "model": model_name, "benchmark": "hellaswag",
        "K": int(K), "N": int(N), "N_borderline": int(N_b),
        "d_within": pa["d"],
        "mp": {k: v for k, v in mp.items() if k != "eigenvalues"},
        "parallel_analysis": {k: v for k, v in pa.items() if k != "observed_eigenvalues"},
        "stratum_confound": confound,
    }
    with open(RESULTS_DIR / f"analysis_hellaswag_{slug}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def main():
    rows = []
    for m in MODELS:
        try:
            rows.append(analyze_one(m))
        except FileNotFoundError as e:
            print(f"[skip] {m}: {e}")
            rows.append({"model": m, "status": "missing", "error": str(e)})
        except Exception as e:
            print(f"[error] {m}: {e}")
            rows.append({"model": m, "status": "failed", "error": str(e)})

    with open(RESULTS_DIR / "analysis_hellaswag_summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("HELLASWAG WITHIN-MODEL SUMMARY")
    print("=" * 70)
    print(f"{'model':45} {'N_b':>5} {'d':>4} {'mp_signal':>10}")
    for r in rows:
        if "d_within" in r and r.get("d_within") is not None:
            mp_signal = r.get("mp", {}).get("n_signal", "-")
            print(f"{r['model']:45} {r['N_borderline']:>5} {r['d_within']:>4} {mp_signal:>10}")
        else:
            print(f"{r['model']:45} {'-':>5} {'-':>4} {'-':>10}  ({r.get('status', 'no data')})")


if __name__ == "__main__":
    main()
