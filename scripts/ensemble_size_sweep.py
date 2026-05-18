"""
Ensemble-size robustness sweep.
================================

For each k in K_VALUES, draw N_SAMPLES random subsets of size k from the full
M-model correctness matrix. For each subset, compute d_across (effective rank
via Shannon entropy of eigenvalues of the M×M Gram on disagreement columns).

If d_across grows approximately linearly with k (slope near 0.8), the
dimensionality claim is intrinsic to the population of models — not a
feature of our particular ensemble composition.

Outputs:
  results/ensemble_size_sweep.json   (per-k mean, IQR, samples)
  paper/figures/ensemble_size_sweep.pdf
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from tueplots import bundles
from tueplots.constants.color import palettes

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_SAMPLES = 500


def d_across_of_submatrix(C_sub: np.ndarray) -> float | None:
    """Effective rank from Shannon entropy of disagreement-column covariance eigenvalues."""
    M, N = C_sub.shape
    if M < 3:
        return None
    col = C_sub.sum(axis=0)
    disagree = (col > 0) & (col < M)
    if disagree.sum() < M:
        return None
    C_d = C_sub[:, disagree].astype(np.float64)
    C_c = C_d - C_d.mean(axis=0, keepdims=True)
    gram = C_c @ C_c.T  # M × M, shares nonzero eigenvalues with column covariance
    eig = np.sort(np.linalg.eigvalsh(gram))[::-1]
    eig = eig[eig > 1e-10]
    if len(eig) == 0:
        return None
    p = eig / eig.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def sweep(C: np.ndarray, k_values: list[int]) -> dict:
    M = C.shape[0]
    results = {}
    for k in k_values:
        if k > M or k < 3:
            continue
        # If k == M there's only one subset.
        n_samples = 1 if k == M else N_SAMPLES
        ds = []
        for _ in range(n_samples):
            idx = RNG.choice(M, size=k, replace=False)
            d = d_across_of_submatrix(C[idx])
            if d is not None:
                ds.append(d)
        if not ds:
            continue
        ds = np.array(ds)
        results[k] = {
            "mean": float(ds.mean()),
            "median": float(np.median(ds)),
            "q25": float(np.quantile(ds, 0.25)),
            "q75": float(np.quantile(ds, 0.75)),
            "min": float(ds.min()),
            "max": float(ds.max()),
            "n": len(ds),
        }
    return results


def make_plot(results: dict, M_total: int, out_path: pathlib.Path):
    rc = bundles.icml2024(column="half", usetex=False, family="serif")
    rc["figure.constrained_layout.use"] = False
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.linewidth"] = 0.8
    rc["xtick.major.width"] = 0.7
    rc["ytick.major.width"] = 0.7
    rc["xtick.major.size"] = 3.2
    rc["ytick.major.size"] = 3.2
    rc["mathtext.fontset"] = "cm"
    rc["axes.labelsize"] = 9.5
    rc["xtick.labelsize"] = 8.6
    rc["ytick.labelsize"] = 8.6
    plt.rcParams.update(rc)

    _tol = palettes.paultol_bright
    BLUE = f"#{_tol[0]}"
    GREY_LT = "#c5cdd4"
    GREY_MID = "#5c6773"

    # Wider-than-tall aspect matched to the column width.  Slight extra height
    # vs. v0 (2.10 vs. 1.95) gives the gap annotation room to breathe.
    fig, ax = plt.subplots(figsize=(3.3, 2.60))
    fig.subplots_adjust(left=0.150, right=0.975, bottom=0.160, top=0.960)

    ks = sorted(results.keys())
    medians = np.array([results[k]["median"] for k in ks])
    q25 = np.array([results[k]["q25"] for k in ks])
    q75 = np.array([results[k]["q75"] for k in ks])

    ks_arr = np.array(ks, dtype=float)
    ceiling = ks_arr - 1.0

    # Shade the gap between the ceiling and the observed median — the
    # quantitative story is "each added model ≈ one new dimension, but we
    # miss ~20% of the ceiling".
    ax.fill_between(ks_arr, medians, ceiling, color=GREY_LT, alpha=0.45,
                    lw=0, zorder=1,
                    label=r"gap = $(k{-}1)-d_{\mathrm{across}}$")

    # Ceiling: grey dashed, label inline on the right so the legend stays small
    ax.plot(ks_arr, ceiling, "--", color=GREY_MID, lw=1.1, zorder=2)
    ax.text(ks_arr[-3] - 0.1, ceiling[-3] + 0.9, r"ceiling $k{-}1$",
            color=GREY_MID, fontsize=8.6, ha="center", va="bottom",
            rotation=28, rotation_mode="anchor")

    # IQR band across subsampled runs (thin, mostly hidden by markers — a
    # finding in itself: subset variability is small relative to the trend).
    ax.fill_between(ks_arr, q25, q75, color=BLUE, alpha=0.22, lw=0, zorder=3)

    # Median curve + markers; k=M separated into a star marker (n=1 point, not
    # a distribution — different visual weight avoids misreading the IQR band
    # as extending that far).
    ks_sub = [k for k in ks if k < M_total]
    meds_sub = [results[k]["median"] for k in ks_sub]
    ax.plot(ks_sub, meds_sub, "-o", color=BLUE, markersize=5.8, lw=1.8,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=4, label=r"median $d_{\mathrm{across}}$ ($k<M$)")
    ax.plot([M_total], [results[M_total]["median"]],
            marker="*", color=BLUE, markersize=13, markerfacecolor="white",
            markeredgewidth=1.5, lw=0, zorder=5,
            label=r"full ensemble ($k\!=\!M$)")

    # Fit slope to the medians (excluding k=M_total which has only 1 sample)
    ks_fit = [k for k in ks if k < M_total and k >= 4]
    meds_fit = [results[k]["median"] for k in ks_fit]
    slope_txt = ""
    if len(ks_fit) >= 2:
        slope, intercept = np.polyfit(ks_fit, meds_fit, 1)
        slope_txt = f"slope $\\approx {slope:.2f}$"
        # Top-left, no box — relies on whitespace above the curve
        ax.text(0.03, 0.97, slope_txt, transform=ax.transAxes,
                fontsize=10.0, color=BLUE, fontweight="bold",
                va="top", ha="left")
        ax.text(0.03, 0.88, r"(ideal $1.0$)", transform=ax.transAxes,
                fontsize=8.4, color=GREY_MID,
                va="top", ha="left")

    ax.set_xlabel(r"Ensemble size $k$")
    ax.set_ylabel(r"Dimensionality $d_{\mathrm{across}}$")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlim(2.8, M_total + 0.8)
    ax.set_ylim(0, M_total - 0.5)

    # Legend: compact, below the slope text in the upper-left.  The curve
    # climbs into the lower-right, so placing the legend there clashed with
    # the data; upper-left under the slope box keeps it out of the way.
    leg = ax.legend(fontsize=7.8, loc="upper left",
                    bbox_to_anchor=(0.02, 0.78),
                    framealpha=0.0,
                    handlelength=1.6, borderpad=0.26,
                    labelspacing=0.32, handletextpad=0.5)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(1.0)
        except Exception:
            pass

    for ext in ("pdf", "png"):
        save_path = out_path.with_suffix(f".{ext}")
        fig.savefig(save_path, dpi=300,
                    bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", edgecolor="none")
    plt.close(fig)


def main():
    C = np.load(RESULTS / "ensemble_expanded_correctness.npy")
    M = C.shape[0]
    print(f"Loaded correctness matrix: M={M}, N={C.shape[1]}")

    # Choose k values that bracket the full range, M is always included.
    base_ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    k_values = sorted(set(k for k in base_ks if k <= M) | {M})

    results = sweep(C, k_values)
    output = {
        "M_total": M,
        "N_samples_per_k": N_SAMPLES,
        "k_values": k_values,
        "per_k": results,
    }
    with open(RESULTS / "ensemble_size_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {RESULTS / 'ensemble_size_sweep.json'}")

    for k in sorted(results.keys()):
        r = results[k]
        print(f"  k={k:2d}: median d={r['median']:5.2f} "
              f"(IQR {r['q25']:.2f}-{r['q75']:.2f}, n={r['n']})")

    out_pdf = OUT / "ensemble_size_sweep.pdf"
    make_plot(results, M, out_pdf)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
