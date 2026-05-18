"""
Main-text figure: MP noise margin lambda_+ - lambda_max vs borderline
threshold epsilon for the five MMLU models (K=100).

Promotes former Fig A1b (panel b of supplementary_figure) into a standalone,
column-width ICML figure.

Style: tueplots ICML half-column + Paul Tol bright palette. Consistent
method->color mapping with supplementary_figure.py.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from tueplots import bundles
from tueplots.constants.color import palettes

# ── paths ────────────────────────────────────────────────────────────
PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── style ───────────────────────────────────────────────────────────
rc = bundles.icml2024(column="half", usetex=False, family="serif")
# Slightly taller to accommodate legend + annotations without crowding.
rc["figure.figsize"] = (3.25, 2.95)
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
rc["axes.titlesize"] = 10.0
plt.rcParams.update(rc)

_tol = palettes.paultol_bright
BLUE    = f"#{_tol[0]}"
RED     = f"#{_tol[1]}"
GREEN   = f"#{_tol[2]}"
YELLOW  = f"#{_tol[3]}"
PURPLE  = f"#{_tol[5]}"
GREY_MID = "#5c6773"

# ── data ────────────────────────────────────────────────────────────
C_q    = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
C_m    = np.load(RESULTS / "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy")
C_smol = np.load(RESULTS / "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy")
C_phi  = np.load(RESULTS / "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy")
C_lla  = np.load(RESULTS / "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy")

epsilons = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

# Consistent colour mapping across figures (matches supplementary_figure.py)
series = [
    ("Qwen-7B",      C_q,    BLUE,   "o"),
    ("Mistral-7B",   C_m,    RED,    "s"),
    ("Phi-3-mini",   C_phi,  GREEN,  "^"),
    ("Llama-3-8B",   C_lla,  PURPLE, "D"),
    ("SmolLM2-1.7B", C_smol, YELLOW, "v"),
]


def mp_and_tw(C_data: np.ndarray, eps: float):
    """Return (gap = lambda_+ - lambda_max, sigma_TW, n_b). NaNs if n_b<2.

    Fix M3 (ddof consistency): construct the correlation matrix via the same
    manual normalization used in ``src/stats.py::marchenko_pastur_test`` —
    center per column, divide by population std (``ddof=0``), guard
    zero-variance columns, then form ``R = (Cn.T @ Cn) / K``. For Pearson
    correlation the ddof cancels analytically, so numbers match
    ``np.corrcoef`` to machine precision on non-degenerate data; we unify the
    code path to make degenerate (zero-variance) columns handled identically
    across the codebase.
    """
    flip = C_data.mean(axis=0)
    K = C_data.shape[0]
    bord = (flip > eps) & (flip < 1 - eps)
    nb = int(bord.sum())
    if nb < 2:
        return np.nan, np.nan, nb
    Cb = C_data[:, bord].astype(np.float64)
    mu = Cb.mean(axis=0, keepdims=True)
    sd = Cb.std(axis=0)                      # ddof=0 (matches src/stats.py)
    sd = np.where(sd == 0, 1.0, sd)          # guard zero-variance columns
    Cn = (Cb - mu) / sd
    Rc = (Cn.T @ Cn) / K
    ev = np.sort(np.linalg.eigvalsh(Rc))[::-1]
    gam = nb / K
    lplus = (1 + np.sqrt(gam)) ** 2
    # Tracy-Widom scale: sigma_TW = (1+sqrt(gamma)) (1 + 1/sqrt(gamma))^(1/3) K^(-2/3)
    sigma_tw = (1 + np.sqrt(gam)) * (1 + 1 / np.sqrt(gam)) ** (1.0 / 3.0) * K ** (-2.0 / 3.0)
    return lplus - ev[0], sigma_tw, nb


gaps = {}
tw_sigmas = np.zeros(len(epsilons))
tw_counts = np.zeros(len(epsilons))
for label, C_data, _, _ in series:
    arr = np.full(len(epsilons), np.nan)
    for k, eps in enumerate(epsilons):
        g, s, _nb = mp_and_tw(C_data, eps)
        arr[k] = g
        if np.isfinite(s):
            tw_sigmas[k] += s
            tw_counts[k] += 1
    gaps[label] = arr
tw_sigma_mean = np.where(tw_counts > 0, tw_sigmas / np.maximum(tw_counts, 1), np.nan)

# ── figure ─────────────────────────────────────────────────────────
# Use evenly-spaced categorical x to prevent 0.01/0.02 labels overlapping;
# tick labels show the true epsilon values. The curve is qualitative in
# epsilon (monotone ordering preserved), so categorical spacing is honest.
x = np.arange(len(epsilons))

fig, ax = plt.subplots(figsize=rc["figure.figsize"])
fig.subplots_adjust(left=0.180, right=0.965, bottom=0.155, top=0.970)

# Signal region: y < 0 shaded yellow (above-noise eigenvalue).
ax.axhspan(-1.0, 0.0, facecolor=YELLOW, alpha=0.15, zorder=0)
# Tracy-Widom +/-1 sigma strip (averaged across models per epsilon).
# Margin = lambda_+ - lambda_max; under MP null lambda_max - lambda_+
# fluctuates at scale sigma_TW, so the margin's ~1 sigma band around 0
# is [-sigma_TW_mean, +sigma_TW_mean].
ax.fill_between(x, -tw_sigma_mean, tw_sigma_mean,
                color=GREY_MID, alpha=0.12, linewidth=0,
                zorder=0.5, label=r"TW $\pm 1\sigma$")
ax.axhline(0, color=GREY_MID, lw=0.9, ls="-", alpha=0.75, zorder=1)

# Plot non-SmolLM2 lines first, then emphasise SmolLM2 on top.
for label, _C, color, marker in series[:-1]:
    ax.plot(x, gaps[label], f"-{marker}", color=color,
            markersize=5.0, label=label, lw=1.4,
            markeredgecolor="white", markeredgewidth=0.45,
            zorder=3, alpha=0.92)
ax.plot(x, gaps["SmolLM2-1.7B"], "-v", color=YELLOW,
        markersize=6.4, label="SmolLM2-1.7B", lw=2.0,
        markeredgecolor="#8a7a0c", markeredgewidth=0.85, zorder=5)

# Region labels (right margin, inside the axes).
_tr = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.965, 0.04, "noise", transform=_tr,
        fontsize=8.4, color=GREY_MID, ha="right", va="bottom",
        style="italic")
ax.text(0.965, -0.04, "signal", transform=_tr,
        fontsize=8.4, color="#8a7a0c", ha="right", va="top",
        style="italic", fontweight="bold")

ax.set_xlabel(r"Borderline threshold $\varepsilon$")
ax.set_ylabel(r"MP noise margin $\lambda_+ - \lambda_{\max}$")
ax.set_xlim(-0.35, len(epsilons) - 0.65)
ax.set_ylim(-0.35, 0.80)
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.2f}" for e in epsilons])
ax.tick_params(axis="both", labelsize=8.6)

leg = ax.legend(fontsize=7.6, loc="upper left",
                bbox_to_anchor=(0.005, 0.995),
                framealpha=0.95, edgecolor="#cccccc",
                handlelength=1.6, borderpad=0.32, labelspacing=0.28,
                ncol=1)
leg.get_frame().set_linewidth(0.55)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"epsilon_sweep.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'epsilon_sweep.pdf'}")
plt.close(fig)
