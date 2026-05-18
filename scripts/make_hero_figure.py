"""
Hero figure for EIML@ICML 2026.

Three-panel layout telling the dimensionality-gap story in one glance:
  (a) eigenvalue spectrum of one model's within-model correlation matrix
      conforms to the Marchenko-Pastur noise bulk (no structure);
  (b) same (near-zero) dimensionality replicates across 5 LLMs and
      3 benchmarks, while the 24-model ensemble has d_across ~= 19;
  (c) split-half agreement of per-question flip rates at r = 0.994
      (per-question estimation is excellent).

Style: tueplots ICML full-column + Paul Tol bright colorblind-safe palette.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory
from scipy.stats import pearsonr
from tueplots import bundles
from tueplots.constants.color import palettes

# ── paths ────────────────────────────────────────────────────────────
PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── tueplots ICML style ─────────────────────────────────────────────
rc = bundles.icml2024(column="full", usetex=False, family="serif")
rc["figure.figsize"] = (6.75, 2.30)
rc["figure.constrained_layout.use"] = False
rc["axes.spines.top"] = False
rc["axes.spines.right"] = False
rc["axes.linewidth"] = 0.8
rc["xtick.major.width"] = 0.7
rc["ytick.major.width"] = 0.7
rc["xtick.major.size"] = 3.2
rc["ytick.major.size"] = 3.2
rc["xtick.direction"] = "out"
rc["ytick.direction"] = "out"
rc["lines.linewidth"] = 1.5
rc["patch.linewidth"] = 0.5
rc["mathtext.fontset"] = "cm"
rc["axes.labelsize"] = 9.8
rc["xtick.labelsize"] = 8.8
rc["ytick.labelsize"] = 8.8
rc["axes.titlesize"] = 10.2
plt.rcParams.update(rc)

# ── Paul Tol bright (colorblind-safe) ───────────────────────────────
_tol = palettes.paultol_bright
BLUE     = f"#{_tol[0]}"   # 4477AA  — within-model
RED      = f"#{_tol[1]}"   # EE6677  — MP null / annotations
GREEN    = f"#{_tol[2]}"   # 228833
LTBLUE   = f"#{_tol[4]}"   # 66CCEE  — histogram fill
GREY_MID = "#5c6773"
GREY_LT  = "#d7dce2"
ENSEMBLE = "#CC3311"       # vibrant red — across-model (distinct from MP RED)

# ── load data ────────────────────────────────────────────────────────
C = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
K, N = C.shape

flip_rates = C.mean(axis=0)
borderline = (flip_rates > 0.05) & (flip_rates < 0.95)
C_b = C[:, borderline]
K_b, N_b = C_b.shape

R = np.corrcoef(C_b.T)
eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]
gamma = N_b / K_b
lambda_plus  = (1 + np.sqrt(gamma)) ** 2
lambda_minus = (1 - np.sqrt(gamma)) ** 2

x_mp = np.linspace(max(lambda_minus * 0.9, 0.001), lambda_plus * 1.04, 400)
mp_density = np.zeros_like(x_mp)
mask = (x_mp >= lambda_minus) & (x_mp <= lambda_plus)
mp_density[mask] = (
    (1 / (2 * np.pi * gamma))
    * np.sqrt((lambda_plus - x_mp[mask]) * (x_mp[mask] - lambda_minus))
    / x_mp[mask]
)

first50 = C[:50, :].mean(axis=0)
last50  = C[50:, :].mean(axis=0)
r_val, _ = pearsonr(first50, last50)

# ── figure layout ────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(6.75, 2.30),
    gridspec_kw={"width_ratios": [0.90, 1.15, 1.40], "wspace": 0.80},
)
fig.subplots_adjust(left=0.070, right=0.985, bottom=0.185, top=0.915)

# Shift panel (b) slightly right so its y-labels have more breathing room
# from panel (a), without widening the figure or disturbing (a)/(c).
_pos_b = axes[1].get_position()
axes[1].set_position([_pos_b.x0 + 0.022, _pos_b.y0,
                      _pos_b.width, _pos_b.height])

# ════════════════════════════════════════════════════════════════════
# (a) Eigenvalue spectrum vs Marchenko-Pastur noise bulk
# ════════════════════════════════════════════════════════════════════
ax = axes[0]

# Fewer, wider bins render at paper scale with crisper bars.  Bin edges match
# the MP support so the rightmost bin terminates near λ_+ — visually connects
# the histogram to the MP curve.
_nbins = 14
_bin_edges = np.linspace(0.0, lambda_plus * 1.02, _nbins + 1)
ax.hist(
    eigvals, bins=_bin_edges, density=True,
    color=LTBLUE, edgecolor="white", linewidth=0.65, zorder=2,
)
# MP density curve — slightly bolder so it reads over the histogram
ax.plot(x_mp, mp_density, color=RED, lw=1.9, zorder=3,
        solid_capstyle="round")

# MP upper edge: dashed vertical at λ_+
ax.axvline(lambda_plus, color=RED, ls="--", lw=1.1, zorder=3,
           dashes=(3, 2))
# Short vertical tick at λ_max so the reader can locate it on the axis
# (it sits just inside the λ_+ line; the small gap is the whole story).
ax.plot([eigvals[0], eigvals[0]], [0, 0.15], color=BLUE, lw=1.5,
        zorder=5, solid_capstyle="butt")
ax.plot(eigvals[0], 0.17, marker="v", color=BLUE, markersize=5.2,
        markeredgecolor="white", markeredgewidth=0.5, zorder=5)

ax.set_xlabel("Eigenvalue $\\lambda$")
ax.set_ylabel("Density")
ax.set_title(r"$\mathbf{(a)}$  Within-model spectrum",
             loc="left", fontweight="normal", pad=4)
ax.set_xlim(-0.05, 5.8)
ax.set_ylim(0, 1.55)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# λ_+ and λ_max labels to the LEFT of the dashed edge, stacked vertically.
# Placed above the histogram bars so they don't overlap the MP curve at that
# height.  Small visual indicator (tick) shows where λ_max sits on the axis.
ax.text(lambda_plus - 0.09, 1.50,
        r"$\lambda_+\!=\!" + f"{lambda_plus:.2f}" + r"$",
        fontsize=9.0, color=RED, ha="right", va="top")
ax.text(lambda_plus - 0.09, 1.22,
        r"$\lambda_{\max}\!=\!" + f"{eigvals[0]:.2f}" + r"$",
        fontsize=9.0, color=BLUE, ha="right", va="top")
# Inline label on the MP density curve — names the theoretical prediction
# directly where it's drawn, rather than as floating text.
ax.text(1.25, 0.44, "MP\nprediction",
        fontsize=8.4, color=RED, ha="center", va="center",
        style="italic", linespacing=1.15, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

# ════════════════════════════════════════════════════════════════════
# (b) Dimensionality gap: all 9 within-model configs vs 1 ensemble
# ════════════════════════════════════════════════════════════════════
ax = axes[1]

# Data: 5 MMLU + 3 HellaSwag + 1 GSM8K within-model rows, one ensemble bar.
# d values are MP signal counts (# eigenvalues above λ_+).
rows = [
    ("Qwen-7B",         "MMLU",      0.0,    "within"),
    ("Mistral-7B",      "MMLU",      0.0,    "within"),
    ("SmolLM2-1.7B",    "MMLU",      1.0,    "within"),
    ("Phi-3-mini",      "MMLU",      0.0,    "within"),
    ("Llama-3-8B",      "MMLU",      0.0,    "within"),
    ("Qwen-7B",         "HellaSwag", 0.0,    "within"),
    ("Mistral-7B",      "HellaSwag", 0.0,    "within"),
    ("Llama-3-8B",      "HellaSwag", 0.0,    "within"),
    ("Qwen-7B",         "GSM8K",     0.0,    "within"),
    ("24-model ensemble", "MMLU",   4.0,    "across"),
]
# Tighter row spacing (was 0.62) so 10 rows fit without looking stretched.
row_gap = 0.40
positions = [i * row_gap for i in range(len(rows) - 1)]
positions.append(positions[-1] + row_gap * 1.6)  # extra gap before ensemble

D_MAX = 5.0

# Guide line at x=0 for the "no signal" baseline
ax.axvline(0.0, color=GREY_LT, lw=0.8, zorder=1)

# Within-model: horizontal stub bars; SmolLM2-MMLU (d=1) is visibly longer.
for pos, (name, bench, val, kind) in zip(positions, rows):
    if kind == "within":
        stick_end = max(val, 0.08)
        ax.barh(pos, stick_end, height=0.40, color=BLUE,
                edgecolor="white", lw=0.5, zorder=3)
        ax.plot(stick_end, pos, "o", color=BLUE, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.45, zorder=4)

# Across-model: a thick bar, label inside.  Visibly dominant.
val_ens = 4.0
ax.barh(positions[-1], val_ens, height=0.78, color=ENSEMBLE,
        edgecolor="white", lw=0.6, zorder=3)
ax.text(val_ens - 0.10, positions[-1],
        r"$d_{\mathrm{across}}^{\mathrm{MP}} = " + f"{int(val_ens)}" + r"$",
        va="center", ha="right", fontsize=10.0, fontweight="bold",
        color="white", zorder=4)

# Background stripes by benchmark group — subtle row shading so the reader
# can parse the three benchmark blocks visually without reading the labels.
_bench_groups = [
    ("MMLU",      0, 4, "#eef3f9"),
    ("HellaSwag", 5, 7, "#f8f0ea"),
    ("GSM8K",     8, 8, "#f1f2ec"),
]
for bench_name, i0, i1, shade in _bench_groups:
    y_top = positions[i0] - row_gap / 2
    y_bot = positions[i1] + row_gap / 2
    ax.axhspan(y_top, y_bot, xmin=0.0, xmax=1.0,
               facecolor=shade, zorder=0)

# Within-model callout, placed in the empty right half of the panel.
# Same metric as the ensemble bar: MP signal count (# eigenvalues > λ_+).
mid_y = (positions[0] + positions[8]) / 2
ax.text(
    1.3, mid_y,
    r"$\mathbf{d_{\mathrm{within}}^{\mathrm{MP}} \leq 1}$",
    fontsize=10.0, color=BLUE, ha="left", va="center",
)

# Dotted separator between within-model block and ensemble row
sep_y = (positions[8] + positions[9]) / 2
ax.axhline(sep_y, color=GREY_LT, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)

# Y-tick labels: "Model (benchmark)" so each row is self-describing.
# The ensemble row just reads as "24-model ensemble".
ax.set_yticks(positions)
_ylabels = []
for n, b, _, k in rows:
    if k == "across":
        _ylabels.append(n)
    else:
        _ylabels.append(f"{n} ({b})")
ax.set_yticklabels(_ylabels, fontsize=7.5)
ax.tick_params(axis="y", length=0, pad=2)
ax.tick_params(axis="x", labelsize=8.8)

ax.set_xlabel(r"MP signal count (\# eigenvalues $>\!\lambda_+$)")
ax.set_title(r"$\mathbf{(b)}$  Dimensionality gap",
             loc="left", fontweight="normal", pad=4)
ax.set_xlim(-0.35, D_MAX)
ax.set_ylim(positions[-1] + 0.55, positions[0] - 0.55)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.spines["left"].set_visible(False)

# ════════════════════════════════════════════════════════════════════
# (c) Split-half consistency of per-question flip rates
# ════════════════════════════════════════════════════════════════════
ax = axes[2]

# Identity line first (beneath data) so overplot order reads correctly.
ax.plot([0, 1], [0, 1], color=GREY_MID, lw=0.9, zorder=1, alpha=0.8)

# Scatter of per-question flip-rate estimates.  Heavier dots (size 9) with
# moderate alpha render as a clear diagonal cloud even at paper scale,
# without the mushiness a KDE contour would add.
ax.scatter(
    first50, last50, s=14, alpha=0.38, color=BLUE,
    edgecolors="none", zorder=3, rasterized=True,
)

ax.set_xlabel("Accuracy (runs 1\u201350)")
ax.set_ylabel("Accuracy (runs 51\u2013100)")
ax.set_title(r"$\mathbf{(c)}$  Split-half consistency",
             loc="left", fontweight="normal", pad=4)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
# Deliberately not set_aspect('equal'): row heights must match (a) and (b)
# at paper scale.  The 0-1 range is already near-square given width_ratios.
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.25))

# Stat box: top-left.  r prominent (Pearson split-half agreement), n dim.
ax.text(
    0.04, 0.97,
    r"$r = " + f"{r_val:.3f}" + r"$",
    transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=BLUE,
    va="top", ha="left",
)
ax.text(
    0.04, 0.85,
    r"$n = " + str(N) + r"$ questions",
    transform=ax.transAxes, fontsize=8.4, color=GREY_MID,
    va="top", ha="left",
)

# y=x label in the empty lower-right — placed along the identity line.
ax.text(0.83, 0.72, r"$y\!=\!x$",
        transform=ax.transAxes, fontsize=8.6, color=GREY_MID,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))

# ── save ─────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"hero_figure.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'hero_figure.pdf'}")
plt.close(fig)
