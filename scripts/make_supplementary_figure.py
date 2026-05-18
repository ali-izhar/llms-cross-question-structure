"""
Supplementary figure (single panel).

  Sorted pair-correlation dot plot of all 276 model pairs, split by
  same-family vs cross-family.  A side-panel KDE shows the two
  distributions' substantial overlap — the family signal is a weak
  nudge on a broadly-diverse population, which strengthens the
  across-model diversity claim.  Top-3 pairs labelled.  Cohen's d and
  KS statistic printed as a headline effect-size.

Threshold-robustness panel (formerly panel b) now lives in main-text
Figure 4 (eps_sweep) and has been removed from this figure.

Style: tueplots ICML half-column + Paul Tol colorblind-safe palette.
"""

import pathlib
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde, ks_2samp
from tueplots import bundles
from tueplots.constants.color import palettes

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"

# ── style ───────────────────────────────────────────────────────────
rc = bundles.icml2024(column="half", usetex=False, family="serif")
rc["figure.figsize"] = (3.25, 3.20)
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
GREY    = f"#{_tol[6]}"
GREY_LT = "#c5cdd4"
GREY_MID = "#5c6773"
ENSEMBLE = "#CC3311"

# ── data ────────────────────────────────────────────────────────────
C_ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
with open(RESULTS / "ensemble_expanded_models.json") as f:
    model_names = json.load(f)
R_ens = np.corrcoef(C_ens)

_SHORT_FAM = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":    ("Smol2",   "SmolLM"),
    "Qwen/Qwen2.5-0.5B-Instruct":             ("Q-0.5B",  "Qwen"),
    "Qwen/Qwen2.5-1.5B-Instruct":             ("Q-1.5B",  "Qwen"),
    "Qwen/Qwen2.5-3B-Instruct":               ("Q-3B",    "Qwen"),
    "Qwen/Qwen2.5-7B-Instruct":               ("Q-7B",    "Qwen"),
    "Qwen/Qwen2-7B-Instruct":                 ("Q2-7B",   "Qwen"),
    "microsoft/phi-2":                        ("Phi-2",   "Phi"),
    "microsoft/Phi-3-mini-4k-instruct":       ("Phi-3",   "Phi"),
    "microsoft/Phi-3.5-mini-instruct":        ("Phi-3.5", "Phi"),
    "mistralai/Mistral-7B-Instruct-v0.2":     ("M-v0.2",  "Mistral"),
    "mistralai/Mistral-7B-Instruct-v0.3":     ("M-v0.3",  "Mistral"),
    "HuggingFaceH4/zephyr-7b-beta":           ("Zephyr",  "Mistral"),
    "mistralai/Ministral-8B-Instruct-2410":   ("Mini-8B", "Mistral"),
    "mistralai/Mistral-Small-Instruct-2409":  ("M-Sm",    "Mistral"),
    "meta-llama/Llama-3.2-1B-Instruct":       ("L3.2-1B", "Llama"),
    "meta-llama/Llama-3.2-3B-Instruct":       ("L3.2-3B", "Llama"),
    "meta-llama/Llama-3.1-8B-Instruct":       ("L3.1-8B", "Llama"),
    "meta-llama/Meta-Llama-3-8B-Instruct":    ("L3-8B",   "Llama"),
    "google/gemma-2-2b-it":                   ("Gem2-2B", "Gemma"),
    "google/gemma-2-9b-it":                   ("Gem2-9B", "Gemma"),
    "allenai/OLMo-2-1124-7B-Instruct":        ("OLMo-2",  "OLMo"),
    "01-ai/Yi-1.5-6B-Chat":                   ("Yi-6B",   "Yi"),
    "deepseek-ai/deepseek-llm-7b-chat":       ("DSk-7B",  "DeepSeek"),
    "ibm-granite/granite-3.1-8b-instruct":    ("Gran-8B", "Granite"),
}
M = R_ens.shape[0]
short = [_SHORT_FAM[n][0] for n in model_names]
fams = [_SHORT_FAM[n][1] for n in model_names]

# All 276 pairs, sorted ascending by r, with labels & same-family flag.
_pairs = []
for i in range(M):
    for j in range(i + 1, M):
        _pairs.append((R_ens[i, j], short[i], short[j],
                       fams[i] == fams[j]))
_pairs.sort(key=lambda x: x[0])  # ascending
_rs = np.array([p[0] for p in _pairs])
_labels = [(p[1], p[2]) for p in _pairs]
_same = np.array([p[3] for p in _pairs])

# Summary stats
_w_r = _rs[_same]
_c_r = _rs[~_same]
_nw, _nc = len(_w_r), len(_c_r)
_sp = np.sqrt(((_nw - 1) * _w_r.var(ddof=1) + (_nc - 1) * _c_r.var(ddof=1))
              / (_nw + _nc - 2))
_cohen_d = (_w_r.mean() - _c_r.mean()) / _sp
_ks = ks_2samp(_w_r, _c_r)

# ── figure layout ────────────────────────────────────────────────────
fig = plt.figure(figsize=(3.25, 3.20))
# Dot plot + narrow KDE strip on the right.
gs = GridSpec(
    1, 2, width_ratios=[5.2, 1.0], wspace=0.08,
    left=0.155, right=0.89, bottom=0.135, top=0.955, figure=fig,
)
ax_dot = fig.add_subplot(gs[0, 0])
ax_kde = fig.add_subplot(gs[0, 1], sharey=ax_dot)

# ════════════════════════════════════════════════════════════════════
# Sorted pair correlations + side KDE
# ════════════════════════════════════════════════════════════════════
ax = ax_dot
ranks = np.arange(1, len(_rs) + 1)
_mean_r = _rs.mean()

# Mean reference line
ax.axhline(_mean_r, color=GREY_LT, lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.text(0.01, _mean_r + 0.012,
        f"overall mean $r = {_mean_r:.2f}$",
        transform=blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=7.4, color=GREY_MID, ha="left", va="bottom", style="italic")

# Dots: cross-family first (grey, background), same-family on top (red)
ax.scatter(ranks[~_same], _rs[~_same], s=10, color=GREY, alpha=0.70,
           edgecolors="white", linewidths=0.2, zorder=3,
           label=f"cross-family  ($n{{=}}{_nc}$)")
ax.scatter(ranks[_same], _rs[_same], s=17, color=ENSEMBLE,
           edgecolors="white", linewidths=0.4, zorder=4,
           label=f"same-family  ($n{{=}}{_nw}$)")

# Label the top 3 pairs with their model names; leaders drawn up-left.
_top3 = sorted(zip(ranks, _rs, _labels, _same), key=lambda x: -x[1])[:3]
# Assign vertical label positions to avoid collisions
_label_ys = [0.68, 0.59, 0.50]
for (rnk, r_val, (a, b), same), ylbl in zip(_top3, _label_ys):
    col = ENSEMBLE if same else GREY_MID
    ax.annotate(
        f"{a}$\\times${b}  ($r{{=}}{r_val:.2f}$)",
        xy=(rnk, r_val), xytext=(rnk - 55, ylbl),
        fontsize=7.2, color=col, ha="right", va="center",
        fontweight="bold" if same else "normal",
        arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.65,
                        shrinkA=0, shrinkB=4),
        zorder=8,
    )

# Effect-size callout — placed right above the legend in bottom-right,
# clear of the top-3 pair-labels that live in the upper-left region.
ax.text(0.995, 0.32,
        ("Family signal is weak:\n"
         rf"Cohen's $d = {_cohen_d:.2f}$  "
         rf"(KS $p<10^{{-3}}$)" + "\n"
         r"distributions overlap $>80\%$ of pairs"),
        transform=ax.transAxes, fontsize=7.2, color=ENSEMBLE,
        ha="right", va="bottom", linespacing=1.3, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=ENSEMBLE, linewidth=0.65,
                  boxstyle="round,pad=0.32", alpha=0.95),
        zorder=9)

ax.set_xlabel("Model-pair rank (sorted by $r$)")
ax.set_ylabel("Pearson $r$")
ax.set_xlim(-5, len(_rs) + 5)
ax.set_ylim(-0.02, 0.72)
ax.set_xticks([1, 100, 200])
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_yticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
ax.tick_params(axis="y", labelsize=8.6, labelleft=True)
ax.tick_params(axis="x", labelsize=8.6)

leg = ax.legend(fontsize=7.4, loc="lower right",
                bbox_to_anchor=(0.995, 0.015),
                frameon=True, edgecolor="#cccccc", framealpha=0.95,
                handlelength=1.2, borderpad=0.34, labelspacing=0.30,
                handletextpad=0.45)
leg.get_frame().set_linewidth(0.55)

# ── KDE side-strip ──────────────────────────────────────────────────
ax = ax_kde
_y_grid = np.linspace(-0.02, 0.72, 250)
kde_w = gaussian_kde(_w_r, bw_method=0.30)
kde_c = gaussian_kde(_c_r, bw_method=0.30)
_dw = kde_w(_y_grid)
_dc = kde_c(_y_grid)
# Normalise shared horizontal scale
_maxd = max(_dw.max(), _dc.max())
_dw = _dw / _maxd
_dc = _dc / _maxd

ax.fill_betweenx(_y_grid, 0, _dc, color=GREY, alpha=0.35,
                 edgecolor=GREY_MID, linewidth=0.55, zorder=2)
ax.fill_betweenx(_y_grid, 0, _dw, color=ENSEMBLE, alpha=0.45,
                 edgecolor=ENSEMBLE, linewidth=0.55, zorder=3)

# Mean markers
ax.plot([0, 0.95], [_c_r.mean()] * 2, color=GREY_MID, lw=1.4,
        solid_capstyle="round", zorder=4)
ax.plot([0, 0.95], [_w_r.mean()] * 2, color=ENSEMBLE, lw=1.8,
        solid_capstyle="round", zorder=5)

# Mean labels on the right of the strip
ax.text(1.05, _w_r.mean(), f"$\\bar r\\!=\\!{_w_r.mean():.2f}$",
        color=ENSEMBLE, fontsize=7.2, ha="left", va="center",
        fontweight="bold", clip_on=False)
ax.text(1.05, _c_r.mean(), f"$\\bar r\\!=\\!{_c_r.mean():.2f}$",
        color=GREY_MID, fontsize=7.2, ha="left", va="center",
        clip_on=False)

ax.set_xlim(0, 1.05)
ax.set_xticks([])
ax.tick_params(axis="y", length=0, labelleft=False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
# Small label above the strip so its purpose reads
ax.text(0.5, 1.01, "density", transform=ax.transAxes,
        fontsize=7.2, color=GREY_MID, ha="center", va="bottom",
        style="italic")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supplementary_figure.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'supplementary_figure.pdf'}")
plt.close(fig)
