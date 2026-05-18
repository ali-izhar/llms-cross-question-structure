"""
Supplementary figure — OPTION 3: replace (b) with small multiples — one
mini-panel per within-model configuration — so SmolLM2's dip below zero
stands out as the only panel with the line crossing y=0.

Panel (a) is unchanged (the Reds-keyed ensemble correlation heatmap).
"""

import pathlib
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from tueplots import bundles
from tueplots.constants.color import palettes

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"

rc = bundles.icml2024(column="full", usetex=False, family="serif")
rc["figure.figsize"] = (6.75, 3.05)
rc["figure.constrained_layout.use"] = False
rc["axes.spines.top"] = False
rc["axes.spines.right"] = False
rc["axes.linewidth"] = 0.6
rc["xtick.major.width"] = 0.5
rc["ytick.major.width"] = 0.5
rc["xtick.major.size"] = 2.4
rc["ytick.major.size"] = 2.4
rc["mathtext.fontset"] = "cm"
rc["axes.labelsize"] = 8.2
rc["axes.titlesize"] = 8.8
plt.rcParams.update(rc)

_tol = palettes.paultol_bright
BLUE    = f"#{_tol[0]}"
RED     = f"#{_tol[1]}"
GREEN   = f"#{_tol[2]}"
YELLOW  = f"#{_tol[3]}"
PURPLE  = f"#{_tol[5]}"
GREY_MID = "#5c6773"
ENSEMBLE = "#CC3311"

_heat_cmap = LinearSegmentedColormap.from_list(
    "paper_ensemble",
    ["#ffffff", "#ffecdc", "#f7b191", "#e86b45", ENSEMBLE, "#7a1e0a"],
)

C_ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
with open(RESULTS / "ensemble_expanded_models.json") as f:
    model_names = json.load(f)
R_ens = np.corrcoef(C_ens)
C_q = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
C_m = np.load(RESULTS / "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy")
C_smol = np.load(RESULTS / "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy")
C_phi  = np.load(RESULTS / "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy")
C_lla  = np.load(RESULTS / "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy")

# ── GridSpec: (a) big heatmap on the left; (b) 5 small multiples on the right
fig = plt.figure(figsize=(6.75, 3.05))
outer = GridSpec(1, 2, width_ratios=[1.12, 1.00], wspace=0.30,
                 left=0.06, right=0.975, bottom=0.17, top=0.88, figure=fig)
ax_heat = fig.add_subplot(outer[0, 0])

# 5 small multiples inside the right cell: 1×5 row
smult = GridSpec(
    2, 3, wspace=0.18, hspace=0.35,
    left=outer[0, 1].get_position(fig).x0,
    right=outer[0, 1].get_position(fig).x1,
    bottom=outer[0, 1].get_position(fig).y0,
    top=outer[0, 1].get_position(fig).y1 - 0.02,
)
# 6 cells → 5 plots + 1 legend placeholder
sm_axes = [
    fig.add_subplot(smult[0, 0]),
    fig.add_subplot(smult[0, 1]),
    fig.add_subplot(smult[0, 2]),
    fig.add_subplot(smult[1, 0]),
    fig.add_subplot(smult[1, 1]),
]
# Last cell (bottom-right) kept for headline annotation
ax_callout = fig.add_subplot(smult[1, 2])

# ═════════ (a) Heatmap ═════════
ax = ax_heat
M = R_ens.shape[0]

_SHORT_MAP = {
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
_FAMILY_ORDER = ["Qwen", "Phi", "Mistral", "Llama", "Gemma",
                 "SmolLM", "OLMo", "Yi", "DeepSeek", "Granite"]
import re
def _size_from_short(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", s)
    if m: return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0
def _sort_key(i):
    name = model_names[i]; short_lbl, fam = _SHORT_MAP[name]
    return (_FAMILY_ORDER.index(fam), _size_from_short(short_lbl), short_lbl)

perm = sorted(range(M), key=_sort_key)
R_ord = R_ens[np.ix_(perm, perm)]
short = [_SHORT_MAP[model_names[i]][0] for i in perm]
fams  = [_SHORT_MAP[model_names[i]][1] for i in perm]
R_plot = R_ord.copy(); np.fill_diagonal(R_plot, np.nan)
_off = R_plot[np.isfinite(R_plot)]
vmin = 0.0; vmax = float(_off.max())
im = ax.imshow(R_plot, cmap=_heat_cmap, vmin=vmin, vmax=vmax,
               aspect="equal", interpolation="nearest")

for i in range(1, M):
    if fams[i] != fams[i - 1]:
        ax.axvline(i - 0.5, color="white", lw=1.0, zorder=4)
        ax.axhline(i - 0.5, color="white", lw=1.0, zorder=4)

runs = []; _i0 = 0
for _i in range(1, M + 1):
    if _i == M or fams[_i] != fams[_i0]:
        runs.append((_i0, _i - 1, fams[_i0])); _i0 = _i

ax.set_xticks(range(M)); ax.set_yticks(range(M))
ax.set_xticklabels(short, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticklabels(short, fontsize=6.5)
ax.set_title(r"$\mathbf{(a)}$  Ensemble correlations ($M\!=\!24$)",
             loc="left", fontweight="normal", pad=12)
ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
ax.tick_params(axis="y", length=0, pad=7)
ax.tick_params(axis="x", length=0, pad=5.5)

_muted = palettes.paultol_muted
_FAM_COLORS = {fam: f"#{_muted[i]}" for i, fam in enumerate(_FAMILY_ORDER)}
_strip_w = 0.50; _strip_gap = 0.18
_tr_x = blended_transform_factory(ax.transData, ax.transData)
for a, b, fam in runs:
    ax.add_patch(Rectangle((a - 0.5, -0.5 - _strip_gap - _strip_w),
                           (b - a + 1), _strip_w,
                           facecolor=_FAM_COLORS[fam], edgecolor="white",
                           lw=0.5, clip_on=False, zorder=5, transform=_tr_x))
for a, b, fam in runs:
    ax.add_patch(Rectangle((-0.5 - _strip_gap - _strip_w, a - 0.5),
                           _strip_w, (b - a + 1),
                           facecolor=_FAM_COLORS[fam], edgecolor="white",
                           lw=0.5, clip_on=False, zorder=5, transform=_tr_x))

cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.035, shrink=0.92,
                    ticks=[0.0, 0.2, 0.4, 0.6])
cbar.ax.tick_params(labelsize=7, length=2, width=0.4)
cbar.outline.set_linewidth(0.4)
cbar.set_label("Pearson $r$", fontsize=8, labelpad=3)

# ═════════ (b) Small multiples ═════════
epsilons = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
series = [
    ("Qwen-7B",      C_q,    BLUE,   "o", False),
    ("Mistral-7B",   C_m,    RED,    "s", False),
    ("Phi-3",        C_phi,  GREEN,  "^", False),
    ("Llama-3-8B",   C_lla,  PURPLE, "D", False),
    ("SmolLM2",      C_smol, YELLOW, "v", True),
]
y_min, y_max = -0.35, 0.78

for sax, (label, C_data, color, marker, is_outlier) in zip(sm_axes, series):
    flip = C_data.mean(axis=0); K = C_data.shape[0]
    gaps = []
    for eps in epsilons:
        bord = (flip > eps) & (flip < 1 - eps); nb = bord.sum()
        if nb < 2: gaps.append(np.nan); continue
        Cb = C_data[:, bord]; Rc = np.corrcoef(Cb.T)
        ev = np.sort(np.linalg.eigvalsh(Rc))[::-1]
        gam = nb / K; lplus = (1 + np.sqrt(gam)) ** 2
        gaps.append(lplus - ev[0])
    gaps = np.array(gaps, dtype=float)

    # Signal region (below 0) always shaded; bolder for outlier.
    sax.axhspan(y_min, 0.0, facecolor=YELLOW,
                alpha=0.30 if is_outlier else 0.10, zorder=0)
    sax.axhline(0, color=GREY_MID, lw=0.5, alpha=0.75, zorder=1)

    sax.plot(epsilons, gaps, f"-{marker}",
             color=color, lw=1.5 if is_outlier else 1.0,
             markersize=4.0 if is_outlier else 3.2,
             markeredgecolor="#8a7a0c" if is_outlier else "white",
             markeredgewidth=0.6 if is_outlier else 0.3,
             zorder=4)

    # Panel-title in-box so models are labelled without eating y-axis space.
    sax.text(0.04, 0.94, label,
             transform=sax.transAxes,
             fontsize=6.8, color=color if not is_outlier else "#8a7a0c",
             fontweight="bold", ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="none",
                       alpha=0.88, pad=1.5))

    sax.set_xlim(-0.005, 0.22)
    sax.set_ylim(y_min, y_max)
    sax.set_xticks([0.0, 0.1, 0.2])
    sax.set_yticks([-0.2, 0.0, 0.4])
    sax.tick_params(axis="both", labelsize=6.3, pad=1.5)
    # Only leftmost column shows y-tick labels
    if sax not in (sm_axes[0], sm_axes[3]):
        sax.set_yticklabels([])
    # Only bottom row shows x-tick labels
    if sax in (sm_axes[0], sm_axes[1], sm_axes[2]):
        sax.set_xticklabels([])

# Shared axis labels positioned below/left of the small-multiples grid.
# Use figure-level text anchored to the grid region.
_gpos = outer[0, 1].get_position(fig)
fig.text((_gpos.x0 + _gpos.x1) / 2, _gpos.y0 - 0.015,
         r"Borderline threshold $\varepsilon$",
         fontsize=8.2, ha="center", va="top")
fig.text(_gpos.x0 - 0.04, (_gpos.y0 + _gpos.y1) / 2,
         r"Noise margin $\lambda_+ - \lambda_{\max}$",
         fontsize=8.2, ha="center", va="center", rotation=90)

# Panel (b) title, aligned to the grid's top-left
fig.text(_gpos.x0 + 0.003, _gpos.y1 + 0.012,
         r"$\mathbf{(b)}$  Threshold robustness (per model)",
         fontsize=8.8, ha="left", va="bottom")

# Headline callout in the 6th (empty) small-multiple cell.
ax_callout.set_xlim(0, 1); ax_callout.set_ylim(0, 1)
ax_callout.set_xticks([]); ax_callout.set_yticks([])
for s in ax_callout.spines.values():
    s.set_visible(False)
ax_callout.text(0.5, 0.60,
                r"$\mathbf{4\,/\,5}$" + "\n"
                r"$\mathbf{no\ signal}$" + "\n"
                r"at any $\varepsilon$",
                fontsize=7.0, color=GREY_MID, ha="center", va="center",
                linespacing=1.25)
ax_callout.text(0.5, 0.18,
                r"SmolLM2: $\leq\!1$ dim" + "\n"
                r"(only at $\varepsilon{=}0.15$)",
                fontsize=6.4, color="#8a7a0c", ha="center", va="center",
                linespacing=1.2, fontweight="bold")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supplementary_option3.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'supplementary_option3.pdf'}")
plt.close(fig)
