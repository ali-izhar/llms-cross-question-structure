"""
Supplementary figure — OPTION 2: replace (a) heatmap with a sorted
pair-correlation dot plot, separating same-family vs cross-family pairs.

Shows the family-structure story as a distribution: same-family pairs
concentrate at higher r, cross-family pairs distribute lower.  More
"punchy" than a heatmap, but individual pair lookup is lost.

Panel (b) is unchanged from the current figure.
"""

import pathlib
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from tueplots import bundles
from tueplots.constants.color import palettes

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"

rc = bundles.icml2024(column="full", usetex=False, family="serif")
rc["figure.figsize"] = (6.75, 3.00)
rc["figure.constrained_layout.use"] = False
rc["axes.spines.top"] = False
rc["axes.spines.right"] = False
rc["axes.linewidth"] = 0.6
rc["xtick.major.width"] = 0.5
rc["ytick.major.width"] = 0.5
rc["xtick.major.size"] = 2.8
rc["ytick.major.size"] = 2.8
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
GREY    = f"#{_tol[6]}"
GREY_LT = "#c5cdd4"
GREY_MID = "#5c6773"
ENSEMBLE = "#CC3311"

C_ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
with open(RESULTS / "ensemble_expanded_models.json") as f:
    model_names = json.load(f)
R_ens = np.corrcoef(C_ens)
C_q = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
C_m = np.load(RESULTS / "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy")
C_smol = np.load(RESULTS / "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy")
C_phi  = np.load(RESULTS / "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy")
C_lla  = np.load(RESULTS / "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy")

_SHORT_FAM = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":    "SmolLM",
    "Qwen/Qwen2.5-0.5B-Instruct":             "Qwen",
    "Qwen/Qwen2.5-1.5B-Instruct":             "Qwen",
    "Qwen/Qwen2.5-3B-Instruct":               "Qwen",
    "Qwen/Qwen2.5-7B-Instruct":               "Qwen",
    "Qwen/Qwen2-7B-Instruct":                 "Qwen",
    "microsoft/phi-2":                        "Phi",
    "microsoft/Phi-3-mini-4k-instruct":       "Phi",
    "microsoft/Phi-3.5-mini-instruct":        "Phi",
    "mistralai/Mistral-7B-Instruct-v0.2":     "Mistral",
    "mistralai/Mistral-7B-Instruct-v0.3":     "Mistral",
    "HuggingFaceH4/zephyr-7b-beta":           "Mistral",
    "mistralai/Ministral-8B-Instruct-2410":   "Mistral",
    "mistralai/Mistral-Small-Instruct-2409":  "Mistral",
    "meta-llama/Llama-3.2-1B-Instruct":       "Llama",
    "meta-llama/Llama-3.2-3B-Instruct":       "Llama",
    "meta-llama/Llama-3.1-8B-Instruct":       "Llama",
    "meta-llama/Meta-Llama-3-8B-Instruct":    "Llama",
    "google/gemma-2-2b-it":                   "Gemma",
    "google/gemma-2-9b-it":                   "Gemma",
    "allenai/OLMo-2-1124-7B-Instruct":        "OLMo",
    "01-ai/Yi-1.5-6B-Chat":                   "Yi",
    "deepseek-ai/deepseek-llm-7b-chat":       "DeepSeek",
    "ibm-granite/granite-3.1-8b-instruct":    "Granite",
}

M = R_ens.shape[0]
fams = [_SHORT_FAM[n] for n in model_names]

# Build arrays of pair correlations tagged by same-family flag.
pairs = []
for i in range(M):
    for j in range(i + 1, M):
        pairs.append((R_ens[i, j], fams[i] == fams[j]))
pairs.sort(key=lambda x: x[0])   # ascending by r
_rs = np.array([p[0] for p in pairs])
_same = np.array([p[1] for p in pairs])

fig, axes = plt.subplots(
    1, 2,
    figsize=(6.75, 3.00),
    gridspec_kw={"width_ratios": [1.20, 1.00], "wspace": 0.32},
)
fig.subplots_adjust(left=0.065, right=0.975, bottom=0.17, top=0.88)

# ═════════ (a) Sorted pair correlations ═════════
ax = axes[0]
ranks = np.arange(1, len(_rs) + 1)

# Faint grey guide line at the overall mean r.
_mean_r = _rs.mean()
ax.axhline(_mean_r, color=GREY_LT, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.text(0.99, _mean_r + 0.01, f"mean $r = {_mean_r:.2f}$",
        transform=blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=6.5, color=GREY_MID, ha="right", va="bottom", style="italic")

# Cross-family dots (larger population, grey) drawn first.
ax.scatter(ranks[~_same], _rs[~_same], s=13, color=GREY, alpha=0.75,
           edgecolors="white", linewidths=0.25, zorder=3,
           label=f"cross-family ($n{{=}}{(~_same).sum()}$)")
# Same-family dots (smaller population, red) on top.
ax.scatter(ranks[_same], _rs[_same], s=22, color=ENSEMBLE,
           edgecolors="white", linewidths=0.4, zorder=4,
           label=f"same-family ($n{{=}}{_same.sum()}$)")

# Mean of each group, drawn as short horizontal bars at the right edge.
_w = _rs[_same]; _c = _rs[~_same]
_x_right = len(_rs) + 6
ax.plot([_x_right - 4, _x_right + 4], [_w.mean()] * 2,
        color=ENSEMBLE, lw=2.0, solid_capstyle="round", clip_on=False, zorder=5)
ax.plot([_x_right - 4, _x_right + 4], [_c.mean()] * 2,
        color=GREY_MID, lw=2.0, solid_capstyle="round", clip_on=False, zorder=5)
ax.text(_x_right + 6, _w.mean(), f"$\\bar r = {_w.mean():.2f}$",
        color=ENSEMBLE, fontsize=6.8, ha="left", va="center",
        fontweight="bold", clip_on=False)
ax.text(_x_right + 6, _c.mean(), f"$\\bar r = {_c.mean():.2f}$",
        color=GREY_MID, fontsize=6.8, ha="left", va="center",
        clip_on=False)

ax.set_xlabel("Model-pair rank (sorted by $r$)")
ax.set_ylabel("Pearson $r$")
ax.set_title(r"$\mathbf{(a)}$  Pairwise correlations, $\binom{24}{2}\!=\!276$ pairs",
             loc="left", fontweight="normal", pad=4)
ax.set_xlim(-5, _x_right + 40)
ax.set_ylim(-0.02, 0.72)
ax.set_xticks([1, 50, 100, 150, 200, 250])
leg = ax.legend(fontsize=7.0, loc="upper left", frameon=True,
                edgecolor="#cccccc", framealpha=0.95,
                handlelength=1.2, borderpad=0.32, labelspacing=0.3,
                handletextpad=0.5)
leg.get_frame().set_linewidth(0.45)

# ═════════ (b) Threshold robustness (unchanged) ═════════
ax = axes[1]
epsilons = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
series = [
    ("Qwen-7B",      C_q,    BLUE,   "o"),
    ("Mistral-7B",   C_m,    RED,    "s"),
    ("Phi-3-mini",   C_phi,  GREEN,  "^"),
    ("Llama-3-8B",   C_lla,  PURPLE, "D"),
    ("SmolLM2-1.7B", C_smol, YELLOW, "v"),
]
_gaps_all = {}
for label, C_data, color, marker in series:
    flip = C_data.mean(axis=0); K = C_data.shape[0]
    gaps = []
    for eps in epsilons:
        bord = (flip > eps) & (flip < 1 - eps); nb = bord.sum()
        if nb < 2: gaps.append(np.nan); continue
        Cb = C_data[:, bord]; Rc = np.corrcoef(Cb.T)
        ev = np.sort(np.linalg.eigvalsh(Rc))[::-1]
        gam = nb / K; lplus = (1 + np.sqrt(gam)) ** 2
        gaps.append(lplus - ev[0])
    _gaps_all[label] = np.array(gaps, dtype=float)

ax.axhspan(-1.0, 0.0, facecolor=YELLOW, alpha=0.16, zorder=0)
ax.axhline(0, color=GREY_MID, lw=0.7, ls="-", alpha=0.75, zorder=1)

for label, C_data, color, marker in series[:-1]:
    ax.plot(epsilons, _gaps_all[label], f"-{marker}", color=color,
            markersize=4.4, label=label, lw=1.15,
            markeredgecolor="white", markeredgewidth=0.35,
            zorder=3, alpha=0.92)
ax.plot(epsilons, _gaps_all["SmolLM2-1.7B"], "-v", color=YELLOW,
        markersize=5.5, label="SmolLM2-1.7B", lw=1.8,
        markeredgecolor="#8a7a0c", markeredgewidth=0.7, zorder=5)

_tr = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.975, 0.58, r"below $\lambda_+$", transform=_tr,
        fontsize=7.2, color=GREY_MID, ha="right", va="bottom", style="italic")
ax.text(0.975, 0.50, "(noise)", transform=_tr,
        fontsize=7.2, color=GREY_MID, ha="right", va="bottom", style="italic")
ax.text(0.975, -0.08, r"above $\lambda_+$", transform=_tr,
        fontsize=7.2, color="#8a7a0c", ha="right", va="top",
        style="italic", fontweight="bold")
ax.text(0.975, -0.16, "(signal)", transform=_tr,
        fontsize=7.2, color="#8a7a0c", ha="right", va="top",
        style="italic", fontweight="bold")

ax.set_xlabel(r"Borderline threshold $\varepsilon$")
ax.set_ylabel(r"Noise margin $\lambda_+ - \lambda_{\max}$")
ax.set_title(r"$\mathbf{(b)}$  Threshold robustness",
             loc="left", fontweight="normal", pad=4)
ax.set_xlim(-0.005, 0.22)
ax.set_ylim(-0.35, 0.78)
ax.tick_params(axis="both", labelsize=7.5)
leg = ax.legend(fontsize=7.0, loc="upper left",
                bbox_to_anchor=(0.005, 0.995),
                framealpha=0.95, edgecolor="#cccccc",
                handlelength=1.4, borderpad=0.32, labelspacing=0.30)
leg.get_frame().set_linewidth(0.45)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supplementary_option2.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'supplementary_option2.pdf'}")
plt.close(fig)
