"""
Supplementary figure — OPTION 1: current layout + headline annotations.

Keeps (a) heatmap + (b) threshold robustness unchanged, but adds compact
summary boxes so each panel has a headline-level quantitative claim.
"""

import pathlib
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
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

fig, axes = plt.subplots(
    1, 2,
    figsize=(6.75, 3.00),
    gridspec_kw={"width_ratios": [1.15, 1.00], "wspace": 0.36},
)
fig.subplots_adjust(left=0.06, right=0.975, bottom=0.17, top=0.88)

# ═════════ (a) Heatmap ═════════
ax = axes[0]
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
R_plot = R_ord.copy()
np.fill_diagonal(R_plot, np.nan)
_off = R_plot[np.isfinite(R_plot)]
vmin = 0.0; vmax = float(_off.max())
im = ax.imshow(R_plot, cmap=_heat_cmap, vmin=vmin, vmax=vmax,
               aspect="equal", interpolation="nearest")

boundaries = [i - 0.5 for i in range(1, M) if fams[i] != fams[i - 1]]
for s in boundaries:
    ax.axvline(s, color="white", lw=1.0, zorder=4)
    ax.axhline(s, color="white", lw=1.0, zorder=4)

runs = []
_i0 = 0
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

# ─── HEADLINE ANNOTATION for (a) ───
# Compute within vs cross-family correlation stats on-the-fly
fams_all = [_SHORT_MAP[n][1] for n in model_names]
_w, _c = [], []
for i in range(M):
    for j in range(i + 1, M):
        (_w if fams_all[i] == fams_all[j] else _c).append(R_ens[i, j])
_w, _c = np.array(_w), np.array(_c)
# Box in upper-right quadrant of the (a) axes area
ax.text(0.98, -0.11,
        (rf"within-family $r = {_w.mean():.2f} \pm {_w.std():.2f}$  "
         rf"($n{{=}}{len(_w)}$)" + "\n"
         rf"cross-family $r = {_c.mean():.2f} \pm {_c.std():.2f}$  "
         rf"($n{{=}}{len(_c)}$)"),
        transform=ax.transAxes, fontsize=6.8, color=ENSEMBLE,
        ha="right", va="top", linespacing=1.3, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=ENSEMBLE, linewidth=0.55,
                  boxstyle="round,pad=0.35", alpha=0.96),
        zorder=10)

# ═════════ (b) Threshold robustness ═════════
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

# ─── HEADLINE ANNOTATION for (b) ───
# Summarise the robustness story in one box.
ax.text(0.97, 0.96,
        r"$\mathbf{4\,/\,5\ models}$: no signal at any $\varepsilon$" + "\n"
        r"$\mathbf{SmolLM2}$: $\leq\!1$ signal dim (at $\varepsilon{=}0.15$ only)",
        transform=ax.transAxes, fontsize=6.5, color="#8a7a0c",
        ha="right", va="top", linespacing=1.3,
        bbox=dict(facecolor="white", edgecolor="#c99b2f", linewidth=0.55,
                  boxstyle="round,pad=0.35", alpha=0.96),
        zorder=10)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"supplementary_option1.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'supplementary_option1.pdf'}")
plt.close(fig)
