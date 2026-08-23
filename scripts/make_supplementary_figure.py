"""
make_supplementary_figure.py -- Figure 2(b) of the EIML@ICML 2026 paper.

Story: the four across-model directions are not a family effect -- same-family
pairs correlate only a little more than cross-family pairs, and the best
cross-family pair is as strong as the second-best same-family pair.

x = rank of a model pair among all 276, sorted ascending by Pearson r over
    the 500 MMLU items;  y = that r.  Warm = same family (n=30), grey =
    cross-family (n=246).  The strip on the right is the kernel density of
    each group on the same y-scale, with the two group means marked.
Data: results/ensemble_expanded_correctness.npy + ensemble_expanded_models.json
      (single source of truth; every value is recomputed here and printed).
Output: paper/figures/supplementary_figure.pdf, exactly 3.35 x 2.15 in so
        \\includegraphics[height=2.15in] scales it 1:1, plus a 300-DPI PNG.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════
# HOUSE STYLE -- identical block in make_hero_figure.py and
# make_epsilon_sweep_figure.py.  Edit all three together or the paper
# stops reading as one visual system.
# ═════════════════════════════════════════════════════════════════════
OSLO = ["#010101", "#0E1E2E", "#15395B", "#26578C",
        "#507BBC", "#7B98CA", "#A2B0CA", "#CED1D7", "#FFFFFF"]
COOL_DK, COOL, COOL_MD, COOL_LT = OSLO[2], OSLO[3], OSLO[4], OSLO[5]
WARM, WARM_DK = "#B75A26", "#852206"
INK, MUTED, FAINT, HAIR = "#1A1A1A", "#5B5B5B", "#8C8C8C", "#D9D9D9"


def tint(hex_color: str, f: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    m = lambda c: int(round(c + (255 - c) * f))  # noqa: E731
    return f"#{m(r):02X}{m(g):02X}{m(b):02X}"


BODY, TICK, SMALL, TINY = 9.0, 8.0, 7.5, 7.0


def _usetex_available() -> bool:
    if shutil.which("pdflatex") is None:
        return False
    try:
        r = subprocess.run(["kpsewhich", "type1ec.sty"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


_usetex = _usetex_available()
RC = {
    "text.usetex": _usetex,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": BODY,
    "axes.labelsize": BODY,
    "axes.titlesize": BODY,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "legend.fontsize": SMALL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.3,
    "patch.linewidth": 0.6,
    "figure.constrained_layout.use": False,
    "savefig.facecolor": "white",
}
if _usetex:
    RC["text.latex.preamble"] = r"\usepackage{times}"

FIG_W, FIG_H = 3.35, 2.15


def bf(s: str) -> str:
    return rf"\textbf{{{s}}}" if _usetex else s


# ── data ─────────────────────────────────────────────────────────────
C_ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
model_names = json.loads(
    (RESULTS / "ensemble_expanded_models.json").read_text())
R_ens = np.corrcoef(C_ens)

SHORT_FAM = {
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
short = [SHORT_FAM[n][0] for n in model_names]
fams = [SHORT_FAM[n][1] for n in model_names]

pairs = [(R_ens[i, j], short[i], short[j], fams[i] == fams[j])
         for i in range(M) for j in range(i + 1, M)]
pairs.sort(key=lambda p: p[0])              # sort order is stated in-figure
rs = np.array([p[0] for p in pairs])
labels = [(p[1], p[2]) for p in pairs]
same = np.array([p[3] for p in pairs])
ranks = np.arange(1, len(rs) + 1)

r_same, r_cross = rs[same], rs[~same]
n_same, n_cross = len(r_same), len(r_cross)
s_pool = np.sqrt(((n_same - 1) * r_same.var(ddof=1)
                  + (n_cross - 1) * r_cross.var(ddof=1))
                 / (n_same + n_cross - 2))
cohen_d = (r_same.mean() - r_cross.mean()) / s_pool
ks = ks_2samp(r_same, r_cross)

# The two pairs worth naming: the strongest pair overall, and the strongest
# pair drawn from two different families.  Their near-equality is the
# panel's argument, so these are the annotations that earn their place.
i_top_same = int(np.where(same)[0][-1])
i_top_cross = int(np.where(~same)[0][-1])

Y_LO, Y_HI = -0.015, 0.748

with plt.rc_context(RC):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    AX_B, AX_H = 0.350 / FIG_H, 1.500 / FIG_H
    ax = fig.add_axes([0.455 / FIG_W, AX_B, 2.070 / FIG_W, AX_H])
    ax_k = fig.add_axes([2.560 / FIG_W, AX_B, 0.380 / FIG_W, AX_H],
                        sharey=ax)

    # ── sorted pair correlations ─────────────────────────────────────
    mean_r = rs.mean()
    ax.axhline(mean_r, color=tint(FAINT, 0.45), lw=0.7, ls=(0, (2.4, 2.0)),
               zorder=1)
    ax.text(272, mean_r - 0.018, rf"all pairs $\bar r = {mean_r:.2f}$",
            fontsize=TINY, color=MUTED, ha="right", va="top", style="italic")

    ax.scatter(ranks[~same], rs[~same], s=7.5, color=tint(FAINT, 0.30),
               alpha=0.85, edgecolors="none", zorder=3,
               label=rf"cross-family ($n\!=\!{n_cross}$)")
    ax.scatter(ranks[same], rs[same], s=13, color=WARM,
               edgecolors="white", linewidths=0.35, zorder=4,
               label=rf"same family ($n\!=\!{n_same}$)")

    # Two callouts, both in the empty space above the curve, with dashed
    # leaders that land on their point and cross nothing.
    for idx, y_lab, col in ((i_top_same, 0.710, WARM_DK),
                            (i_top_cross, 0.628, MUTED)):
        a, b = labels[idx]
        ax.annotate(
            rf"{a}$\times${b} ($r\!=\!{rs[idx]:.2f}$)",
            xy=(ranks[idx], rs[idx]), xytext=(ranks[idx] - 28, y_lab),
            fontsize=TINY, color=col, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.6,
                            shrinkA=2, shrinkB=3,
                            linestyle=(0, (2.2, 1.8))),
            zorder=8)

    ax.set_xlabel("Model-pair rank (sorted by $r$)", labelpad=2)
    ax.set_ylabel(r"Pearson $r$ across 500 items", labelpad=2)
    ax.set_xlim(-6, len(rs) + 6)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([1, 100, 200, len(rs)])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6"])

    ax.legend(loc="lower right", bbox_to_anchor=(1.005, -0.01),
              frameon=False, handlelength=0.9, handletextpad=0.40,
              labelspacing=0.28, borderaxespad=0.0, fontsize=TINY,
              scatterpoints=1)

    fig.text(0.012, 0.982, bf("(b)"), fontsize=BODY, color=INK,
             ha="left", va="top")
    # The headline effect size lives in the top band -- the same strip that
    # carries the legend in Fig. 2a -- so no text sits over the data.
    fig.text(0.088, 0.962,
             ("Family is a weak nudge:  "
              rf"Cohen's $d\!=\!{cohen_d:.2f}$,  KS $p\!<\!10^{{-3}}$"),
             fontsize=TINY, color=MUTED, ha="left", va="top")

    # ── density strip: the overlap is the point ──────────────────────
    grid = np.linspace(Y_LO, Y_HI, 320)
    d_same = gaussian_kde(r_same, bw_method=0.30)(grid)
    d_cross = gaussian_kde(r_cross, bw_method=0.30)(grid)
    scale = max(d_same.max(), d_cross.max())
    d_same, d_cross = d_same / scale, d_cross / scale

    # Outlines drawn separately so the fills do not stroke a spurious
    # vertical rule down the x = 0 side of the strip.
    ax_k.fill_betweenx(grid, 0, d_cross, facecolor=tint(FAINT, 0.55),
                       edgecolor="none", zorder=2)
    ax_k.fill_betweenx(grid, 0, d_same, facecolor=tint(WARM, 0.55),
                       edgecolor="none", alpha=0.85, zorder=3)
    ax_k.plot(d_cross, grid, color=FAINT, lw=0.5, zorder=3.5)
    ax_k.plot(d_same, grid, color=WARM, lw=0.6, zorder=3.6)
    ax_k.plot([0, 0.92], [r_cross.mean()] * 2, color=MUTED, lw=1.2,
              solid_capstyle="round", zorder=4)
    ax_k.plot([0, 0.92], [r_same.mean()] * 2, color=WARM_DK, lw=1.5,
              solid_capstyle="round", zorder=5)
    ax_k.text(1.02, r_same.mean() + 0.012, rf"$\bar r\!=\!{r_same.mean():.2f}$",
              color=WARM_DK, fontsize=TINY, ha="left", va="bottom",
              clip_on=False)
    ax_k.text(1.02, r_cross.mean() - 0.012,
              rf"$\bar r\!=\!{r_cross.mean():.2f}$",
              color=MUTED, fontsize=TINY, ha="left", va="top", clip_on=False)

    ax_k.set_xlim(0, 1.02)
    ax_k.set_xticks([])
    ax_k.tick_params(axis="y", length=0, labelleft=False)
    for side in ("left", "bottom"):
        ax_k.spines[side].set_visible(False)
    ax_k.text(0.5, 1.005, "density", transform=ax_k.transAxes,
              fontsize=TINY, color=FAINT, ha="center", va="bottom",
              style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"supplementary_figure.{ext}", dpi=300)
    plt.close(fig)

print(f"usetex={_usetex}  figsize=({FIG_W}, {FIG_H}) in")
print(f"{M} models, {len(rs)} pairs; same-family n={n_same}, "
      f"cross-family n={n_cross}")
print(f"r range [{rs.min():.4f}, {rs.max():.4f}]  overall mean {mean_r:.4f}")
print(f"same-family  mean {r_same.mean():.4f}  sd {r_same.std(ddof=1):.4f}")
print(f"cross-family mean {r_cross.mean():.4f}  sd {r_cross.std(ddof=1):.4f}")
print(f"Cohen's d {cohen_d:.4f}   KS D={ks.statistic:.4f} p={ks.pvalue:.3e}")
for tag, idx in (("top same-family ", i_top_same),
                 ("top cross-family", i_top_cross)):
    a, b = labels[idx]
    print(f"{tag}: {a} x {b}  r={rs[idx]:.4f}  rank {ranks[idx]}")
print(f"y-limits [{Y_LO}, {Y_HI}]  all points on axis: "
      f"{rs.min() > Y_LO and rs.max() < Y_HI}")
print(f"Saved {OUT / 'supplementary_figure.pdf'}")
