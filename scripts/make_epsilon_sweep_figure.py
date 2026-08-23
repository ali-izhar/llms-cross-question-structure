"""
make_epsilon_sweep_figure.py -- Figure 2(a) of the EIML@ICML 2026 paper.

Story: the within-model ceiling is not an artefact of the borderline cut --
four of the five MMLU models keep a positive MP noise margin at every
threshold, and the fifth only ever grazes zero.

x = borderline threshold epsilon (categorical, evenly spaced)
y = MP noise margin lambda_+ - lambda_max; positive means no eigenvalue
    reached the edge.  Grey band = Tracy-Widom +/- 1 sigma, averaged over
    the five models at each epsilon.
Data: results/correctness_matrix_*.npy (single source of truth; every value
      is recomputed here and printed at the end of the run).
Output: paper/figures/epsilon_sweep.pdf, exactly 3.15 x 2.15 in so
        \\includegraphics[height=2.15in] scales it 1:1, plus a 300-DPI PNG.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════
# HOUSE STYLE -- identical block in make_hero_figure.py and
# make_supplementary_figure.py.  Edit all three together or the paper
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

FIG_W, FIG_H = 3.15, 2.15


def bf(s: str) -> str:
    return rf"\textbf{{{s}}}" if _usetex else s


# ── entity -> encoding ───────────────────────────────────────────────
# The four models whose margin never leaves the noise side share one cool
# ramp: the claim is about them as a group, so hue does not have to name
# them -- the legend does, and monotone lightness keeps them separable in
# a grayscale print.  SmolLM2 is the single focal series and carries the
# warm accent, which means the same thing here as in Fig. 1b: an
# eigenvalue at or past the MP edge.
SERIES = [
    ("Qwen-7B",      "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy",
     COOL_DK, "o", 1.25, 3.9),
    ("Mistral-7B",   "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy",
     COOL,    "s", 1.25, 3.7),
    ("Phi-3-mini",   "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy",
     COOL_MD, "^", 1.25, 4.1),
    ("Llama-3-8B",   "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy",
     COOL_LT, "D", 1.25, 3.4),
    ("SmolLM2-1.7B", "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy",
     WARM,    "v", 2.0,  5.4),
]
EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def margin_and_tw(C: np.ndarray, eps: float):
    """(lambda_+ - lambda_max, sigma_TW, N_b) for one model at one cut.

    Normalisation matches src/stats.py::marchenko_pastur_test: centre each
    column, divide by the population std (ddof=0), guard zero-variance
    columns, then R = (Cn.T @ Cn) / K.
    """
    C = C.astype(np.float64)
    K = C.shape[0]
    rate = C.mean(axis=0)
    keep = (rate > eps) & (rate < 1 - eps)
    n_b = int(keep.sum())
    if n_b < 2:
        return np.nan, np.nan, n_b
    Cb = C[:, keep]
    sd = Cb.std(axis=0)
    Cn = (Cb - Cb.mean(axis=0, keepdims=True)) / np.where(sd == 0, 1.0, sd)
    ev = np.sort(np.linalg.eigvalsh((Cn.T @ Cn) / K))[::-1]
    gamma = n_b / K
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    sigma_tw = ((1 + np.sqrt(gamma))
                * (1 + 1 / np.sqrt(gamma)) ** (1 / 3) * K ** (-2 / 3))
    return lam_plus - ev[0], sigma_tw, n_b


margins, nbs = {}, {}
tw_sum = np.zeros(len(EPSILONS))
tw_n = np.zeros(len(EPSILONS))
for name, fname, *_ in SERIES:
    C = np.load(RESULTS / fname)
    m = np.full(len(EPSILONS), np.nan)
    nb = np.zeros(len(EPSILONS), dtype=int)
    for k, eps in enumerate(EPSILONS):
        g, s, n_b = margin_and_tw(C, eps)
        m[k], nb[k] = g, n_b
        if np.isfinite(s):
            tw_sum[k] += s
            tw_n[k] += 1
    margins[name], nbs[name] = m, nb
tw_sigma = np.where(tw_n > 0, tw_sum / np.maximum(tw_n, 1), np.nan)

x = np.arange(len(EPSILONS), dtype=float)

with plt.rc_context(RC):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Axes box matches make_supplementary_figure.py exactly (bottom 0.350 in,
    # height 1.500 in) so the two halves of Fig. 2 share one plot frame when
    # they are set side by side.
    ax = fig.add_axes([0.575 / FIG_W, 0.350 / FIG_H,
                       2.455 / FIG_W, 1.500 / FIG_H])

    # Tracy-Widom strip first, then the zero line on top of it: the line is
    # what every margin is read against, so it must never be buried.
    ax.fill_between(x, -tw_sigma, tw_sigma, facecolor=tint(FAINT, 0.80),
                    edgecolor="none", zorder=0.5,
                    label=r"TW $\pm 1\sigma$")
    ax.axhline(0.0, color=INK, lw=0.9, zorder=2)

    for name, _f, color, marker, lw, ms in SERIES:
        ax.plot(x, margins[name], marker=marker, color=color, lw=lw,
                markersize=ms, markeredgecolor="white",
                markeredgewidth=0.45, label=name,
                zorder=6 if color == WARM else 4,
                clip_on=False)

    Y_LO, Y_HI = -0.312, 0.545
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlim(-0.32, len(EPSILONS) - 0.50)

    # The zero line named at its right end, in the two words the reader
    # needs to orient: above it the spectrum is pure noise, below it an
    # eigenvalue has passed the edge.
    # Just clear of the TW strip below and of the Phi-3-mini marker above:
    # a 0.02 offset leaves about 2.5 pt to the fill and 4 pt to the marker.
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.995, float(tw_sigma[-1]) + 0.020, "noise", transform=tr,
            fontsize=TINY, color=MUTED, ha="right", va="bottom",
            style="italic")
    ax.text(0.995, -float(tw_sigma[-1]) - 0.020, "signal", transform=tr,
            fontsize=TINY, color=MUTED, ha="right", va="top",
            style="italic")

    ax.set_xlabel(r"Borderline threshold $\varepsilon$", labelpad=2)
    ax.set_ylabel("MP noise margin\n" r"$\lambda_+ - \lambda_{\max}$",
                  labelpad=2, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in EPSILONS])
    yticks = [-0.2, 0.0, 0.2, 0.4]
    ax.set_yticks(yticks)
    # Plain-text signed ticks: under a usetex Times stack the math-mode
    # minus can silently fail to render.
    ax.set_yticklabels([f"{t:.1f}" for t in yticks])

    # Legend above the axes so it never sits over data, frameless.
    # Matplotlib fills a multi-column legend column-major, so the handles
    # are reordered here to make each column a sensible pair and to put
    # the focal series at the head of the last one.
    h, lab = ax.get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]     # Qwen, Mistral, Phi-3, Llama, Smol, TW
    ax.legend([h[i] for i in order], [lab[i] for i in order],
              loc="lower left", bbox_to_anchor=(-0.16, 1.005), ncol=3,
              frameon=False, handlelength=1.5, handletextpad=0.42,
              columnspacing=0.95, labelspacing=0.30, borderaxespad=0.0,
              fontsize=TINY)

    fig.text(0.012, 0.982, bf("(a)"), fontsize=BODY, color=INK,
             ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"epsilon_sweep.{ext}", dpi=300)
    plt.close(fig)

print(f"usetex={_usetex}  figsize=({FIG_W}, {FIG_H}) in")
print("MP noise margin  lambda_+ - lambda_max  (N_b in brackets)")
print(f"{'model':<14s}" + "".join(f"{e:>10.2f}" for e in EPSILONS))
for name, *_ in SERIES:
    print(f"{name:<14s}"
          + "".join(f"{v:>10.3f}" for v in margins[name])
          + "   N_b=" + str(list(nbs[name])))
print(f"{'TW sigma':<14s}" + "".join(f"{v:>10.3f}" for v in tw_sigma))
lo = np.nanmin([margins[n].min() for n, *_ in SERIES])
hi = np.nanmax([margins[n].max() for n, *_ in SERIES])
print(f"data range [{lo:.3f}, {hi:.3f}]  y-limits [{Y_LO}, {Y_HI}]  "
      f"all points on axis: {lo > Y_LO and hi < Y_HI}")
print(f"Saved {OUT / 'epsilon_sweep.pdf'}")
