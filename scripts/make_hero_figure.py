"""
make_hero_figure.py -- Figure 1 of the EIML@ICML 2026 paper.

Story: one model sampled a hundred times produces at most one direction above
the Marchenko-Pastur noise edge; twenty-four models produce four.

  (a) Qwen2.5-7B's question-question eigenvalue density sits inside the
      Marchenko-Pastur prediction; lambda_max = 4.17 falls short of
      lambda_+ = 4.41.
  (b) MP signal count for all nine within-model settings of Table 1 against
      the 24-model ensemble, drawn as a unit chart: one filled chip per
      eigenvalue above that setting's own edge.
  (c) Split-half per-question pass rates (r = 0.994); the 121 borderline
      items that panel (a) analyses are picked out of the cloud.

Data: results/correctness_matrix_*.npy and results/ensemble_expanded_*.
      Every number is recomputed from those files at render time -- nothing
      about the spectra is hardcoded here.
Output: paper/figures/hero_figure.pdf (vector master, exactly 6.75 x 2.45 in
        so \\includegraphics[width=\\textwidth] scales it 1:1) + a 300-DPI PNG.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import pearsonr

# ── paths ────────────────────────────────────────────────────────────
PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════
# HOUSE STYLE -- identical block in make_epsilon_sweep_figure.py and
# make_supplementary_figure.py.  Edit all three together or the paper
# stops reading as one visual system.
# ═════════════════════════════════════════════════════════════════════
# Palette: Fabio Crameri's Scientific colour maps (perceptually uniform,
# colour-vision-deficiency safe, monotone L* so they survive a grayscale
# print).  Cool stops are `oslo`, warm stops are the warm arm of `vik`.
#
# Roles -- one entity, one encoding, in every figure of the paper:
#   COOL_*  within-model data (one model, many samples)
#   WARM    the editorially focal item: an eigenvalue ABOVE the MP edge
#           in Fig. 1b, SmolLM2 in Fig. 2a, same-family pairs in Fig. 2b
#   INK     theory (the MP law, the edge, the identity line) and text
#   grey    the background population
OSLO = ["#010101", "#0E1E2E", "#15395B", "#26578C",
        "#507BBC", "#7B98CA", "#A2B0CA", "#CED1D7", "#FFFFFF"]
COOL_DK, COOL, COOL_MD, COOL_LT = OSLO[2], OSLO[3], OSLO[4], OSLO[5]
WARM, WARM_DK = "#B75A26", "#852206"          # vik warm arm
INK, MUTED, FAINT, HAIR = "#1A1A1A", "#5B5B5B", "#8C8C8C", "#D9D9D9"


def tint(hex_color: str, f: float) -> str:
    """Blend `hex_color` a fraction `f` of the way toward white."""
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
# The paper loads \usepackage{times} only -- Times text, Computer Modern
# math.  Match that exactly so figure type is indistinguishable from body
# type at 1:1.
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

# ── canvas: physically \textwidth so \includegraphics scales 1:1 ─────
FIG_W, FIG_H = 6.75, 2.45


def bf(s: str) -> str:
    return rf"\textbf{{{s}}}" if _usetex else s


# ═════════════════════════════════════════════════════════════════════
# DATA -- recomputed from the raw correctness matrices
# ═════════════════════════════════════════════════════════════════════
EPS = 0.05  # borderline cut used throughout the paper


def mp_spectrum(C: np.ndarray, eps: float = EPS):
    """Correlation spectrum of the borderline columns of a K x N 0/1 matrix.

    Normalisation matches src/stats.py::marchenko_pastur_test: centre each
    column, divide by the population std (ddof=0), guard zero-variance
    columns, then R = (Cn.T @ Cn) / K.
    """
    C = C.astype(np.float64)
    K = C.shape[0]
    rate = C.mean(axis=0)
    keep = (rate > eps) & (rate < 1 - eps)
    Cb = C[:, keep]
    sd = Cb.std(axis=0)
    Cn = (Cb - Cb.mean(axis=0, keepdims=True)) / np.where(sd == 0, 1.0, sd)
    R = (Cn.T @ Cn) / K
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    gamma = Cb.shape[1] / K
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    lam_minus = (1 - np.sqrt(gamma)) ** 2
    return dict(ev=ev, K=K, N_b=int(keep.sum()), gamma=gamma,
                lam_plus=lam_plus, lam_minus=lam_minus,
                count=int((ev > lam_plus).sum()), keep=keep)


# --- panel (a) + (c): Qwen2.5-7B on MMLU -----------------------------
C_qwen = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
K, N = C_qwen.shape
S = mp_spectrum(C_qwen)
ev, lam_plus, lam_minus, gamma, N_b = (
    S["ev"], S["lam_plus"], S["lam_minus"], S["gamma"], S["N_b"])
lam_max = ev[0]

x_mp = np.linspace(lam_minus, lam_plus, 800)
mp_pdf = (np.sqrt(np.clip((lam_plus - x_mp) * (x_mp - lam_minus), 0, None))
          / (2 * np.pi * gamma * x_mp))

first50, last50 = C_qwen[:50].mean(axis=0), C_qwen[50:].mean(axis=0)
r_split, _ = pearsonr(first50, last50)
borderline = S["keep"]

# --- panel (b): all nine within-model settings + the ensemble --------
WITHIN = [
    ("MMLU",      "Qwen2.5-7B",   "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy"),
    ("MMLU",      "Mistral-7B",   "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy"),
    ("MMLU",      "SmolLM2-1.7B", "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy"),
    ("MMLU",      "Phi-3-mini",   "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy"),
    ("MMLU",      "Llama-3-8B",   "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy"),
    ("HellaSwag", "Qwen2.5-7B",   "correctness_matrix_hellaswag_Qwen_Qwen2.5-7B-Instruct.npy"),
    ("HellaSwag", "Mistral-7B",   "correctness_matrix_hellaswag_mistralai_Mistral-7B-Instruct-v0.3.npy"),
    ("HellaSwag", "Llama-3-8B",   "correctness_matrix_hellaswag_meta-llama_Meta-Llama-3-8B-Instruct.npy"),
    ("GSM8K",     "Qwen2.5-7B",   "correctness_matrix_gsm8k_medium_Qwen_Qwen2.5-7B-Instruct.npy"),
]
within_rows = []
for bench, name, fname in WITHIN:
    s = mp_spectrum(np.load(RESULTS / fname))
    within_rows.append(dict(bench=bench, name=name, count=s["count"],
                            N_b=s["N_b"], K=s["K"],
                            lam_plus=s["lam_plus"], lam_max=s["ev"][0]))

# 24 models, one greedy pass each, on the items where they do not all agree.
C_ens = np.load(RESULTS / "ensemble_expanded_correctness.npy").astype(np.float64)
M_ens = C_ens.shape[0]
col = C_ens.sum(axis=0)
split = (col > 0) & (col < M_ens)
Ce = C_ens[:, split]
sd_e = Ce.std(axis=0)
Cn_e = (Ce - Ce.mean(axis=0, keepdims=True)) / np.where(sd_e == 0, 1.0, sd_e)
ev_e = np.sort(np.linalg.eigvalsh((Cn_e.T @ Cn_e) / M_ens))[::-1]
gam_e = Ce.shape[1] / M_ens
lam_plus_e = (1 + np.sqrt(gam_e)) ** 2
count_e = int((ev_e > lam_plus_e).sum())

MAX_SLOTS = max(count_e, max(r["count"] for r in within_rows))

# ═════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════
with plt.rc_context(RC):
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def rect(x0, y0, w, h):
        """Axes rectangle given in INCHES on the canvas."""
        return [x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H]

    AX_BOT, AX_H = 0.42, 1.70
    ax_a = fig.add_axes(rect(0.50, AX_BOT, 1.34, AX_H))
    ax_b = fig.add_axes(rect(2.26, AX_BOT, 2.22, AX_H))
    ax_c = fig.add_axes(rect(4.92, AX_BOT, 1.54, AX_H))

    def panel_title(ax, tag, text, pad=4.5):
        ax.set_title(f"{bf('(' + tag + ')')}\\ \\ {text}" if _usetex
                     else f"({tag})  {text}",
                     loc="left", pad=pad, color=INK)

    # ═════════════════════════════════════════════════════════════════
    # (a) observed spectrum inside the Marchenko-Pastur prediction
    # ═════════════════════════════════════════════════════════════════
    ax = ax_a
    # No grid here: the panel is read for shape, and gridlines would cross
    # all three annotation blocks that live in the empty upper region.
    ax.set_axisbelow(True)

    # Theory as a filled envelope, observation as a dark outline on top:
    # containment reads at a glance, and fill-vs-stroke survives grayscale.
    ax.fill_between(x_mp, 0, mp_pdf, facecolor=tint(COOL_MD, 0.82),
                    edgecolor="none", zorder=1)
    ax.plot(x_mp, mp_pdf, color=COOL_MD, lw=1.0, zorder=2,
            solid_capstyle="round")

    n_bins = 12
    edges = np.linspace(0.0, lam_plus, n_bins + 1)
    hist, _ = np.histogram(ev, bins=edges, density=True)
    ax.stairs(hist, edges, color=INK, lw=1.05, zorder=4)

    # Room for the whole histogram (never clip a bar) plus annotation
    # headroom.  The MP density has an integrable singularity at
    # lambda_- and is allowed to run off the top over a ~1% x-sliver.
    Y_TOP = float(hist.max()) * 1.06
    ax.set_ylim(0, Y_TOP)
    ax.set_xlim(-0.10, 4.78)

    # The edge (theory) and the largest observed eigenvalue (data).
    ax.axvline(lam_plus, color=INK, ls=(0, (3.2, 2.0)), lw=0.95, zorder=5)
    ax.axvline(lam_max, color=COOL, lw=1.7, zorder=5)

    # Both values as one right-aligned block that stops short of the two
    # rules it names, so no text ever crosses a reference line.  Colour
    # keys each value to its own line.
    ax.text(3.85, Y_TOP * 0.99, rf"$\lambda_+ = {lam_plus:.2f}$",
            fontsize=SMALL, color=INK, ha="right", va="top")
    ax.text(3.85, Y_TOP * 0.855, rf"$\lambda_{{\max}} = {lam_max:.2f}$",
            fontsize=SMALL, color=COOL, ha="right", va="top")

    # Explains the one place the data leaves the envelope: with gamma > 1
    # the law carries a point mass at zero.
    n_zero = int((ev < 1e-8).sum())
    ax.text(0.44, Y_TOP * 0.685, rf"{n_zero} zeros ($\gamma\!>\!1$)",
            fontsize=TINY, color=FAINT, ha="left", va="center")

    ax.text(1.62, Y_TOP * 0.44,
            ("Marchenko--Pastur\nprediction" if _usetex
             else "Marchenko–Pastur\nprediction"),
            fontsize=TINY, color=COOL_MD, ha="center", va="center",
            style="italic", linespacing=1.3)

    ax.set_xlabel(r"Eigenvalue $\lambda$", labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    panel_title(ax, "a", "Within-model spectrum")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    # ═════════════════════════════════════════════════════════════════
    # (b) unit chart: one chip per eigenvalue above that setting's edge
    # ═════════════════════════════════════════════════════════════════
    # A unit chart, not a bar chart: nine within-model settings put at most
    # one chip on their rail, the ensemble fills four.  Absence of signal is
    # the finding, so absence has to be visible -- an empty rail shows it,
    # a zero-length bar does not.
    ax = ax_b
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Layout columns, in axes-x units (0-1 spans the 2.00 in panel).
    X_BENCH, X_NAME = 0.014, 0.474
    X_SLOT0, X_PITCH = 0.547, 0.079
    X_NUM = X_SLOT0 + (MAX_SLOTS - 1) * X_PITCH + 0.090
    slots = [X_SLOT0 + k * X_PITCH for k in range(MAX_SLOTS)]

    GROUP_GAP = 0.42          # whitespace, not a rule, separates benchmarks
    ENS_GAP = 1.45
    ys = []
    y = 0.0
    for i, r in enumerate(within_rows):
        if i > 0 and r["bench"] != within_rows[i - 1]["bench"]:
            y += GROUP_GAP
        ys.append(y)
        y += 1.0
    y_ens = y - 1.0 + ENS_GAP
    ys = np.array(ys)

    ax.set_xlim(0, 1)
    ax.set_ylim(y_ens + 0.90, ys[0] - 0.90)

    def rail(y_row, count, size, rail_lw, rail_col):
        ax.plot([slots[0], slots[-1]], [y_row, y_row],
                color=rail_col, lw=rail_lw, zorder=2,
                solid_capstyle="round")
        for k, xs in enumerate(slots):
            if k < count:
                ax.plot(xs, y_row, "o", markersize=size,
                        markerfacecolor=WARM, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=4)
            else:
                ax.plot(xs, y_row, "o", markersize=1.7,
                        color=tint(FAINT, 0.42), zorder=3)

    for r, y_row in zip(within_rows, ys):
        rail(y_row, r["count"], 5.6, 0.7, tint(FAINT, 0.62))
        ax.text(X_NAME, y_row, r["name"], fontsize=SMALL, color=INK,
                ha="right", va="center")
        ax.text(X_NUM, y_row, str(r["count"]), fontsize=SMALL,
                color=MUTED if r["count"] == 0 else WARM_DK,
                ha="center", va="center")

    # benchmark tag: once per block, in its own column
    seen = set()
    for r, y_row in zip(within_rows, ys):
        if r["bench"] not in seen:
            seen.add(r["bench"])
            ax.text(X_BENCH, y_row, r["bench"], fontsize=TINY - 0.3,
                    color=FAINT, ha="left", va="center", style="italic")

    # ensemble row -- the figure's single accent, marked by a tinted band,
    # heavier chips and a larger numeral, so it carries focus by weight as
    # well as by colour and survives a grayscale print.
    ax.axhspan(y_ens - 0.62, y_ens + 0.62, color=tint(WARM, 0.925), zorder=0)
    ax.axhline(0.5 * (ys[-1] + y_ens), xmin=0.0, xmax=1.0,
               color=tint(HAIR, 0.05), lw=0.6, zorder=1)
    rail(y_ens, count_e, 8.6, 0.9, tint(WARM, 0.62))
    ax.text(X_NAME, y_ens, "24 models", fontsize=SMALL, color=INK,
            ha="right", va="center")
    ax.text(X_BENCH, y_ens, "ensemble", fontsize=TINY - 0.3, color=FAINT,
            ha="left", va="center", style="italic")
    ax.text(X_NUM, y_ens, bf(str(count_e)), fontsize=BODY + 4.0,
            color=WARM_DK, ha="center", va="center")

    # Axis label centred on the chip rails, not on the whole panel box.
    ax.text((X_SLOT0 + slots[-1]) / 2, -0.10,
            r"MP signal count ($\lambda > \lambda_+$)",
            transform=ax.get_xaxis_transform(), fontsize=BODY,
            color=INK, ha="center", va="top", clip_on=False)
    panel_title(ax, "b", "Signals above the edge")

    # ═════════════════════════════════════════════════════════════════
    # (c) split-half agreement of the per-question pass rates
    # ═════════════════════════════════════════════════════════════════
    ax = ax_c
    ax.grid(color=tint(HAIR, 0.35), lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, zorder=2, alpha=0.55)

    # The 379 items the model is already certain about pile into the two
    # corners; the 121 the MP test actually sees are the diagonal band.
    ax.scatter(first50[~borderline], last50[~borderline], s=9,
               color=tint(FAINT, 0.52), alpha=0.60, edgecolors="none",
               zorder=3, rasterized=True)
    ax.scatter(first50[borderline], last50[borderline], s=13,
               color=COOL, alpha=0.72, edgecolors="none",
               zorder=4, rasterized=True)

    ax.set_xlabel("Pass rate, runs 1--50" if _usetex
                  else "Pass rate, runs 1–50", labelpad=2)
    ax.set_ylabel("Pass rate, runs 51--100" if _usetex
                  else "Pass rate, runs 51–100", labelpad=2)
    panel_title(ax, "c", "Split-half consistency")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    ax.text(0.045, 0.965, rf"$r = {r_split:.3f}$", transform=ax.transAxes,
            fontsize=BODY + 1.0, color=INK, va="top", ha="left")
    ax.text(0.045, 0.845, rf"$n = {N}$ items", transform=ax.transAxes,
            fontsize=TINY, color=MUTED, va="top", ha="left")
    # Both notes live in the empty lower-right triangle, clear of the cloud.
    ax.text(0.975, 0.145,
            f"{int(borderline.sum())} borderline\nitems (panel a)",
            transform=ax.transAxes, fontsize=TINY, color=COOL,
            style="italic", ha="right", va="bottom", linespacing=1.3)
    ax.text(0.975, 0.045, r"$y = x$", transform=ax.transAxes,
            fontsize=TINY, color=MUTED, ha="right", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"hero_figure.{ext}", dpi=300)
    plt.close(fig)

# ── self-audit: every plotted number, printed ────────────────────────
print(f"usetex={_usetex}  figsize=({FIG_W}, {FIG_H}) in")
print(f"(a) Qwen2.5-7B MMLU  K={K} N={N} N_b={N_b} gamma={gamma:.4f}")
print(f"    lambda_-={lam_minus:.4f} lambda_+={lam_plus:.4f} "
      f"lambda_max={lam_max:.4f} margin={lam_plus - lam_max:+.4f} "
      f"count={S['count']}")
print(f"    histogram bins={n_bins} max density={hist.max():.3f} "
      f"(y-limit {Y_TOP})")
print("(b) MP signal counts")
for r in within_rows:
    print(f"    {r['bench']:<10s} {r['name']:<13s} K={r['K']:3d} "
          f"N_b={r['N_b']:3d} lambda_+={r['lam_plus']:6.2f} "
          f"lambda_max={r['lam_max']:6.2f} count={r['count']}")
print(f"    ensemble   {M_ens} models  N_split={Ce.shape[1]} "
      f"lambda_+={lam_plus_e:.2f} top4={np.round(ev_e[:4], 1)} "
      f"count={count_e}")
print(f"(c) split-half r={r_split:.4f} n={N} borderline={int(borderline.sum())}")
print(f"Saved {OUT / 'hero_figure.pdf'}")
