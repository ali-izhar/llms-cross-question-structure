"""
Cost-AUROC Pareto frontier for selective prediction on Qwen2.5-7B.

Visual story, reading top-to-bottom:
  - Red Ensemble curve sits ENTIRELY above the blue SC operational curve,
    even at 1/10th the cost.  Pareto dominance is immediate.
  - Two headline points are emphasised with outlined markers:
      Ensemble_2  (k=2,   AUROC=0.81, cost=2.46)  ← top-left  (cheap & good)
      SC_100      (K=100, AUROC=0.71, cost=100)   ← bottom-right (costly & worse)
  - A single summary box states the quantitative gap, so the reader never
    has to measure arrow endpoints that don't land on real data.

Style: tueplots ICML single-column + Paul Tol bright palette.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots.constants.color import palettes

# ── paths ────────────────────────────────────────────────────────────
PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── tueplots ICML single-column ──────────────────────────────────────
rc = bundles.icml2024(column="half", usetex=False, family="serif")
rc["figure.figsize"] = (3.3, 2.75)
rc["figure.constrained_layout.use"] = False
rc["axes.spines.top"] = False
rc["axes.spines.right"] = False
rc["axes.linewidth"] = 0.8
rc["xtick.major.width"] = 0.7
rc["ytick.major.width"] = 0.7
rc["xtick.major.size"] = 3.2
rc["ytick.major.size"] = 3.2
rc["lines.linewidth"] = 1.6
rc["mathtext.fontset"] = "cm"
rc["axes.labelsize"] = 9.5
rc["xtick.labelsize"] = 8.6
rc["ytick.labelsize"] = 8.6
plt.rcParams.update(rc)

_tol = palettes.paultol_bright
BLUE = f"#{_tol[0]}"
ENSEMBLE = "#CC3311"
GREY_LT = "#c5cdd4"
GREY_MID = "#5c6773"
ORACLE = "#cdd2d8"  # faint — "context only", not the comparison

# ── load results ─────────────────────────────────────────────────────
with open(RESULTS / "selective_prediction.json") as f:
    R = json.load(f)

sc_K = sorted(R["SC_K"], key=lambda r: r["K_or_k"])
ens_K = sorted(R["Ensemble_familydiverse"], key=lambda r: r["K_or_k"])

sc_costs = np.array([r["cost_units_qwen7b_equivalent"] for r in sc_K])
sc_auroc_op = np.array([r["AUROC_operational"] for r in sc_K])
sc_auroc_or = np.array([r["AUROC_oracle_upper_bound"] for r in sc_K])
sc_K_vals = np.array([r["K_or_k"] for r in sc_K])

ens_costs = np.array([r["cost_units_qwen7b_equivalent"] for r in ens_K])
ens_auroc = np.array([r["AUROC"] for r in ens_K])
ens_K_vals = np.array([r["K_or_k"] for r in ens_K])

ens2_idx = int(np.where(ens_K_vals == 2)[0][0])
ens2_cost = ens_costs[ens2_idx]
ens2_auroc = ens_auroc[ens2_idx]
sc100_cost = sc_costs[-1]
sc100_auroc = sc_auroc_op[-1]

# Headline values, rounded the way the paper reports them.
HEAD_ENS2_AUROC = round(ens2_auroc, 2)     # 0.81
HEAD_SC100_AUROC = round(sc100_auroc, 2)   # 0.71
HEAD_DAUROC = HEAD_ENS2_AUROC - HEAD_SC100_AUROC  # 0.10
HEAD_COST_RATIO = int(sc100_cost / ens2_cost)     # 40

# ── figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 2.75))
fig.subplots_adjust(left=0.165, right=0.97, bottom=0.175, top=0.965)

# ── horizontal reference line at SC_100's AUROC: the "SC ceiling" ───
# Lets the reader see at a glance that EVERY ensemble point sits above it.
ax.axhline(sc100_auroc, color=GREY_LT, lw=0.9, ls=(0, (2, 2)), zorder=1)
# Label BELOW the dashed line on the left margin.
ax.text(0.015, sc100_auroc - 0.009, r"SC$_{100}$ ceiling",
        transform=matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData),
        fontsize=7.0, color=GREY_MID, ha="left", va="top",
        style="italic")

# ── SC oracle (faint dotted grey — context, not deployable) ─────────
ax.plot(sc_costs, sc_auroc_or, ":", color=ORACLE, lw=1.0,
        marker="s", ms=2.6, alpha=0.75,
        markeredgecolor="white", markeredgewidth=0.25,
        label="SC (oracle, not deployable)", zorder=2)

# ── Ensemble curve: red, diamond markers — the hero ─────────────────
ax.plot(ens_costs, ens_auroc, "-", color=ENSEMBLE, lw=2.0, zorder=5)
ax.plot(ens_costs, ens_auroc, "D", color=ENSEMBLE, ms=5.4,
        markeredgecolor="white", markeredgewidth=0.6, zorder=6,
        label="Ensemble (cross-model)")

# ── SC operational: blue, circle markers ────────────────────────────
ax.plot(sc_costs, sc_auroc_op, "-", color=BLUE, lw=1.6, zorder=4)
ax.plot(sc_costs, sc_auroc_op, "o", color=BLUE, ms=4.8,
        markeredgecolor="white", markeredgewidth=0.5, zorder=5,
        label="Self-consistency (operational)")

# ── Headline-point emphasis: large outlined markers on Ens_2 & SC_100 ─
ax.plot(ens2_cost, ens2_auroc, "D", color=ENSEMBLE, ms=10.5,
        markeredgecolor="white", markeredgewidth=1.5, zorder=7)
ax.plot(sc100_cost, sc100_auroc, "o", color=BLUE, ms=9.5,
        markeredgecolor="white", markeredgewidth=1.5, zorder=7)

# No arrow between SC_100 and Ensemble_2 — the visual story (red curve
# sits entirely above blue curve, crossing the SC_100 ceiling) already
# makes the Pareto-dominance claim.  Any connecting arrow would cross
# the Ensemble curve itself and add clutter.

# ── Summary callout in the empty post-Ensemble mid-right region ─────
# Ensemble_k data ends at k=10 (cost 11.4); everything right of x=12 and
# above y=0.72 (SC ceiling) is free of data points and curves until the
# oracle line at y~0.93.  Positioning at x~55 puts the box well clear of
# both the Ensemble curve's rightmost marker and SC_100.
ax.text(
    55, 0.81,
    r"$\mathbf{Ensemble_{2}\ \succ\ SC_{100}}$" + "\n"
    + rf"${HEAD_COST_RATIO}\times$ cheaper · "
    + rf"$+{HEAD_DAUROC:.2f}$ AUROC",
    fontsize=8.2, color=ENSEMBLE, ha="center", va="center",
    linespacing=1.4,
    bbox=dict(facecolor="white", edgecolor=ENSEMBLE, linewidth=0.7,
              boxstyle="round,pad=0.38", alpha=0.96),
    zorder=8,
)

# ── Point labels: placed in genuinely empty space, no crossing leaders ─
# Ensemble_2: label ABOVE the marker, short straight leader up.  This avoids
# crossing the Ensemble k=1 marker (which is at x=1.14, upper-left area).
ax.annotate(
    rf"$k{{=}}2$: $\mathbf{{{HEAD_ENS2_AUROC:.2f}}}$" + "\n"
    + rf"cost ${ens2_cost:.1f}$",
    xy=(ens2_cost, ens2_auroc),
    xytext=(ens2_cost, ens2_auroc + 0.055),
    fontsize=7.6, color=ENSEMBLE, ha="center", va="bottom",
    linespacing=1.2,
    arrowprops=dict(arrowstyle="-", color=ENSEMBLE, lw=0.6, alpha=0.6,
                    shrinkA=1, shrinkB=6),
    zorder=8,
)
# SC_100: label to the RIGHT and SLIGHTLY BELOW the marker so it sits
# clear of the "SC_100 ceiling" dashed line (which passes through the
# marker at y=sc100_auroc).  Equals sign uses {=} (no negative thinspace)
# so it reads cleanly at small font size.
ax.annotate(
    rf"$K{{=}}100$: $\mathbf{{{HEAD_SC100_AUROC:.2f}}}$" + "\n"
    + rf"cost ${int(sc100_cost)}$",
    xy=(sc100_cost, sc100_auroc),
    xytext=(sc100_cost * 1.05, sc100_auroc - 0.060),
    fontsize=7.6, color=BLUE, ha="left", va="top",
    linespacing=1.2,
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6, alpha=0.6,
                    shrinkA=1, shrinkB=6),
    zorder=8,
)

# ── axes ─────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlabel(r"Inference cost (Qwen-7B-equivalent forward passes)")
ax.set_ylabel("AUROC")
ax.set_xlim(0.85, 200)
ax.set_ylim(0.47, 1.00)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# ── legend: compact, bottom-right (empty quadrant) ──────────────────
leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02),
                fontsize=7.4, frameon=True,
                fancybox=False, edgecolor="#cccccc",
                facecolor="white", framealpha=0.95,
                handlelength=1.6, borderpad=0.34, labelspacing=0.28,
                handletextpad=0.45)
leg.get_frame().set_linewidth(0.55)

# ── save ─────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"selective_prediction.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
print(f"Saved {OUT / 'selective_prediction.pdf'}")
plt.close(fig)
