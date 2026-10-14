"""Figure: power of the within-model test (\\columnwidth), from item03_power.json.

Detection rate vs planted latent correlation rho (rho = 0 is the null) at the Qwen2.5-7B and Llama-3-8B
pass rates; color = planted structure, line style = rule (MP edge, 1.05 x edge, Horn); Wilson 95% whiskers.
Run: python scripts/make_fig_power.py -> paper/figures/fig_power.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as S

W, H = S.COLUMNWIDTH, 2.25
MODELS = [("MMLU Qwen2.5-7B T1.0", "Qwen2.5-7B"), ("MMLU Meta-Llama-3-8B", "Llama-3-8B")]
RHO_POS = {0.0: 0, 0.01: 1, 0.02: 2, 0.05: 3, 0.1: 4, 0.2: 5}
SCEN = [("diffuse", "diffuse", S.COOL, "o"),
        ("cluster30", "cluster of 30", S.WARM, "s"),
        ("cluster9", "cluster of 9", S.MUTED, "^")]
RULES = [("power_mp_strict", r"$\lambda_1>\lambda_+$", "-", 1.0, True),
         ("power_mp_buffered_1p05", r"$\lambda_1>1.05\lambda_+$", "--", 0.9, False),
         ("power_horn_contiguous", "Horn", ":", 0.9, False)]
DX = {"power_mp_strict": -0.11, "power_mp_buffered_1p05": 0.0, "power_horn_contiguous": 0.11}


def wilson(p: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def n_rule(m: dict, s: dict, rkey: str) -> int:
    """Draws behind rule `rkey` in scenario `s`: Horn is computed on the first meta.n_sim_per_scenario draws only."""
    if rkey != "power_horn_contiguous":
        return s["n_sim"]
    if s["scenario"] == "null":
        return m["calibration_from_null_scenario"]["n_null_draws_with_horn"]
    return min(s["n_sim"], m["meta"]["n_sim_per_scenario"])



if __name__ == "__main__":
    it3 = S.load("item03_power.json")
    log = []
    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        BOT, TOP, L, GAPX = 0.45, 1.70, 0.42, 0.12
        PW = (W - L - 0.06 - GAPX) / 2
        axes = [fig.add_axes([(L + i * (PW + GAPX)) / W, BOT / H, PW / W, (TOP - BOT) / H]) for i in range(2)]
        for ax, (mkey, mname) in zip(axes, MODELS):
            m = it3["models"][mkey]
            scen = m["scenarios"]
            null = [s for s in scen if s["scenario"] == "null"][0]
            nb = m["observed"]["n_borderline"]
            S.ygrid(ax)
            ax.axhline(0.05, color=S.FAINT, lw=0.7, ls="-", zorder=1)
            for skey, slab, col, mk in SCEN:
                pts = [null] + sorted([s for s in scen if s["scenario"] == skey], key=lambda s: s["rho_latent"])
                for rkey, rlab, ls, lw, filled in RULES:
                    xs = np.array([RHO_POS[round(s["rho_latent"], 3)] for s in pts]) + DX[rkey]
                    ys = np.array([s[rkey] for s in pts])
                    ci = np.array([wilson(s[rkey], n_rule(m, s, rkey)) for s in pts])
                    ax.plot(xs, ys, ls=ls, lw=lw, color=col, zorder=3)
                    ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], fmt="none",
                                ecolor=S.WHISK, elinewidth=0.6, capsize=1.2, zorder=2)
                    ax.plot(xs[1:], ys[1:], ls="none", marker=mk, ms=3.4, mfc=col if filled else "white",
                            mec=col, mew=0.8, zorder=4)
                    for s, x_, y_, (lo, hi) in zip(pts, xs, ys, ci):
                        log.append((mname, skey, s["rho_latent"], rkey, n_rule(m, s, rkey), y_, lo, hi))
            # rho = 0 (shared null) drawn once per rule in ink
            for rkey, rlab, ls, lw, filled in RULES:
                ax.plot(RHO_POS[0.0] + DX[rkey], null[rkey], ls="none", marker="o", ms=3.0,
                        mfc=S.INK if filled else "white", mec=S.INK, mew=0.8, zorder=5)
            ax.set_xlim(-0.45, 6.15)
            ax.set_ylim(-0.03, 1.03)
            ax.xaxis.set_major_locator(FixedLocator(list(RHO_POS.values())))
            ax.set_xticklabels(["0", ".01", ".02", ".05", ".1", ".2"])
            ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
            ax.text(0.04, 0.80, rf"{mname}" + "\n" + rf"$N_b={nb}$", transform=ax.transAxes, ha="left", va="top",
                    fontsize=S.SMALL, color=S.INK)
            ax.set_xlabel(r"Latent correlation $\rho$", labelpad=2)
        axes[0].set_ylabel("Detection rate", labelpad=2)
        axes[1].set_yticklabels([])
        S.ref_label(axes[1], 0.05, "5%" if not S.USETEX else r"5\%", above=True, color=S.FAINT)
        S.panel_tag(axes[0], "a")
        S.panel_tag(axes[1], "b")

        handles = [Line2D([], [], color=c, marker=mk, ms=3.4, lw=1.0, label=l) for _, l, c, mk in SCEN]
        rules_h = [Line2D([], [], color=S.INK, ls=ls, lw=lw, label=l) for _, l, ls, lw, _ in RULES]
        handles = [h for pair in zip(handles, rules_h) for h in pair]   # column-major: row 1 structure, row 2 rule
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (TOP + 0.10) / H), ncol=3,
                   frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=0.9,
                   labelspacing=0.2, borderaxespad=0.0, fontsize=S.SMALL)
        S.save(fig, "fig_power")

    print(f"figsize={W}x{H} usetex={S.USETEX}")
    for row in log:
        print("  %-11s %-10s rho=%.2f %-24s n=%4d rate=%.3f CI=(%.3f,%.3f)" % row)
