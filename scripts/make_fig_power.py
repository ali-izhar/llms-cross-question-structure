"""Figure: power of the within-model test (\\columnwidth, appendix), from item03_power.json.

Story: the MP count detects a weak factor spread over all borderline questions, but rarely a 9-question
cluster; more borderline questions (Llama-3-8B, N_b=173) raise power for the diffuse factor and lower it for
a fixed-size cluster.

One panel per planted structure (diffuse, cluster of 30, cluster of 9), each on its own *linear* rho axis
(the diffuse factor was simulated at rho <= 0.05, the clusters at rho in {0.1, 0.2}); rho = 0 is the shared
null. Color + marker = model (both are independent-draw runs, so both take the cool hue, dark vs light). Only the MP count
(lambda_1 > lambda_+) is drawn: it is the count whose zero the paper reports and the most powerful of the
three. The buffered and Horn counts are printed in the log for the caption/body (Horn is within 0.06 of MP
at every point; buffered has lower power and ~0.5% false positives). Wilson 95% whiskers.
Run: .venv/bin/python scripts/make_fig_power.py -> paper/figures/fig_power.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as S

W, H = S.COLUMNWIDTH, 1.75
MS_LINE = 3.6  # marker size on lines and dense scatters, shared by all four figures
# model key -> (display, color, marker, linestyle); both runs are independent draws -> cool hue, two shades,
# one marker shape (square/triangle mean shared seed/prompt in the other figures); linestyle separates the models
MODELS = [("MMLU Qwen2.5-7B T1.0", "Qwen2.5-7B", S.OSLO[5], "o", "--"),
          ("MMLU Meta-Llama-3-8B", "Llama-3-8B", S.COOL, "o", "-")]
PANELS = [("diffuse", "diffuse", [0, 0.01, 0.02, 0.05], (-0.004, 0.054)),
          ("cluster30", "cluster of 30", [0, 0.1, 0.2], (-0.016, 0.216)),
          ("cluster9", "cluster of 9", [0, 0.1, 0.2], (-0.016, 0.216))]
RULE = "power_mp_strict"
OTHER = ("power_mp_buffered_1p05", "power_horn_contiguous")


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
    log, gap = [], 0.0
    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        BOT, TOP, L, R, GAPX = 0.40, 1.36, 0.36, 0.05, 0.10
        PW = (W - L - R - 2 * GAPX) / 3
        axes = [fig.add_axes(((L + i * (PW + GAPX)) / W, BOT / H, PW / W, (TOP - BOT) / H)) for i in range(3)]
        for ax, (skey, title, xt, xl) in zip(axes, PANELS):
            S.ygrid(ax)
            ax.axhline(0.05, color=S.FAINT, lw=0.8, ls="--", zorder=1)
            for mkey, mname, col, mk, ls in MODELS:
                m = it3["models"][mkey]
                scen = m["scenarios"]
                pts = [s for s in scen if s["scenario"] == "null"] + \
                    sorted([s for s in scen if s["scenario"] == skey], key=lambda s: s["rho_latent"])
                xs = np.array([s["rho_latent"] for s in pts])
                ys = np.array([s[RULE] for s in pts])
                ci = np.array([wilson(s[RULE], n_rule(m, s, RULE)) for s in pts])
                ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], fmt="none",
                            ecolor=S.WHISK, elinewidth=0.6, capsize=1.5, zorder=2)
                ax.plot(xs, ys, ls=ls, lw=1.1, color=col, marker=mk, ms=MS_LINE, mfc=col, mec=col, zorder=3)
                for s, (lo, hi) in zip(pts, ci):
                    row = [mname, skey, s["rho_latent"], s[RULE], lo, hi]
                    row += [s[o] for o in OTHER]
                    gap = max(gap, abs(s["power_horn_contiguous"] - s[RULE]))
                    log.append(tuple(row))
            ax.set_xlim(*xl)
            ax.set_ylim(-0.03, 1.04)
            ax.xaxis.set_major_locator(FixedLocator(xt))
            ax.set_xticklabels(["" if t == 0.01 else f"{t:g}" for t in xt])  # .01 ticked, unlabeled (crowded)
            ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
            ax.set_title(title, fontsize=S.SMALL, color=S.INK, pad=2)
        axes[0].set_yticklabels(["0", "", "0.5", "", "1"])
        for ax in axes[1:]:
            ax.set_yticklabels([])
        axes[0].set_ylabel("Detection rate", labelpad=2)
        fig.text((L + (W - L - R) / 2) / W, 0.03 / H, r"Latent correlation $\rho$", ha="center", va="bottom",
                 fontsize=S.BODY)
        S.ref_label(axes[0], 0.05, "5%" if not S.USETEX else r"5\%", above=True, color=S.FAINT)

        handles = [Line2D([], [], color=c, marker=mk, ms=MS_LINE, lw=1.1, ls=ls,
                          label=rf"{n} ($N_b={it3['models'][k]['observed']['n_borderline']}$)")
                   for k, n, c, mk, ls in MODELS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (TOP + 0.17) / H), ncol=2,
                   frameon=False, handlelength=2.0, handletextpad=0.4, columnspacing=1.2,
                   borderaxespad=0.0, fontsize=S.SMALL)
        S.save(fig, "fig_power")

    print(f"figsize={W}x{H} usetex={S.USETEX}  plotted rule={RULE}")
    print("  model       structure  rho   MP     MP-CI            buffered  Horn")
    for r in log:
        print("  %-11s %-10s %.2f  %.3f  (%.3f,%.3f)   %.4f    %.3f" % r)
    print(f"  max |Horn - MP| over all plotted points = {gap:.3f}")
