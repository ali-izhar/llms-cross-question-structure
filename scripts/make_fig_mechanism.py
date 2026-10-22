"""Appendix figure: what the shared component couples (item02_shared_randomness.json).

Story: a shared seed (Qwen) and a shared prompt (Llama) couple the questions of a pass through the
answer letter -- letter shares vary far beyond their shuffle null and the top eigenvector's loadings
group by gold letter -- while the independent runs and Qwen with a shared prompt stay at the null.

(a) across-pass variance of the letter shares / its shuffle-null mean (log x), tick = shuffle-null p99, dashed line = shuffle mean;
(b) eta^2 of the top eigenvector's question loadings by gold letter, tick = permutation-null p95.
Significance is read as "marker beyond its tick" (permutation p < .01 in (a), < .05 in (b) coincide
with that for every row; asserted below). The exemplar-letter correlations (old panel c) are stated
in the appendix text.
Data: results/analysis/item02_shared_randomness.json (read live).
Run: .venv/bin/python scripts/make_fig_mechanism.py -> paper/figures/fig_mechanism.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

import figstyle as S

W, H = S.COLUMNWIDTH, 1.75
MS = 4.6  # single-point marker size, shared by all four figures
S1 = "mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct"
Q2 = "mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct"
L2 = "mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"
P_A, P_B = 0.01, 0.05          # thresholds matching the plotted null percentiles (p99, p95)


if __name__ == "__main__":
    it2 = S.load("item02_shared_randomness.json")["runs"]
    assert it2[S1]["iid_control"] == it2[Q2]["iid_control"]

    # rows top -> bottom: (label, series key, letter-share dict, eta dict)
    ROWS = [
        ("Qwen, independent", "iid", it2[S1]["iid_control"]["letter_share_dispersion"],
         it2[S1]["iid_control"]["top_pc"]["loadings_eta_sq_by_gold_letter"]),
        ("Llama, independent", "iid", it2[L2]["iid_control"]["letter_share_dispersion"],
         it2[L2]["iid_control"]["top_pc"]["loadings_eta_sq_by_gold_letter"]),
        ("Qwen, shared seed", "sharedseed", it2[S1]["letter_share_dispersion"],
         it2[S1]["top_pc"]["loadings_eta_sq_by_gold_letter"]),
        ("Qwen, shared prompt", "sharedprompt", it2[Q2]["letter_share_dispersion"],
         it2[Q2]["top_pc"]["loadings_eta_sq_by_gold_letter"]),
        ("Llama, shared prompt", "sharedprompt", it2[L2]["letter_share_dispersion"],
         it2[L2]["top_pc"]["loadings_eta_sq_by_gold_letter"]),
    ]
    Y = np.arange(len(ROWS))[::-1].astype(float)

    log = []
    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        BOT, TOP = 0.50, H - 0.17
        L0, PW, GAP = 1.13, 0.93, 0.17
        axa = fig.add_axes((L0 / W, BOT / H, PW / W, (TOP - BOT) / H))
        axb = fig.add_axes(((L0 + PW + GAP) / W, BOT / H, PW / W, (TOP - BOT) / H), sharey=axa)

        XA, XB = (0.6, 20.0), (-0.03, 0.8)

        def nullmark(ax, x0, x, y):
            # tick = null p99 (a) / p95 (b), drawn below the marker; open markers have a white face, so a
            # near-tie shows the marker whole with the tick visible above and below it
            ax.plot([x, x], [y - 0.30, y + 0.30], color=S.MUTED, lw=1.2, solid_capstyle="butt", zorder=3)

        for ax in (axa, axb):
            ax.xaxis.grid(True, color=S.GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)

        for (lab, key, lsd, eta), y in zip(ROWS, Y):
            _, col, _, mk = S.SERIES[key]
            ratio, p99 = lsd["ratio_obs_over_null_mean"], lsd["null_p99"] / lsd["null_mean"]
            # the "beyond the tick" reading must agree with the permutation test
            assert (ratio > p99) == (lsd["perm_p"] < P_A), lab
            assert (eta["eta_sq"] > eta["null_p95"]) == (eta["perm_p"] < P_B), lab
            nullmark(axa, XA[0], p99, y)
            # filled = beyond the null tick (significant), open = not, as in fig_main
            fa, fb = ratio > p99, eta["eta_sq"] > eta["null_p95"]
            axa.plot(ratio, y, ls="none", marker=mk, ms=MS * (1.15 if mk == "^" else 1.0), mfc=col if fa else "white", mec=col, mew=0.9, zorder=4)
            nullmark(axb, 0.0, eta["null_p95"], y)
            axb.plot(eta["eta_sq"], y, ls="none", marker=mk, ms=MS * (1.15 if mk == "^" else 1.0), mfc=col if fb else "white", mec=col, mew=0.9, zorder=4)
            log.append((lab, round(ratio, 3), round(p99, 3), lsd["perm_p"],
                        round(eta["eta_sq"], 3), round(eta["null_p95"], 3), eta["perm_p"]))

        axa.axvline(1.0, color=S.FAINT, lw=0.8, ls="--", zorder=1)   # shuffle mean
        axa.set_xscale("log")
        axa.set_xlim(*XA)
        axa.xaxis.set_major_locator(FixedLocator([1, 3, 10]))
        axa.xaxis.set_minor_locator(NullLocator())
        axa.set_xticklabels(["1", "3", "10"])
        axa.set_yticks(Y)
        axa.set_yticklabels([r[0] for r in ROWS])
        for t, (_, key, _, _) in zip(axa.get_yticklabels(), ROWS):
            t.set_color(S.SERIES[key][1])
        axa.set_ylim(-0.55, len(ROWS) - 0.45)
        axa.set_xlabel("Letter-share variance\n" + r"($\times$ shuffle mean)", labelpad=2, linespacing=1.0)
        S.panel_tag(axa, "a")

        axb.set_xlim(*XB)
        axb.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75]))
        axb.set_xticklabels(["0", "0.25", "0.5", "0.75"])
        axb.tick_params(axis="y", labelleft=False)
        axb.set_xlabel(r"$\eta^2$ by gold letter" + "\n(top eigenvector)", labelpad=2, linespacing=1.0)
        S.panel_tag(axb, "b")

        S.save(fig, "fig_mechanism")

    print(f"figsize={W}x{H} usetex={S.USETEX}")
    print("   (row, ratio, null p99/mean, perm_p, eta2, null p95, perm_p)")
    for row in log:
        print("  ", row)
