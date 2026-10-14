"""Appendix figure: what the shared component couples (item02_shared_randomness.json).

(a) letter-share variance / its shuffle-null mean; (b) eta^2 of the top eigenvector's loadings by gold
letter; (c) per-letter r between a pass's exemplar gold-letter count and its answer share (S2 runs).
Run: python scripts/make_fig_mechanism.py -> paper/figures/fig_mechanism.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import figstyle as S

W, H = S.TEXTWIDTH, 1.95
S1 = "mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct"
Q2 = "mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct"
L2 = "mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"
P_A, P_B, P_C = 0.01, 0.05, 0.05
LETTERS = "ABCD"


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
    ECHO = {"Qwen": it2[Q2]["exemplar_alignment"]["letter_echo_exemplar_count_vs_answer_share"],
            "Llama": it2[L2]["exemplar_alignment"]["letter_echo_exemplar_count_vs_answer_share"]}

    log = []
    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        BOT, TOP = 0.42, H - 0.30
        axa = fig.add_axes([1.30 / W, BOT / H, 1.75 / W, (TOP - BOT) / H])
        axb = fig.add_axes([3.25 / W, BOT / H, 1.45 / W, (TOP - BOT) / H], sharey=axa)
        axc = fig.add_axes([5.15 / W, BOT / H, 1.53 / W, (TOP - BOT) / H])

        def nullmark(ax, x, y):
            ax.plot([x, x], [y - 0.28, y + 0.28], color=S.WHISK, lw=1.1, solid_capstyle="butt", zorder=2)

        for ax in (axa, axb):
            ax.xaxis.grid(True, color=S.GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        # (a)
        for (lab, key, lsd, eta), y in zip(ROWS, Y):
            _, col, _, mk = S.SERIES[key]
            ratio = lsd["ratio_obs_over_null_mean"]
            p99 = lsd["null_p99"] / lsd["null_mean"]
            nullmark(axa, p99, y)
            filled = lsd["perm_p"] < P_A
            axa.plot(ratio, y, ls="none", marker=mk, ms=5.0, mfc=col if filled else "white", mec=col,
                     mew=0.9, zorder=4)
            axa.annotate(f"{ratio:.1f}" + r"$\times$", xy=(max(ratio, p99), y), xytext=(6, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=S.SMALL, color=col)
            log.append(("a", lab, ratio, p99, lsd["perm_p"]))
        axa.set_xscale("log")
        axa.set_xlim(0.6, 40)
        axa.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
        axa.xaxis.set_minor_locator(NullLocator())
        axa.set_xticklabels(["1", "3", "10", "30"])
        axa.axvline(1.0, color=S.FAINT, lw=0.7, zorder=1)
        axa.set_yticks(Y)
        axa.set_yticklabels([r[0] for r in ROWS])
        for t, (lab, key, _, _) in zip(axa.get_yticklabels(), ROWS):
            t.set_color(S.SERIES[key][1])
        axa.set_ylim(-0.6, len(ROWS) - 0.4)
        axa.set_xlabel("Letter-share variance / null mean", labelpad=2)
        S.panel_tag(axa, "a", x=-0.02)

        # (b)
        for (lab, key, lsd, eta), y in zip(ROWS, Y):
            _, col, _, mk = S.SERIES[key]
            nullmark(axb, eta["null_p95"], y)
            filled = eta["perm_p"] < P_B
            axb.plot(eta["eta_sq"], y, ls="none", marker=mk, ms=5.0, mfc=col if filled else "white", mec=col,
                     mew=0.9, zorder=4)
            axb.annotate(f"{eta['eta_sq']:.2f}", xy=(max(eta["eta_sq"], eta["null_p95"]), y), xytext=(6, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=S.SMALL, color=col)
            log.append(("b", lab, eta["eta_sq"], eta["null_p95"], eta["perm_p"]))
        axb.set_xlim(-0.02, 1.0)
        axb.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
        axb.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        axb.tick_params(axis="y", labelleft=False)
        axb.set_xlabel(r"$\eta^2$ of top loadings by gold letter", labelpad=2)
        S.panel_tag(axb, "b", x=-0.02)

        # (c)
        S.ygrid(axc)
        axc.axhline(0, color=S.WHISK, lw=0.9, zorder=1)
        xs = np.arange(4)
        for name, dx, mk in (("Qwen", -0.12, "^"), ("Llama", 0.12, "v")):
            col = S.WARM if name == "Qwen" else S.WARM_DK
            col = S.SERIES["sharedprompt"][1] if name == "Llama" else S.tint(S.SERIES["sharedprompt"][1], 0.35)
            for j, L in enumerate(LETTERS):
                r, p = ECHO[name][L]["r"], ECHO[name][L]["perm_p_two_sided"]
                axc.plot(j + dx, r, ls="none", marker=mk, ms=5.0, mfc=col if p < P_C else "white", mec=col,
                         mew=0.9, zorder=4)
                log.append(("c", name, L, r, p))
        axc.set_xticks(xs)
        axc.set_xticklabels(LETTERS)
        axc.set_xlim(-0.55, 3.55)
        axc.set_ylim(-0.2, 0.62)
        axc.yaxis.set_major_locator(FixedLocator([-0.2, 0, 0.2, 0.4, 0.6]))
        axc.set_yticklabels(["-0.2", "0", "0.2", "0.4", "0.6"])   # plain-string minus survives usetex/Times
        axc.set_xlabel("Gold letter $L$ of the exemplars", labelpad=2)
        axc.set_ylabel(r"Per-letter $r$", labelpad=2)
        axc.text(3.45, 0.555, "Llama, shared prompt", ha="right", va="top", fontsize=S.SMALL,
                 color=S.SERIES["sharedprompt"][1])
        axc.text(3.45, 0.16, "Qwen, shared prompt", ha="right", va="bottom", fontsize=S.SMALL,
                 color=S.tint(S.SERIES["sharedprompt"][1], 0.35))
        S.panel_tag(axc, "c", x=-0.22)

        h = [Line2D([], [], color=S.WHISK, lw=1.1, ls="none", marker="|", ms=7, mew=1.1,
                    label="null p99 (a), p95 (b)"),
             Line2D([], [], ls="none", marker="o", ms=5, mfc=S.MUTED, mec=S.MUTED,
                    label=r"filled: permutation $p<.01$ (a), $<.05$ (b, c)"),
             Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=S.MUTED, mew=0.9, label="open: not significant")]
        fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, (TOP + 0.06) / H), ncol=3,
                   frameon=False, borderaxespad=0, handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.6, fontsize=S.SMALL)
        S.save(fig, "fig_mechanism")

    print(f"figsize={W}x{H} usetex={S.USETEX}")
    for row in log:
        print("  ", row)
    for name in ("Qwen", "Llama"):
        run = Q2 if name == "Qwen" else L2
        ps = it2[run]["exemplar_alignment"]["pass_pair_similarity_shared_exemplars_borderline"]
        print(f"   pass-pair similarity (shares exemplar vs not) {name}: z={ps['z_vs_null']:.2f}")
