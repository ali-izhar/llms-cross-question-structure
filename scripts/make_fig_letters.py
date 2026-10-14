"""Figure 2 (\\columnwidth): eigenvector 2 of the 24-model rotation-0 matrix is answer-letter preference.

(a) each model's share of answers D vs its eigenvector-2 score (v3_B_extra); (b) per rotation, r of the
loadings with the shown-letter, original-letter and same-question profiles (v3_C1b_letter_following).
Run: python scripts/make_fig_letters.py -> paper/figures/fig_letters.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as S
import v3_across as V

W, H = S.COLUMNWIDTH, 1.95

# direct labels: model -> (display, dx_pt, dy_pt, ha)
OFFSETS = {
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2-1B", 5, -1, "left"),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ("SmolLM2-1.7B", -5, -1, "right"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B", -2, 7, "right"),
}
ROTS = ["1", "2", "3"]
MEAS = [("r_shown", "shown letter", S.INK, "o", -0.18),
        ("r_orig", "original letter", S.COOL_MD, "s", 0.0),
        ("r_direct", "same question", S.FAINT, "^", 0.18)]


if __name__ == "__main__":
    bx = S.load("v3_B_extra.json")
    c1b = S.load("v3_C1b_letter_following.json")["A"]["eigvec2"]
    ev2 = bx["letter_analysis"]["eigenvectors"][1]

    d = V.load_rep("A")
    names = d["names"]
    shareD = V.shares_of(d["shown_choice"][:, :, 0])[:, 3]
    score = np.array(ev2["model_scores"])
    r_json = ev2["model_score_corr"]["share_D"]["r"]
    r_re = float(np.corrcoef(score, shareD)[0, 1])
    assert abs(r_re - r_json) < 1e-3, (r_re, r_json)   # scores are stored to 4 decimals

    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        BOT, TOP = 0.40, 1.58
        axa = fig.add_axes([0.42 / W, BOT / H, 1.32 / W, (TOP - BOT) / H])
        axb = fig.add_axes([2.28 / W, BOT / H, 1.00 / W, (TOP - BOT) / H])

        # (a)
        ax = axa
        S.ygrid(ax)
        ax.axhline(0, color=S.WHISK, lw=0.6, zorder=1)
        hi = np.array([n in OFFSETS for n in names])
        ax.plot(shareD[~hi], score[~hi], ls="none", marker="D", ms=3.4, mfc="white", mec=S.INK, mew=0.8, zorder=3)
        ax.plot(shareD[hi], score[hi], ls="none", marker="D", ms=3.8, mfc=S.INK, mec=S.INK, mew=0.8, zorder=4)
        for n, (lab, dx, dy, ha) in OFFSETS.items():
            i = names.index(n)
            ax.annotate(lab, xy=(shareD[i], score[i]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=S.SMALL, color=S.INK)
        xspan = shareD.max() - 0
        ax.set_xlim(-0.02, shareD.max() + 0.06 * xspan)
        yspan = score.max() - score.min()
        ax.set_ylim(score.min() - 0.08 * yspan, score.max() + 0.10 * yspan)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
        ax.set_xticklabels(["0", ".2", ".4", ".6"])
        yt = [-0.4, 0, 0.4]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.set_yticklabels(["-0.4", "0", "0.4"])          # plain-string minus survives usetex/Times
        ax.text(0.97, 0.66, rf"$r={r_json:.2f}$".replace("-", "{-}") if S.USETEX else f"r = {r_json:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=S.SMALL, color=S.INK)
        ax.set_xlabel("Share of answers D", labelpad=2)
        ax.set_ylabel("Eigenvector-2 model score", labelpad=2)
        S.panel_tag(ax, "a")

        # (b)
        ax = axb
        S.ygrid(ax)
        ax.axhline(0, color=S.WHISK, lw=0.9, zorder=1)
        log = []
        for key, lab, col, mk, dx in MEAS:
            for k, rot in enumerate(ROTS):
                v = c1b[rot][key]
                ax.plot(k + dx, v, ls="none", marker=mk, ms=4.2, mfc=col, mec=col, mew=0.8, zorder=4)
                log.append((key, rot, v))
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(ROTS)
        ax.set_ylim(-0.5, 0.92)
        ax.yaxis.set_major_locator(FixedLocator([-0.4, 0, 0.4, 0.8]))
        ax.set_yticklabels(["-0.4", "0", "0.4", "0.8"])
        ax.set_xlabel("Rotation", labelpad=2)
        ax.set_ylabel(r"Correlation $r$", labelpad=2)
        S.panel_tag(ax, "b")

        handles = [Line2D([], [], ls="none", marker=mk, ms=4.2, mfc=col, mec=col, label=lab)
                   for _, lab, col, mk, _ in MEAS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (TOP + 0.08) / H), ncol=3,
                   frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=1.0, labelspacing=0.2,
                   borderaxespad=0.0, fontsize=S.SMALL)
        S.save(fig, "fig_letters")

    # audit log
    print(f"figsize={W}x{H} usetex={S.USETEX}")
    print(f"(a) r(score, share D): json={r_json:.4f} recomputed={r_re:.4f}")
    for n, s_, u in zip(names, shareD, score):
        print(f"    {n:42s} shareD={s_:.3f} score={u:+.4f}")
    m = np.array([n != "meta-llama/Llama-3.2-1B-Instruct" for n in names])
    acc = d["CR"][:, :, 0].mean(axis=1)
    print(f"    (not plotted) without Llama-3.2-1B: r(score, share D)={np.corrcoef(score[m], shareD[m])[0, 1]:.3f}, "
          f"r(score, accuracy)={np.corrcoef(score[m], acc[m])[0, 1]:.3f}; all 24: r(score, accuracy)="
          f"{np.corrcoef(score, acc)[0, 1]:.3f} (json {ev2['model_score_corr']['accuracy']['r']:.3f})")
    print("(b) C1b eigvec2:")
    for row in log:
        print("    %-9s rot %s r=%+.3f" % row)
    for rot in ROTS:
        print(f"    rot {rot}: model-score |r|={c1b[rot]['model_score_match_abs_r']:.3f} p={c1b[rot]['perm_p_two_sided']:.4f}")
