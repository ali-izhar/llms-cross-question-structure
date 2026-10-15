"""Figure (\\columnwidth): eigenvector 2 of the 24-model rotation-0 matrix scores models by their share of D answers.

Each model's share of answers D vs its eigenvector-2 model score (v3_B_extra). The rotation test that used to be
panel (b) is now panel (b) of fig_main (scripts/make_fig_main.py).
Story: the eigenvector-2 model scores track the share of D answers.
Run: python scripts/make_fig_letters.py -> paper/figures/fig_letters.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

import figstyle as S
import v3_across as V

W, H = S.COLUMNWIDTH, 1.70
MS, MS_LINE = 4.6, 3.6  # single-point / dense-scatter marker sizes, shared by all figures

# direct labels: model -> (display, dx_pt, dy_pt, ha)
OFFSETS = {
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2-1B", 5, -1, "left"),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ("SmolLM2-1.7B", -5, -1, "right"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B", 4, 8, "right"),
}


def signed(t: float) -> str:
    """Plain-string tick label with an en-dash minus (math minus can vanish under the Times usetex stack)."""
    s_ = f"{abs(t):g}"
    return (S.dash() if S.USETEX else "\u2212") + s_ if t < 0 else s_


if __name__ == "__main__":
    bx = S.load("v3_B_extra.json")
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
        BOT, TOP = 0.40, H - 0.14
        axa = fig.add_axes((0.47 / W, BOT / H, (W - 0.47 - 0.08) / W, (TOP - BOT) / H))

        # (a)
        ax = axa
        S.ygrid(ax)
        ax.axhline(0, color=S.WHISK, lw=0.8, ls="--", zorder=1)
        # one encoding for all 24 models (fill means "beyond the null" in fig_main, so no fill code here)
        ax.plot(shareD, score, ls="none", marker="D", ms=MS_LINE, mfc=S.INK, mec="white", mew=0.4, zorder=3)
        for n, (lab, dx, dy, ha) in OFFSETS.items():
            i = names.index(n)
            ax.annotate(lab, xy=(float(shareD[i]), float(score[i])), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=S.SMALL, color=S.INK)
        xspan = shareD.max() - 0
        ax.set_xlim(-0.02, shareD.max() + 0.06 * xspan)
        yspan = score.max() - score.min()
        ax.set_ylim(score.min() - 0.08 * yspan, score.max() + 0.10 * yspan)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
        yt = [-0.4, 0, 0.4]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.set_yticklabels([signed(t) for t in yt])
        ax.set_xlabel("Share of D answers", labelpad=2)
        ax.set_ylabel("Eigenvector-2 model score", labelpad=2)

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
