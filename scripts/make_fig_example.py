"""Figure (\\columnwidth): the running example -- Llama-3.2-1B's accuracy follows the letter A, not the question.

Story: Llama-3.2-1B is right on about nine in ten questions whose answer is shown at A and on fewer than one in
four elsewhere, in every rotation; Gemma-2-9B is about equally accurate at every letter.
x = the letter at which the correct answer is SHOWN (rotation r shows option j at position (j + r) mod 4, so
    shown gold = (gold_idx + r) mod 4); y = accuracy (unparsed answers count as wrong, as in `correct`).
Small open dots (outside each pooled marker, rotations 0-3 left to right) = the four rotations; each rotation's
letter group is a different quarter of the 500 questions.
Filled marker = the pooled accuracy at that letter; pooled over the four rotations each letter holds every one
of the 500 questions exactly once, so the whisker is a Wilson 95% CI over 500 questions.
Data: results/mmlu_ensemble_rotations.npz (read live).
Run: .venv/bin/python scripts/make_fig_example.py -> paper/figures/fig_example.pdf + .png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

import figstyle as S

W, H = S.COLUMNWIDTH, 1.33
NPZ = S.ROOT / "results" / "mmlu_ensemble_rotations.npz"
LETTERS = "ABCD"

# model -> (display, color, linestyle (unused: categorical x, no connecting lines), marker, x offset in a letter group)
# local to this figure: figstyle.SERIES has no per-model entries (warm = the letter-biased model, cool = reference)
MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2-1B", S.WARM, "-", "s", -0.12),
    "google/gemma-2-9b-it":             ("Gemma-2-9B",   S.COOL, "-", "o", +0.12),
}
# direct labels: model -> (letter index anchor, y data, dx_pt, dy_pt, ha)
OFFSETS = {
    "meta-llama/Llama-3.2-1B-Instruct": (-0.12, 0.918, 7, 0, "left"),
    "google/gemma-2-9b-it":             (2.5, 0.90, 0, 0, "center"),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


if __name__ == "__main__":
    d = np.load(NPZ, allow_pickle=True)
    names = [str(m) for m in d["models"]]
    correct, gold, parsed = d["correct"], d["gold_idx"].astype(int), d["parsed"]
    n_q, n_rot = correct.shape[1], correct.shape[2]
    shown = (gold[:, None] + np.arange(n_rot)[None, :]) % 4          # (question, rotation)

    stats = {}
    for key in MODELS:
        m = names.index(key)
        per_rot = np.array([[correct[m, shown[:, r] == L, r].mean() for r in range(n_rot)] for L in range(4)])
        n_rot_L = np.array([[(shown[:, r] == L).sum() for r in range(n_rot)] for L in range(4)])
        k = np.array([sum(correct[m, shown[:, r] == L, r].sum() for r in range(n_rot)) for L in range(4)])
        n = n_rot_L.sum(axis=1)
        assert (n == n_q).all()                                        # each letter pool = all questions once
        pooled = k / n
        ci = np.array([wilson(int(k[L]), int(n[L])) for L in range(4)])
        share = np.array([(parsed[m] == c).mean() for c in LETTERS])
        unparsed = float((parsed[m] == "").mean())
        stats[key] = dict(per_rot=per_rot, n=n_rot_L, pooled=pooled, ci=ci, share=share, unparsed=unparsed,
                          acc=float(correct[m].mean()))

    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))
        L_, R_, BOT, TOP = 0.42, 0.06, 0.29, 0.06
        ax = fig.add_axes((L_ / W, BOT / H, (W - L_ - R_) / W, (H - BOT - TOP) / H))
        S.ygrid(ax)
        x = np.arange(4)
        for key, (lab, col, ls, mk, dx) in MODELS.items():
            st = stats[key]
            xs = x + dx
            # the four rotations: small open dots in a strip outside the pooled marker, one per question quarter
            side = np.sign(dx)
            rx = side * (0.11 + 0.045 * np.arange(n_rot)) if side > 0 else side * (0.11 + 0.045 * np.arange(n_rot)[::-1])
            for L in range(4):
                ax.plot(xs[L] + rx, st["per_rot"][L], ls="none", marker="o", ms=2.4, mfc="white", mec=col,
                        mew=0.6, zorder=2)
            ax.errorbar(xs, st["pooled"], yerr=[st["pooled"] - st["ci"][:, 0], st["ci"][:, 1] - st["pooled"]],
                        ls="none", ecolor=S.WHISK, elinewidth=0.9, capsize=2.0, zorder=4)
            ax.plot(xs, st["pooled"], ls="none", marker=mk, ms=4.2, mfc=col, mec="white", mew=0.5, zorder=5)
            xa, ya, ddx, ddy, ha = OFFSETS[key]
            ax.annotate(lab, xy=(xa, ya), xytext=(ddx, ddy), textcoords="offset points", ha=ha, va="center",
                        fontsize=S.SMALL, color=col)

        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.set_xticklabels(list(LETTERS))
        ax.tick_params(axis="x", length=0, pad=2)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_xlabel("Letter at which the correct answer is shown", labelpad=2)
        ax.set_ylabel("Accuracy", labelpad=2)
        S.save(fig, "fig_example")

    # audit log: every plotted number
    print(f"figsize={W}x{H} usetex={S.USETEX}  data={NPZ.name}  questions={n_q} rotations={n_rot}")
    for key, st in stats.items():
        lab = MODELS[key][0]
        print(f"{lab}: overall acc={st['acc']:.3f}  answer shares A-D={np.round(st['share'], 3).tolist()} "
              f"unparsed={st['unparsed']:.3f}")
        for L in range(4):
            rot = " ".join(f"r{r}={st['per_rot'][L, r]:.3f}(n={st['n'][L, r]})" for r in range(n_rot))
            print(f"  shown {LETTERS[L]}: pooled={st['pooled'][L]:.3f} "
                  f"CI=[{st['ci'][L, 0]:.3f},{st['ci'][L, 1]:.3f}]  {rot}")
