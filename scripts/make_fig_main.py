"""Main figure (\\textwidth, 1.80 in): one panel per result step; every detected direction has a non-knowledge source.

Story: within one model only a shared seed or prompt lifts lambda_max above the MP edge (a); across models the
eigenvector-2 loading of a rotated question follows the letter it now shows, not the question (b); rank 2 of the
rotation-0 matrix exceeds the one-ability nulls, and under all-4-correct no rank does (c).
(a) lambda_max / lambda_+ per within-model run with its MC null 5-95% bar (item01, item02);
(b) for rotations 1-3, r of each question's rotated eigenvector-2 loading with the rotation-0 mean loading of the
    letter it now shows, and with its own rotation-0 loading (v3_C1b_letter_following);
(c) eigenvalues 2-4 of the 24-model matrix divided by the rank-matched p99 of two valid one-ability nulls each
    (rotation 0: 3PL, uniform know-or-guess = v3_B_extra suite; all-4-correct: 2PL, Rasch = v3_C_rotations_A).
Data are read live from results/analysis; every plotted value is printed below.
Run: .venv/bin/python scripts/make_fig_main.py -> paper/figures/fig_main.{pdf,png}
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

import figstyle as S

W, H = S.TEXTWIDTH, 1.80
MS = 4.6  # single-point marker size, shared by all figures
NR = 4
EDGE_LS = (0, (3.2, 2.0))

QWEN_S1 = "mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct"
QWEN_S2 = "mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct"
LLAMA_S2 = "mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"
IID_ORDER = [
    ("MMLU", "mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct"),
    ("MMLU", "mmlu_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3"),
    ("MMLU", "mmlu_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"),
    ("MMLU", "mmlu_iid-T1.0_HuggingFaceTB_SmolLM2-1.7B-Instruct"),
    ("MMLU", "mmlu_iid-T1.0_microsoft_Phi-3-mini-4k-instruct"),
    ("HS", "hellaswag_iid-T1.0_Qwen_Qwen2.5-7B-Instruct"),
    ("HS", "hellaswag_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3"),
    ("HS", "hellaswag_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"),
    ("GSM", "gsm8k_iid-T1.0_Qwen_Qwen2.5-7B-Instruct"),
    ("GSM", "gsm8k_iidlong-T1.0_Qwen_Qwen2.5-7B-Instruct"),
    ("TEMP", "mmlu_iid-T0.5_Qwen_Qwen2.5-7B-Instruct"),
    ("TEMP", "mmlu_iid-T1.5_Qwen_Qwen2.5-7B-Instruct"),
]
SHARED_ORDER = [("sharedseed", QWEN_S1), ("sharedprompt", QWEN_S2), ("sharedprompt", LLAMA_S2)]
# (a) direct labels for the shared-randomness runs: run -> (text, dx_pt, dy_pt, ha)
OFFSETS_A = {QWEN_S1: ("Qwen, shared seed", -5, 0, "right"),
             QWEN_S2: ("Qwen, shared prompt", -5, 0, "right"),
             LLAMA_S2: ("Llama, shared prompt", -5, 0, "right")}

# (b) series: key -> (direct label, colour, marker, label x (rotation), label y (data))
ROTS = ["1", "2", "3"]
# filled circles in two inks: no diamond (the ensemble/null-test mark) and no fill code here
SER_B = [("r_shown", "shown letter", S.INK, "o", 2.0, 0.695),
         ("r_direct", "same question", S.FAINT, "o", 2.0, 0.235)]

# (c) the two valid nulls per group, left / right of the rank; short names used as direct labels
NULL_NAME = {"3pl_post": "3PL", "kg_uniform4": "KG-u", "2pl_post": "2PL", "rasch_jml": "Rasch"}
DXC = 0.15  # horizontal offset of the two markers at one rank
# null label beside its rank-2 marker: key -> (dx_pt, dy_pt, ha, va); tuned on the rendered PNG
OFFSETS_C = {"3pl_post": (-2, -5, "center", "top"), "kg_uniform4": (5, 0, "left", "center"),
             "2pl_post": (-5, 0, "right", "center"), "rasch_jml": (1, -6, "left", "top")}


def ratio_row(r: dict) -> dict:
    lp, s = r["lambda_plus"], r["sigma_tw"]
    q = r["mc_null_z"]
    return dict(ratio=r["lambda_max"] / lp, lo=1 + q["q05"] * s / lp, hi=1 + q["q95"] * s / lp,
                count=r["mp_count_strict"])


def signed(t: float) -> str:
    """Plain-string tick label with a text minus (math minus can vanish under the Times usetex stack)."""
    s_ = f"{abs(t):g}"
    return (S.dash() if S.USETEX else "−") + s_ if t < 0 else s_


if __name__ == "__main__":
    it1 = S.load("item01_within_table.json")
    it2 = S.load("item02_shared_randomness.json")
    it7 = S.load("item07_across_model.json")
    bx = S.load("v3_B_extra.json")
    rotA = S.load("v3_C_rotations_A.json")
    c1b = S.load("v3_C1b_letter_following.json")["A"]["eigvec2"]
    RUNS1 = {r["run_id"]: r for r in it1["runs"]}
    RUNS2 = it2["runs"]
    assert len(IID_ORDER) == 12 and set(r for _, r in IID_ORDER) == set(RUNS1)

    # (a) data
    rows_a: list[dict[str, Any]] = []
    x, GAP, prev = 0.0, 4.2, None
    for grp, rid in IID_ORDER:
        if prev is not None and grp != prev:
            x += GAP
        rows_a.append(dict(x=x, kind="iid", grp=grp, rid=rid, **ratio_row(RUNS1[rid])))
        prev, x = grp, x + 1
    x += GAP + 0.4
    for kind, rid in SHARED_ORDER:
        rows_a.append(dict(x=x, kind=kind, grp="shared", rid=rid, **ratio_row(RUNS2[rid])))
        x += 1

    # (c) data: observed eigenvalues and the stored null draws they are compared with
    obs_r0 = np.array(bx["suite"]["observed"]["top_eigs"][:NR])
    assert np.allclose(obs_r0, it7["observed"]["top_eigs_1to4"])
    a4 = rotA["C3_rotation_invariant"]["all4_correct"]
    assert np.allclose(a4["suite"]["observed"]["top_eigs"], a4["spectrum"]["top_eigs"])
    EIG_R0 = np.load(S.AN / "raw_v3_null_eigs_A_rot0_extra.npz")
    EIG_ROT = np.load(S.AN / "raw_v3_null_eigs_A_rotations.npz")
    GROUPS_C: list[dict[str, Any]] = [
        dict(tag="rotation 0", obs=obs_r0, nulls=[
            ("3pl_post", bx["suite"]["nulls"]["3pl_post"], EIG_R0["A_rot0__3pl_post"]),
            ("kg_uniform4", bx["suite"]["nulls"]["kg_uniform4"], EIG_R0["A_rot0__kg_uniform4"])]),
        dict(tag="all-4-correct", obs=np.array(a4["spectrum"]["top_eigs"][:NR]), nulls=[
            ("2pl_post", a4["suite"]["nulls"]["2pl_post"], EIG_ROT["A_all4_correct__2pl_post"]),
            ("rasch_jml", a4["suite"]["nulls"]["rasch_jml"], EIG_ROT["A_all4_correct__rasch_jml"])]),
    ]
    for G in GROUPS_C:
        G["rows"] = []
        for key, summ, draws in G["nulls"]:
            p99 = np.array(summ["rank_eigs_p99"][:NR])
            assert np.allclose(np.percentile(draws[:, :NR], 99, axis=0), p99, atol=1e-9), key
            pct1 = 100.0 * float(np.mean(draws[:, 0] < G["obs"][0]))
            assert 1.0 <= pct1 <= 99.0, (G["tag"], key, pct1)   # null-validity rule on lambda_1
            above = [j + 1 for j in range(NR) if G["obs"][j] > p99[j]]
            assert above == summ["observed_ranks_above_p99"], (key, above)
            G["rows"].append(dict(key=key, p99=p99, ratio=G["obs"] / p99, pct1=pct1, above=above,
                                  n=draws.shape[0]))

    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))

        def rect(x0, y0, w, h):
            return (x0 / W, y0 / H, w / W, h / H)

        BOT, TOP = 0.40, H - 0.22
        ax_a = fig.add_axes(rect(0.40, BOT, 2.20, TOP - BOT))
        ax_b = fig.add_axes(rect(3.13, BOT, 1.20, TOP - BOT))
        ax_c = fig.add_axes(rect(4.88, BOT, 1.84, TOP - BOT))

        # (a) lambda_max / lambda_+ per within-model run, grey bar = its MC null 5-95%
        ax = ax_a
        ax.set_yscale("log")
        ax.set_ylim(0.71, 4.15)
        yt = [0.75, 1, 1.5, 2, 3, 4]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in yt])
        S.ygrid(ax)
        xs = np.array([r["x"] for r in rows_a])
        ax.set_xlim(xs.min() - 0.9, xs.max() + 3.2)   # right-end room for the lambda_+ label
        ax.axhline(1.0, color=S.INK, ls=EDGE_LS, lw=0.9, zorder=2)
        for r in rows_a:
            _, col, _, mk = S.SERIES[r["kind"]]
            ax.add_patch(Rectangle((r["x"] - 0.40, r["lo"]), 0.80, r["hi"] - r["lo"],
                                   facecolor=S.tint(S.FAINT, 0.45), edgecolor="none", zorder=1))
            filled = r["count"] > 0
            ax.plot(r["x"], r["ratio"], ls="none", marker=mk, ms=(MS if filled else 3.6) * (1.15 if mk == "^" else 1.0),
                    mfc=col if filled else "none", mec=col, mew=0.8, zorder=4)   # open marker smaller than its bar
            if r["rid"] in OFFSETS_A:
                lab, dx, dy, ha = OFFSETS_A[r["rid"]]
                ax.annotate(lab, xy=(r["x"], r["ratio"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va="center", fontsize=S.SMALL, color=col)
        ax.set_xticks([])
        GROUPS_A = [("MMLU", "MMLU"), ("HS", "Hella-\nSwag"), ("GSM", "GSM\n8K"),
                    ("TEMP", r"$T{=}0.5$," + "\n" + r"$1.5$"), ("shared", "shared")]
        for g, text in GROUPS_A:
            gx = [r["x"] for r in rows_a if r["grp"] == g]
            ax.text(float(np.mean(gx)), -0.035, text, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=S.SMALL, color=S.INK, linespacing=1.05)
        x_sep = (max(r["x"] for r in rows_a if r["grp"] != "shared")
                 + min(r["x"] for r in rows_a if r["grp"] == "shared")) / 2
        ax.axvline(x_sep, color=S.HAIR, lw=0.7, zorder=0)
        ax.set_ylabel(r"$\lambda_{\max}/\lambda_+$", labelpad=2)
        S.ref_label(ax, 1.0, r"$\lambda_+$", above=True)
        S.panel_tag(ax, "a")

        # (b) rotation test: r with the shown letter's mean loading vs with the question's own loading
        ax = ax_b
        S.ygrid(ax)
        ax.axhline(0, color=S.WHISK, lw=0.8, ls="--", zorder=1)
        log_b = []
        for key, lab, col, mk, lx, ly in SER_B:
            v = [c1b[rot][key] for rot in ROTS]
            ax.plot([int(t) for t in ROTS], v, ls="none", marker=mk, ms=MS, mfc=col, mec="white",
                    mew=0.4, zorder=4)
            ax.text(lx, ly, lab, ha="center", va="bottom", fontsize=S.SMALL, color=col)
            log_b += [(key, rot, val) for rot, val in zip(ROTS, v)]
        ax.set_xlim(0.5, 3.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(ROTS)
        ax.set_ylim(-0.16, 0.82)
        ybt = [0, 0.2, 0.4, 0.6, 0.8]
        ax.yaxis.set_major_locator(FixedLocator(ybt))
        ax.set_yticklabels([signed(t) for t in ybt])
        ax.set_xlabel("Rotation", labelpad=2)
        ax.set_ylabel(r"Correlation $r$", labelpad=2)
        S.panel_tag(ax, "b")

        # (c) lambda_j / null p99 for ranks 2-4; 1 = threshold; filled = above that null's p99
        ax = ax_c
        S.ygrid(ax)
        RK = np.arange(2, NR + 1)
        GX0 = [0.0, 4.25]   # x of rank 2 in each group
        _, col, _, mk = S.SERIES["ensemble"]
        ax.axhline(1.0, color=S.INK, ls=EDGE_LS, lw=0.9, zorder=2)
        ticks, ticklabs = [], []
        for G, x0 in zip(GROUPS_C, GX0):
            for side, r in zip((-1, 1), G["rows"]):
                for j in RK:
                    xj = x0 + (j - 2) + side * DXC
                    f = r["ratio"][j - 1] > 1.0
                    ax.plot(xj, r["ratio"][j - 1], ls="none", marker=mk, ms=MS, mfc=col if f else "white",
                            mec=col, mew=0.9, zorder=5)
                # direct label once per null, beside its rank-2 marker
                dx, dy, ha, va = OFFSETS_C[r["key"]]
                ax.annotate(NULL_NAME[r["key"]], xy=(x0 + side * DXC, r["ratio"][1]),
                            xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=S.SMALL, color=S.MUTED)
            ticks += [x0 + (j - 2) for j in RK]
            ticklabs += [str(j) for j in RK]
            ax.text(x0 + 1.0, 1.0, G["tag"], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=S.SMALL, color=S.MUTED)
        ax.axvline((GX0[0] + 2 + GX0[1]) / 2, color=S.HAIR, lw=0.7, zorder=0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabs)
        ax.set_xlim(GX0[0] - 0.7, GX0[1] + 2 + 1.25)
        ax.set_ylim(0.82, 1.24)
        yct = [0.9, 1.0, 1.1, 1.2]
        ax.yaxis.set_major_locator(FixedLocator(yct))
        ax.set_yticklabels([f"{t:g}" for t in yct])
        ax.set_xlabel("Rank $j$", labelpad=2)
        ax.set_ylabel(r"$\lambda_j$ / null p99", labelpad=2)
        S.ref_label(ax, 1.0, "p99", above=True)
        S.panel_tag(ax, "c")

        S.save(fig, "fig_main")

    # audit log
    print(f"figsize={W}x{H} usetex={S.USETEX}")
    print("(a) run, lambda_max/lambda_+, null 5-95% (lo,hi), MP count (filled if >0)")
    for r in rows_a:
        print(f"    {r['rid'][:60]:60s} {r['ratio']:.3f} ({r['lo']:.3f},{r['hi']:.3f}) count={r['count']}")
    print("(b) C1b eigvec2 r:")
    for row in log_b:
        print("    %-9s rot %s r=%+.3f" % row)
    for G in GROUPS_C:
        print(f"(c) {G['tag']!r}: obs lambda_1..4={np.round(G['obs'], 2).tolist()}")
        for r in G["rows"]:
            print(f"     {r['key']:11s} p99={np.round(r['p99'], 2).tolist()} ratio={np.round(r['ratio'], 3).tolist()} "
                  f"above={r['above']} lambda_1 pct={r['pct1']:.1f} (n={r['n']})")
