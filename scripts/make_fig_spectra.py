"""Figure 1 (\\textwidth): within-model and across-model spectra against their nulls.

(a) Qwen2.5-7B MMLU eigenvalues / own MP edge, iid histogram vs MP law; (b) lambda_max / lambda_+ per
run with its MC null band; (c) 24 models, rotation 0, vs letter-aware and 3PL null p99; (d) the
all-4-correct matrix vs 2PL and Rasch null p99. Run: python scripts/make_fig_spectra.py -> paper/figures/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

import figstyle as S
from null_models import borderline_mask, corr_eigs, mp_edge

EPS = 0.05
W, H = S.TEXTWIDTH, 2.30
NR = 4  # ranks drawn in (c, d): the nulls are summarized at ranks 1-4


QWEN_IID = "mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct"
QWEN_S1 = "mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct"
QWEN_S2 = "mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct"
LLAMA_S2 = "mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct"


def spectrum(run_id: str, stored: dict) -> tuple[np.ndarray, float]:
    C = np.load(S.MAT / f"{run_id}.npy").astype(np.float64)
    keep = borderline_mask(C, EPS)
    e = corr_eigs(C[:, keep])
    lp = mp_edge(int(keep.sum()), C.shape[0])
    assert int(keep.sum()) == stored["N_b"], run_id
    assert np.allclose(e[:5], stored["top_eigs"], atol=1e-8), run_id
    assert abs(lp - stored["lambda_plus"]) < 1e-10, run_id
    return e, lp

def mp_pdf(x):
    return np.sqrt(np.clip((lp_iid - x) * (x - lm_iid), 0, None)) / (2 * np.pi * gam * x)


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


def ratio_row(r: dict) -> dict:
    lp, s = r["lambda_plus"], r["sigma_tw"]
    q = r["mc_null_z"]
    return dict(ratio=r["lambda_max"] / lp, lo=1 + q["q05"] * s / lp, hi=1 + q["q95"] * s / lp,
                z=r["tw_z"], count=r["mp_count_strict"], N_b=r["N_b"])


if __name__ == "__main__":
    it1 = S.load("item01_within_table.json")
    it2 = S.load("item02_shared_randomness.json")
    it7 = S.load("item07_across_model.json")
    raw7 = S.load("raw_item07_across_two_margin_null.json")
    bx = S.load("v3_B_extra.json")
    rotA = S.load("v3_C_rotations_A.json")
    RUNS1 = {r["run_id"]: r for r in it1["runs"]}
    RUNS2 = it2["runs"]

    # panel (a) data
    SPEC = {
        "iid": spectrum(QWEN_IID, RUNS1[QWEN_IID]),
        "sharedseed": spectrum(QWEN_S1, RUNS2[QWEN_S1]),
        "sharedprompt": spectrum(QWEN_S2, RUNS2[QWEN_S2]),
    }
    e_iid, lp_iid = SPEC["iid"]
    K = RUNS1[QWEN_IID]["K"]
    Nb = RUNS1[QWEN_IID]["N_b"]
    gam = Nb / K
    lm_iid = (1 - np.sqrt(gam)) ** 2

    BINS = np.geomspace(0.035, 1.35, 11)
    hist_obs, _ = np.histogram(e_iid / lp_iid, bins=BINS)
    mp_counts = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        xs = np.linspace(max(lo * lp_iid, lm_iid), min(hi * lp_iid, lp_iid), 4001)
        mp_counts.append(Nb * np.trapezoid(mp_pdf(xs), xs) if xs[-1] > xs[0] else 0.0)
    mp_counts = np.array(mp_counts)

    assert len(IID_ORDER) == 12 and set(r for _, r in IID_ORDER) == set(RUNS1)

    rows_b = []
    x = 0.0
    GAP = 3.4
    prev = None
    for grp, rid in IID_ORDER:
        if prev is not None and grp != prev:
            x += GAP
        rows_b.append(dict(x=x, kind="iid", grp=grp, rid=rid, **ratio_row(RUNS1[rid])))
        prev, x = grp, x + 1
    x += GAP + 0.4
    for kind, rid in SHARED_ORDER:
        rows_b.append(dict(x=x, kind=kind, grp="shared", rid=rid, **ratio_row(RUNS2[rid])))
        x += 1

    # panels (c, d) data
    obs_r0 = np.array(bx["suite"]["observed"]["top_eigs"][:NR])
    assert np.allclose(obs_r0, it7["observed"]["top_eigs_1to4"])
    assert np.allclose(obs_r0, raw7["observed"]["top_eigs"][:NR])
    a4 = rotA["C3_rotation_invariant"]["all4_correct"]
    EIG_R0 = np.load(S.AN / "raw_v3_null_eigs_A_rot0_extra.npz")
    EIG_ROT = np.load(S.AN / "raw_v3_null_eigs_A_rotations.npz")
    assert np.allclose(a4["suite"]["observed"]["top_eigs"], a4["spectrum"]["top_eigs"])
    # Each panel: the two drawn nulls as (SERIES key, JSON null summary, stored null eigenvalue draws).
    ACROSS = {
        "c": dict(tag="rotation 0", obs=obs_r0, lp=it7["observed"]["lambda_plus"], nulls=[
            ("null_letter", bx["suite"]["nulls"]["letter_post"], EIG_R0["A_rot0__letter_post"]),
            ("null_3pl", bx["suite"]["nulls"]["3pl_post"], EIG_R0["A_rot0__3pl_post"])]),
        "d": dict(tag="all-4-correct", obs=np.array(a4["spectrum"]["top_eigs"][:NR]),
                  lp=a4["spectrum"]["lambda_plus"], nulls=[
            ("null_2pl", a4["suite"]["nulls"]["2pl_post"], EIG_ROT["A_all4_correct__2pl_post"]),
            ("null_rasch", a4["suite"]["nulls"]["rasch_jml"], EIG_ROT["A_all4_correct__rasch_jml"])]),
    }
    assert abs(ACROSS["c"]["lp"] - bx["suite"]["observed"]["lambda_plus"]) < 1e-9
    for P in ACROSS.values():
        P["rows"] = []
        for skey, summ, draws in P["nulls"]:
            p99 = np.array(summ["rank_eigs_p99"][:NR])
            # the stored draws are the ones the JSON summarizes
            assert np.allclose(np.percentile(draws[:, :NR], 99, axis=0), p99, atol=1e-9), skey
            # null-validity rule: observed lambda_1 between the null's 1st and 99th percentile
            pct1 = 100.0 * float(np.mean(draws[:, 0] < P["obs"][0]))
            assert 1.0 <= pct1 <= 99.0, (P["tag"], skey, pct1)
            above = [j + 1 for j in range(NR) if P["obs"][j] > p99[j]]
            assert above == summ["observed_ranks_above_p99"], (skey, above, summ["observed_ranks_above_p99"])
            P["rows"].append(dict(key=skey, p99=p99, pct1=pct1, above=above, n=draws.shape[0]))
        P["filled"] = [j for j in range(1, NR + 1) if all(j in r["above"] for r in P["rows"])]

    # figure
    with plt.rc_context(S.RC):
        fig = plt.figure(figsize=(W, H))

        def rect(x0, y0, w, h):
            return [x0 / W, y0 / H, w / W, h / H]

        BOT, TOP = 0.50, 1.82
        HIST_H = 0.66
        ax_at = fig.add_axes(rect(0.40, BOT + HIST_H + 0.04, 1.46, TOP - BOT - HIST_H - 0.04))
        ax_a = fig.add_axes(rect(0.40, BOT, 1.46, HIST_H))
        ax_b = fig.add_axes(rect(2.34, BOT, 2.10, TOP - BOT))
        ax_c = fig.add_axes(rect(4.90, BOT, 0.84, TOP - BOT))
        ax_d = fig.add_axes(rect(5.84, BOT, 0.84, TOP - BOT), sharey=ax_c)

        # (a)
        XL = (0.035, 4.9)
        XT = [0.1, 0.3, 1, 3]
        for ax in (ax_a, ax_at):
            ax.set_xscale("log")
            ax.set_xlim(*XL)
            ax.xaxis.set_major_locator(FixedLocator(XT))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axvline(1.0, color=S.INK, ls=(0, (3.2, 2.0)), lw=0.9, zorder=5)
        ax_a.set_xticklabels([f"{t:g}" for t in XT])
        ax_a.stairs(mp_counts, BINS, fill=True, facecolor=S.tint(S.COOL_MD, 0.80), edgecolor="none", zorder=1)
        ax_a.stairs(mp_counts, BINS, color=S.COOL_MD, lw=0.9, zorder=2)
        ax_a.stairs(hist_obs, BINS, color=S.INK, lw=1.05, zorder=4)
        ymax_a = max(hist_obs.max(), mp_counts.max()) * 1.12
        ax_a.set_ylim(0, ymax_a)
        ax_a.set_xlabel(r"Eigenvalue / own MP edge, $\lambda/\lambda_+$", labelpad=2)
        ax_a.set_ylabel("Eigenvalues per bin", labelpad=2)
        S.ygrid(ax_a)
        ax_a.text(1.25, ymax_a * 0.96, "observed\n" + rf"$N_b={Nb}$", fontsize=S.SMALL,
                  color=S.INK, ha="left", va="top")
        ax_a.text(1.25, ymax_a * 0.42, "MP law", fontsize=S.SMALL, color=S.COOL_MD,
                  ha="left", va="top", style="italic")

        ROWS_A = [("iid", 2.0), ("sharedseed", 1.0), ("sharedprompt", 0.0)]
        ax_at.set_ylim(-0.35, 2.75)
        ax_at.set_yticks([])
        for side in ("left", "bottom"):
            ax_at.spines[side].set_visible(False)
        ax_at.tick_params(axis="x", which="both", length=0, labelbottom=False)
        for key, yrow in ROWS_A:
            e, lp = SPEC[key]
            lab, col, _, mk = S.SERIES[key]
            r = e[e > 1e-9] / lp
            above = r > 1.0
            ax_at.plot([XL[0], XL[1]], [yrow, yrow], color=S.HAIR, lw=0.5, zorder=1)
            ax_at.plot(r[~above], np.full((~above).sum(), yrow), ls="none", marker="|",
                       ms=4.0, mew=0.6, color=S.tint(col, 0.35), zorder=2)
            if above.any():
                ax_at.plot(r[above], np.full(above.sum(), yrow), ls="none", marker=mk, ms=4.6,
                           mfc=col, mec="white", mew=0.5, zorder=4)
            n_above = int(above.sum())
            ax_at.text(XL[0] * 1.05, yrow + 0.16, lab, fontsize=S.SMALL, color=col,
                       ha="left", va="bottom")
            ax_at.text(XL[1], yrow + 0.16, f"{n_above} above", fontsize=S.SMALL,
                       color=col if n_above else S.MUTED, ha="right", va="bottom")
        ax_at.text(1.0, 1.0, r"$\lambda_+$", transform=ax_at.get_xaxis_transform(),
                   fontsize=S.SMALL, color=S.REFLAB, ha="center", va="bottom")
        S.panel_tag(ax_at, "a", x=-0.02, y=1.0)

        # (b)
        ax = ax_b
        ax.set_yscale("log")
        YB = (0.70, 4.6)
        ax.set_ylim(*YB)
        yt = [0.75, 1, 1.5, 2, 3, 4]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in yt])
        S.ygrid(ax)
        xs = np.array([r["x"] for r in rows_b])
        ax.set_xlim(xs.min() - 0.8, xs.max() + 7.4)   # right-end room for the reference labels
        ax.axhline(1.0, color=S.INK, ls=(0, (3.2, 2.0)), lw=0.9, zorder=2)
        ax.axhline(1.05, color=S.FAINT, ls=":", lw=0.9, zorder=2)
        for r in rows_b:
            lab, col, _, mk = S.SERIES[r["kind"]]
            ax.add_patch(plt.Rectangle((r["x"] - 0.32, r["lo"]), 0.64, r["hi"] - r["lo"],
                                       facecolor=S.tint(S.FAINT, 0.62), edgecolor="none", zorder=1))
            filled = r["count"] > 0
            ax.plot(r["x"], r["ratio"], ls="none", marker=mk, ms=4.4 if r["kind"] == "iid" else 5.0,
                    mfc=col if filled else "white", mec=col, mew=0.9, zorder=4)
        MODEL = {QWEN_S1: "Qwen", QWEN_S2: "Qwen", LLAMA_S2: "Llama"}
        for r in rows_b:
            if r["kind"] in ("sharedseed", "sharedprompt"):
                ax.annotate(rf"{MODEL[r['rid']]}, $z={r['z']:+.1f}$", xy=(r["x"], r["ratio"]), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=S.SMALL,
                            color=S.SERIES[r["kind"]][1])
        ax.set_xticks([])
        GROUPS = [("MMLU", "MMLU"), ("HS", "Hella-\nSwag"), ("GSM", "GSM\n8K"),
                  ("TEMP", r"$T{=}0.5$" + "\n" + r"$T{=}1.5$"), ("shared", "shared")]
        # TEMP and shared are MMLU runs of Qwen2.5-7B (and Llama-3-8B for one shared prompt run); caption says so
        for g, text in GROUPS:
            gx = [r["x"] for r in rows_b if r["grp"] == g]
            ax.text(np.mean(gx), -0.035, text, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=S.SMALL, color=S.INK, linespacing=1.1)
        ax.set_ylabel(r"$\lambda_{\max}/\lambda_+$", labelpad=2)
        S.ref_label(ax, 1.0, r"$\lambda_+$", above=False)
        S.ref_label(ax, 1.05, r"$1.05\lambda_+$", above=True, color=S.FAINT)
        S.panel_tag(ax, "b")

        # (c, d)
        YC = (17, 125)
        ranks = np.arange(1, NR + 1)
        for key, ax in (("c", ax_c), ("d", ax_d)):
            P = ACROSS[key]
            ax.set_yscale("log")
            ax.set_ylim(*YC)
            ax.yaxis.set_major_locator(FixedLocator([20, 30, 50, 100]))
            ax.yaxis.set_minor_locator(NullLocator())
            S.ygrid(ax)
            ax.set_xlim(0.45, NR + 1.45)   # right-end room for the lambda_+ label
            ax.set_xticks(ranks)
            ax.axhline(P["lp"], color=S.INK, ls=(0, (3.2, 2.0)), lw=0.9, zorder=2)
            for r in P["rows"]:
                _, col, ls, _ = S.SERIES[r["key"]]
                ax.plot(ranks, r["p99"], ls=ls, lw=0.9, color=col, marker="_", ms=9, mew=1.6, zorder=3)
            _, col, _, mk = S.SERIES["ensemble"]
            ax.plot(ranks, P["obs"], ls="-", lw=0.8, color=col, zorder=4)
            for j, v in zip(ranks, P["obs"]):
                f = j in P["filled"]
                ax.plot(j, v, ls="none", marker=mk, ms=4.6, mfc=col if f else "white", mec=col, mew=0.9, zorder=5)
            ax.text(0.97, 0.74, P["tag"], transform=ax.transAxes, ha="right", va="top",
                    fontsize=S.SMALL, color=S.MUTED)
            ax.set_xlabel("Rank $j$", labelpad=2)
            S.ref_label(ax, P["lp"], r"$\lambda_+$", above=False)
            S.panel_tag(ax, key, x=-0.04 if key == "d" else -0.02)
        ax_c.set_yticklabels(["20", "30", "50", "100"])
        ax_c.set_ylabel(r"Eigenvalue $\lambda_j$ (24 models)", labelpad=2)
        ax_d.tick_params(axis="y", labelleft=False)

        # shared legend
        handles = [
            Line2D([], [], ls="none", marker="o", ms=4.4, mfc="white", mec=S.COOL, mew=0.9,
                   label="independent draws"),
            Line2D([], [], ls="none", marker="s", ms=4.6, mfc=S.WARM, mec=S.WARM, label="shared seed"),
            Line2D([], [], ls="none", marker="^", ms=4.8, mfc=S.WARM_DK, mec=S.WARM_DK, label="shared prompt"),
            Patch(facecolor=S.tint(S.FAINT, 0.62), edgecolor="none",
                  label=(f"MC null 5{S.dash()}95\\%" if S.USETEX else "MC null 5–95%")),
            Line2D([], [], ls="-", lw=0.8, marker="D", ms=4.2, color=S.INK, mfc=S.INK, label="24 models"),
            Line2D([], [], ls="none", label=" "),   # spacer: column 4 = the (c) nulls, column 5 = the (d) nulls
        ] + [Line2D([], [], ls=S.SERIES[k][2], color=S.SERIES[k][1], marker="_", ms=8, mew=1.5,
                    label=S.SERIES[k][0]) for k in ("null_letter", "null_3pl", "null_2pl", "null_rasch")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (TOP + 0.10) / H), ncol=5,
                   frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=1.1, labelspacing=0.25,
                   borderaxespad=0.0, fontsize=S.SMALL)

        S.save(fig, "fig_spectra")

    # audit log
    print(f"figsize={W}x{H} usetex={S.USETEX}")
    print(f"(a) Qwen iid N_b={Nb} K={K} lambda+={lp_iid:.4f} lambda_max={e_iid[0]:.4f}")
    print("    observed counts:", hist_obs.tolist(), " MP expected:", np.round(mp_counts, 2).tolist())
    for k in SPEC:
        e, lp = SPEC[k]
        print(f"    {k}: lambda+={lp:.4f} top5/lambda+={np.round(e[:5] / lp, 3).tolist()} n_above={(e > lp).sum()}")
    print("(b) x, run, ratio, band(lo,hi), z, MP count")
    for r in rows_b:
        print(f"    {r['x']:5.2f} {r['rid'][:58]:58s} {r['ratio']:.3f} ({r['lo']:.3f},{r['hi']:.3f}) "
              f"z={r['z']:+.2f} count={r['count']}")
    for key, P in ACROSS.items():
        print(f"({key}) {P['tag']!r}: obs={np.round(P['obs'], 2).tolist()} lambda+={P['lp']:.2f}")
        for r in P["rows"]:
            print(f"     {r['key']:11s} p99={np.round(r['p99'], 2).tolist()} above={r['above']} "
                  f"lambda_1 pct={r['pct1']:.1f} (n={r['n']})")
        print(f"     filled={P['filled']}  obs-p99 rank4: "
              f"{[round(float(P['obs'][3] - r['p99'][3]), 3) for r in P['rows']]}")
