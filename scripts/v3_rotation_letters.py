"""
C1b: does each across-model direction move with the DISPLAYED letters when the options rotate?

Within one rotation the shown gold letter is a one-to-one relabelling of the original gold
letter, so eta^2 by shown letter equals eta^2 by original letter: a single rotation cannot say
which one a direction follows. Across rotations it can. For eigenvector j of rotation 0
(question loadings v_0j, mean loading per gold letter = profile_j, 4 values) and rotation r:

  * match: the eigenvector k of rotation r (k = 1..4) whose model scores have the largest
    |corr| with those of v_0j (sign aligned); `model_score_match_abs_r`;
  * pred_shown_i = profile_j[(g_i + r) mod 4]  (the direction follows the displayed letter),
    pred_orig_i  = profile_j[g_i]               (it follows the original option / content),
    and r_shown = corr(v_rk, pred_shown), r_orig = corr(v_rk, pred_orig), r_direct =
    corr(v_rk, v_0j), over the questions split in both rotations.
  * permutation null for D = r_shown - r_orig: gold letters permuted over the questions
    (5000 draws, the same permutation in both rotations); two-sided p.

A letter direction gives r_shown > 0 and r_orig ~ 0 or below (D > 0); a content direction gives
r_orig ~ r_direct > r_shown. Output: results/analysis/v3_C1b_letter_following.json
"""

from __future__ import annotations

import v2lib  # noqa: F401
import numpy as np
from v2lib import write_json

from null_models import perm_pvalue_upper
from v3_across import N_RANKS, eig_vectors, load_rep

N_PERM = 5000
SEED = 20261014


def profile(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.array([v[g == L].mean() if np.any(g == L) else 0.0 for L in range(4)])


def follow_stats(v0: np.ndarray, vr: np.ndarray, g: np.ndarray, r: int) -> tuple[float, float, float]:
    p = profile(v0, g)
    pred_shown = p[(g + r) % 4]
    pred_orig = p[g]
    return (float(np.corrcoef(vr, pred_shown)[0, 1]), float(np.corrcoef(vr, pred_orig)[0, 1]),
            float(np.corrcoef(vr, v0)[0, 1]))


def analyse(rep: str) -> dict:
    d = load_rep(rep)
    CR, gold = d["CR"], d["gold"]
    rng = np.random.default_rng(SEED)
    dec = {}
    for r in range(4):
        lam, U, V, keep = eig_vectors(CR[:, :, r])
        full = np.full((CR.shape[1], N_RANKS), np.nan)
        full[keep] = V
        dec[r] = (lam, U, full, keep)
    out = {}
    for j in range(N_RANKS):
        rec = {}
        for r in range(1, 4):
            _, U0, V0, k0 = dec[0]
            _, Ur, Vr, kr = dec[r]
            cors = [np.corrcoef(U0[:, j], Ur[:, k])[0, 1] for k in range(N_RANKS)]
            k = int(np.argmax(np.abs(cors)))
            sg = np.sign(cors[k])
            both = k0 & kr
            v0, vr, g = V0[both, j], sg * Vr[both, k], gold[both]
            rs, ro, rd = follow_stats(v0, vr, g, r)
            null = []
            for _ in range(N_PERM):
                gp = rng.permutation(g)
                a, b, _ = follow_stats(v0, vr, gp, r)
                null.append(a - b)
            null = np.array(null)
            D = rs - ro
            rec[str(r)] = {"matched_rank_in_rotation": k + 1, "model_score_match_abs_r": float(abs(cors[k])),
                           "r_shown": rs, "r_orig": ro, "r_direct": rd, "D_shown_minus_orig": D,
                           "perm_p_two_sided": perm_pvalue_upper(abs(D), np.abs(null)),
                           "n_questions": int(both.sum())}
        rec["profile_rotation0_by_letter"] = dict(zip("ABCD", profile(dec[0][2][dec[0][3], j], gold[dec[0][3]]).round(4).tolist()))
        out[f"eigvec{j + 1}"] = rec
        print(rep, j + 1, {r: (rec[r]["matched_rank_in_rotation"], round(rec[r]["model_score_match_abs_r"], 2),
                              round(rec[r]["r_shown"], 2), round(rec[r]["r_orig"], 2), round(rec[r]["r_direct"], 2),
                              rec[r]["perm_p_two_sided"]) for r in ("1", "2", "3")})
    return out


def main() -> None:
    write_json("v3_C1b_letter_following", {
        "meta": {"n_perm": N_PERM, "seed": SEED, "doc": __doc__.strip().splitlines()[0]},
        "A": analyse("A"), "B": analyse("B")})


if __name__ == "__main__":
    main()
