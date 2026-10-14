"""B: leave one model out (24-model MMLU, 4 rotations) -> results/analysis/v3_B_leave_one_out.json.

Per model: rotation-0 and all-4-correct spectra without it, gold-letter eta^2 of rotation-0 eigenvectors 1-4.
Without Llama-3.2-1B (pi_A = 0.90): the rotation-0 and all-4 null suites refitted, seeds and draws as v3_across.
"""

from __future__ import annotations

import v2lib  # noqa: F401
import numpy as np
from v2lib import write_json

from null_models import eta_squared
from v3_across import (N_DRAWS, N_RANKS, SEED, Context, eig_vectors, kg_fits, load_rep, run_suite, spectrum)

DROP = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    d = load_rep("A")
    CR, gold, names = d["CR"], d["gold"], d["names"]
    loo = {}
    for m, n in enumerate(names):
        keep = np.arange(len(names)) != m
        X0 = CR[keep][:, :, 0]
        lam, _, V, k = eig_vectors(X0)
        loo[n] = {"rot0": spectrum(X0), "all4": spectrum(CR[keep].all(axis=2).astype(np.int8)),
                  "rot0_eta_sq_gold_letter_eigvec_1to4": [eta_squared(V[:, j], gold[k]) for j in range(N_RANKS)]}
    keep = np.array([n != DROP for n in names])
    d23 = {**d, "CR": CR[keep], "shown_choice": d["shown_choice"][keep], "names": [n for n in names if n != DROP]}
    fits, info = kg_fits(d23)
    pi = fits["kg_letter4"][1]
    C = d23["CR"][:, :, 0].astype(np.int8)
    rot0 = run_suite(Context(C, d23["CR"], gold, shown_gold=np.broadcast_to(gold, C.shape), pi=pi,
                             derive=lambda T: T[:, :, 0], kg_fits=fits),
                     ["2pl_post", "3pl_post", "letter_post", "kg_letter4"], SEED, N_DRAWS, "loo_rot0")
    X4 = d23["CR"].all(axis=2).astype(np.int8)
    all4 = run_suite(Context(X4, d23["CR"], gold, pi=pi, derive=lambda T: T.all(axis=2).astype(np.int8), kg_fits=fits),
                     ["rasch_jml", "fixed_margin", "2pl_post", "ba_a4", "kg_letter4"], SEED, N_DRAWS, "loo_all4")
    write_json("v3_B_leave_one_out", {"meta": {"seed": SEED, "n_draws_per_null": N_DRAWS, "dropped_for_suites": DROP},
                                      "leave_one_out_spectra": loo,
                                      "without_llama_3_2_1b": {"kg_fits": info, "rot0_suite": rot0, "all4_suite": all4}})
    for n, v in loo.items():
        print(f"{n:42s} rot0 {np.round(v['rot0']['top_eigs'][:4], 1)} buf {v['rot0']['mp_count_buffered']} "
              f"eta {np.round(v['rot0_eta_sq_gold_letter_eigvec_1to4'], 2)} all4 {np.round(v['all4']['top_eigs'][:3], 1)} "
              f"buf {v['all4']['mp_count_buffered']}")


if __name__ == "__main__":
    main()
