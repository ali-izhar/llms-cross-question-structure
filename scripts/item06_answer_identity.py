"""Item 6: answer identity on the Qwen2.5-7B MMLU iid run.

(a) share of a borderline question's errors on its modal wrong letter; (b) max pairwise Cramer's V and
MI over borderline questions vs a column-shuffle null (letters for Qwen; correctness per MMLU iid run).
Output: results/analysis/item06_answer_identity.json
"""

from __future__ import annotations

import time

import v2lib
import numpy as np
from v2lib import EPS, LABEL, QWEN_IID, RUNS, letters_to_codes, load_npz, modal_wrong_share, write_json

import answer_identity_replication as air
from null_models import borderline_mask, perm_pvalue_upper

N_PERM = 1000
SEED = air.SEED


def family_test(A: np.ndarray, n_cats: int, rng: np.random.Generator) -> dict:
    t0 = time.time()
    v, mi = air.max_pairwise_stats(A, n_cats)
    nv, nmi = air.permutation_null(A, n_cats, N_PERM, rng)
    return {"n_items": int(A.shape[1]), "n_pairs": int(A.shape[1] * (A.shape[1] - 1) // 2),
            "max_cramers_v": {"observed": v, "null_p95": float(np.percentile(nv, 95)),
                              "null_p99": float(np.percentile(nv, 99)), "perm_p": perm_pvalue_upper(v, nv)},
            "max_mi_nats": {"observed": mi, "null_p95": float(np.percentile(nmi, 95)),
                            "null_p99": float(np.percentile(nmi, 99)), "perm_p": perm_pvalue_upper(mi, nmi)},
            "exceeds_p99": bool(v > np.percentile(nv, 99) or mi > np.percentile(nmi, 99)),
            "n_permutations": N_PERM, "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    z = load_npz(QWEN_IID)
    C = z["correct"].astype(np.int8)
    codes = letters_to_codes(z["parsed"])
    gold = letters_to_codes(z["gold"])
    bm = borderline_mask(C, EPS)
    share = modal_wrong_share(codes[:, bm], gold[bm])
    n_unparse_b = int((codes[:, bm] < 0).sum())
    out = {"run_id": QWEN_IID, "eps": EPS, "N_b": int(bm.sum()),
           "modal_wrong_letter_share": {
               "mean_over_borderline": float(np.nanmean(share)), "median": float(np.nanmedian(share)),
               "min": float(np.nanmin(share)), "n_questions": int(np.isfinite(share).sum()),
               "n_unparseable_cells_in_borderline": n_unparse_b},
           "letter_test_qwen": family_test(codes[:, bm].astype(np.int8), 4, np.random.default_rng(SEED)),
           "correctness_pattern_tests": {}}
    assert (codes[:, bm] >= 0).all(), "letters must be parseable for the 4-category test"
    for r, _, g in RUNS:
        if g == "iid" and r.startswith("mmlu_iid-T1.0"):
            Cr = np.load(v2lib.matrix_path(r)).astype(np.int8)
            out["correctness_pattern_tests"][LABEL[r]] = family_test(
                Cr[:, borderline_mask(Cr, EPS)], 2, np.random.default_rng(SEED))
    write_json("item06_answer_identity", out)
    print(out["modal_wrong_letter_share"], out["letter_test_qwen"])
    for k, v in out["correctness_pattern_tests"].items():
        print(k, v["max_cramers_v"], v["max_mi_nats"], v["exceeds_p99"])


if __name__ == "__main__":
    main()
