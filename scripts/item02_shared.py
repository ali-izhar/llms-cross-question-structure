"""Item 2: shared-randomness runs (S1 shared seed, S2 shared 5-shot prompt per pass).

Per run: the item-1 statistics, pass-level overdispersion, letter-share dispersion, and the top PC's
alignment with gold letters and subjects; for S2 also exemplar-letter echo and subject match.
Output: results/analysis/item02_shared_randomness.json
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import (EPS, LABEL, LETTERS, LLAMA_IID, OUT, QWEN_IID, RUNS, V2, label_eta_sq, letter_share_dispersion,
                   letters_to_codes, load_npz, matrix_path, perm_r2, subject_match_effect, top_pc_scores,
                   write_json, gold_letter_eta_sq, exemplar_overlap_similarity)

import pass_dispersion
from item01_within import within_stats
from null_models import borderline_mask, perm_pvalue_upper

N_PERM = 5000

SHARED = [r for r, _, g in RUNS if g == "shared"]
CONTROL = {"mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct": QWEN_IID,
           "mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct": QWEN_IID,
           "mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct": LLAMA_IID}


def letter_echo(codes: np.ndarray, counts: np.ndarray, n_perm: int, rng) -> dict:
    """Per letter: r over passes between exemplar count of L and pass share of answer L."""
    out = {}
    for j, L in enumerate(LETTERS):
        share = (codes == j).mean(axis=1)                                  # (K,)
        x = counts[:, j].astype(float)
        if x.std() == 0:
            out[L] = {"r": None}
            continue
        r = float(np.corrcoef(x, share)[0, 1])
        null = np.array([np.corrcoef(x[rng.permutation(x.size)], share)[0, 1] for _ in range(n_perm)])
        out[L] = {"r": r, "perm_p_two_sided": perm_pvalue_upper(abs(r), np.abs(null))}
    return out


def structure_tests(run_id: str, rng: np.random.Generator) -> dict:
    z = load_npz(run_id)
    C = z["correct"].astype(np.int8)
    codes = letters_to_codes(z["parsed"])                                   # (K, N)
    gold = letters_to_codes(z["gold"])                                      # (N,)
    bm = borderline_mask(C, EPS)
    scores, load, lam1 = top_pc_scores(C[:, bm])
    # Letter shares over the NON-borderline questions only: on the borderline
    # columns the pass score and the letter shares are mechanically linked
    # (a correct answer is the gold letter), which gives R^2 ~ 0.2 even for iid.
    shares = np.stack([(codes[:, ~bm] == j).mean(axis=1) for j in range(4)], axis=1)   # (K, 4)
    subj_all = np.array([q["subject"] for q in json.loads((V2 / "questions_mmlu.json").read_text())])
    rec = {
        "letter_share_dispersion": letter_share_dispersion(codes, 2000, rng),
        "top_pc": {
            "lambda1": lam1,
            "loadings_eta_sq_by_gold_letter": gold_letter_eta_sq(load, gold[bm], 5000, rng),
            # 57 subjects; the label permutation keeps group sizes
            "loadings_eta_sq_by_subject": label_eta_sq(load, subj_all[bm], 5000, rng),
            "pass_score_on_nonborderline_letter_shares": perm_r2(scores, shares[:, :3], 2000, rng),
        },
    }
    if "exemplar_dev_rows" in z.files:
        dev = json.loads((V2 / "questions_mmlu_dev.json").read_text())
        ex = z["exemplar_dev_rows"]                                         # (K, 5)
        ex_gold = np.array([[LETTERS.index(dev[r]["gold_letter"]) for r in row] for row in ex])
        counts = np.stack([(ex_gold == j).sum(axis=1) for j in range(4)], axis=1)   # (K, 4)
        pass_subj = [set(dev[r]["subject"] for r in row) for row in ex]
        subj = np.array([q["subject"] for q in json.loads((V2 / "questions_mmlu.json").read_text())])
        total_b = C[:, bm].sum(axis=1).astype(float)
        rec["exemplar_alignment"] = {
            "exemplar_rows_identical_across_models": None,  # filled in main
            "top_pc_scores_on_exemplar_gold_counts": perm_r2(scores, counts[:, :3], 5000, rng),
            "borderline_total_on_exemplar_gold_counts": perm_r2(total_b, counts[:, :3], 5000, rng),
            "letter_echo_exemplar_count_vs_answer_share": letter_echo(codes, counts, 5000, rng),
            "pass_pair_similarity_shared_exemplars_borderline": exemplar_overlap_similarity(C[:, bm], ex, 2000, rng),
            "subject_match_borderline": subject_match_effect(C[:, bm], subj[bm], pass_subj, 1000, rng),
            "subject_match_all_questions": subject_match_effect(C, subj, pass_subj, 1000, rng),
            "pass_score_top_abs_corr_with_any_exemplar_letter_count": float(
                max(abs(np.corrcoef(scores, counts[:, j])[0, 1]) for j in range(4) if counts[:, j].std() > 0)),
        }
    return rec


def main() -> None:
    rng = np.random.default_rng(20261010)
    out = {"meta": {"eps": EPS, "n_perm_pass_dispersion": pass_dispersion.N_PERM,
                    "seed": 20261010, "controls": "iid T=1 runs of the same models (item 1)"},
           "runs": {}}
    disp = pass_dispersion.main(models=[(LABEL[r], str(matrix_path(r))) for r in SHARED],
                                out_path=OUT / "raw_item02_pass_dispersion.json",
                                row_order="row k = pass k; passes ran in index order in vLLM batches (src/collect.py)")
    disp_by = {m["model"]: m for m in disp["models"]}
    ctrl_cache = {}
    for r in SHARED:
        base = within_stats(r)
        base["pass_dispersion"] = disp_by[LABEL[r]]
        base.update(structure_tests(r, rng))
        c = CONTROL[r]
        if c not in ctrl_cache:
            ctrl_cache[c] = structure_tests(c, rng)
        base["iid_control"] = {"run_id": c, **ctrl_cache[c]}
        out["runs"][r] = base
    ex_q = load_npz(SHARED[1])["exemplar_dev_rows"]
    ex_l = load_npz(SHARED[2])["exemplar_dev_rows"]
    same = bool(np.array_equal(ex_q, ex_l))
    for r in SHARED[1:]:
        out["runs"][r]["exemplar_alignment"]["exemplar_rows_identical_across_models"] = same

    # Committed verdict per run: above-noise structure if any calibrated test rejects at 1%.
    verdict = {}
    for r, rec in out["runs"].items():
        flags = {
            "mp_count_buffered>=1": rec["mp_count_buffered"] >= 1,
            "horn_contiguous>=1": rec["horn_count_contiguous"] >= 1,
            "z_above_mc_null_p99": rec["mc_null_z"]["percentile_of_obs"] > 0.99,
            "pass_overdispersion_p<0.01": rec["pass_dispersion"]["p_overdispersion"] < 0.01,
            "letter_share_dispersion_p<0.01": rec["letter_share_dispersion"]["perm_p"] < 0.01,
        }
        verdict[r] = {"flags": flags,
                      "correctness_spectrum_detects": flags["mp_count_buffered>=1"] or flags["horn_contiguous>=1"]
                      or flags["z_above_mc_null_p99"],
                      "pass_level_correctness_coupling": flags["pass_overdispersion_p<0.01"],
                      "letter_level_coupling": flags["letter_share_dispersion_p<0.01"]}
    out["verdict"] = verdict
    write_json("item02_shared_randomness", out)
    for r, v in verdict.items():
        rec = out["runs"][r]
        print(LABEL[r], "Nb", rec["N_b"], "z", round(rec["tw_z"], 2), "pct", rec["mc_null_z"]["percentile_of_obs"],
              "MP", rec["mp_count_strict"], "buf", rec["mp_count_buffered"], "Horn", rec["horn_count_contiguous"],
              "disp ratio", round(rec["pass_dispersion"]["dispersion_ratio_obs_over_expected"], 3),
              "p", rec["pass_dispersion"]["p_overdispersion"], v)


if __name__ == "__main__":
    main()
