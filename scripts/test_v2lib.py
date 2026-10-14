"""Tests for v2lib.py: brute-force equivalence, permutation-test calibration, the paper's reference
numbers (matrices in results/provenance/), and bit-identical reruns of the reused scripts' main().
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy: within_power forks 10 workers

import json
from collections import Counter

import numpy as np
import numpy.testing as npt
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import v2lib
from v2lib import (ROOT, exemplar_overlap_similarity, impute_column_mean, letter_share_dispersion,
                   letters_to_codes, modal_wrong_share, ols_r2, pair_correlations, spectrum_counts,
                   split_half_r, subject_match_effect, top_pc_scores)

RES = ROOT / "results" / "provenance"   # matrices behind the paper's reference numbers
FAST = settings(max_examples=1000, deadline=None)


@FAST
@given(arrays(np.dtype("<U1"), (7, 5), elements=st.sampled_from(list("ABCD") + [""])))
def test_letters_to_codes_roundtrip(P):
    c = letters_to_codes(P)
    back = np.where(c >= 0, np.array(list("ABCD"))[np.clip(c, 0, 3)], "")
    assert (back == P).all()


@FAST
@given(arrays(np.int8, (9, 6), elements=st.integers(-1, 3)), arrays(np.int8, (6,), elements=st.integers(0, 3)))
def test_modal_wrong_share_brute_force(codes, gold):
    got = modal_wrong_share(codes, gold)
    for i in range(codes.shape[1]):
        wrong = [int(x) for x in codes[:, i] if x != gold[i]]
        if not wrong:
            assert np.isnan(got[i])
        else:
            assert got[i] == Counter(wrong).most_common(1)[0][1] / len(wrong)


@FAST
@given(arrays(np.int8, (8, 5), elements=st.integers(0, 1)), arrays(np.bool_, (8, 5)))
def test_impute_column_mean(C, miss):
    X = impute_column_mean(C, miss)
    npt.assert_array_equal(X[~miss], C[~miss])
    for j in range(C.shape[1]):
        obs = ~miss[:, j]
        if obs.any():
            npt.assert_allclose(X[:, j].mean(), C[obs, j].mean(), rtol=1e-12, atol=1e-15)
        else:
            assert (X[:, j] == 0).all()


@FAST
@given(st.integers(0, 10_000), st.floats(0.1, 100), st.floats(-50, 50))
def test_ols_r2_single_predictor_is_r_squared_and_affine_invariant(seed, scale, shift):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(40)
    y = 0.3 * x + rng.standard_normal(40)
    r = np.corrcoef(x, y)[0, 1]
    npt.assert_allclose(ols_r2(y, x[:, None]), r ** 2, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(ols_r2(y, (scale * x + shift)[:, None]), r ** 2, rtol=1e-9, atol=1e-12)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000))
def test_top_pc_matches_corr_eigs(seed):
    from null_models import corr_eigs, standardize_columns
    rng = np.random.default_rng(seed)
    X = (rng.random((30, 12)) < rng.uniform(0.2, 0.8, 12)).astype(np.int8)
    X[0], X[1] = 1, 0                                   # no constant columns
    scores, v, lam1 = top_pc_scores(X)
    npt.assert_allclose(lam1, corr_eigs(X)[0], rtol=1e-10)
    npt.assert_allclose(scores, standardize_columns(X.astype(float)) @ v, rtol=1e-9, atol=1e-10)
    assert v.sum() >= 0


def test_spectrum_counts_matches_archive_z_at():
    import archive_claims
    C = np.load(RES / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy").astype(np.int8)
    from null_models import borderline_mask
    s = spectrum_counts(C[:, borderline_mask(C, 0.05)].astype(float))
    z, nb, n_mp, n_buf = archive_claims.z_at(C, 0.05)
    assert (s["n_cols"], s["mp_count_strict"], s["mp_count_buffered"]) == (nb, n_mp, n_buf) == (121, 0, 0)
    npt.assert_allclose(s["tw_z"], z, rtol=1e-12)
    npt.assert_allclose([s["lambda_plus"], s["lambda_max"]], [4.41, 4.17], atol=0.006)   # paper Table 1


def test_split_half_matches_paper_old_value():
    C = np.load(RES / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
    npt.assert_allclose(split_half_r(C), 0.994, atol=5e-4)                  # paper Fig. 1c


def _naive_subject_stat(C, subj, sets):
    diffs = []
    for i in range(C.shape[1]):
        m = np.array([subj[i] in sets[k] for k in range(C.shape[0])])
        if m.any() and (~m).any():
            diffs.append(C[m, i].mean() - C[~m, i].mean())
    return np.mean(diffs)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000))
def test_subject_match_effect_equals_naive(seed):
    rng = np.random.default_rng(seed)
    K, n = 25, 12
    C = (rng.random((K, n)) < 0.5).astype(np.int8)
    subj = np.array(rng.choice(list("abcdef"), n))
    sets = [set(rng.choice(list("abcdefgh"), 2, replace=False)) for _ in range(K)]
    got = subject_match_effect(C, subj, sets, 5, np.random.default_rng(0))
    npt.assert_allclose(got["mean_diff_matched_minus_unmatched"], _naive_subject_stat(C, subj, sets), rtol=1e-12)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000))
def test_exemplar_overlap_equals_naive(seed):
    from null_models import standardize_columns
    rng = np.random.default_rng(seed)
    K, n = 15, 10
    X = (rng.random((K, n)) < 0.5).astype(np.int8)
    X[0], X[1] = 1, 0
    ex = np.array([rng.choice(12, 3, replace=False) for _ in range(K)])
    Z = standardize_columns(X.astype(float))
    sh, no = [], []
    for k in range(K):
        for l in range(k + 1, K):
            s = np.corrcoef(Z[k], Z[l])[0, 1]
            (sh if len(set(ex[k]) & set(ex[l])) else no).append(s)
    if not sh or not no:
        return
    got = exemplar_overlap_similarity(X, ex, 5, np.random.default_rng(0))
    npt.assert_allclose(got["mean_sim_sharing_minus_not"], np.mean(sh) - np.mean(no), rtol=1e-10, atol=1e-12)


def test_letter_share_dispersion_calibrated_and_powered():
    rng = np.random.default_rng(1)
    K, n = 100, 200
    probs = rng.dirichlet(np.ones(4), n)                                    # per-question letter law
    iid = np.array([[rng.choice(4, p=probs[i]) for i in range(n)] for _ in range(K)], dtype=np.int8)
    pvals = [letter_share_dispersion(iid, 199, np.random.default_rng(s))["perm_p"] for s in range(3)]
    assert min(pvals) > 0.01
    # Planted pass-level letter bias: on pass k every question moves 30% of its mass to letter b_k.
    b = rng.integers(0, 4, K)
    shared = np.array([[rng.choice(4, p=0.7 * probs[i] + 0.3 * np.eye(4)[b[k]]) for i in range(n)]
                       for k in range(K)], dtype=np.int8)
    assert letter_share_dispersion(shared, 199, rng)["perm_p"] <= 0.01


def test_subject_match_effect_null_calibration():
    """Under no effect, the two-sided permutation p is roughly uniform: few rejections at 5%."""
    rng = np.random.default_rng(3)
    rej = 0
    for _ in range(60):
        C = (rng.random((40, 30)) < 0.5).astype(np.int8)
        subj = np.array(rng.choice(list("abcde"), 30))
        sets = [set(rng.choice(list("abcde"), 2, replace=False)) for _ in range(40)]
        rej += subject_match_effect(C, subj, sets, 99, rng)["perm_p_two_sided"] <= 0.05
    assert rej <= 9                                                         # E = 3; P(>9) < 1e-3


def test_pair_correlations_reproduce_paper_and_naive():
    from null_models import family_of
    C = np.load(RES / "ensemble_expanded_correctness.npy")
    names = json.loads((RES / "ensemble_expanded_models.json").read_text())
    fam = [family_of(n) for n in names]
    p = pair_correlations(C, fam)
    npt.assert_allclose([p["mean_r_same_family"], p["mean_r_cross_family"], p["cohen_d"], p["mean_r"]],
                        [0.41, 0.34, 0.58, 0.35], atol=0.006)                # paper Sec. 3.2 / App. B
    npt.assert_allclose([p["min_r"], p["max_r"]], [0.01, 0.63], atol=0.005)
    assert p["ks_p"] < 1e-3 and (p["n_same_family"], p["n_cross_family"]) == (30, 246)
    naive = [np.corrcoef(C[i], C[j])[0, 1] for i in range(24) for j in range(i + 1, 24)]
    npt.assert_allclose(p["mean_r"], np.mean(naive), rtol=1e-12)


def test_epsilon_margins_match_z_at():
    import make_epsilon_sweep_figure as m
    from null_models import borderline_mask, corr_eigs, mp_edge
    for slug in ("Qwen_Qwen2.5-7B-Instruct", "mistralai_Mistral-7B-Instruct-v0.3",
                 "microsoft_Phi-3-mini-4k-instruct", "meta-llama_Meta-Llama-3-8B-Instruct",
                 "HuggingFaceTB_SmolLM2-1.7B-Instruct"):
        C = np.load(RES / f"correctness_matrix_{slug}.npy")
        for eps in m.EPSILONS:
            margin, _, n_b = m.margin_and_tw(C, eps)
            X = C[:, borderline_mask(C, eps)]
            npt.assert_allclose(margin, mp_edge(X.shape[1], C.shape[0]) - corr_eigs(X)[0], rtol=1e-10)
            assert n_b == X.shape[1]
    C = np.load(RES / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
    npt.assert_allclose(m.margin_and_tw(C, 0.05)[0], 0.24, atol=0.005)       # paper Table 1


# Reused scripts: main() reproduces the saved raw outputs byte for byte.

def test_d_across_null_main_reproduces_item07_raw(tmp_path):
    import d_across_null
    d_across_null.main(matrix_path=v2lib.ENSEMBLE_NPY, out_json=tmp_path / "d.json", fig_path=None)
    assert (tmp_path / "d.json").read_bytes() == (v2lib.OUT / "raw_item07_d_across_null.json").read_bytes()


def test_pass_dispersion_main_reproduces_item04_raw(tmp_path):
    import item04_dispersion
    import pass_dispersion
    iid = [r for r, _, g in v2lib.RUNS if g == "iid"]
    pass_dispersion.main(models=[(v2lib.LABEL[r], str(v2lib.matrix_path(r))) for r in iid],
                         out_path=tmp_path / "p.json", row_order=item04_dispersion.ROW_ORDER)
    assert (tmp_path / "p.json").read_bytes() == (v2lib.OUT / "raw_item04_pass_dispersion.json").read_bytes()


def test_within_power_main_reproduces_item03_raw(tmp_path):
    import item03_power
    import within_power
    within_power.main(matrix_path=v2lib.matrix_path(v2lib.QWEN_IID), questions_path=v2lib.V2 / "questions_mmlu.json",
                      out_path=tmp_path / "w.json", max_workers=item03_power.MAX_WORKERS)
    assert (tmp_path / "w.json").read_bytes() == (v2lib.OUT / "raw_item03_power_qwen.json").read_bytes()
