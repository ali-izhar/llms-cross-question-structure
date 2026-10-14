"""
Tests for scripts/v3_G_familywise.py: brute-force percentiles and LOO T, size of p_T under pure nulls,
reproduction of stored step-1 / step-2b outputs, seed disjointness.
Run: uv run python -m pytest scripts/test_v3_G_familywise.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import binom, kstest

import v3_G_familywise as G
import v3_across as V3
import v3_softpref as V
from v2lib import OUT

FAST = settings(max_examples=1000, deadline=None)
ROOT = Path(__file__).resolve().parents[1]


small_E = st.integers(min_value=2, max_value=12).flatmap(
    lambda n: arrays(np.float64, (n, 4), elements=st.sampled_from([0.0, 1.0, 1.5, 2.0, 3.0, 7.25])))


@FAST
@given(E=small_E, obs=arrays(np.float64, (4,), elements=st.sampled_from([0.0, 1.0, 1.5, 2.0, 2.5, 7.25, 9.0])))
def test_half_counts_equal_pct_of(E, obs):
    h = G.obs_half_counts(obs, E)
    pct = G.half_to_pct(h, E.shape[0])
    for r in range(4):
        npt.assert_allclose(pct[r], V.pct_of(obs[r], E[:, r]), rtol=0, atol=1e-12)


@FAST
@given(E=small_E)
def test_loo_half_counts_brute_force(E):
    n = E.shape[0]
    H = G.loo_half_counts(E)
    for d in range(n):
        others = np.delete(E, d, axis=0)
        for r in range(4):
            assert H[d, r] == 2 * np.sum(others[:, r] < E[d, r]) + np.sum(others[:, r] == E[d, r])


def _brute_familywise(obs, E):
    n = E.shape[0]
    T_obs = max(V.pct_of(obs[r], E[:, r]) for r in (1, 2, 3))
    T = []
    for d in range(n):
        o = np.delete(E, d, axis=0)
        T.append(max(V.pct_of(E[d, r], o[:, r]) for r in (1, 2, 3)))
    T = np.array(T)
    return T_obs, T, (1 + np.sum(T >= T_obs - 1e-9)) / (1 + n)


@settings(max_examples=300, deadline=None)
@given(seed=st.integers(0, 10 ** 6), n=st.integers(100, 140), ties=st.booleans())
def test_p_T_brute_force(seed, n, ties):
    rng = np.random.default_rng(seed)
    E = np.sort(rng.standard_normal((n, 4)), axis=1)[:, ::-1] + np.arange(4)[::-1]
    obs = np.sort(rng.standard_normal(4))[::-1] + np.arange(4)[::-1]
    if ties:
        E, obs = np.round(E, 1), np.round(obs, 1)
    fw = G.familywise(obs, E)
    T_obs, T, p = _brute_familywise(obs, E)
    npt.assert_allclose(fw["T_pct"], T_obs, rtol=0, atol=1e-12)
    # the brute force compares floats with a 1e-9 slack; the module compares exact integer cross products
    assert fw["p_T"] == pytest.approx(p, abs=1e-12)
    S = E[:, 1:4].sum(axis=1)
    assert fw["p_S"] == (1 + np.sum(S >= obs[1:4].sum())) / (1 + n)


@FAST
@given(seed=st.integers(0, 10 ** 6), n=st.integers(100, 160), ties=st.booleans())
def test_obs_equal_to_a_draw_gives_that_draws_loo_T(seed, n, ties):
    """With obs = null draw d (null = the other draws), T equals the leave-one-out T_d."""
    rng = np.random.default_rng(seed)
    E_all = rng.gamma(2.0, 1.0, (n + 1, 4)).cumsum(axis=1)[:, ::-1]
    if ties:
        E_all = np.round(E_all, 0)
    d = int(rng.integers(n + 1))
    fw = G.familywise(E_all[d], np.delete(E_all, d, axis=0))
    H = G.loo_half_counts(E_all)
    t_loo = H[d, 1:4].max()
    assert fw["T_pct"] == G.half_to_pct(t_loo, n)        # exact (same integer half-counts, same denominator)


def _size_sim(n_rep, n_null, rng, make):
    pts = np.empty(n_rep)
    for k in range(n_rep):
        E_all = make(n_null + 1, rng)
        pts[k] = G.familywise(E_all[0], E_all[1:])["p_T"]
    return pts


def _sorted_gauss(n, rng):
    Z = rng.standard_normal((n, 6))
    Z[:, :4] += 0.8 * Z[:, 4:5]          # correlated ranks
    return np.sort(Z[:, :4], axis=1)[:, ::-1]


def _eigs_random(n, rng):
    return np.array([V3.corr_eigs(rng.standard_normal((12, 20)))[:4] for _ in range(n)])


@pytest.mark.parametrize("make,name", [(_sorted_gauss, "gauss"), (_eigs_random, "eigs")])
def test_size_under_pure_null(make, name):
    """obs drawn from the null itself -> p_T ~ U(0, 1); FP at 0.05 within Monte Carlo error."""
    rng = np.random.default_rng(12345)
    n_rep = 3000 if name == "gauss" else 1000
    pts = _size_sim(n_rep, 199, rng, make)
    k = int(np.sum(pts < 0.05))
    lo, hi = binom.ppf(0.0005, n_rep, 0.05), binom.ppf(0.9995, n_rep, 0.05)
    assert lo <= k <= hi, (k, lo, hi)
    # p grid is 1/200: compare with U(0,1) by KS (grid D <= 0.005, far below the critical value)
    assert kstest(pts, "uniform").pvalue > 1e-3


def test_size_with_ties_is_not_anticonservative():
    rng = np.random.default_rng(7)
    pts = _size_sim(3000, 199, rng, lambda n, r: np.round(_sorted_gauss(n, r), 0))
    k = int(np.sum(pts < 0.05))
    assert k <= binom.ppf(0.9995, 3000, 0.05)


def test_size_on_stored_2pl_draws():
    """Size on realistic spectra: 400 stored 2PL draws (all-4), each as obs against the other 999."""
    E = np.load(OUT / "raw_v3_F_family_eigs.npz")["full_ge4__2pl_post"]
    rng = np.random.default_rng(3)
    idx = rng.choice(E.shape[0], 400, replace=False)
    pts = np.array([G.familywise(E[d], np.delete(E, d, axis=0))["p_T"] for d in idx])
    k = int(np.sum(pts < 0.05))
    assert binom.ppf(0.0005, 400, 0.05) <= k <= binom.ppf(0.9995, 400, 0.05)


def test_sibling_subsets():
    names = V.variant_data("full")["names"]
    assert len(names) == 24
    for c in G.CLUSTERS:
        for n in c:
            assert names.count(n) == 1, n
    subs = G.sibling_subsets(names)
    assert len(subs) == 24
    seen = set()
    in_cluster = {n for c in G.CLUSTERS for n in c}
    for s in subs:
        assert len(s["keep"]) == 19 and len(s["removed"]) == 5
        for c in G.CLUSTERS:
            assert sum(n in s["keep"] for n in c) == 1
        assert set(names) - in_cluster <= set(s["keep"])
        assert s["keep"] == [n for n in names if n in s["keep"]]       # original order
        seen.add(tuple(s["keep"]))
    assert len(seen) == 24
    assert len(names) - len(in_cluster) == 15


def test_random_control_subsets():
    names = V.variant_data("full")["names"]
    rnd = G.random_control_subsets(names, G.rng_g(G.STREAM_G["g4_random_pick"], 1))
    in_cluster = {n for c in G.CLUSTERS for n in c}
    assert len(rnd) == 24 and len({tuple(sorted(r["removed"])) for r in rnd}) == 24
    for r in rnd:
        assert len(r["removed"]) == 5 and not set(r["removed"]) & in_cluster and len(r["keep"]) == 19
    rnd2 = G.random_control_subsets(names, G.rng_g(G.STREAM_G["g4_random_pick"], 1))
    assert rnd == rnd2                                                   # deterministic


def test_subset_rows_matches_variant_data():
    d = V.variant_data("full")
    a = G.subset_rows(d, [n for n in d["names"] if n != V.LLAMA_1B])
    b = V.variant_data("no_llama1b")
    for k in ("CR", "shown_choice", "fam", "lp", "chosen", "gold", "subj"):
        npt.assert_array_equal(a[k], b[k])
    assert a["names"] == b["names"]


def test_subset_cols_matches_variant_data():
    d = V.variant_data("full")
    drop = {r["index"] for r in V.find_option_referencing(V.load_questions())}
    keep = np.array([i not in drop for i in range(d["CR"].shape[1])])
    a, b = G.subset_cols(d, keep), V.variant_data("no_optref")
    for k in ("CR", "shown_choice", "lp", "chosen", "gold", "subj", "question_index"):
        npt.assert_array_equal(a[k], b[k])


def _repo_integer_literals() -> set[int]:
    out = set()
    for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").glob("*.py")):
        if p.name in ("v3_G_familywise.py", "test_v3_G_familywise.py"):
            continue
        for m in re.finditer(r"\b\d[\d_]{3,}\b", p.read_text()):
            out.add(int(m.group(0).replace("_", "")))
    return out


def test_seed_namespace_is_new():
    lits = _repo_integer_literals()
    assert G.SEED_G not in lits
    earlier = [V.SEED_F, V3.SEED, 20261012, 20261011, 20261010, 20260417, 20260418, 20260419, 20260420, 20260421,
               20260414]
    for s in earlier:
        assert s in lits or s == V.SEED_F                  # the scan finds the earlier seeds
    # integer seeds of the form SEED + k, SEED_F + 1000 (rep + 1), ... are all within 1e8 of an earlier seed
    assert all(abs(G.SEED_G - s) > 10 ** 8 for s in lits if s > 10 ** 6)
    assert 0 < G.SEED_G < 2 ** 32
    # every G stream key is distinct, starts with SEED_G and has positive components
    keys = G.all_seed_keys()
    assert len(keys) == len(set(keys))
    for k in keys:
        assert G.seed_g(*k)[0] == G.SEED_G
    with pytest.raises(ValueError):
        G.seed_g(0)
    with pytest.raises(ValueError):
        G.seed_g(1, 0)
    # a G entropy differs from every earlier SeedSequence entropy form ([SEED_F, ...], SEED, SEED_F + 1000 r)
    ss = np.random.SeedSequence(G.seed_g(1, 1))
    assert ss.entropy[0] == G.SEED_G


def test_all_items_cover_every_seed_key_once():
    items = G.all_items()
    assert len(items) == len(set((k, repr(j)) for k, j in items))
    assert sum(k == "G2" for k, _ in items) == G.N_G2
    assert sum(k == "G3" for k, _ in items) == G.N_G3 * len(G.S_G3)
    assert sum(k == "G4" for k, _ in items) == 48
    assert sum(k == "G1" for k, _ in items) == 6


@pytest.fixture(scope="module")
def fam_table():
    return json.loads((OUT / "v3_F_softpref.json").read_text())["step1_construction_family"]["table"]


def test_stored_draws_reproduce_step1_table(fam_table):
    """Per-rank percentiles and validity of every step-1 null and score (full data, stored draws)."""
    z = np.load(OUT / "raw_v3_F_family_eigs.npz")
    d = V.variant_data("full")
    for sc in V.SCORES:
        X = V.SCORES[sc](d["CR"])
        obs = V3.Context(X).obs
        npt.assert_allclose(obs[:4], fam_table["full"][sc]["top_eigs"], rtol=1e-12)
        for nm in V.family_nulls(sc):
            fw = G.familywise(obs, z[f"full_{sc}__{nm}"])
            npt.assert_allclose(fw["pct_rank1to4"], fam_table["full"][sc]["observed_pct_rank1to4_by_null"][nm],
                                rtol=0, atol=1e-9)
            assert fw["valid"] == fam_table["full"][sc]["null_valid"][nm]
            assert fw["ranks_beyond_ability"] == fam_table["full"][sc]["ranks_beyond_ability_all_nulls"][nm]
            npt.assert_allclose(fw["rank_eigs_p99"], fam_table["full"][sc]["rank_p99_by_null"][nm], rtol=1e-12)


@pytest.mark.parametrize("score,nulls", [("ge4", ["rasch_jml", "2pl_post"]),
                                         ("rot0", ["3pl_post", "letter_post", "kg_uniform4", "fixed_margin"])])
def test_context_and_seed_reproduce_stored_draws(score, nulls):
    """score_context + run_nulls with the step-1 seed reproduce the first 25 stored draws exactly."""
    z = np.load(OUT / "raw_v3_F_family_eigs.npz")
    d = V.variant_data("full")
    ctx = G.score_context(d, score, V.family_nulls(score))
    E = G.run_nulls(ctx, nulls, V.SEED_FAMILY, 25, f"test_{score}")
    for nm in nulls:
        npt.assert_allclose(E[nm], z[f"full_{score}__{nm}"][:25], rtol=1e-12, atol=1e-12)


def test_g2_pipeline_reproduces_step2b_exactly():
    """Simulation + 2PL/Rasch refit of step 2b (calib_task), with step 2b's own seeds and 200 draws."""
    raw = json.loads((OUT / "raw_v3_F_calib.json").read_text())["per_dataset"]
    for rep in (0, 1):
        CR = G.g2_simulate(V.rng_for(*V.calib_key(rep)))
        want = {"ge4": ["rasch_jml", "2pl_post"], "ge3": ["rasch_jml", "2pl_post"], "ge2": ["2pl_post"]}
        for sc, nulls in want.items():
            res = G.nulls_on_scores(CR, (sc,), nulls, lambda _s: V.SEED_F + 1000 * (rep + 1), V.N_CALIB_NULL, "t")
            ref = raw[rep][sc]
            npt.assert_allclose(res[sc][nulls[0]]["observed_top_eigs"], ref["obs_top_eigs"], rtol=1e-12)
            for nm in nulls:
                r, q = res[sc][nm], ref["nulls"][nm]
                assert r["ranks_above_p99"] == q["ranks_above_p99"], (rep, sc, nm)
                assert r["ranks_beyond_ability"] == q["beyond_ability"]
                assert r["valid"] == q["valid"]
                npt.assert_allclose(r["pct_rank1to4"][0], q["lambda1_pct"], rtol=0, atol=1e-9)


def test_planted_dataset_equals_power_dataset(monkeypatch):
    """The G3 planted data equal v3_softpref.power_dataset's data for the same RNG (captured at its SP refit)."""
    captured = {}

    class _Stop(Exception):
        pass

    def fake_fit(data, spec, *a, **k):
        captured["CR"] = data.correct()
        raise _Stop

    d = V.variant_data("full")
    par = V.load_base("full")
    stem = V.stem_mask(d["subj"])
    for s in (1.0, 2.0):
        monkeypatch.setattr(V, "fit_sp", fake_fit)
        with pytest.raises(_Stop):
            V.power_dataset(par, d["gold"], stem, d["chosen"] < 0, s, np.random.default_rng(77))
        monkeypatch.undo()
        mine = G.planted_correct(par, d["gold"], stem, d["chosen"] < 0, s, np.random.default_rng(77))
        npt.assert_array_equal(mine, captured["CR"])


@settings(max_examples=500, deadline=None)
@given(seed=st.integers(0, 10 ** 6))
def test_all4_unknown_mask_brute_force(seed):
    rng = np.random.default_rng(seed)
    M, N = 3, 5
    truth = rng.integers(0, 2, (M, N, 4))
    unp = rng.random((M, N, 4)) < 0.3
    chosen = np.where(unp, -1, 0)
    CR = np.where(unp, 0, truth).astype(np.int8)
    mask = G.all4_unknown_mask(CR, chosen)
    for m in range(M):
        for i in range(N):
            vals = set()
            u = np.flatnonzero(unp[m, i])
            for bits in range(2 ** u.size):
                c = CR[m, i].copy()
                for b, r in enumerate(u):
                    c[r] = (bits >> b) & 1
                vals.add(int(c.all()))
            assert mask[m, i] == (len(vals) > 1)
            if not mask[m, i]:
                assert vals == {int(CR[m, i].all())}


def test_impute_patch_identity_and_restore():
    d = V.variant_data("full")
    ctx = V3.Context(V.SCORES["ge4"](d["CR"]))
    orig = V3.corr_eigs
    E0 = G.run_nulls(ctx, ["rasch_jml"], [G.SEED_G, 999], 15, "t0")["rasch_jml"]
    with G.impute_null_draws(np.zeros((ctx.X.shape[0], int(ctx.keep.sum())), dtype=bool)):
        E1 = G.run_nulls(ctx, ["rasch_jml"], [G.SEED_G, 999], 15, "t1")["rasch_jml"]
    assert V3.corr_eigs is orig
    # not bitwise: the imputed copy changes the BLAS summation order (max abs diff ~1e-13)
    npt.assert_allclose(E0, E1, rtol=1e-12, atol=1e-12)
    # with a mask: equals impute_column_mean of the same draw
    rng = np.random.default_rng(5)
    Y = (rng.random((24, 30)) < 0.5).astype(np.int8)
    mk = rng.random((24, 30)) < 0.1
    from v2lib import impute_column_mean
    Xi = impute_column_mean(Y, mk)
    nc = Xi.std(axis=0) > 0
    npt.assert_allclose(G.imputed_eigs(Y, mk, orig), orig(Xi[:, nc]), rtol=1e-12)
    with pytest.raises(ValueError):
        G.imputed_eigs(Y, mk[:, :5], orig)


def test_g6_mask_counts():
    d = V.variant_data("full")
    m = G.g6_masks(d)
    assert int(m["rot0"].any(axis=0).sum()) == 112
    assert int(m["ge4_any"].any(axis=0).sum()) == 217
    assert int(m["ge4"].any(axis=0).sum()) == 95 and int(m["ge4"].sum()) == 124
    # unknown all-4 entries are scored 0 as-scored
    assert np.all(V.SCORES["ge4"](d["CR"])[m["ge4"]] == 0)


def test_calibration_helpers():
    pts = np.array([0.001] * 30 + [0.5] * 70)
    c = G.calibration_of(pts)
    assert c["fp_raw"] == 0.3 and c["uses_calibrated_threshold"]
    assert c["cal_threshold_p_T_q05"] == pytest.approx(0.001)
    assert G.cal_p(0.001, pts) == 0.3
    fw = {"valid": True, "p_T": 0.04}
    assert G.g1_reject(fw) and not G.g1_reject(fw, 0.01) and not G.g1_reject({**fw, "valid": False})
