"""
Tests for v3_softpref.py against loop oracles, scipy quad, central finite differences and Monte Carlo.
Run: uv run python -m pytest scripts/test_v3_softpref.py
"""

from __future__ import annotations

import ast
import inspect
import os
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, settings, strategies as st
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import kstest, norm

import v3_softpref as F

FAST = settings(max_examples=1000, deadline=None)
SLOW = settings(max_examples=60, deadline=None)


def _problem(seed: int, M: int = 5, N: int = 7, with_tau: bool = True, p_unp: float = 0.1, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 4, N)
    chosen = rng.integers(0, 4, (M, N, 4))
    chosen[rng.random((M, N, 4)) < p_unp] = -1
    data = F.SPData(chosen, gold)
    spec = F.SPSpec(M, N, with_tau)
    x = scale * rng.standard_normal(spec.n_params)
    return data, spec, x, rng


def _naive_logits(par, gold, m, i, r, e=0.0):
    out = np.empty(4)
    for j in range(4):
        letter = (j + r) % 4
        out[j] = par["b"][i, j] + (par["a"][i] * par["theta"][m] + e if j == gold[i] else 0.0) + par["logpi"][m, letter]
    return out


def _naive_loglik(data, spec, x):
    """Unpenalized log-lik of the parsed choices by explicit loops (quad over e when tau > 0)."""
    par = F.unpack(x, spec, data.nongold)
    total = 0.0
    for m in range(data.M):
        tau = float(par["tau"][m]) if np.ndim(par["tau"]) else par["tau"]
        for i in range(data.N):
            rs = [r for r in range(4) if data.chosen[m, i, r] >= 0]

            def logf(e):
                s = 0.0
                for r in rs:
                    L = _naive_logits(par, data.gold, m, i, r, e)
                    s += L[data.chosen[m, i, r]] - logsumexp(L)
                return s

            if tau == 0.0 or not rs:
                total += logf(0.0)
            else:
                c = logf(0.0)
                val, _ = quad(lambda e: np.exp(logf(e) - c) * norm.pdf(e, 0, tau), -14 * tau, 14 * tau,
                              epsabs=0, epsrel=1e-13, limit=400)
                total += c + np.log(val)
    return total


def _penalty(spec, par):
    return -0.5 * (np.sum(par["theta"] ** 2) + np.sum(par["a"] ** 2) / spec.sigma_a ** 2
                   + np.sum(par["bfree"] ** 2) / spec.sigma_b ** 2 + np.sum(par["lam"] ** 2) / spec.sigma_lam ** 2)


def test_letter_of_matches_rotation_rule():
    # rotation r shows original option j at position (j + r) mod 4
    for r in range(4):
        for j in range(4):
            assert F.LETTER_OF[r, j] == (j + r) % 4
        assert sorted(F.LETTER_OF[r]) == [0, 1, 2, 3]


def test_letter_of_matches_parsed_letters_in_data():
    d = F.variant_data("full")
    shown = np.where(d["chosen"] >= 0, F.LETTER_OF[np.arange(4)[None, None, :], np.maximum(d["chosen"], 0)], -1)
    npt.assert_array_equal(shown, d["shown_choice"])


@FAST
@given(st.integers(0, 2 ** 31 - 1))
def test_scores_brute_force(seed):
    rng = np.random.default_rng(seed)
    T = (rng.random((3, 6, 4)) < rng.random()).astype(np.int8)
    for k in range(1, 5):
        exp = np.array([[int(sum(T[m, i]) >= k) for i in range(6)] for m in range(3)])
        npt.assert_array_equal(F.SCORES[f"ge{k}"](T), exp)
    npt.assert_array_equal(F.SCORES["rot0"](T), T[:, :, 0])
    npt.assert_allclose(F.SCORES["mean"](T), T.sum(2) / 4.0, rtol=0, atol=0)
    npt.assert_array_equal(F.SCORES["ge4"](T), T.all(axis=2).astype(np.int8))


@pytest.mark.parametrize("choices,expected", [
    (["ejecta", "raised rims", "central peaks", "A and B only"], {3}),
    (["x", "y", "z", "Both A and C."], {3}),
    (["x", "y", "z", "(B) (C) and (D)."], {3}),
    (["x", "y", "Both of these", "Neither of these"], {2, 3}),
    (["x", "y", "z", "None of the above."], {3}),
    (["x", "y", "z", "all of the above"], {3}),
    (["LC oscillators", "RC oscillators", "Both LC and RC Oscillators", "None"], {2}),
    (["causal harm", "expressive harm", "both causal and expressive harm", "auxiliary harm"], {2}),
    (["above of the above", "q", "r", "s"], {0}),
])
def test_option_reference_positive(choices, expected):
    assert {k for k, _, _ in F.option_reference_rule(choices)} == expected


@pytest.mark.parametrize("choices", [
    ["B", "Al", "In", "Tl"],                                       # boron
    ["Valid", "Invalid. Counterexample when C and D are true", "Invalid. C true", "Invalid. D true"],
    ["It supports garbage collection.", "It integrates with C, C++, COM", "Python", "Java"],
    ["Both fox and hare populations will decrease.", "Both fox and hare populations will increase.",
     "Fox populations will decrease and hare populations will increase.",
     "Fox populations will increase and hare populations will decrease."],
    ["I only", "II only", "I and II only", "I, II and III"],      # stem statements
    ["1 only", "3 only", "1 and 2 because they both reduce", "1 and 3"],
    ["vitamin A", "vitamin B12", "coenzyme A (CoA).", "hepatitis B"],
    ["None", "III only", "I and II only", "I, II, and III"],      # 78: "None" of the stem statements
    ["All of them", "The largest 2", "The smallest 2", "The second largest"],   # 137: stem eigenvalues
    ["none", "one", "two", "three"],                               # 259: a count
    ["America", "stands", "indivisible", "all"],                   # 416: a word
    ["x", "y", "z", "prior to the passage of these Carlsbad Resolutions"],     # 214
    ["The Aardvarks do not win.", "The Bees win.", "The Chipmunks do not win.",
     "Either the Aardvarks or the Chipmunks do not win."],         # 167: logic content
])
def test_option_reference_negative(choices):
    assert F.option_reference_rule(choices) == []


def test_option_referencing_pinned_on_mmlu():
    # Pinned after reading every hit by hand.
    idx = [r["index"] for r in F.find_option_referencing(F.load_questions())]
    assert idx == [28, 121, 124, 126, 145, 146, 187, 204, 207, 241, 298, 327, 334, 376, 377, 379, 421, 422,
                   428, 429, 444, 450, 460]


def test_variants_shapes():
    full = F.variant_data("full")
    nol = F.variant_data("no_llama1b")
    nor = F.variant_data("no_optref")
    assert full["CR"].shape == (24, 500, 4)
    assert nol["CR"].shape == (23, 500, 4) and F.LLAMA_1B not in nol["names"]
    assert nor["CR"].shape == (24, 477, 4) and nor["gold"].shape == (477,) and nor["subj"].shape == (477,)
    i = full["names"].index(F.LLAMA_1B)
    npt.assert_array_equal(np.delete(full["CR"], i, 0), nol["CR"])
    npt.assert_array_equal(full["CR"][:, nor["question_index"]], nor["CR"])


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.floats(0.1, 3.0))
def test_loglik_tau0_equals_naive(seed, scale):
    data, spec, x, _ = _problem(seed, M=3, N=4, with_tau=False, scale=scale)
    obj, _, ll = F.sp_terms(x, data, spec)
    npt.assert_allclose(ll, _naive_loglik(data, spec, x), rtol=1e-12, atol=1e-10)
    npt.assert_allclose(obj - ll, _penalty(spec, F.unpack(x, spec, data.nongold)), rtol=1e-12, atol=1e-12)


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.floats(-2.0, 1.0), st.sampled_from([True, "model"]))
def test_loglik_tau_equals_quadrature(seed, logtau, mode):
    data, spec, x, rng = _problem(seed, M=3, N=3, with_tau=mode)
    x[spec.n_params - spec.n_tau:] = np.minimum(logtau + 0.5 * rng.standard_normal(spec.n_tau), 1.0)
    # K = 201: Gauss-Hermite truncation error at tau = e^1 is 8e-12 (41 nodes: 4e-5), so tau <= e^1 here.
    spec = F.SPSpec(spec.M, spec.N, mode, K=201)
    _, _, ll = F.sp_terms(x, data, spec)
    npt.assert_allclose(ll, _naive_loglik(data, spec, x), rtol=1e-10, atol=1e-9)


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.floats(-2.0, 1.2))
def test_per_model_tau_equal_values_equals_common(seed, logtau):
    data, spec_m, x_m, _ = _problem(seed, M=3, N=4, with_tau="model")
    x_m[-3:] = logtau
    spec_c = F.SPSpec(3, 4, True)
    x_c = np.concatenate([x_m[:-3], [logtau]])
    o_m, g_m, l_m = F.sp_terms(x_m, data, spec_m)
    o_c, g_c, l_c = F.sp_terms(x_c, data, spec_c)
    npt.assert_allclose(l_m, l_c, rtol=1e-13, atol=1e-11)
    npt.assert_allclose(g_m[:-3], g_c[:-1], rtol=1e-11, atol=1e-11)
    npt.assert_allclose(g_m[-3:].sum(), g_c[-1], rtol=1e-11, atol=1e-11)   # chain rule: d/dlogtau = sum_m


def test_tau_to_zero_limit():
    data, spec, x, _ = _problem(3, with_tau=True)
    x[-1] = -40.0
    spec0 = F.SPSpec(spec.M, spec.N, False)
    _, _, ll = F.sp_terms(x, data, spec)
    _, _, ll0 = F.sp_terms(x[:-1], data, spec0)
    npt.assert_allclose(ll, ll0, rtol=1e-13, atol=1e-11)


def test_all_unparsed_cell_contributes_zero():
    data, spec, x, _ = _problem(5, with_tau=True)
    ch = data.chosen.copy()
    ch[0, 0, :] = -1
    d2 = F.SPData(ch, data.gold)
    par = F.unpack(x, spec, d2.nongold)
    ll_ref = _naive_loglik(d2, spec, x)
    npt.assert_allclose(F.sp_terms(x, d2, spec)[2], ll_ref, rtol=1e-9)
    assert par["tau"] > 0


@pytest.mark.parametrize("with_tau", [False, True, "model"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_all_coordinates_fd(with_tau, seed):
    data, spec, x, _ = _problem(seed, M=4, N=5, with_tau=with_tau)
    _, g, _ = F.sp_terms(x, data, spec)
    h = 1e-6
    fd = np.empty_like(x)
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = h
        fd[k] = (F.sp_terms(x + e, data, spec)[0] - F.sp_terms(x - e, data, spec)[0]) / (2 * h)
    # NOTE: central FD error ~ h^2 f''' + eps |f| / h ~ 1e-12 + 1e-16 * 50 / 1e-6 = 5e-9.
    npt.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([False, True, "model"]), st.floats(0.2, 2.5))
def test_gradient_fd_random_points(seed, with_tau, scale):
    data, spec, x, rng = _problem(seed, M=3, N=4, with_tau=with_tau, scale=scale)
    if spec.n_tau:
        x[spec.n_params - spec.n_tau:] = np.clip(x[spec.n_params - spec.n_tau:], -2.0, 1.2)
    _, g, _ = F.sp_terms(x, data, spec)
    h = 1e-6
    for k in rng.choice(x.size, 8, replace=False):
        e = np.zeros_like(x)
        e[k] = h
        fd = (F.sp_terms(x + e, data, spec)[0] - F.sp_terms(x - e, data, spec)[0]) / (2 * h)
        assert abs(fd - g[k]) <= 1e-6 * max(1.0, abs(g[k]))


@FAST
@given(st.integers(0, 2 ** 31 - 1))
def test_standardize_keeps_likelihood_and_probabilities(seed):
    data, spec, x, _ = _problem(seed, M=4, N=5, with_tau=False)
    par = F.unpack(x, spec, data.nongold)
    sp = F.standardize(par, data.nongold)
    npt.assert_allclose(sp["theta"].mean(), 0, atol=1e-12)
    npt.assert_allclose(sp["theta"].std(), 1, rtol=1e-12)
    x2 = F.pack(sp["theta"], sp["a"], sp["bfree"], sp["lam"])
    npt.assert_allclose(F.sp_terms(x2, data, spec)[2], F.sp_terms(x, data, spec)[2], rtol=1e-11, atol=1e-9)
    npt.assert_allclose(F.choice_probs(sp, data.gold), F.choice_probs(par, data.gold), rtol=1e-9, atol=1e-13)


def test_choice_probs_sum_to_one_and_match_naive():
    data, spec, x, _ = _problem(11, with_tau=False)
    par = F.unpack(x, spec, data.nongold)
    P = F.choice_probs(par, data.gold)
    npt.assert_allclose(P.sum(axis=3), 1.0, rtol=1e-13)
    for m, i, r in [(0, 0, 0), (2, 3, 1), (4, 6, 3)]:
        L = _naive_logits(par, data.gold, m, i, r)
        npt.assert_allclose(P[m, i, r], np.exp(L - logsumexp(L)), rtol=1e-12)


@pytest.mark.parametrize("tau", [0.0, 1.3, "model"])
def test_simulation_frequencies(tau):
    rng = np.random.default_rng(7)
    M, N = 3, 4
    mode = "model" if tau == "model" else tau > 0
    data, spec, x, _ = _problem(21, M=M, N=N, with_tau=mode, p_unp=0.0)
    if mode == "model":
        x[-3:] = np.log([0.5, 1.3, 2.2])
    elif mode:
        x[-1] = np.log(tau)
    par = F.unpack(x, spec, data.nongold)
    tau_m = np.broadcast_to(np.asarray(par["tau"], dtype=float), (M,))
    rho = np.array([0.0, 0.1, 0.3])
    n = 20000
    counts = np.zeros((M, N, 4, 5))
    for _ in range(n):
        ch = F.simulate_choices(par, data.gold, rho, rng)
        for k in range(5):
            counts[..., k] += (ch == (k if k < 4 else -1))
    freq = counts / n
    # Expected: (1 - rho) * E_e[softmax], E_e by 81-node Gauss-Hermite.
    if not mode:
        P = F.choice_probs(par, data.gold)
    else:
        z, w = np.polynomial.hermite_e.hermegauss(81)
        w = w / w.sum()
        P = sum(wk * F.choice_probs(par, data.gold, e=np.broadcast_to(tau_m[:, None] * zk, (M, N)))
                for zk, wk in zip(z, w))
    exp = np.concatenate([(1 - rho)[:, None, None, None] * P, np.broadcast_to(rho[:, None, None, None], (M, N, 4, 1))],
                         axis=3)
    sd = np.sqrt(exp * (1 - exp) / n)
    assert np.all(np.abs(freq - exp) <= 5 * sd + 1e-12), np.max(np.abs(freq - exp) / (sd + 1e-12))


def test_rho_is_unparsed_share_and_unparsed_is_wrong():
    data, _ = F.sp_data_of("full")
    npt.assert_allclose(data.rho, (data.chosen < 0).reshape(data.M, -1).mean(axis=1), rtol=0, atol=0)
    rng = np.random.default_rng(1)
    par = {"theta": np.zeros(data.M), "a": np.zeros(data.N), "b": np.zeros((data.N, 4)),
           "logpi": np.log(np.full((data.M, 4), 0.25)), "tau": 0.0}
    ch = F.simulate_choices(par, data.gold, np.ones(data.M), rng)
    assert np.all(ch == -1)
    assert not np.any(ch == data.gold[None, :, None])


def test_unparsed_loglik_closed_form():
    data, _ = F.sp_data_of("full")
    exp = 0.0
    for m in range(data.M):
        nu, npar = data.n_unp[m], data.n_par[m]
        r = nu / (nu + npar)
        exp += (nu * np.log(r) if nu else 0.0) + npar * np.log1p(-r)
    npt.assert_allclose(data.unparsed_loglik(), exp, rtol=1e-13)


def test_null_draw_conditional_columns_and_eigs():
    rng = np.random.default_rng(3)
    data, spec, x, _ = _problem(4, M=6, N=30, with_tau=True)
    par = F.unpack(x, spec, data.nongold)
    C = data.correct()
    keeps = {s: F.keep_mask(F.SCORES[s](C)) for s in F.SCORES}
    keeps = {s: k for s, k in keeps.items() if k.sum() >= 2}
    out = F.null_draw_scores(par, data.gold, data.chosen < 0, rng, keeps)
    for s, e in out.items():
        assert e.shape == (min(6, int(keeps[s].sum())),)
        npt.assert_allclose(e.sum(), keeps[s].sum(), rtol=1e-10)   # trace of R = number of kept columns
        assert np.all(np.diff(e) <= 1e-12)


def test_pct_of_and_rank_summary():
    null = np.arange(100.0)
    assert F.pct_of(-1, null) == 0.0 and F.pct_of(1000, null) == 100.0
    npt.assert_allclose(F.pct_of(10.0, null), 10.5)
    E = np.tile(np.array([10.0, 5, 4, 3]), (200, 1)) + np.random.default_rng(0).normal(0, 0.1, (200, 4))
    s = F.rank_summary(E, np.array([10.0, 9.0, 4.0, 3.0]))
    assert s["observed_ranks_above_p99"] == [2] and s["ranks_beyond_ability"] == 1
    assert s["null_valid_lambda1_in_p01_p99"]


@FAST
@given(st.integers(0, 2 ** 31 - 1))
def test_cubic_residual_orthogonal(seed):
    rng = np.random.default_rng(seed)
    p = rng.random(40)
    v = rng.standard_normal(40) + 3 * p ** 2
    r = F.cubic_residual(v, p)
    A = np.column_stack([np.ones_like(p), p, p ** 2, p ** 3])
    npt.assert_allclose(A.T @ r, 0, atol=1e-9)


@pytest.mark.parametrize("with_tau", [False, True, "model"])
def test_fit_recovers_parameters_moderate(with_tau):
    rng = np.random.default_rng(12)
    M, N = 24, 150
    gold = rng.integers(0, 4, N)
    lam = rng.normal(0, 0.8, (M, 3))
    lam_full = np.concatenate([np.zeros((M, 1)), lam], 1)
    par = {"theta": rng.standard_normal(M), "a": rng.lognormal(0.5, 0.4, N), "lam": lam,
           "logpi": lam_full - logsumexp(lam_full, axis=1, keepdims=True),
           "tau": (np.linspace(0.5, 2.5, M) if with_tau == "model" else 1.5 if with_tau else 0.0)}
    b = rng.normal(-0.5, 1.0, (N, 4))
    b[np.arange(N), gold] = 0
    par["b"] = b
    data = F.SPData(F.simulate_choices(par, gold, np.full(M, 0.02), rng), gold)
    fit = F.fit_sp(data, F.SPSpec(M, N, with_tau))
    assert fit.converged, fit.message
    assert fit.max_abs_grad < 1e-3
    assert np.corrcoef(fit.par["theta"], par["theta"])[0, 1] > 0.97
    assert np.corrcoef(fit.par["a"], par["a"] * par["theta"].std())[0, 1] > 0.6
    assert np.max(np.abs(np.exp(fit.par["logpi"]) - np.exp(par["logpi"]))) < 0.12
    if with_tau is True:
        assert abs(fit.par["tau"] - 1.5) < 0.4
    if with_tau == "model":
        # NOTE: 150 questions per model; tau_m is estimated from within-cell agreement only.
        assert np.corrcoef(fit.par["tau"], par["tau"])[0, 1] > 0.6


def test_null_draw_failure_is_reported_not_filled():
    # Every cell unparsed -> every column constant (all wrong) -> the conditional draw must fail.
    rng = np.random.default_rng(5)
    data, spec, x, _ = _problem(6, M=4, N=6, with_tau=False)
    par = F.unpack(x, spec, data.nongold)
    keeps = {"rot0": np.ones(6, bool), "ge4": np.ones(6, bool)}
    all_unp = np.ones((4, 6, 4), bool)
    out = F.null_draw_scores(par, data.gold, all_unp, rng, keeps, max_redraws=20)
    assert out == {"rot0": None, "ge4": None}
    E, fail = F.sp_null(par, data.gold, all_unp, keeps, 3, rng, max_redraws=20)
    assert fail == {"rot0": 3, "ge4": 3} and E["rot0"].shape == (0, 4)


def _clustered_mask(rng, M, N, frac_q=0.3, frac_m=0.5, p_bg=0.02):
    """Unparsed cells that cluster by question: on frac_q of the questions, frac_m of the models fail all 4."""
    mask = rng.random((M, N, 4)) < p_bg
    for i in np.flatnonzero(rng.random(N) < frac_q):
        mask[rng.random(M) < frac_m, i, :] = True
    return mask


def _par(seed, M, N, tau=0.8):
    """Parameter dict built directly from the model definition (b_gold = 0, log pi = log_softmax([0, lam]))."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 4, N)
    lam = rng.normal(0, 0.5, (M, 3))
    th = rng.standard_normal(M)
    bfree = rng.normal(-0.5, 0.8, (N, 3))
    b = np.zeros((N, 4))
    for i in range(N):
        b[i, [j for j in range(4) if j != gold[i]]] = bfree[i]
    lf = np.concatenate([np.zeros((M, 1)), lam], axis=1)
    return {"theta": (th - th.mean()) / th.std(), "a": rng.lognormal(0.3, 0.3, N), "bfree": bfree, "b": b,
            "lam": lam, "logpi": lf - logsumexp(lf, axis=1, keepdims=True), "tau": tau, "logtau": np.log(tau)}, gold


def _spy_simulate(monkeypatch):
    calls = []
    real = F.simulate_choices

    def spy(*args, **kw):
        calls.append({"rho": args[2] if len(args) > 2 else kw.get("rho"), **{k: v for k, v in kw.items() if k != "rho"}})
        return real(*args, **kw)

    monkeypatch.setattr(F, "simulate_choices", spy)
    return calls


def test_simulate_with_mask_reproduces_mask_exactly():
    par, gold = _par(1, 6, 15)
    rng = np.random.default_rng(2)
    mask = _clustered_mask(rng, 6, 15)
    for _ in range(20):
        ch = F.simulate_choices(par, gold, None, rng, unp_mask=mask)
        npt.assert_array_equal(ch < 0, mask)
    idx = np.array([3, 7, 11])
    npt.assert_array_equal(F.simulate_choices(par, gold, None, rng, idx=idx, unp_mask=mask) < 0, mask[:, idx])
    with pytest.raises(ValueError):
        F.simulate_choices(par, gold, np.zeros(6), rng, unp_mask=mask)       # both
    with pytest.raises(ValueError):
        F.simulate_choices(par, gold, None, rng)                             # neither


def test_every_simulation_call_in_module_is_conditional_on_the_mask():
    # Static check over the whole module: each call simulate_choices(par, gold, None, rng, ..., unp_mask=...).
    tree = ast.parse(inspect.getsource(F))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "simulate_choices"]
    assert len(calls) >= 8
    for c in calls:
        assert any(k.arg == "unp_mask" for k in c.keywords), ast.unparse(c)
        assert isinstance(c.args[2], ast.Constant) and c.args[2].value is None, ast.unparse(c)


def test_null_draws_and_ppc_pass_the_mask(monkeypatch):
    par, gold = _par(3, 6, 20)
    rng = np.random.default_rng(4)
    mask = _clustered_mask(rng, 6, 20)
    calls = _spy_simulate(monkeypatch)
    keeps = {"rot0": np.ones(20, bool), "ge2": np.ones(20, bool)}
    F.sp_null(par, gold, mask, keeps, 5, rng)
    data = F.SPData(F.simulate_choices(par, gold, None, rng, unp_mask=mask), gold)
    F.ppc(SimpleNamespace(par=par, rho=data.rho), data, 3, rng)
    assert len(calls) > 9
    for c in calls:
        assert c["rho"] is None and c["unp_mask"] is mask or np.array_equal(c["unp_mask"], mask)


def _uniform_u(obs, null, rng):
    """Randomized rank of obs among (null, obs): exactly U(0, 1) under exchangeability."""
    less, eq = int(np.sum(null < obs)), int(np.sum(null == obs))
    return (less + rng.integers(0, eq + 1) + rng.random()) / (null.size + 1)


def _exchangeability_u(null_rho: bool, n_rep=300, n_null=19, M=8, N=40, scores=("rot0", "ge2", "mean")):
    par, gold = _par(5, M, N)
    rng = np.random.default_rng(6)
    mask = _clustered_mask(rng, M, N)
    rho = mask.reshape(M, -1).mean(axis=1)
    U = {(s, j): [] for s in scores for j in (0, 1)}
    for _ in range(n_rep):
        C = (F.simulate_choices(par, gold, None, rng, unp_mask=mask) == gold[None, :, None]).astype(np.int8)
        keeps = {s: F.keep_mask(F.SCORES[s](C)) for s in scores}
        obs = {s: F.corr_eigs(F.SCORES[s](C)[:, keeps[s]].astype(float)) for s in scores}
        if null_rho:   # unparsed cells Bernoulli(rho_m), independent of the question
            E = {s: [] for s in scores}
            for _ in range(n_null):
                T = (F.simulate_choices(par, gold, rho, rng) == gold[None, :, None]).astype(np.int8)
                for s in scores:
                    X = F.SCORES[s](T[:, keeps[s]]).astype(float)
                    if np.all(X.std(axis=0) > 0):
                        E[s].append(F.corr_eigs(X))
            E = {s: np.array(v) for s, v in E.items()}
        else:
            E, fail = F.sp_null(par, gold, mask, keeps, n_null, rng)
            assert all(v == 0 for v in fail.values())
        for s in scores:
            if E[s].shape[0] == 0:
                continue
            for j in (0, 1):
                U[(s, j)].append(_uniform_u(obs[s][j], E[s][:, j], rng))
    return U


def test_exchangeability_at_known_parameters_with_clustered_mask():
    # Data and null from the same known parameters: the randomized rank of lambda_1, lambda_2 is U(0, 1).
    U = _exchangeability_u(null_rho=False)
    for k, u in U.items():
        assert len(u) == 300
        assert kstest(u, "uniform").pvalue > 1e-3, (k, kstest(u, "uniform"))


def test_exchangeability_fails_with_a_per_model_unparsed_rate():
    # Negative control: a per-model unparsed rate is not exchangeable with question-clustered unparsed cells.
    U = _exchangeability_u(null_rho=True, n_rep=150)
    assert min(kstest(u, "uniform").pvalue for u in U.values() if len(u) > 50) < 1e-6


def test_redraw_boundary_checks_last_redraw(monkeypatch):
    M, N = 3, 2
    gold = np.zeros(N, dtype=int)
    calls = {"n": 0}

    def fake(par, gold_, rho, rng, idx=None, extra_gold=None, unp_mask=None):
        calls["n"] += 1
        n = gold_.size if idx is None else len(idx)
        if idx is None:
            return np.full((M, n, 4), 1)                  # first draw: everyone wrong -> constant columns
        ch = np.full((M, n, 4), 1)
        ch[0] = 0                                         # redraw: model 0 right -> non-constant
        return ch

    monkeypatch.setattr(F, "simulate_choices", fake)
    keeps = {"rot0": np.ones(N, bool)}
    mask = np.zeros((M, N, 4), bool)
    rng = np.random.default_rng(0)
    out1 = F.null_draw_scores({"theta": np.zeros(M)}, gold, mask, rng, keeps, max_redraws=1)
    assert out1["rot0"] is not None and calls["n"] == 2   # one redraw allowed, and it fixed the columns
    out0 = F.null_draw_scores({"theta": np.zeros(M)}, gold, mask, rng, keeps, max_redraws=0)
    assert out0["rot0"] is None                            # no redraw allowed


def test_detect_min_draw_guard():
    obs = np.array([10.0, 5.0, 3.0, 2.0])
    for n in (0, 3):
        d = F._detect(np.random.default_rng(1).normal(size=(n, 4)), obs)
        assert d["insufficient_draws"] and not d["rank2_above_p99"] and not d["valid"] and d["n_draws"] == n
    for n in (F.MIN_NULL_DRAWS - 1,):
        d = F._detect(np.random.default_rng(1).normal(size=(n, 4)), obs)
        assert d["insufficient_draws"] and not d["rank2_above_p99"] and not d["valid"] and d["n_draws"] == n
    d = F._detect(np.random.default_rng(1).normal(size=(F.MIN_NULL_DRAWS, 4)), obs)
    assert not d["insufficient_draws"] and d["rank2_above_p99"]


def test_validity_uses_the_same_percentile_call_as_p99():
    # null lambda_1 = 0..99: np.percentile p01 = 0.99, so 0.5 is invalid although its mid-rank percentile is 1.0.
    E = np.column_stack([np.arange(100.0), np.zeros((100, 3))])
    obs = np.array([0.5, 0.0, 0.0, 0.0])
    assert F.pct_of(0.5, E[:, 0]) == 1.0
    assert F._detect(E, obs)["valid"] is False
    assert F.lambda1_valid(E, obs) is False
    assert F.rank_summary(E, obs)["null_valid_lambda1_in_p01_p99"] is False
    p01, p99 = F.null_bands(E)
    npt.assert_allclose([p01[0], p99[0]], np.percentile(E[:, 0], [1, 99]))
    assert F._detect(E, np.array([0.99, 0, 0, 0]))["valid"] is True


def test_pool_equal_weight_excludes_short_refits():
    a, b, c = np.ones((10, 4)), 2 * np.ones((10, 4)), 100 * np.ones((3, 4))
    pooled, n_excl = F.pool_equal_weight([a, b, c], 10)
    assert pooled.shape == (20, 4) and n_excl == 1 and not np.any(pooled == 100)
    with pytest.raises(ValueError):
        F.pool_equal_weight([np.ones((11, 4))], 10)


def test_sp_null_target_reaches_target_or_stops_at_cap():
    par, gold = _par(7, 6, 20)
    rng = np.random.default_rng(8)
    keeps = {"rot0": np.ones(20, bool), "ge4": np.ones(20, bool)}
    E, att = F.sp_null_target(par, gold, np.zeros((6, 20, 4), bool), keeps, 10, 50, rng)
    assert all(E[s].shape[0] == 10 for s in keeps) and all(10 <= att[s] <= 50 for s in keeps)
    E, att = F.sp_null_target(par, gold, np.ones((6, 20, 4), bool), keeps, 10, 7, rng)    # every draw fails
    assert all(E[s].shape[0] == 0 and att[s] == 7 for s in keeps)


def _det(p2, valid=True, r2=False, z2=None, ins=False):
    if ins:
        return {"insufficient_draws": True, "rank2_above_p99": False, "valid": False}
    return {"insufficient_draws": False, "pct_rank1to4": [50.0, p2, 40.0, 30.0],
            "z_rank1to4": [0.0, p2 / 10 if z2 is None else z2, 0.0, 0.0], "rank2_above_p99": r2, "valid": valid}


def test_calibrate_fraction_threshold_and_fpr():
    boot = [_det(p) for p in np.linspace(0, 100, 101)] + [_det(99.5, r2=True), _det(99.5, r2=True, valid=False),
                                                           _det(0, ins=True)]
    obs = {"insufficient_draws": False, "pct_rank1to4": [50.0, 95.0, 40.0, 30.0], "z_rank1to4": [0, 9.5, 0, 0]}
    c = F.calibrate(obs, boot)
    B = 103
    assert c["n_boot"] == B and c["n_boot_insufficient_draws"] == 1
    P2 = np.r_[np.linspace(0, 100, 101), 99.5, 99.5]
    npt.assert_allclose(c["cal_p_rank1to4"][1], np.mean(P2 >= 95.0))
    npt.assert_allclose(c["cal_p_mc_rank1to4"][1], (np.sum(P2 >= 95.0) + 1) / (B + 1))
    npt.assert_allclose(c["cal_threshold_pct_99_rank1to4"][1], np.percentile(P2, 99))
    assert 2 not in c["ranks_above_cal_threshold"]
    npt.assert_allclose(c["fpr_rank2_above_p99_and_valid"], 1 / B)
    assert c["cal_p_rank1to4"][0] == np.mean(np.full(B, 50.0) >= 50.0) == 1.0
    assert c["lambda1_calibrated_valid"] is True and c["lambda1_cal_p_z_two_sided"] == 1.0    # all z_1 = 0 = obs
    obs_out = {**obs, "z_rank1to4": [-1.0, 9.5, 0, 0]}                     # lambda_1 z below every bootstrap z_1
    c2 = F.calibrate(obs_out, boot)
    assert c2["lambda1_calibrated_valid"] is False and c2["ranks_beyond_ability_calibrated_valid"] is None
    assert c2["lambda1_cal_p_two_sided"] == 1.0                            # the percentile version cannot see it
    assert F.calibrate({"insufficient_draws": True}, boot)["calibrated"] is False


def test_bias_corrected_sets():
    rng = np.random.default_rng(9)
    M, N = 5, 12
    par, gold = _par(10, M, N, tau=1.2)
    hat = {k: par[k] for k in ("theta", "a", "bfree", "lam", "tau")}
    B = 4
    refits = {"a": np.stack([1.1 * hat["a"]] * B), "bfree": np.stack([1.2 * hat["bfree"]] * B),
              "lam": np.stack([hat["lam"]] * B), "theta": np.stack([hat["theta"]] * B), "tau": np.full(B, 1.0)}
    sets, diag = F.bias_corrected_sets(hat, refits, gold)
    npt.assert_allclose(sets["boot_bias_corrected"]["a"], 0.9 * hat["a"])
    npt.assert_allclose(sets["boot_bias_corrected"]["bfree"], 0.8 * hat["bfree"])
    npt.assert_allclose(sets["boot_bias_corrected"]["tau"], 1.4)
    npt.assert_allclose(sets["mean_square_rescaled"]["a"], hat["a"] / 1.1)
    npt.assert_allclose(sets["mean_square_rescaled"]["tau"], 1.44)
    npt.assert_allclose(diag["refit_inflation_of_mean_sq"]["a"], 1.21)
    for p in sets.values():                         # valid parameter dicts: b_gold = 0, pi on the simplex
        assert np.all(p["b"][np.arange(N), gold] == 0)
        npt.assert_allclose(np.exp(p["logpi"]).sum(axis=1), 1.0)
        F.simulate_choices(p, gold, None, rng, unp_mask=np.zeros((M, N, 4), bool))


def test_power_dataset_planted_term_only_on_stem_and_null_without_it(monkeypatch):
    M, N = 6, 16
    par, gold = _par(11, M, N)
    stem = np.zeros(N, bool)
    stem[:6] = True
    mask = _clustered_mask(np.random.default_rng(12), M, N, frac_q=0.1)
    calls = _spy_simulate(monkeypatch)
    out = F.power_dataset(par, gold, stem, mask, 1.5, np.random.default_rng(13), n_null=F.MIN_NULL_DRAWS,
                          twopl_scores=())
    first, rest = calls[0], calls[1:]
    eg = first["extra_gold"]
    assert eg.shape == (M, N) and np.all(eg[:, ~stem] == 0) and np.all(eg[:, stem] != 0)
    npt.assert_allclose(eg[:, stem] / eg[:, stem][:, :1], 1.0)          # theta2_m * s: same on every STEM column
    assert len(rest) >= F.MIN_NULL_DRAWS and all(c.get("extra_gold") is None for c in rest)
    assert all(c["unp_mask"] is mask for c in calls)
    assert set(out["sp"]) == set(F.POWER_SCORES) and out["s"] == 1.5


def test_power_summary_primary_column_and_calibrated_threshold():
    def rec(s, p2, r2, valid, ins=False):
        d = ({"insufficient_draws": True, "rank2_above_p99": False, "valid": False, "ranks_above_p99": [],
              "observed_top_eigs": [9, 5, 3, 2], "n_draws": 3} if ins else
             {"insufficient_draws": False, "rank2_above_p99": r2, "valid": valid, "ranks_above_p99": [2] if r2 else [],
              "observed_top_eigs": [9, 5, 3, 2], "p99": [10, 4, 3, 2], "pct_rank1to4": [50, p2, 50, 50],
              "z_rank1to4": [0, p2 / 10, 0, 0]})
        return {"s": s, "mean_pair_r_rot0": {"stem": 0.1, "non_stem": 0.05}, "obs_top_eigs": {},
                "sp_fit": {"converged": True, "tau_mean": 1.0}, "sp": {"ge2": d}, "2pl": {"ge2": d}}
    raw = {"results": [rec(0.0, 99.9, True, True), rec(0.0, 99.5, True, False), rec(0.0, 98.0, False, True),
                       rec(0.0, 0, False, False, ins=True)]}
    base = {"a": np.ones(10), "tau": 1.0}
    subj = np.array(["abstract_algebra"] * 5 + ["philosophy"] * 5)
    out = F.power_summary(raw, base, subj, {"ge2": (99.6, 9.97)})["0"]
    r = out["by_null"]["sp:ge2"]
    assert r["detect_rank2_and_valid"] == 0.25 and r["detect_rank2"] == 0.5 and r["n_insufficient_draws"] == 1
    assert r["detect_rank2_calibrated"] == 0.25 and r["detect_rank2_calibrated_z"] == 0.25
    assert "detect_rank2_calibrated" not in out["by_null"]["2pl:ge2"]
    assert "planted_var_share_stem_mean" in out and "latent_share_of_gold_utility_variance_stem" not in out
    assert F.power_summary(raw, base, subj)["0"]["by_null"]["sp:ge2"]["detect_rank2_and_valid"] == 0.25


def _fake_fit(obj, ll, restarts=(), starts=(), cov=0.5):
    pp = {k: {"coverage_98": cov} for k in ("acc_model", "acc_model_rot", "share_model", "hist_model",
                                            "acc_question_rot", "n_all4_question", "n_none_question")}
    pp["n_cols_by_score"] = {s: {"observed_pct": 50.0} for s in F.SCORES}
    return {"fit": {"objective_penalized": obj, "loglik_total": ll}, "restarts": [{"objective": o} for o in restarts],
            "starts": list(starts), "ppc": pp}


def test_primary_rule_is_computed_from_the_fits():
    fits = {F.fit_key("full", True): _fake_fit(-100.0, -90.0, restarts=(-100.5, -100.0), cov=0.79),
            F.fit_key("full", False): _fake_fit(-200.0, -150.0, cov=0.37),
            F.fit_key("full", "model"): _fake_fit(-80.0, -70.0, restarts=(-38.0, -90.0),
                                                  starts=[{"max_tau": 1.0}, {"max_tau": 1e5}])}
    r = F.primary_rule(fits)
    assert r["restart_max_abs_objective_diff"] == 0.5 and r["loglik_gain_over_tau0"] == 60.0
    assert r["taum_best_restart_minus_fit"] == 42.0 and r["taum_starts_tau_gt_1e3"] == 1
    assert "0.5" in r["rule"] and "60.0" in r["rule"] and "+42.0" in r["rule"] and "within 1e-8" not in r["rule"]


def test_replicate_note_is_computed():
    ab = F.replicate_ab_difference()
    A, B = F.V3.load_rep("A"), F.V3.load_rep("B")
    chA, chB = (np.load(F.V3.REPS[r])["chosen_option"] for r in "AB")
    assert ab["n_cells_correctness_differs"] == int((A["CR"] != B["CR"]).sum()) and ab["n_cells"] == A["CR"].size
    assert ab["n_cells_choice_differs"] == int((chA != chB).sum())
    assert sum(ab["choice_differs_by_model"].values()) == ab["n_cells_choice_differs"]


def test_stream_keys_positive_and_distinct():
    with pytest.raises(ValueError):
        F.rng_for(0)
    with pytest.raises(ValueError):
        F.rng_for(1, 0)
    assert np.random.default_rng([F.SEED_F, 9, 0]).random() == np.random.default_rng([F.SEED_F, 9]).random()
    keys = [j[7] for j in F.sp_jobs()] + [F.taum_key(j[7]) for j in F.sp_jobs() if j[1] == "model"]
    keys += [F.cal_key(*j) for j in F.cal_jobs()] + [F.power_key(*j) for j in F.power_jobs()]
    keys += [F.calib_key(j) for j in F.calib_jobs()] + [F.calib_bc_key(*j) for j in F.calib_bc_jobs()]
    keys += [(F.STREAM["char"],)]
    assert all(all(isinstance(k, int) and k > 0 for k in key) for key in keys)
    assert len(set(keys)) == len(keys)
    first = [F.rng_for(*k).random() for k in keys]
    assert len(set(first)) == len(first)


def test_n_proc_at_most_8_and_no_dead_n_mc():
    assert F.N_PROC <= 8
    assert "n_mc" not in inspect.signature(F._p_correct).parameters
