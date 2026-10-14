"""Tests for v3lib.py and v2lib.label_eta_sq.

Oracles: the 2PL code in null_models at pi = 0, brute-force loops, central finite differences, Monte Carlo.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import numpy.testing as npt
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy.special import expit, logsumexp
from scipy.stats import norm

import v3lib
from null_models import _grid, _twopl_terms, fit_twopl_mml, twopl_sample
from v2lib import label_eta_sq
from v3lib import (_ba_terms, _guess_terms, _ml_theta, _prep, fit_guess_mml, fit_twopl_bock_aitkin,
                   guess_sample, holm, kg_sample_rotations, letter_groups_binary, letter_groups_rotations,
                   letter_preference, qap_family_test, same_minus_cross)

FAST = settings(max_examples=1000, deadline=None)


def _small_problem(seed: int, M: int = 7, N: int = 11):
    rng = np.random.default_rng(seed)
    C = (rng.random((M, N)) < 0.55).astype(np.int8)
    gold = rng.integers(0, 4, N)
    params = np.concatenate([rng.normal(size=M), [0.2, np.log(0.6), -0.1, np.log(1.2)]])
    nodes, logw = _grid(5, 9, 3.0, 3.0)
    return rng, C, gold, params, nodes, logw


def test_guess_terms_equal_twopl_at_pi_zero():
    for seed in range(5):
        rng, C, gold, params, nodes, logw = _small_problem(seed)
        ll0, g0, lp0 = _twopl_terms(params, C.T.astype(float), nodes, logw)
        # one group, pi = 0
        St, Ft, lpi, l1mpi = _prep(C[None].astype(float), np.ones((1,) + C.shape), np.zeros((C.shape[0], 1)))
        ll1, g1, lp1 = _guess_terms(params, St, Ft, lpi, l1mpi, nodes, logw)
        npt.assert_allclose(ll1, ll0, rtol=1e-12)
        npt.assert_allclose(g1, g0, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(lp1, lp0, rtol=1e-10, atol=1e-10)
        # four letter groups, pi = 0: identical likelihood
        S, T = letter_groups_binary(C, gold)
        St, Ft, lpi, l1mpi = _prep(S, T, np.zeros((C.shape[0], 4)))
        ll2, g2, _ = _guess_terms(params, St, Ft, lpi, l1mpi, nodes, logw)
        npt.assert_allclose(ll2, ll0, rtol=1e-12)
        npt.assert_allclose(g2, g0, rtol=1e-10, atol=1e-10)


def _brute_loglik(params, S, T, pi, nodes, logw):
    """log L = sum_i log sum_q w_q prod_{g,m} P^S (1-P)^(T-S), P = pi + (1-pi) sigma(a(theta-b))."""
    L, M, N = S.shape
    theta = params[:M]
    mu_la, lsd_la, mu_b, lsd_b = params[M:]
    total = 0.0
    for i in range(N):
        terms = []
        for q in range(nodes.shape[0]):
            a = np.exp(mu_la + np.exp(lsd_la) * nodes[q, 0])
            b = mu_b + np.exp(lsd_b) * nodes[q, 1]
            lq = logw[q]
            for g in range(L):
                for m in range(M):
                    eta = a * (theta[m] - b)
                    P = pi[m, g] + (1 - pi[m, g]) * expit(eta)
                    one_minus_P = (1 - pi[m, g]) * expit(-eta)      # exact form; 1 - P rounds to 0
                    if S[g, m, i] > 0:
                        lq += S[g, m, i] * np.log(P)
                    if T[g, m, i] - S[g, m, i] > 0:
                        lq += (T[g, m, i] - S[g, m, i]) * np.log(one_minus_P)
            terms.append(lq)
        total += logsumexp(terms)
    return total


def test_guess_terms_brute_force_and_finite_differences():
    for seed in range(3):
        rng, C, gold, params, nodes, logw = _small_problem(seed, M=4, N=5)
        CR = (rng.random((4, 5, 4)) < 0.6).astype(np.int8)
        S, T = letter_groups_rotations(CR, gold)
        pi = rng.uniform(0.0, 0.5, (4, 4))
        pi[0, 1] = 0.0                                                        # exact zero is allowed
        St, Ft, lpi, l1mpi = _prep(S, T, pi)
        ll, g, _ = _guess_terms(params, St, Ft, lpi, l1mpi, nodes, logw)
        npt.assert_allclose(ll, _brute_loglik(params, S, T, pi, nodes, logw), rtol=1e-11)
        # central FD, x64; step 1e-6 -> truncation ~1e-12, rounding ~1e-10 relative
        h = 1e-6
        fd = np.array([(_guess_terms(params + h * e, St, Ft, lpi, l1mpi, nodes, logw)[0]
                        - _guess_terms(params - h * e, St, Ft, lpi, l1mpi, nodes, logw)[0]) / (2 * h)
                       for e in np.eye(params.size)])
        npt.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_fit_guess_mml_equals_fit_twopl_mml_at_pi_zero():
    rng = np.random.default_rng(3)
    M, N = 12, 150
    theta = rng.normal(size=M)
    a = np.exp(rng.normal(0.0, 0.4, N))
    b = rng.normal(0.0, 1.0, N)
    C = (rng.random((M, N)) < expit(a * (theta[:, None] - b))).astype(np.int8)
    grid = dict(n_a=15, n_b=61, hw_a=4.0, hw_b=4.0)
    f0 = fit_twopl_mml(C, **grid)
    S, T = letter_groups_binary(C, rng.integers(0, 4, N))
    f1 = fit_guess_mml(S, T, np.zeros((M, 4)), **grid)
    npt.assert_allclose(f1.loglik, f0.loglik, rtol=1e-8)
    npt.assert_allclose(f1.theta, f0.theta, atol=2e-4)
    npt.assert_allclose(f1.hyper, f0.hyper, atol=2e-4)


def test_fit_guess_mml_recovers_3pl():
    """Recovery: 3PL data (c = 0.25), fitted with pi = 0.25, must recover theta (r > 0.97)."""
    rng = np.random.default_rng(5)
    M, N = 24, 600
    theta = rng.normal(size=M)
    a = np.exp(rng.normal(0.3, 0.3, N))
    b = rng.normal(0.0, 1.0, N)
    P = 0.25 + 0.75 * expit(a * (theta[:, None] - b))
    C = (rng.random((M, N)) < P).astype(np.int8)
    S, T = letter_groups_binary(C, np.zeros(N, dtype=int))
    f = fit_guess_mml(S, T, np.full((M, 4), 0.25), n_a=21, n_b=121, hw_a=4.0, hw_b=4.0)
    assert np.corrcoef(f.theta, theta)[0, 1] > 0.97
    assert abs(f.hyper[1] - 0.3) < 0.15                                       # sd of log a


def test_guess_sample_equals_twopl_sample_at_zero():
    rng = np.random.default_rng(7)
    M, N = 10, 60
    C = (rng.random((M, N)) < 0.6).astype(np.int8)
    f = fit_twopl_mml(C, n_a=9, n_b=41, hw_a=3.0, hw_b=3.0)
    for cond in (False, True):
        for mode in ("posterior", "prior"):
            A = twopl_sample(f, np.random.default_rng(11), mode=mode, condition_on_split=cond)
            B = guess_sample(f, np.zeros((M, N)), np.random.default_rng(11), mode=mode, condition_on_split=cond)
            npt.assert_array_equal(A, B)


def test_guess_sample_marginal_rate():
    """With c = 1 every cell is correct; with c = 0.5 the rate is 0.5 + 0.5 sigma (MC, 6 SD)."""
    rng = np.random.default_rng(8)
    M, N = 6, 40
    C = (rng.random((M, N)) < 0.6).astype(np.int8)
    f = fit_twopl_mml(C, n_a=9, n_b=41, hw_a=3.0, hw_b=3.0)
    assert guess_sample(f, np.full((M, N), 0.999999999), rng).mean() > 0.999
    # prior mode with fixed item params is not available; check rate of mean
    draws = np.mean([guess_sample(f, np.full((M, N), 0.5), rng).mean() for _ in range(400)])
    base = np.mean([twopl_sample(f, rng).mean() for _ in range(400)])
    sd = np.sqrt(0.25 / (M * N * 400)) * 2
    assert abs(draws - (0.5 + 0.5 * base)) < 6 * sd


@FAST
@given(arrays(np.int8, (3, 5, 4), elements=st.integers(0, 1)), arrays(np.int64, (5,), elements=st.integers(0, 3)))
def test_letter_groups_rotations_brute_force(CR, gold):
    S, T = letter_groups_rotations(CR, gold)
    npt.assert_array_equal(T, 1.0)
    for m in range(3):
        for i in range(5):
            for r in range(4):
                assert S[(gold[i] + r) % 4, m, i] == CR[m, i, r]
    npt.assert_array_equal(S.sum(axis=0), CR.sum(axis=2))


@FAST
@given(arrays(np.int8, (3, 6), elements=st.integers(0, 1)), arrays(np.int64, (3, 6), elements=st.integers(0, 3)))
def test_letter_groups_binary_brute_force(C, G):
    S, T = letter_groups_binary(C, G)
    npt.assert_array_equal(T.sum(axis=0), 1.0)
    npt.assert_array_equal(S.sum(axis=0), C)
    for m in range(3):
        for i in range(6):
            assert T[G[m, i], m, i] == 1 and S[G[m, i], m, i] == C[m, i]


def test_letter_preference_exact_on_expected_shares():
    """Feed the estimator data whose shares equal the model's expectations exactly: recovers pi and kbar."""
    M, N = 3, 400
    pi_true = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.6], [0.25, 0.25, 0.25, 0.25]])
    pi_true[1] *= 0.9 / pi_true[1].sum()                                     # 10% unparseable for model 1
    k = np.array([0.5, 0.3, 0.0])
    # Construct expected-share tensors by exact enumeration: N questions, gold uniform; use fractional cells.
    gold = np.arange(N) % 4
    CR = np.zeros((M, N, 4))
    letters = np.zeros((M, N, 4, 5))
    for r in range(4):
        shown = (gold + r) % 4
        for m in range(M):
            p_guess_right = pi_true[m, shown]
            CR[m, :, r] = k[m] + (1 - k[m]) * p_guess_right
            for L in range(4):
                letters[m, :, r, L] = (1 - k[m]) * pi_true[m, L] + k[m] * (shown == L)
            letters[m, :, r, 4] = (1 - k[m]) * (1 - pi_true[m].sum())
    # The estimator uses means of indicators; emulate with a weighted "chosen" by passing shares directly.
    acc = CR.reshape(M, -1).mean(1)
    share = letters[..., :4].reshape(M, -1, 4).mean(1)
    kbar = (acc - share.sum(1) / 4) / 0.75
    npt.assert_allclose(kbar, k, atol=1e-12)
    pi_hat = (share - kbar[:, None] / 4) / (1 - kbar[:, None])
    npt.assert_allclose(pi_hat, pi_true, atol=1e-12)


def test_letter_preference_and_kg_sampler_monte_carlo():
    """Simulate know-or-guess with kg_sample_rotations; letter_preference recovers pi (MC tolerance)."""
    rng = np.random.default_rng(12)
    M, N = 4, 4000
    C = (rng.random((M, 50)) < 0.6).astype(np.int8)
    f = fit_twopl_mml(C, n_a=9, n_b=41, hw_a=3.0, hw_b=3.0)
    pi = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.6], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.0]])
    gold = rng.integers(0, 4, N)
    f.log_post = np.zeros((N, f.log_post.shape[1])) + f.log_post[:1]          # same posterior for all
    f._cdf = None
    rng2 = np.random.default_rng(13)
    CR = kg_sample_rotations(f, pi, gold, rng2, mode="posterior")
    # Reconstruct chosen letters consistently: re-simulate with the same stream to get guesses.
    rng3 = np.random.default_rng(13)
    a, b = v3lib.draw_item_params(f, np.arange(N), rng3, "posterior")
    know = rng3.random((M, N)) < expit(a[None, :] * (f.theta[:, None] - b[None, :]))
    u = rng3.random((M, N, 4))
    guess = (u[..., None] >= np.cumsum(pi, 1)[:, None, None, :]).sum(-1)
    shown = (gold[None, :, None] + np.arange(4)[None, None, :]) % 4
    npt.assert_array_equal(CR, (know[..., None] | (guess == shown)).astype(np.int8))
    chosen = np.where(know[..., None], shown, np.where(guess == 4, -1, guess))
    est = letter_preference(CR, chosen)
    se = np.sqrt(0.25 / (N * 4))
    npt.assert_allclose(est["pi_letters"], pi, atol=12 * se / (1 - est["kbar"].max()))
    # the know indicator is shared across rotations: P(all 4 correct) = p + (1-p) prod_L pi(L)
    p = expit(a[None, :] * (f.theta[:, None] - b[None, :]))
    exp_all4 = (p + (1 - p) * np.prod(pi, axis=1)[:, None]).mean(1)
    npt.assert_allclose(CR.all(axis=2).mean(1), exp_all4, atol=6 * np.sqrt(0.25 / N))


def test_ba_terms_brute_force_and_fd():
    rng = np.random.default_rng(21)
    M, N = 6, 5
    X = (rng.random((M, N)) < 0.5).astype(float)
    nodes = np.linspace(-3, 3, 7)
    logw = norm.logpdf(nodes)
    logw -= logsumexp(logw)
    params = np.concatenate([rng.uniform(0.3, 2.0, N), rng.normal(size=N)])
    ll, g = _ba_terms(params, X, nodes, logw)
    a, d = params[:N], params[N:]
    brute = 0.0
    for m in range(M):
        t = [logw[k] + sum(X[m, i] * np.log(expit(a[i] * nodes[k] + d[i]))
                           + (1 - X[m, i]) * np.log(expit(-(a[i] * nodes[k] + d[i]))) for i in range(N))
             for k in range(nodes.size)]
        brute += logsumexp(t)
    npt.assert_allclose(ll, brute, rtol=1e-12)
    h = 1e-6
    fd = np.array([(_ba_terms(params + h * e, X, nodes, logw)[0] - _ba_terms(params - h * e, X, nodes, logw)[0]) / (2 * h)
                   for e in np.eye(2 * N)])
    npt.assert_allclose(g, fd, rtol=1e-6, atol=1e-7)


def test_ba_fit_recovers_items_and_respects_bounds():
    rng = np.random.default_rng(22)
    M, N = 600, 20
    theta = rng.normal(size=M)
    a = rng.uniform(0.6, 2.0, N)
    d = rng.normal(0, 1, N)
    X = (rng.random((M, N)) < expit(theta[:, None] * a + d)).astype(np.int8)
    f = fit_twopl_bock_aitkin(X, a_max=4.0)
    assert np.all(f.a <= 4.0 + 1e-12) and np.all(f.a >= 1e-3 - 1e-12)
    assert np.corrcoef(f.a, a)[0, 1] > 0.8 and np.corrcoef(f.d, d)[0, 1] > 0.95
    assert np.corrcoef(f.theta, theta)[0, 1] > 0.85


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000))
def test_ml_theta_is_grid_argmax(seed):
    rng = np.random.default_rng(seed)
    N = 30
    a = rng.uniform(0.2, 3.0, N)
    d = rng.normal(0, 1.5, N)
    x = (rng.random(N) < 0.5).astype(float)
    if x.sum() in (0, N):
        return
    grid = np.linspace(-6, 6, 120001)
    eta = grid[:, None] * a + d
    ll = (x * np.log(expit(eta)) + (1 - x) * np.log(expit(-eta))).sum(1)
    t = _ml_theta(x, a, d, 0.0)
    assert abs(t - np.clip(grid[np.argmax(ll)], -6, 6)) < 2e-4


@FAST
@given(arrays(np.float64, (6, 6), elements=st.floats(-1, 1)),
       arrays(np.int64, (6,), elements=st.integers(0, 2)))
def test_same_minus_cross_naive(A, fam):
    R = (A + A.T) / 2
    same, cross = [], []
    for i in range(6):
        for j in range(i + 1, 6):
            (same if fam[i] == fam[j] else cross).append(R[i, j])
    if not same or not cross:
        return
    npt.assert_allclose(same_minus_cross(R, fam), np.mean(same) - np.mean(cross), rtol=1e-12, atol=1e-12)


def test_qap_calibrated_under_null():
    """Rows exchangeable, labels arbitrary: p-values are ~uniform (KS-free check: rejection rate at 5%)."""
    rng = np.random.default_rng(31)
    fam = np.array([0] * 4 + [1] * 4 + [2] * 3 + [3, 4, 5])
    ps = []
    for _ in range(300):
        C = (rng.random((fam.size, 80)) < 0.5).astype(np.int8)
        ps.append(qap_family_test(C, fam, 199, rng)["perm_p_one_sided"])
    rate = np.mean(np.array(ps) <= 0.05)
    assert abs(rate - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 300) + 0.005


def _holm_naive(p):
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    for j, idx in enumerate(order):
        adj[idx] = min(1.0, max((m - k) * p[order[k]] for k in range(j + 1)))
    return adj


@FAST
@given(arrays(np.float64, st.integers(1, 8), elements=st.floats(0, 1)))
def test_holm_matches_naive(p):
    d = {f"t{i}": float(v) for i, v in enumerate(p)}
    out = holm(d)
    npt.assert_allclose([out[f"t{i}"] for i in range(p.size)], _holm_naive(p), rtol=0, atol=1e-15)


def test_label_eta_sq_properties():
    rng = np.random.default_rng(41)
    lab = np.repeat(np.arange(5), 20)
    v = lab.astype(float)                                                     # perfectly separated
    r = label_eta_sq(v, lab, 500, rng)
    npt.assert_allclose(r["eta_sq"], 1.0, rtol=1e-12)
    assert r["perm_p"] <= 2 / 501
    null_means = [label_eta_sq(rng.normal(size=100), lab, 200, rng)["null_mean"] for _ in range(5)]
    npt.assert_allclose(np.mean(null_means), 4 / 99, rtol=0.1)                # (G-1)/(n-1)


def test_indep_and_conditional_samplers():
    rng = np.random.default_rng(51)
    M, N = 5, 30
    C = (rng.random((M, N)) < 0.6).astype(np.int8)
    f = fit_twopl_mml(C, n_a=9, n_b=41, hw_a=3.0, hw_b=3.0)
    idx = np.arange(N)
    T = v3lib.indep_sample_rotations(f, np.random.default_rng(1), idx)
    assert T.shape == (M, N, 4)
    # same RNG stream: rotation 0 of the first draw uses the same (a, b) as twopl_sample's draw
    rates = np.mean([v3lib.indep_sample_rotations(f, rng, idx).mean() for _ in range(300)])
    base = np.mean([twopl_sample(f, rng).mean() for _ in range(300)])
    assert abs(rates - base) < 6 * np.sqrt(0.25 / (M * N * 300))
    X = v3lib.sample_matrix_conditional(lambda i: v3lib.indep_sample_rotations(f, rng, i),
                                        lambda T: T.mean(axis=2), np.ones(N, bool))
    assert np.all(X.std(axis=0) > 0)
    keep = np.zeros(N, bool)
    keep[:7] = True
    X = v3lib.kg_sample_matrix(f, np.full((M, 4), 0.2), rng.integers(0, 4, N), lambda T: T.all(axis=2), keep, rng)
    assert X.shape == (M, 7) and np.all(X.std(axis=0) > 0)


@FAST
@given(st.integers(0, 10_000), st.integers(1, 3))
def test_follow_stats_identifies_letter_and_content_directions(seed, r):
    """A loading vector that is a function of the shown letter gives r_shown = 1; of the original letter, r_orig = 1."""
    from v3_rotation_letters import follow_stats
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 4, 200)
    if np.unique(g).size < 4:
        return
    prof = rng.normal(size=4)
    if np.ptp(prof) < 1e-3:
        return
    v0 = prof[g] + 0.0                                   # rotation 0: shown letter = original letter
    rs, ro, _ = follow_stats(v0, prof[(g + r) % 4], g, r)
    npt.assert_allclose(rs, 1.0, atol=1e-10)
    rs2, ro2, rd2 = follow_stats(v0, v0.copy(), g, r)
    npt.assert_allclose([ro2, rd2], [1.0, 1.0], atol=1e-10)
