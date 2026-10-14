"""Tests for src/null_models.py against brute-force enumeration, independent routines, central finite
differences, exact moment identities, and parameter recovery."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, settings, strategies as st
from scipy.special import expit
from scipy.stats import chisquare, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import null_models as nm  # noqa: E402

FD_STEP = 1e-6


def central_fd(fun, x: np.ndarray, h: float = FD_STEP) -> np.ndarray:
    g = np.empty_like(x)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        g[j] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n_rows=st.integers(3, 30), n_cols=st.integers(3, 30),
       p=st.floats(0.1, 0.9))
def test_corr_eigs_matches_corrcoef(seed, n_rows, n_cols, p):
    """Nonzero spectrum of Z^T Z / n equals that of np.corrcoef on non-constant columns."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n_rows, n_cols)) < p).astype(float)
    X = X[:, nm.split_mask(X)]
    if X.shape[1] < 2:
        return
    ours = nm.corr_eigs(X)
    ref = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1][: ours.size]
    # f64, both O(1)-conditioned symmetric eigenproblems: rounding ~ n * eps * lambda_max
    npt.assert_allclose(ours, ref, rtol=0, atol=1e-10 * max(1.0, ref[0]))


def test_mp_edge_and_tw_z_known_values():
    npt.assert_allclose(nm.mp_edge(121, 100), (1 + np.sqrt(1.21)) ** 2, rtol=1e-15)
    lam_plus = nm.mp_edge(400, 100)
    assert nm.tw_z(lam_plus, 100, 400) == 0.0


def test_effective_rank_matches_d_across_null():
    from d_across_null import shannon_effective_rank
    rng = np.random.default_rng(3)
    C = (rng.random((24, 500)) < rng.random(500)).astype(np.int8)
    ours = nm.shannon_effective_rank_rows(C[:, nm.split_mask(C)].astype(float))
    npt.assert_allclose(ours, shannon_effective_rank(C), rtol=1e-9)  # NOTE: reference adds 1e-12 inside log


def test_effective_rank_known_values():
    # Orthogonal equal-variance directions: exp(H) = number of directions.
    X = np.kron(np.eye(4), np.ones((1, 3)))          # (4, 12)
    X = np.vstack([X, -X])                           # centered rows span 4 equal directions
    npt.assert_allclose(nm.shannon_effective_rank_rows(X), 4.0, rtol=1e-12)


@settings(max_examples=1000, deadline=None)
@given(eps=st.lists(st.floats(0.05, 20.0), min_size=1, max_size=8))
def test_elementary_symmetric_brute_force(eps):
    eps = np.array(eps)
    g = nm.elementary_symmetric(eps)
    ref = [sum(np.prod(eps[list(S)]) for S in itertools.combinations(range(eps.size), s))
           for s in range(eps.size + 1)]
    npt.assert_allclose(g, ref, rtol=1e-12)


def test_cml_gradient_fd():
    rng = np.random.default_rng(0)
    M = 7
    C = (rng.random((M, 300)) < 0.6).astype(int)
    r = C.sum(1).astype(float)
    n_s = np.bincount(C.sum(0), minlength=M + 1).astype(float)
    theta = rng.normal(size=M)
    _, g = nm._rasch_cml_negloglik(theta, r, n_s)
    g_fd = central_fd(lambda t: nm._rasch_cml_negloglik(t, r, n_s)[0], theta)
    # NOTE: FD truncation O(h^2 f''') and cancellation O(eps |f| / h) ~ 1e-7 relative
    npt.assert_allclose(g, g_fd, rtol=1e-6, atol=1e-5)


def test_cml_recovers_theta_and_ignores_extreme_columns():
    rng = np.random.default_rng(1)
    M, N = 8, 6000
    theta = np.linspace(-1.5, 1.5, M)
    b = rng.normal(0, 1.2, N)
    C = (rng.random((M, N)) < expit(theta[:, None] - b[None, :])).astype(int)
    th = nm.rasch_cml_rows(C)
    npt.assert_allclose(th, theta - theta.mean(), atol=0.08)  # SE ~ 0.03 at N=6000
    # Constant columns carry no conditional information: exact invariance.
    C2 = np.hstack([C, np.zeros((M, 50), int), np.ones((M, 30), int)])
    npt.assert_allclose(nm.rasch_cml_rows(C2), th, atol=1e-6)


@settings(max_examples=1000, deadline=None)
@given(offsets=st.lists(st.floats(-8, 8), min_size=2, max_size=40), frac=st.floats(0.02, 0.98))
def test_solve_logistic_location(offsets, frac):
    o = np.array(offsets)
    t = np.array([frac * o.size])
    x = nm.solve_logistic_location(o, t)
    npt.assert_allclose(expit(x[0] - o).sum(), t[0], rtol=0, atol=1e-9)


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_jml_matches_both_margins(seed):
    rng = np.random.default_rng(seed)
    M, N = 10, 200
    theta = rng.normal(0, 0.7, M)
    b = rng.normal(0, 1.0, N)
    C = (rng.random((M, N)) < expit(theta[:, None] - b[None, :])).astype(int)
    C = C[:, nm.split_mask(C)]
    th, bb, _ = nm.rasch_jml(C)
    P = nm.rasch_prob(th, bb)
    npt.assert_allclose(P.sum(1), C.sum(1), atol=1e-6)
    npt.assert_allclose(P.sum(0), C.sum(0), atol=1e-6)


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), M=st.integers(2, 8), N=st.integers(1, 15),
       trades=st.integers(0, 50))
def test_curveball_preserves_margins(seed, M, N, trades):
    rng = np.random.default_rng(seed)
    A = (rng.random((M, N)) < 0.5).astype(np.int8)
    B = nm.curveball(A, trades, rng)
    npt.assert_array_equal(B.sum(0), A.sum(0))
    npt.assert_array_equal(B.sum(1), A.sum(1))
    assert set(np.unique(B)).issubset({0, 1})


def test_curveball_uniform_on_small_margin_class():
    """All 3x3 0/1 matrices with row sums (2,1,1) and column sums (1,2,1), by enumeration."""
    rows, cols = np.array([2, 1, 1]), np.array([1, 2, 1])
    space = [m for m in (np.array(bits).reshape(3, 3) for bits in itertools.product([0, 1], repeat=9))
             if np.array_equal(m.sum(1), rows) and np.array_equal(m.sum(0), cols)]
    keys = {m.tobytes(): i for i, m in enumerate(space)}
    rng = np.random.default_rng(7)
    A = space[0].astype(np.int8)
    counts = np.zeros(len(space))
    for _ in range(30000):
        A = nm.curveball(A, 3, rng)
        counts[keys[A.astype(np.int64).tobytes()]] += 1
    # NOTE: thinned chain draws are weakly dependent; p > 1e-3 is a loose check.
    assert chisquare(counts).pvalue > 1e-3, counts


def test_sample_bernoulli_split_conditional_law():
    """Column pattern frequencies equal P(pattern) / P(split) (M = 3, enumeration)."""
    p = np.array([0.2, 0.5, 0.9])
    pats = [np.array(x) for x in itertools.product([0, 1], repeat=3) if 0 < sum(x) < 3]
    prob = np.array([np.prod(np.where(x == 1, p, 1 - p)) for x in pats])
    prob /= prob.sum()
    rng = np.random.default_rng(11)
    P = np.repeat(p[:, None], 20000, axis=1)
    S = nm.sample_bernoulli_split(P, rng)
    assert nm.split_mask(S).all()
    idx = {tuple(x): i for i, x in enumerate(pats)}
    counts = np.bincount([idx[tuple(c)] for c in S.T], minlength=len(pats))
    assert chisquare(counts, prob * counts.sum()).pvalue > 1e-3


def _tiny_2pl(seed=0, M=6, N=40):
    rng = np.random.default_rng(seed)
    C = (rng.random((M, N)) < rng.random(N)).astype(float)
    nodes, logw = nm._grid(7, 9, 3.0, 3.0)
    params = np.concatenate([rng.normal(size=M), [0.1, np.log(0.6), -0.2, np.log(1.3)]])
    return C, nodes, logw, params


def test_twopl_loglik_matches_naive():
    C, nodes, logw, params = _tiny_2pl()
    M = C.shape[0]
    ll, _, _ = nm._twopl_terms(params, C.T, nodes, logw)
    theta = params[:M]
    mu_la, sd_la, mu_b, sd_b = params[M], np.exp(params[M + 1]), params[M + 2], np.exp(params[M + 3])
    ref = 0.0
    for i in range(C.shape[1]):
        Li = 0.0
        for q in range(nodes.shape[0]):
            a = np.exp(mu_la + sd_la * nodes[q, 0])
            b = mu_b + sd_b * nodes[q, 1]
            pr = expit(a * (theta - b))
            Li += np.exp(logw[q]) * np.prod(np.where(C[:, i] == 1, pr, 1 - pr))
        ref += np.log(Li)
    npt.assert_allclose(ll, ref, rtol=1e-11)


def test_twopl_gradient_fd():
    C, nodes, logw, params = _tiny_2pl(seed=1)
    _, g, _ = nm._twopl_terms(params, C.T, nodes, logw)
    g_fd = central_fd(lambda x: nm._twopl_terms(x, C.T, nodes, logw)[0], params)
    npt.assert_allclose(g, g_fd, rtol=1e-6, atol=1e-6)


def test_twopl_recovers_abilities():
    rng = np.random.default_rng(5)
    M, N = 24, 3000
    theta = np.sort(rng.normal(size=M))
    theta = (theta - theta.mean()) / theta.std()
    a = np.exp(rng.normal(-0.3, 0.6, N))
    b = rng.normal(-1.0, 2.0, N)
    C = (rng.random((M, N)) < expit(a[None, :] * (theta[:, None] - b[None, :]))).astype(np.int8)
    fit = nm.fit_twopl_mml(C)
    assert fit.converged
    assert np.corrcoef(fit.theta, theta)[0, 1] > 0.99
    npt.assert_allclose(fit.hyper, [-0.3, 0.6, -1.0, 2.0], atol=0.25)  # sampling error at N=3000


def test_twopl_marginal_loglik_consistent_with_fit():
    """Public loglik at the standardized fit equals the stored fit loglik; scale invariance."""
    rng = np.random.default_rng(6)
    C = (rng.random((8, 200)) < rng.random(200)).astype(np.int8)
    fit = nm.fit_twopl_mml(C, n_a=21, n_b=81)
    npt.assert_allclose(nm.twopl_marginal_loglik(C, fit.theta, fit.hyper, fit.z_grid), fit.loglik, rtol=1e-12)
    # theta -> s theta + c, a -> a / s, b -> s b + c leaves the likelihood unchanged.
    s_, c_ = 1.7, -0.4
    h = fit.hyper
    h2 = np.array([h[0] - np.log(s_), h[1], s_ * h[2] + c_, s_ * h[3]])
    npt.assert_allclose(nm.twopl_marginal_loglik(C, s_ * fit.theta + c_, h2, fit.z_grid), fit.loglik, rtol=1e-11)


def test_twopl_sample_conditional_is_split():
    rng = np.random.default_rng(2)
    C = (rng.random((10, 300)) < rng.random(300)).astype(np.int8)
    fit = nm.fit_twopl_mml(C, n_a=21, n_b=81)
    S = nm.twopl_sample(fit, rng, mode="posterior", condition_on_split=True)
    assert S.shape == C.shape and nm.split_mask(S).all()


ROT_NPZ = HERE.parent / "results" / "mmlu_ensemble_rotations.npz"
needs_rot = pytest.mark.skipif(not ROT_NPZ.exists(), reason="results/mmlu_ensemble_rotations.npz not present")
# Main 2PL fit on the rotation-0 matrix (41 x 401 grid), from results/analysis/raw_item07_across_two_margin_null.json.
ROT0_LOGLIK = -5839.173115186843
ROT0_HYPER = [-0.2835106781520716, 0.7037656571545311, -0.5888458775185295, 2.7424976570065884]
ROT0_THETA = [
    -1.4432727043976172, -1.9035896300147976, 0.07695658443990955, -0.6796615291508598, 1.0850229441332058,
    0.8838824588833151, 0.5185251549839192, 0.6864999624577246, 1.3394212960893384, -0.5925286232504723,
    -0.2491578492417319, -0.44918515981708423, -0.36820476697176907, 0.5802654898350954, -1.2096415764091468,
    0.36685139727062543, 0.19776558902794936, -0.3384635134096337, 1.0089923774463518, 1.49274308116589,
    1.0220689154287241, -2.4259259992146953, -0.14441374390159484, 0.5450498446173536]


def _rot0():
    return np.load(ROT_NPZ)["correct"][:, :, 0].astype(np.int8)


@needs_rot
def test_twopl_main_fit_rot0_unchanged():
    import across_two_margin_null as A
    fit = nm.fit_twopl_mml(_rot0(), **A.TWOPL_GRID)
    assert fit.converged
    npt.assert_allclose(fit.loglik, ROT0_LOGLIK, rtol=0, atol=1e-6)
    npt.assert_allclose(fit.hyper, ROT0_HYPER, rtol=0, atol=1e-8)
    npt.assert_allclose(fit.theta, ROT0_THETA, rtol=0, atol=1e-8)


@needs_rot
def test_twopl_random_restart_recovers_from_abnormal_line_search():
    """First random restart of across_two_margin_null (seed SEED + 1): plain
    L-BFGS-B overflows exp(mu_la + sd_la z) in a trial step and stops ABNORMAL
    at log-lik -5880.41; the fit must reach the main optimum -5839.17."""
    import warnings

    import across_two_margin_null as A
    from scipy.optimize import minimize

    C = _rot0()
    theta0 = np.random.default_rng(A.SEED + 1).standard_normal(C.shape[0])
    # Precondition: the unbounded optimizer alone fails from this start.
    nodes, logw = nm._grid(**{k: A.TWOPL_GRID[k] for k in ("n_a", "n_b", "hw_a", "hw_b")})
    x0 = np.concatenate([theta0, [np.log(1.5), np.log(0.5), 0.0, np.log(1.5)]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plain = minimize(nm.twopl_negloglik, x0, args=(C.T.astype(np.float64), nodes, logw), jac=True,
                         method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-7, "ftol": 1e-13})
        fit = nm.fit_twopl_mml(C, **A.TWOPL_GRID, theta0=theta0)
    assert not plain.success and "ABNORMAL" in str(plain.message)
    assert fit.converged
    npt.assert_allclose(fit.loglik, ROT0_LOGLIK, rtol=0, atol=1e-6)
    npt.assert_allclose(fit.theta, ROT0_THETA, rtol=0, atol=1e-4)  # same optimum, different path


def test_probit_factor_margins_and_phi():
    rng = np.random.default_rng(9)
    K = 200000
    p = np.array([0.3, 0.6, 0.0, 1.0])
    rho = 0.2
    load = np.array([np.sqrt(rho), np.sqrt(rho), 0.0, 0.0])
    S = nm.probit_factor_sample(p, load, K, rng).astype(float)
    se = np.sqrt(p * (1 - p) / K)
    assert np.all(np.abs(S.mean(0) - p) <= 4 * se + 1e-12)
    from scipy.stats import multivariate_normal
    t = norm.ppf(p[:2])
    p11 = multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf(t)
    phi = (p11 - p[0] * p[1]) / np.sqrt(p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    npt.assert_allclose(np.corrcoef(S[:, 0], S[:, 1])[0, 1], phi, atol=4 / np.sqrt(K))


def test_column_shuffle_row_total_variance_identity():
    """E[var(row totals)] = K/(K-1) sum f(1-f) under independent column shuffles."""
    rng = np.random.default_rng(4)
    K, N = 20, 15
    X = (rng.random((K, N)) < rng.random(N)).astype(float)
    f = X.mean(0)
    expected = K / (K - 1) * np.sum(f * (1 - f))
    sims = np.array([nm.column_shuffle(X, rng).sum(1).var(ddof=1) for _ in range(40000)])
    assert abs(sims.mean() - expected) < 4 * sims.std() / np.sqrt(sims.size)
    npt.assert_array_equal(nm.column_shuffle(X, rng).sum(0), X.sum(0))


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), scale=st.floats(1e-3, 1e3), shift=st.floats(-10, 10))
def test_eta_squared_affine_invariant_and_bounded(seed, scale, shift):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=24)
    lab = rng.integers(0, 5, 24)
    e = nm.eta_squared(v, lab)
    assert -1e-12 <= e <= 1 + 1e-12
    npt.assert_allclose(nm.eta_squared(scale * v + shift, lab), e, rtol=1e-9, atol=1e-12)
    # Alternative formula: 1 - within SS / total SS.
    within = sum(np.sum((v[lab == g] - v[lab == g].mean()) ** 2) for g in np.unique(lab))
    npt.assert_allclose(e, 1 - within / np.sum((v - v.mean()) ** 2), rtol=1e-10)


def test_perm_pvalue_bounds():
    null = np.arange(99.0)
    assert nm.perm_pvalue_upper(1e9, null) == 1 / 100
    assert nm.perm_pvalue_upper(-1e9, null) == 1.0


# Labels of the submitted version's family_of on the 24 ensemble models.
SELECTIVE_PREDICTION_FAMILY = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "smollm", "Qwen/Qwen2.5-0.5B-Instruct": "qwen",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen", "microsoft/phi-2": "phi", "microsoft/Phi-3-mini-4k-instruct": "phi",
    "microsoft/Phi-3.5-mini-instruct": "phi", "Qwen/Qwen2.5-3B-Instruct": "qwen", "Qwen/Qwen2-7B-Instruct": "qwen",
    "Qwen/Qwen2.5-7B-Instruct": "qwen", "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral", "HuggingFaceH4/zephyr-7b-beta": "mistral",
    "allenai/OLMo-2-1124-7B-Instruct": "olmo", "01-ai/Yi-1.5-6B-Chat": "yi",
    "deepseek-ai/deepseek-llm-7b-chat": "deepseek", "ibm-granite/granite-3.1-8b-instruct": "granite",
    "mistralai/Ministral-8B-Instruct-2410": "mistral", "google/gemma-2-2b-it": "gemma",
    "meta-llama/Llama-3.1-8B-Instruct": "llama", "google/gemma-2-9b-it": "gemma",
    "mistralai/Mistral-Small-Instruct-2409": "mistral", "meta-llama/Llama-3.2-1B-Instruct": "llama",
    "meta-llama/Llama-3.2-3B-Instruct": "llama", "meta-llama/Meta-Llama-3-8B-Instruct": "llama",
}


def test_family_labels_match_selective_prediction():
    import json
    names = json.loads((HERE.parent / "results" / "provenance" / "ensemble_expanded_models.json").read_text())
    assert [nm.family_of(n) for n in names] == [SELECTIVE_PREDICTION_FAMILY[n] for n in names]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
