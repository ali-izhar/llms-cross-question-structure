"""IRT fits and simulators for the v3 across-model analysis (tested in test_v3lib.py).

M models (rows), N questions, L = 4 shown letters; C (M, N) in {0, 1}.
Guessing-floor IRT: P(c_mi = 1) = pi_{m,g} + (1 - pi_{m,g}) sigma(a_i (theta_m - b_i)), g = shown gold letter
(pi = 0: 2PL; 0.25: 3PL; letter preference: know-or-guess). Also a bounded-slope Bock-Aitkin 2PL and a QAP family test.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, log_expit, logsumexp  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from null_models import TwoPLFit, _grid, split_mask  # noqa: E402

N_LETTERS = 4
THETA_MAX = 6.0


def letter_groups_binary(C: np.ndarray, shown_gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Successes and trials (S, T), each (L, M, N): T_g = 1[shown_gold = g], S_g = C T_g; shown_gold (M, N) or (N,)."""
    M, N = C.shape
    G = np.broadcast_to(shown_gold, (M, N))
    T = np.stack([(G == g) for g in range(N_LETTERS)]).astype(np.float64)    # (L, M, N)
    S = T * C[None].astype(np.float64)
    return S, T


def letter_groups_rotations(CR: np.ndarray, gold_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, T), each (L, M, N), from CR (M, N, 4); rotation r shows gold at (gold_idx + r) mod 4, so T = 1."""
    M, N, R = CR.shape
    assert R == N_LETTERS
    S = np.zeros((N_LETTERS, M, N))
    for r in range(R):
        g = (gold_idx + r) % N_LETTERS                                       # (N,)
        for L in range(N_LETTERS):
            S[L][:, g == L] += CR[:, g == L, r]
    return S, np.ones_like(S)


def _guess_terms(params: np.ndarray, S: np.ndarray, F: np.ndarray, logpi: np.ndarray,
                 log1mpi: np.ndarray, nodes: np.ndarray, logw: np.ndarray):
    """Marginal log-likelihood, gradient and log-posterior of the guessing model.

    params = [theta (M), mu_la, log sd_la, mu_b, log sd_b]; S, F (L, N, M); logpi, log1mpi (M, L).
    P = pi + (1 - pi) sigma(eta): d log P / d eta = (1 - pi) sigma (1 - sigma) / P, d log(1 - P) / d eta = -sigma.
    """
    L, N, M = S.shape
    theta = params[:M]
    mu_la, lsd_la, mu_b, lsd_b = params[M:]
    sd_la, sd_b = np.exp(lsd_la), np.exp(lsd_b)
    a = np.exp(mu_la + sd_la * nodes[:, 0])                                  # (Q,)
    b = mu_b + sd_b * nodes[:, 1]                                            # (Q,)
    dev = theta[None, :] - b[:, None]                                        # (Q, M)
    eta = a[:, None] * dev
    ls, lf = log_expit(eta), log_expit(-eta)                                  # (Q, M)
    v = np.exp(ls)                                                           # sigma
    ell = np.zeros((N, nodes.shape[0]))
    U = []
    for g in range(L):
        logP = np.logaddexp(logpi[None, :, g], log1mpi[None, :, g] + ls)     # (Q, M)
        log1mP = log1mpi[None, :, g] + lf
        ell += S[g] @ logP.T + F[g] @ log1mP.T                               # (N, Q)
        U.append(np.exp(log1mpi[None, :, g] + ls + lf - logP))               # (Q, M)
    joint = ell + logw[None, :]
    logL = logsumexp(joint, axis=1)                                          # (N,)
    log_post = joint - logL[:, None]
    Pi = np.exp(log_post)                                                    # (N, Q)

    g_theta = np.zeros(M)
    d_loga = np.zeros_like(ell)
    d_b = np.zeros_like(ell)
    Fsum = F.sum(axis=0)                                                     # (N, M)
    av = a[:, None] * v                                                      # (Q, M)
    g_theta -= np.sum(Fsum * (Pi @ av), axis=0)
    d_loga -= Fsum @ (v * dev).T
    d_b += Fsum @ v.T
    for g in range(L):
        u = U[g]
        g_theta += np.sum(S[g] * (Pi @ (a[:, None] * u)), axis=0)
        d_loga += S[g] @ (u * dev).T
        d_b -= S[g] @ u.T
    d_loga *= a[None, :]
    d_b *= a[None, :]
    g_mu_la = np.sum(Pi * d_loga)
    g_lsd_la = np.sum(Pi * d_loga * nodes[None, :, 0]) * sd_la
    g_mu_b = np.sum(Pi * d_b)
    g_lsd_b = np.sum(Pi * d_b * nodes[None, :, 1]) * sd_b
    grad = np.concatenate([g_theta, [g_mu_la, g_lsd_la, g_mu_b, g_lsd_b]])
    return float(logL.sum()), grad, log_post


def _prep(S: np.ndarray, T: np.ndarray, pi: np.ndarray):
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    assert S.shape == T.shape and S.ndim == 3 and np.all(S <= T) and np.all(S >= 0)
    pi = np.asarray(pi, dtype=np.float64)
    assert pi.shape == (S.shape[1], S.shape[0]) and np.all((pi >= 0) & (pi < 1))
    St = S.transpose(0, 2, 1).copy()                                          # (L, N, M)
    Ft = (T - S).transpose(0, 2, 1).copy()
    with np.errstate(divide="ignore"):
        logpi = np.log(pi)
    return St, Ft, logpi, np.log1p(-pi)


def guess_negloglik(params, St, Ft, logpi, log1mpi, nodes, logw):
    ll, g, _ = _guess_terms(params, St, Ft, logpi, log1mpi, nodes, logw)
    return -ll, -g


class GuessFit(TwoPLFit):
    """TwoPLFit plus the guessing floor pi (M, L) it was fitted with."""

    def __init__(self, *args, pi=None, **kw):
        super().__init__(*args, **kw)
        self.pi = pi


def fit_guess_mml(S: np.ndarray, T: np.ndarray, pi: np.ndarray, n_a: int = 41, n_b: int = 401,
                  hw_a: float = 5.0, hw_b: float = 5.0, theta0: np.ndarray | None = None) -> GuessFit:
    """Marginal ML of P = pi_{m,g} + (1 - pi_{m,g}) sigma(a_i (theta_m - b_i)); S, T (L, M, N), pi (M, L).

    Fixed theta (standardized after the fit), normal random (log a_i, b_i) on the fit_twopl_mml grid.
    With four rotations this is a composite likelihood: rotations independent given (a_i, b_i).
    """
    St, Ft, logpi, log1mpi = _prep(S, T, pi)
    M = St.shape[2]
    nodes, logw = _grid(n_a, n_b, hw_a, hw_b)
    if theta0 is None:
        acc = np.clip(St.sum((0, 1)) / (St + Ft).sum((0, 1)), 1e-3, 1 - 1e-3)
        theta0 = np.log(acc / (1 - acc))
        theta0 = (theta0 - theta0.mean()) / theta0.std()
    x0 = np.concatenate([theta0, [np.log(1.5), np.log(0.5), 0.0, np.log(1.5)]])
    res = minimize(guess_negloglik, x0, args=(St, Ft, logpi, log1mpi, nodes, logw), jac=True,
                   method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-7, "ftol": 1e-13})
    theta = res.x[:M]
    mu_la, lsd_la, mu_b, lsd_b = res.x[M:]
    c, s = theta.mean(), theta.std()
    theta_std = (theta - c) / s
    hyper = np.array([mu_la + np.log(s), np.exp(lsd_la), (mu_b - c) / s, np.exp(lsd_b) / s])
    params_std = np.concatenate([theta_std, [hyper[0], np.log(hyper[1]), hyper[2], np.log(hyper[3])]])
    ll, _, log_post = _guess_terms(params_std, St, Ft, logpi, log1mpi, nodes, logw)
    return GuessFit(theta_std, hyper, nodes, log_post, ll, bool(res.success), int(res.nit), pi=pi)


def draw_item_params(fit: TwoPLFit, n_idx: np.ndarray, rng: np.random.Generator,
                     mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) for questions n_idx from the grid posterior or the prior; same RNG use as twopl_sample."""
    if mode == "posterior":
        a_nodes, b_nodes = fit.node_params()
        cdf = fit.posterior_cdf()
        u = rng.random(n_idx.size) * cdf[n_idx, -1]
        q = (cdf[n_idx] < u[:, None]).sum(axis=1)
        return a_nodes[q], b_nodes[q]
    if mode == "prior":
        mu_la, sd_la, mu_b, sd_b = fit.hyper
        a = np.exp(mu_la + sd_la * rng.standard_normal(n_idx.size))
        b = mu_b + sd_b * rng.standard_normal(n_idx.size)
        return a, b
    raise ValueError(mode)


def guess_sample(fit: TwoPLFit, c: np.ndarray, rng: np.random.Generator, mode: str = "posterior",
                 condition_on_split: bool = False, max_redraws: int = 100000) -> np.ndarray:
    """Simulate (M, N) correctness with P = c_mi + (1 - c_mi) sigma(a_i (theta_m - b_i)).

    c = 0 reproduces null_models.twopl_sample exactly; condition_on_split redraws columns until split."""
    M = fit.theta.size
    N = fit.log_post.shape[0]
    c = np.broadcast_to(c, (M, N))

    def draw(n_idx: np.ndarray) -> np.ndarray:
        a, b = draw_item_params(fit, n_idx, rng, mode)
        P = expit(a[None, :] * (fit.theta[:, None] - b[None, :]))
        P = c[:, n_idx] + (1.0 - c[:, n_idx]) * P
        return (rng.random(P.shape) < P).astype(np.int8)

    Sm = draw(np.arange(N))
    if not condition_on_split:
        return Sm
    for _ in range(max_redraws):
        bad = np.flatnonzero(~split_mask(Sm))
        if bad.size == 0:
            return Sm
        Sm[:, bad] = draw(bad)
    raise RuntimeError("conditional split sampling did not terminate")


def letter_preference(CR: np.ndarray, chosen_letter: np.ndarray) -> dict:
    """Know-or-guess moment estimate of each model's guessing distribution pi_m(L) over shown letters.

    share_m(L) = kbar_m / 4 + (1 - kbar_m) pi_m(L) and acc_m = kbar_m + (sum_L share_m(L) - kbar_m) / 4,
    so kbar_m = (acc_m - sum_L share_m(L) / 4) / (3/4); negatives clipped, letters renormalized to 1 - pi_m(U).
    CR (M, N, 4) correctness; chosen_letter (M, N, 4) in 0..3, -1 unparseable.
    """
    M = CR.shape[0]
    acc = CR.reshape(M, -1).mean(axis=1)                                     # (M,)
    share = np.stack([(chosen_letter == L).reshape(M, -1).mean(axis=1) for L in range(N_LETTERS)], axis=1)
    s_u = (chosen_letter < 0).reshape(M, -1).mean(axis=1)
    kbar = (acc - share.sum(axis=1) / 4.0) / 0.75
    kbar = np.clip(kbar, 0.0, 1.0 - 1e-9)
    raw = (share - kbar[:, None] / 4.0) / (1.0 - kbar[:, None])
    pi_u = s_u / (1.0 - kbar)
    clipped = (raw < 0).sum(axis=1)
    pi = np.clip(raw, 0.0, None)
    tot = pi.sum(axis=1, keepdims=True)
    pi = np.where(tot > 0, pi / np.where(tot > 0, tot, 1.0) * (1.0 - pi_u[:, None]), 0.0)
    return {"pi_letters": pi, "pi_unparseable": pi_u, "kbar": kbar, "share": share,
            "n_clipped": clipped, "accuracy_all_rotations": acc}


def kg_sample_rotations(fit: TwoPLFit, pi_letters: np.ndarray, gold_idx: np.ndarray,
                        rng: np.random.Generator, mode: str = "posterior", n_idx: np.ndarray | None = None
                        ) -> np.ndarray:
    """Know-or-guess (M, n, 4) tensor: know_mi ~ Bern(sigma(a_i (theta_m - b_i))) once per cell, else guess ~ pi_m.

    Correct iff the model knows or its guess equals the shown gold letter (gold_idx_i + r) mod 4."""
    M = fit.theta.size
    if n_idx is None:
        n_idx = np.arange(fit.log_post.shape[0])
    a, b = draw_item_params(fit, n_idx, rng, mode)
    know = rng.random((M, n_idx.size)) < expit(a[None, :] * (fit.theta[:, None] - b[None, :]))
    cum = np.cumsum(pi_letters, axis=1)                                      # (M, 4)
    u = rng.random((M, n_idx.size, N_LETTERS))
    guess = (u[..., None] >= cum[:, None, None, :]).sum(axis=-1)             # 0..3, 4 = unparseable
    shown = (gold_idx[n_idx][None, :, None] + np.arange(N_LETTERS)[None, None, :]) % N_LETTERS
    return (know[..., None] | (guess == shown)).astype(np.int8)


def kg_sample_matrix(fit: TwoPLFit, pi_letters: np.ndarray, gold_idx: np.ndarray, derive,
                     keep: np.ndarray, rng: np.random.Generator, mode: str = "posterior",
                     max_redraws: int = 100000) -> np.ndarray:
    """derive((M, n, 4) know-or-guess tensor) -> (M, n) on the columns `keep`, redrawn until non-constant."""
    idx = np.flatnonzero(keep)
    X = derive(kg_sample_rotations(fit, pi_letters, gold_idx, rng, mode, idx)).astype(np.float64)
    for _ in range(max_redraws):
        bad = np.flatnonzero(X.std(axis=0) == 0)
        if bad.size == 0:
            return X
        X[:, bad] = derive(kg_sample_rotations(fit, pi_letters, gold_idx, rng, mode, idx[bad]))
    raise RuntimeError("conditional sampling did not terminate")


class BAFit:
    def __init__(self, a, d, theta_ml, theta_eap, loglik, converged, n_iter, a_max, d_max, nodes):
        self.a, self.d, self.theta, self.theta_eap = a, d, theta_ml, theta_eap
        self.loglik, self.converged, self.n_iter = loglik, converged, n_iter
        self.a_max, self.d_max, self.nodes = a_max, d_max, nodes

    @property
    def b(self):
        return -self.d / self.a

    def prob(self) -> np.ndarray:
        """(M, N) sigma(a_i theta_m + d_i) at the ML abilities."""
        return expit(self.theta[:, None] * self.a[None, :] + self.d[None, :])


def _ba_terms(params: np.ndarray, X: np.ndarray, nodes: np.ndarray, logw: np.ndarray):
    """Marginal log-likelihood over theta ~ N(0, 1) (quadrature) and its gradient in params = [a (N), d (N)].

    eta_ki = a_i theta_k + d_i; d/d d_i = sum_k (R_ki - n_k p_ki), d/d a_i = sum_k theta_k (R_ki - n_k p_ki),
    with h_mk the node posterior, R = h^T X, n_k = sum_m h_mk.
    """
    N = X.shape[1]
    a, d = params[:N], params[N:]
    eta = nodes[:, None] * a[None, :] + d[None, :]                           # (K, N)
    lp, lq = log_expit(eta), log_expit(-eta)
    ell = X @ lp.T + (1.0 - X) @ lq.T                                        # (M, K)
    joint = ell + logw[None, :]
    logL = logsumexp(joint, axis=1)
    h = np.exp(joint - logL[:, None])                                        # (M, K)
    R = h.T @ X                                                              # (K, N)
    nk = h.sum(axis=0)                                                       # (K,)
    resid = R - nk[:, None] * np.exp(lp)                                     # (K, N)
    g_d = resid.sum(axis=0)
    g_a = nodes @ resid
    return float(logL.sum()), np.concatenate([g_a, g_d])


def fit_twopl_bock_aitkin(C: np.ndarray, a_max: float = 4.0, d_max: float = 20.0,
                          n_nodes: int = 81, hw: float = 5.0, a_min: float = 1e-3) -> BAFit:
    """Bock-Aitkin marginal ML 2PL (theta ~ N(0, 1), items fixed) with a_min <= a_i <= a_max, by L-BFGS-B.

    Slopes are bounded because a question the ability order separates has an unbounded ML slope.
    Returns bounded ML abilities given the items (used for simulation) and EAP abilities.
    """
    X = C.astype(np.float64)
    M, N = X.shape
    nodes = np.linspace(-hw, hw, n_nodes)
    logw = norm.logpdf(nodes)
    logw -= logsumexp(logw)
    p = np.clip(X.mean(axis=0), 0.02, 0.98)
    x0 = np.concatenate([np.ones(N), np.log(p / (1 - p))])
    bounds = [(a_min, a_max)] * N + [(-d_max, d_max)] * N
    res = minimize(lambda x: tuple(-v for v in _ba_terms(x, X, nodes, logw)), x0, jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 20000, "gtol": 1e-6, "ftol": 1e-14})
    a, d = res.x[:N], res.x[N:]
    ll = _ba_terms(res.x, X, nodes, logw)[0]
    grid = np.linspace(-THETA_MAX, THETA_MAX, 2401)
    eta = grid[:, None] * a[None, :] + d[None, :]
    ellg = X @ log_expit(eta).T + (1.0 - X) @ log_expit(-eta).T             # (M, G)
    theta_ml0 = grid[np.argmax(ellg, axis=1)]
    theta_ml = np.array([_ml_theta(X[m], a, d, theta_ml0[m]) for m in range(M)])
    lw = norm.logpdf(grid)
    post = np.exp(ellg + lw - logsumexp(ellg + lw, axis=1, keepdims=True))
    theta_eap = post @ grid
    return BAFit(a, d, theta_ml, theta_eap, ll, bool(res.success), int(res.nit), a_max, d_max, nodes)


def _ml_theta(x: np.ndarray, a: np.ndarray, d: np.ndarray, t0: float, n_iter: int = 100) -> float:
    """Newton on the concave log-likelihood sum_i x_i log p_i + (1 - x_i) log(1 - p_i) in theta."""
    t = t0
    for _ in range(n_iter):
        p = expit(a * t + d)
        g = np.sum(a * (x - p))
        h = -np.sum(a * a * p * (1 - p))
        step = g / h
        t_new = float(np.clip(t - step, -THETA_MAX, THETA_MAX))
        if abs(t_new - t) < 1e-12:
            return t_new
        t = t_new
    return t


def same_minus_cross(R: np.ndarray, fam: np.ndarray) -> float:
    """Mean of R over same-family pairs minus mean over cross-family pairs (upper triangle)."""
    iu = np.triu_indices(R.shape[0], 1)
    same = fam[iu[0]] == fam[iu[1]]
    r = R[iu]
    return float(r[same].mean() - r[~same].mean())


def qap_family_test(C: np.ndarray, families, n_perm: int, rng: np.random.Generator) -> dict:
    """QAP test of mean row r (same family) - mean r (cross family); null permutes family labels over models.

    Unlike a pair-level test, the null keeps the dependence among pairs that share a model. One-sided p."""
    from null_models import perm_pvalue_upper
    fam = np.asarray(families)
    R = np.corrcoef(C.astype(np.float64))
    obs = same_minus_cross(R, fam)
    null = np.array([same_minus_cross(R, rng.permutation(fam)) for _ in range(n_perm)])
    iu = np.triu_indices(R.shape[0], 1)
    same = fam[iu[0]] == fam[iu[1]]
    return {"stat_mean_r_same_minus_cross": obs, "mean_r_same": float(R[iu][same].mean()),
            "mean_r_cross": float(R[iu][~same].mean()), "n_same_pairs": int(same.sum()),
            "n_cross_pairs": int((~same).sum()), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "perm_p_one_sided": perm_pvalue_upper(obs, null),
            "n_perm": n_perm}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a dict name -> p."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out, running = {}, 0.0
    for j, k in enumerate(names):
        running = max(running, min(1.0, (m - j) * pvals[k]))
        out[k] = running
    return {k: out[k] for k in pvals}


def indep_sample_rotations(fit: TwoPLFit, rng: np.random.Generator, n_idx: np.ndarray,
                           n_rot: int = N_LETTERS, mode: str = "posterior") -> np.ndarray:
    """(M, n, n_rot) correctness, rotations independent given (a_i, b_i): P = sigma(a_i (theta_m - b_i)).

    No shared 'know' state: the liberal extreme for the rotation-averaged matrix."""
    a, b = draw_item_params(fit, n_idx, rng, mode)
    P = expit(a[None, :] * (fit.theta[:, None] - b[None, :]))
    return (rng.random(P.shape + (n_rot,)) < P[..., None]).astype(np.int8)


def sample_matrix_conditional(sampler, derive, keep: np.ndarray, max_redraws: int = 100000) -> np.ndarray:
    """derive(sampler(idx)) -> (M, n) on the columns `keep`, redrawn until non-constant; sampler -> (M, n, R)."""
    idx = np.flatnonzero(keep)
    X = derive(sampler(idx)).astype(np.float64)
    for _ in range(max_redraws):
        bad = np.flatnonzero(X.std(axis=0) == 0)
        if bad.size == 0:
            return X
        X[:, bad] = derive(sampler(idx[bad]))
    raise RuntimeError("conditional sampling did not terminate")
