"""Null models and spectral helpers for the robustness analyses.

Z = columns of the 0/1 C (rows = passes or models) centered, / population SD (ddof=0, 0 -> 1); R = Z^T Z / n_rows.
MP edge lambda_+ = (1 + sqrt(n_cols / n_rows))^2; buffered count #{lambda > 1.05 lambda_+}.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import norm

MP_BUFFER = 1.05


def split_mask(C: np.ndarray) -> np.ndarray:
    """Columns with at least one 1 and at least one 0 (0 < col sum < n_rows)."""
    s = C.sum(axis=0)
    return (s > 0) & (s < C.shape[0])


def borderline_mask(C: np.ndarray, eps: float) -> np.ndarray:
    """Columns with eps < mean < 1 - eps (paper's borderline filter)."""
    f = C.mean(axis=0)
    return (f > eps) & (f < 1.0 - eps)


def mp_edge(n_cols: int, n_rows: int) -> float:
    """lambda_+ = (1 + sqrt(n_cols / n_rows))^2."""
    return float((1.0 + np.sqrt(n_cols / n_rows)) ** 2)


def standardize_columns(X: np.ndarray) -> np.ndarray:
    """Center each column; divide by its population SD (zero SD -> 1)."""
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - X.mean(axis=0, keepdims=True)) / sd


def corr_eigs(X: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of R = Z^T Z / n_rows, length min(n_rows, n_cols) (via the smaller Gram)."""
    n_rows, n_cols = X.shape
    Z = standardize_columns(X.astype(np.float64))
    G = (Z @ Z.T) / n_rows if n_rows <= n_cols else (Z.T @ Z) / n_rows
    return sla.eigvalsh(G)[::-1]


def corr_eigh_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending eigenpairs of Z Z^T / n_rows; eigenvectors live in row (model) space."""
    Z = standardize_columns(X.astype(np.float64))
    w, V = sla.eigh((Z @ Z.T) / X.shape[0])
    return w[::-1], V[:, ::-1]


def split_spectrum(C: np.ndarray) -> tuple[np.ndarray, float, int]:
    """(eigenvalues desc, lambda_+, N_split) on the split columns of C."""
    X = C[:, split_mask(C)]
    n_split = X.shape[1]
    return corr_eigs(X), mp_edge(n_split, C.shape[0]), n_split


def buffered_count(eigs: np.ndarray, lam_plus: float, buffer: float = MP_BUFFER) -> int:
    return int((eigs > buffer * lam_plus).sum())


def tw_z(lam1: float, K: int, Nb: int) -> float:
    """(lambda_max - lambda_+) / sigma_TW, sigma_TW as in tw_z_diagnostic.py."""
    gamma = Nb / K
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    sigma_tw = (1.0 + np.sqrt(gamma)) * (1.0 + 1.0 / np.sqrt(gamma)) ** (1.0 / 3.0) * K ** (-2.0 / 3.0)
    return float((lam1 - lam_plus) / sigma_tw)


def shannon_effective_rank_rows(X: np.ndarray, tol: float = 1e-10) -> float:
    """exp(entropy) of the normalized column-centered covariance spectrum (= d_across_null's version)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T if X.shape[0] <= X.shape[1] else Xc.T @ Xc
    w = sla.eigvalsh(G)
    w = w[w > tol]
    p = w / w.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def perm_pvalue_upper(obs: float, null: np.ndarray) -> float:
    """(1 + #{null >= obs}) / (1 + B), Phipson & Smyth (2010)."""
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + null.size))


def column_shuffle(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute every column independently (keeps column sums exactly)."""
    return rng.permuted(X, axis=0)


def elementary_symmetric(eps: np.ndarray) -> np.ndarray:
    """gamma_s = sum over size-s subsets S of prod_{m in S} eps_m, s = 0..len(eps) (summation algorithm)."""
    g = np.zeros(eps.size + 1)
    g[0] = 1.0
    for e in eps:
        g[1:] = g[1:] + e * g[:-1]
    return g


def _rasch_cml_negloglik(theta: np.ndarray, r: np.ndarray, n_s: np.ndarray) -> tuple[float, np.ndarray]:
    """-log L and gradient, log L = sum_m theta_m r_m - sum_s n_s log gamma_s(exp theta).

    r: (M,) row sums; n_s: (M+1,) number of columns with sum s.
    """
    M = theta.size
    shift = theta.mean()
    eps = np.exp(theta - shift)                       # (M,) shift cancels in ratios
    g = elementary_symmetric(eps)                     # (M+1,)
    # log gamma_s(exp theta) = log gamma_s(eps) + s * shift
    s_idx = np.arange(M + 1)
    log_g = np.log(g) + s_idx * shift
    nll = -(theta @ r - n_s @ log_g)
    grad_log_g = np.zeros((M + 1, M))                 # (s, m)
    for m in range(M):
        g_m = elementary_symmetric(np.delete(eps, m))  # (M,) s = 0..M-1
        grad_log_g[1:, m] = eps[m] * g_m / g[1:]
    grad = -(r - n_s @ grad_log_g)
    return float(nll), grad


def rasch_cml_rows(C: np.ndarray) -> np.ndarray:
    """CML row parameters theta (mean zero) given the column sums; consistent for n_cols -> inf at fixed n_rows."""
    M = C.shape[0]
    r = C.sum(axis=1).astype(np.float64)
    n_s = np.bincount(C.sum(axis=0).astype(int), minlength=M + 1).astype(np.float64)
    res = minimize(_rasch_cml_negloglik, np.zeros(M), args=(r, n_s), jac=True,
                   method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
    theta = res.x - res.x.mean()
    return theta


def rasch_column_ml(theta: np.ndarray, col_sums: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """ML b_i solving sum_m sigma(theta_m - b_i) = s_i by bisection; +-inf for s_i in {0, M}."""
    M = theta.size
    lo = np.full(col_sums.shape, theta.min() - 60.0)
    hi = np.full(col_sums.shape, theta.max() + 60.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        expected = expit(theta[None, :] - mid[:, None]).sum(axis=1)
        too_easy = expected > col_sums                 # b too small
        lo = np.where(too_easy, mid, lo)
        hi = np.where(too_easy, hi, mid)
    b = 0.5 * (lo + hi)
    b = np.where(col_sums <= 0, np.inf, b)
    b = np.where(col_sums >= M, -np.inf, b)
    return b


def solve_logistic_location(offsets: np.ndarray, targets: np.ndarray,
                            n_iter: int = 200) -> np.ndarray:
    """x_k with sum_j sigma(x_k - offsets_j) = t_k by bisection; needs 0 < t_k < len(offsets)."""
    lo = np.full(targets.shape, offsets.min() - 60.0)
    hi = np.full(targets.shape, offsets.max() + 60.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_big = expit(mid[:, None] - offsets[None, :]).sum(axis=1) > targets
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return 0.5 * (lo + hi)


def rasch_jml(C: np.ndarray, tol: float = 1e-9, max_iter: int = 5000) -> tuple[np.ndarray, np.ndarray, int]:
    """Joint ML Rasch fit (theta mean zero, b, iterations) matching both margins; C must have split columns.

    Not bias-corrected (JML b is biased outward at fixed M); the CML and curveball nulls bracket it.
    """
    r = C.sum(axis=1).astype(np.float64)
    s = C.sum(axis=0).astype(np.float64)
    theta = np.zeros(C.shape[0])
    b = -solve_logistic_location(-theta, s)
    for it in range(1, max_iter + 1):
        theta = solve_logistic_location(b, r)
        b_new = -solve_logistic_location(-theta, s)
        shift = b_new.mean()
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    theta, b = theta - shift, b - shift
    return theta, b, it


def sample_bernoulli_split(P: np.ndarray, rng: np.random.Generator,
                           max_redraws: int = 100000) -> np.ndarray:
    """C ~ Bernoulli(P) conditional on every column being split (constant columns are redrawn)."""
    S = (rng.random(P.shape) < P).astype(np.int8)
    for _ in range(max_redraws):
        bad = np.flatnonzero(~split_mask(S))
        if bad.size == 0:
            return S
        S[:, bad] = (rng.random((P.shape[0], bad.size)) < P[:, bad]).astype(np.int8)
    raise RuntimeError("conditional split sampling did not terminate")


def rasch_prob(theta: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(M, N) matrix sigma(theta_m - b_i); +-inf b gives exact 0/1."""
    return expit(theta[:, None] - b[None, :])


def curveball(A: np.ndarray, n_trades: int, rng: np.random.Generator) -> np.ndarray:
    """A after n_trades curveball trades: margins kept; the stationary law is uniform (Carstens 2015).

    Margins are sufficient under Rasch, so that uniform law is the exact conditional Rasch null.
    """
    B = A.astype(bool).copy()
    M = B.shape[0]
    for _ in range(n_trades):
        i, j = rng.choice(M, size=2, replace=False)
        only_i = B[i] & ~B[j]
        only_j = B[j] & ~B[i]
        pool = np.flatnonzero(only_i | only_j)
        if pool.size == 0:
            continue
        k = int(only_i.sum())
        perm = rng.permutation(pool)
        B[i, pool] = False
        B[j, pool] = False
        B[i, perm[:k]] = True
        B[j, perm[k:]] = True
    return B.astype(np.int8)


class TwoPLFit:
    """Fitted 2PL: theta (M,), hyper (mu_loga, sd_loga, mu_b, sd_b), grid, log-posterior."""

    def __init__(self, theta, hyper, z_grid, log_post, loglik, converged, n_iter):
        self.theta = theta
        self.hyper = hyper
        self.z_grid = z_grid            # (Q, 2) standard-normal grid nodes
        self.log_post = log_post        # (N, Q) log posterior weights per column
        self.loglik = loglik
        self.converged = converged
        self.n_iter = n_iter

    def node_params(self) -> tuple[np.ndarray, np.ndarray]:
        mu_la, sd_la, mu_b, sd_b = self.hyper
        a = np.exp(mu_la + sd_la * self.z_grid[:, 0])
        b = mu_b + sd_b * self.z_grid[:, 1]
        return a, b

    def posterior_cdf(self) -> np.ndarray:
        """(N, Q) cumulative posterior weights per column (cached)."""
        if getattr(self, "_cdf", None) is None:
            self._cdf = np.cumsum(np.exp(self.log_post), axis=1)
        return self._cdf

    def eap(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior means of (a_i, b_i)."""
        a, b = self.node_params()
        W = np.exp(self.log_post)
        return W @ a, W @ b


def _grid(n_a: int, n_b: int, hw_a: float, hw_b: float) -> tuple[np.ndarray, np.ndarray]:
    """(Q, 2) grid of (log-slope z, difficulty z) with normalized N(0, I) log-weights.

    b needs a finer grid: for a steep question the likelihood moves on the scale of the ability gaps.
    """
    za = np.linspace(-hw_a, hw_a, n_a)
    zb = np.linspace(-hw_b, hw_b, n_b)
    Z1, Z2 = np.meshgrid(za, zb, indexing="ij")
    nodes = np.column_stack([Z1.ravel(), Z2.ravel()])            # (Q, 2)
    logw = norm.logpdf(nodes).sum(axis=1)
    logw -= logsumexp(logw)
    return nodes, logw


def _twopl_terms(params: np.ndarray, X: np.ndarray, nodes: np.ndarray, logw: np.ndarray):
    """(log L, gradient, log-posterior) for params = [theta (M), mu_la, log sd_la, mu_b, log sd_b].

    L_i = sum_q w_q prod_m p_qm^x_mi (1 - p_qm)^(1 - x_mi), p_qm = sigma(a_q (theta_m - b_q)); X is (N, M).
    """
    M = X.shape[1]
    theta = params[:M]
    mu_la, lsd_la, mu_b, lsd_b = params[M:]
    sd_la, sd_b = np.exp(lsd_la), np.exp(lsd_b)
    a = np.exp(mu_la + sd_la * nodes[:, 0])                       # (Q,)
    b = mu_b + sd_b * nodes[:, 1]                                 # (Q,)
    eta = a[:, None] * (theta[None, :] - b[:, None])             # (Q, M)
    lp, lq = log_expit(eta), log_expit(-eta)                      # (Q, M)
    P = np.exp(lp)                                                # (Q, M)
    ell = X @ lp.T + (1.0 - X) @ lq.T                             # (N, Q)
    joint = ell + logw[None, :]
    logL = logsumexp(joint, axis=1)                               # (N,)
    log_post = joint - logL[:, None]
    Pi = np.exp(log_post)                                         # (N, Q)

    # d ell_iq / d theta_m = a_q (x_mi - p_qm)
    g_theta = X * (Pi @ a)[:, None] - Pi @ (a[:, None] * P)       # (N, M)
    g_theta = g_theta.sum(axis=0)
    # d ell_iq / d log a_q = a_q [ sum_m x_mi (theta_m - b_q) - sum_m p_qm (theta_m - b_q) ]
    s_i = X.sum(axis=1)                                           # (N,)
    xt = X @ theta                                                # (N,)
    c_q = (P * (theta[None, :] - b[:, None])).sum(axis=1)         # (Q,)
    d_loga = a[None, :] * (xt[:, None] - s_i[:, None] * b[None, :] - c_q[None, :])  # (N, Q)
    # d ell_iq / d b_q = -a_q (s_i - sum_m p_qm)
    d_b = -a[None, :] * (s_i[:, None] - P.sum(axis=1)[None, :])   # (N, Q)
    g_mu_la = np.sum(Pi * d_loga)
    g_lsd_la = np.sum(Pi * d_loga * nodes[None, :, 0]) * sd_la
    g_mu_b = np.sum(Pi * d_b)
    g_lsd_b = np.sum(Pi * d_b * nodes[None, :, 1]) * sd_b
    grad = np.concatenate([g_theta, [g_mu_la, g_lsd_la, g_mu_b, g_lsd_b]])
    return float(logL.sum()), grad, log_post


def twopl_negloglik(params, X, nodes, logw):
    ll, g, _ = _twopl_terms(params, X, nodes, logw)
    return -ll, -g


def fit_twopl_mml(C: np.ndarray, n_a: int = 41, n_b: int = 401, hw_a: float = 5.0,
                  hw_b: float = 5.0, theta0: np.ndarray | None = None) -> TwoPLFit:
    """MML fit of P(c_mi=1) = sigma(a_i (theta_m - b_i)), log a_i ~ N, b_i ~ N, on C (M, N).

    Random question effects keep a_i finite where joint ML diverges; theta is returned standardized.
    """
    X = C.T.astype(np.float64)                                    # (N, M)
    M = C.shape[0]
    nodes, logw = _grid(n_a, n_b, hw_a, hw_b)
    if theta0 is None:
        acc = np.clip(C.mean(axis=1), 1e-3, 1 - 1e-3)
        theta0 = np.log(acc / (1 - acc))
        theta0 = (theta0 - theta0.mean()) / theta0.std()
    x0 = np.concatenate([theta0, [np.log(1.5), np.log(0.5), 0.0, np.log(1.5)]])
    opts = {"maxiter": 5000, "gtol": 1e-7, "ftol": 1e-13}
    res = minimize(twopl_negloglik, x0, args=(X, nodes, logw), jac=True, method="L-BFGS-B", options=opts)
    n_iter = int(res.nit)
    if not res.success:
        # From a poor start a trial step can overflow a = exp(mu_la + sd_la z) and abort the line search.
        # Restart with box bounds on the hyperparameters; only here, because bounds change the L-BFGS-B
        # path of fits that already converge.
        bounds = [(None, None)] * M + [(-20.0, 20.0), (-10.0, np.log(10.0)),
                                       (None, None), (-10.0, np.log(100.0))]
        x1 = np.clip(res.x, [lo if lo is not None else -np.inf for lo, _ in bounds],
                     [hi if hi is not None else np.inf for _, hi in bounds])
        res = minimize(twopl_negloglik, x1, args=(X, nodes, logw), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=opts)
        n_iter += int(res.nit)
    theta = res.x[:M]
    mu_la, lsd_la, mu_b, lsd_b = res.x[M:]
    c, s = theta.mean(), theta.std()
    theta_std = (theta - c) / s
    hyper = np.array([mu_la + np.log(s), np.exp(lsd_la), (mu_b - c) / s, np.exp(lsd_b) / s])
    params_std = np.concatenate([theta_std, [hyper[0], np.log(hyper[1]), hyper[2], np.log(hyper[3])]])
    ll, _, log_post = _twopl_terms(params_std, X, nodes, logw)
    return TwoPLFit(theta_std, hyper, nodes, log_post, ll, bool(res.success), n_iter)


def twopl_marginal_loglik(C: np.ndarray, theta: np.ndarray, hyper: np.ndarray,
                          z_grid: np.ndarray) -> float:
    """Marginal log-likelihood of C (M, N) at theta and hyper = (mu_la, sd_la, mu_b, sd_b)."""
    logw = norm.logpdf(z_grid).sum(axis=1)
    logw -= logsumexp(logw)
    params = np.concatenate([theta, [hyper[0], np.log(hyper[1]), hyper[2], np.log(hyper[3])]])
    return _twopl_terms(params, C.T.astype(np.float64), z_grid, logw)[0]


def twopl_sample(fit: TwoPLFit, rng: np.random.Generator, mode: str = "posterior",
                 condition_on_split: bool = False, max_redraws: int = 100000) -> np.ndarray:
    """(M, N) draw; (a_i, b_i) from each question's posterior ("posterior") or the fitted law ("prior").

    condition_on_split redraws a column until it is split.
    """
    a_nodes, b_nodes = fit.node_params()
    N = fit.log_post.shape[0]
    cdf = fit.posterior_cdf()
    mu_la, sd_la, mu_b, sd_b = fit.hyper

    def draw(n_idx: np.ndarray) -> np.ndarray:
        if mode == "posterior":
            u = rng.random(n_idx.size) * cdf[n_idx, -1]
            q = (cdf[n_idx] < u[:, None]).sum(axis=1)
            a, b = a_nodes[q], b_nodes[q]
        elif mode == "prior":
            a = np.exp(mu_la + sd_la * rng.standard_normal(n_idx.size))
            b = mu_b + sd_b * rng.standard_normal(n_idx.size)
        else:
            raise ValueError(mode)
        P = expit(a[None, :] * (fit.theta[:, None] - b[None, :]))   # (M, n)
        return (rng.random(P.shape) < P).astype(np.int8)

    S = draw(np.arange(N))
    if not condition_on_split:
        return S
    for _ in range(max_redraws):
        bad = np.flatnonzero(~split_mask(S))
        if bad.size == 0:
            return S
        S[:, bad] = draw(bad)
    raise RuntimeError("conditional split sampling did not terminate")


def probit_factor_sample(p: np.ndarray, loading: np.ndarray, K: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(K, N) c_ki = 1[l_i u_k + sqrt(1 - l_i^2) e_ki < Phi^{-1}(p_i)], l = loading.

    P(c_ki = 1) = p_i exactly; latent correlation of two questions is l_i l_j.
    """
    thr = norm.ppf(p)                                             # (N,) +-inf at 0/1
    u = rng.standard_normal((K, 1))
    e = rng.standard_normal((K, p.size))
    Z = loading[None, :] * u + np.sqrt(1.0 - loading[None, :] ** 2) * e
    return (Z < thr[None, :]).astype(np.int8)


def family_of(model: str) -> str:
    m = model.lower()
    for key, fam in (("qwen", "qwen"), ("mistral", "mistral"), ("ministral", "mistral"),
                     ("zephyr", "mistral"), ("phi", "phi"), ("smollm", "smollm"),
                     ("olmo", "olmo"), ("yi", "yi"), ("deepseek", "deepseek"),
                     ("granite", "granite"), ("gemma", "gemma"), ("llama", "llama")):
        if key in m:
            return fam
    raise ValueError(f"unknown family for {model}")


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-group sum of squares / total sum of squares."""
    total = np.sum((values - values.mean()) ** 2)
    between = sum(np.sum(labels == g) * (values[labels == g].mean() - values.mean()) ** 2
                  for g in np.unique(labels))
    return float(between / total)
