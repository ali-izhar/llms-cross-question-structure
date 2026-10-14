"""
Across-model null suite and rotation analyses on the 24-model, 4-rotation MMLU data (replicates A, B).

Run: python v3_across.py {B_extra | C [--rep A|B] | near}  (B_extra: extra rotation-0 nulls; C: per-rotation,
rotation-randomized and all-4/mean analyses; near: near-tie sensitivity). R = Z^T Z / M on non-constant columns;
a null rejects rank j when observed lambda_j > null p99 at rank j. Outputs: results/analysis/v3_B_extra.json,
v3_C_rotations_<rep>.json, v3_C5_near_ties.json, raw_v3_null_eigs_<tag>.npz.
"""

from __future__ import annotations

import argparse
import json
import time

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import OUT, V2, label_eta_sq, write_json

import across_two_margin_null as atmn
import d_across_null
from null_models import (MP_BUFFER, TwoPLFit, corr_eigs, curveball, eta_squared, family_of, fit_twopl_mml,
                         mp_edge, perm_pvalue_upper, rasch_jml, rasch_prob, split_mask, standardize_columns)
from v3lib import (fit_guess_mml, fit_twopl_bock_aitkin, guess_sample, indep_sample_rotations,
                   kg_sample_rotations, letter_groups_binary, letter_groups_rotations, letter_preference,
                   sample_matrix_conditional)

LETTERS = "ABCD"
N_DRAWS = 1000
N_RANKS = 4
N_RANDOM_ROT = 1000           # C2 rotation-randomized ensembles
N_ROBUST_DRAWS = 10           # C2: draws refitted with their own 2PL and letter-aware nulls
N_ROBUST_NULL = 300           # null draws per refitted draw
N_PERM_ETA = 5000
N_PERM_R = 10000
STRESS = (1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
SEED = 20261013
# Each null's child stream is its position here, so its draws do not depend on which other nulls run.
ALL_NULLS = (["rasch_jml", "fixed_margin", "2pl_post", "2pl_prior", "3pl_post", "letter_post",
              "ba_a4", "ba_a8", "kg_letter4", "kg_uniform4", "indep4_2pl", "kg_letter4_randrot",
              "bern_question", "bern_model"]
             + [f"2pl_post_stress_x{f:g}" for f in STRESS])
REPS = {"A": V2 / "mmlu_ensemble_rotations.npz", "B": V2 / "mmlu_ensemble_rotations_rep.npz"}


def load_rep(rep: str) -> dict:
    z = np.load(REPS[rep], allow_pickle=False)
    CR = z["correct"].astype(np.int8)                                         # (M, N, 4)
    gold = z["gold_idx"].astype(int)                                          # (N,)
    ch = z["chosen_option"].astype(int)
    shown_choice = np.where(ch >= 0, (ch + np.arange(4)[None, None, :]) % 4, -1)   # shown letter 0..3, -1
    codes = np.full(z["parsed"].shape, -1)
    for j, L in enumerate(LETTERS):
        codes[z["parsed"] == L] = j
    assert np.array_equal(codes, shown_choice), "parsed letter must equal the rotated chosen option"
    names = z["models"].tolist()
    if rep == "A":
        g = np.load(V2 / "mmlu_ensemble_greedy.npz")
        assert np.array_equal(g["correct"], CR[:, :, 0]) and g["models"].tolist() == names
    subj = np.array([q["subject"] for q in json.loads((V2 / "questions_mmlu.json").read_text())])
    return {"rep": rep, "CR": CR, "gold": gold, "shown_choice": shown_choice, "names": names,
            "fam": np.array([family_of(n) for n in names]), "subj": subj,
            "lp": z["letter_logprob_first_by_option"]}


def keep_mask(X: np.ndarray) -> np.ndarray:
    """Non-constant columns (binary: split columns)."""
    return X.std(axis=0) > 0


def spectrum(X: np.ndarray, n_eigs: int = 6) -> dict:
    keep = keep_mask(X)
    M = X.shape[0]
    e = corr_eigs(X[:, keep].astype(np.float64))
    lp = mp_edge(int(keep.sum()), M)
    return {"M": int(M), "n_cols": int(keep.sum()), "lambda_plus": lp, "lambda_plus_buffered": MP_BUFFER * lp,
            "top_eigs": e[:n_eigs].tolist(), "lambda2_over_lambda1": float(e[1] / e[0]),
            "mp_count_strict": int((e > lp).sum()), "mp_count_buffered": int((e > MP_BUFFER * lp).sum())}


def ratio_stats(null_eigs: np.ndarray, obs_eigs: np.ndarray) -> dict:
    """lambda_2 / lambda_1: observed vs the null distribution (upper tail)."""
    r = null_eigs[:, 1] / null_eigs[:, 0]
    o = float(obs_eigs[1] / obs_eigs[0])
    return {"observed": o, "null_mean": float(r.mean()), "null_p99": float(np.percentile(r, 99)),
            "null_max": float(r.max()), "mc_p_upper": perm_pvalue_upper(o, r), "n_draws": int(r.size)}


def summarize(name: str, e: np.ndarray, obs: np.ndarray, edge: float, notes: str) -> dict:
    atmn._M_ROWS = obs.size
    s = atmn.summarize_null(name, e, np.full(e.shape[0], edge), obs, edge, notes)
    s["lambda2_over_lambda1"] = ratio_stats(e, obs)
    s["rank_eigs_p99_1to6"] = np.percentile(e[:, :6], 99, axis=0).tolist()
    return s


def _draws(fn, n: int) -> np.ndarray:
    return np.array([fn() for _ in range(n)])


class Context:
    """Everything a null needs: the matrix, its kept columns, letters, and the 4-rotation fits."""

    def __init__(self, X: np.ndarray, CR: np.ndarray | None = None, gold: np.ndarray | None = None,
                 shown_gold: np.ndarray | None = None, pi: np.ndarray | None = None, derive=None,
                 kg_fits: dict | None = None):
        self.X = X
        self.keep = keep_mask(X)
        self.binary = set(np.unique(X).tolist()) <= {0, 1}
        self.CR, self.gold, self.shown_gold, self.pi = CR, gold, shown_gold, pi
        self.derive = derive
        self.kg_fits = kg_fits or {}
        self.obs = corr_eigs(X[:, self.keep].astype(np.float64))
        self.edge = mp_edge(int(self.keep.sum()), X.shape[0])
        self._fits = {}

    def fit_2pl(self) -> TwoPLFit:
        if "2pl" not in self._fits:
            self._fits["2pl"] = fit_twopl_mml(self.X.astype(np.int8))
        return self._fits["2pl"]


def run_null(name: str, ctx: Context, rng: np.random.Generator, n_draws: int, fit_info: dict) -> np.ndarray:
    """(n_draws, M) null eigenvalues of the kept columns for the named null."""
    X, keep, M = ctx.X, ctx.keep, ctx.X.shape[0]
    Xk = X[:, keep].astype(np.int8) if ctx.binary else None
    if name == "rasch_jml":
        th, b, it = rasch_jml(Xk)
        fit_info[name] = {"iterations": it}
        return _draws(lambda: corr_eigs(atmn.sample_bernoulli_split(rasch_prob(th, b), rng)), n_draws)
    if name == "fixed_margin":
        S = curveball(Xk, atmn.CURVEBALL_BURNIN, rng)
        out = np.empty((n_draws, M))
        for d in range(n_draws):
            S = curveball(S, atmn.CURVEBALL_THIN, rng)
            out[d] = corr_eigs(S)
        assert np.array_equal(S.sum(0), Xk.sum(0)) and np.array_equal(S.sum(1), Xk.sum(1))
        return out
    if name in ("2pl_post", "2pl_prior") or name.startswith("2pl_post_stress"):
        f = ctx.fit_2pl()
        fit_info["2pl"] = {"loglik": f.loglik, "converged": f.converged, "hyper": f.hyper.tolist(),
                           "corr_theta_vs_accuracy": float(np.corrcoef(f.theta, X.mean(1))[0, 1])}
        mode = "prior" if name == "2pl_prior" else "posterior"
        if name.startswith("2pl_post_stress"):
            factor = float(name.split("_x")[1])
            f = TwoPLFit(f.theta, f.hyper.copy(), f.z_grid, f.log_post, f.loglik, f.converged, f.n_iter)
            f.hyper[0] += np.log(factor)
        return _draws(lambda: corr_eigs(atmn.twopl_sample(f, rng, mode=mode, condition_on_split=True)[:, keep]),
                      n_draws)
    if name in ("3pl_post", "letter_post"):
        if name == "3pl_post":
            pi = np.full((M, 4), 0.25)
            S, T = letter_groups_binary(X.astype(np.int8), np.zeros(X.shape[1], dtype=int))
        else:
            pi = ctx.pi
            S, T = letter_groups_binary(X.astype(np.int8), ctx.shown_gold)
        f = fit_guess_mml(S, T, pi)
        G = np.broadcast_to(ctx.shown_gold if name == "letter_post" else 0, X.shape)
        c = np.take_along_axis(pi, G, axis=1)                                 # (M, N) guessing floor
        fit_info[name] = {"loglik": f.loglik, "converged": f.converged, "hyper": f.hyper.tolist(),
                          "corr_theta_vs_accuracy": float(np.corrcoef(f.theta, X.mean(1))[0, 1])}
        return _draws(lambda: corr_eigs(guess_sample(f, c, rng, condition_on_split=True)[:, keep]), n_draws)
    if name in ("ba_a4", "ba_a8"):
        a_max = float(name[4:])
        f = fit_twopl_bock_aitkin(Xk, a_max=a_max)
        fit_info[name] = {"a_max": a_max, "loglik": f.loglik, "converged": f.converged,
                          "frac_slopes_at_bound": float(np.mean(f.a >= a_max - 1e-6)),
                          "slope_quantiles_0_10_50_90_100": np.quantile(f.a, [0, .1, .5, .9, 1]).tolist(),
                          "theta_ml": f.theta.tolist(), "theta_eap": f.theta_eap.tolist(),
                          "corr_theta_ml_vs_accuracy": float(np.corrcoef(f.theta, X.mean(1))[0, 1])}
        P = f.prob()
        return _draws(lambda: corr_eigs(atmn.sample_bernoulli_split(P, rng)), n_draws)
    if name in ("kg_letter4", "kg_uniform4"):
        fk, pik = ctx.kg_fits[name]
        return _draws(lambda: corr_eigs(sample_matrix_conditional(
            lambda idx: kg_sample_rotations(fk, pik, ctx.gold, rng, n_idx=idx), ctx.derive, keep)), n_draws)
    if name == "indep4_2pl":
        fk, _ = ctx.kg_fits[name]
        return _draws(lambda: corr_eigs(sample_matrix_conditional(
            lambda idx: indep_sample_rotations(fk, rng, idx), ctx.derive, keep)), n_draws)
    raise ValueError(name)


NOTES = {
    "rasch_jml": "Rasch, joint ML (row and column sums matched); conditional on split",
    "fixed_margin": "curveball chain (exact row and column sums; = exact conditional Rasch null)",
    "2pl_post": "project 2PL: theta_m fixed effects, (log a_i, b_i) normal random effects integrated "
                "(marginal ML over questions, 41 x 401 grid); question-matched posterior draws",
    "2pl_prior": "project 2PL; (a_i, b_i) from the fitted population law",
    "3pl_post": "3PL with c = 0.25 (P = 0.25 + 0.75 sigma(a_i (theta_m - b_i))), project MML; posterior draws",
    "letter_post": "letter-aware know-or-guess: P = pi_m(g) + (1 - pi_m(g)) sigma(a_i (theta_m - b_i)), g = shown "
                   "gold letter, pi_m = model m's letter preference (4-rotation moment estimate); project MML",
    "ba_a4": "Bock-Aitkin 2PL: theta ~ N(0,1) integrated, (a_i, d_i) fixed effects, 0 < a_i <= 4; "
             "simulated at ML theta_m (fixed effects given the items)",
    "ba_a8": "Bock-Aitkin 2PL as ba_a4 with a_i <= 8",
    "kg_letter4": "know-or-guess fitted on all 4 rotations (composite MML with pi_m letter preference); "
                  "know once per cell, guess per rotation from pi_m; derived matrix",
    "kg_uniform4": "know-or-guess on all 4 rotations with uniform guessing (pi = 0.25 per letter, 3PL-type)",
    "indep4_2pl": "2PL on all 4 rotations with rotations independent given (a_i, b_i) (no shared know state); "
                  "liberal: cell averages have the smallest variance",
}


def run_suite(ctx: Context, names: list[str], seed: int, n_draws: int, tag: str,
              store: dict | None = None) -> dict:
    children = np.random.default_rng(seed).spawn(len(ALL_NULLS))
    out, fit_info = {}, {}
    for name in names:
        t0 = time.time()
        e = run_null(name, ctx, children[ALL_NULLS.index(name)], n_draws, fit_info)
        note = NOTES.get(name, "STRESS TEST (not fitted): project 2PL posterior draws, every slope x"
                         + name.split("_x")[-1] if "stress" in name else name)
        out[name] = summarize(name, e, ctx.obs, ctx.edge, note)
        if store is not None:
            store[f"{tag}__{name}"] = e
        print(f"[{tag}] {name}: p99 {np.round(out[name]['rank_eigs_p99'], 1)} above {out[name]['observed_ranks_above_p99']} "
              f"l2/l1 obs {out[name]['lambda2_over_lambda1']['observed']:.3f} p99 "
              f"{out[name]['lambda2_over_lambda1']['null_p99']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"observed": {"n_cols": int(ctx.keep.sum()), "lambda_plus": ctx.edge, "top_eigs": ctx.obs[:6].tolist(),
                         "lambda2_over_lambda1": float(ctx.obs[1] / ctx.obs[0]),
                         "count_above_1p05_edge": int((ctx.obs > MP_BUFFER * ctx.edge).sum())},
            "nulls": out, "fits": fit_info, "seed": seed, "n_draws_per_null": n_draws,
            "ranks_above_p99_by_null": {k: v["observed_ranks_above_p99"] for k, v in out.items()},
            "ranks_beyond_ability": {k: contiguous_from_2(v["observed_ranks_above_p99"]) for k, v in out.items()}}


def contiguous_from_2(ranks: list[int]) -> int:
    """Number of ranks 2, 3, ... (contiguous from rank 2) above p99: directions beyond ability."""
    n = 0
    for j in range(2, N_RANKS + 1):
        if j not in ranks:
            break
        n += 1
    return n


def eig_vectors(X: np.ndarray):
    """(eigenvalues, model scores U (M, k), question loadings V (n_keep, k), keep) of the standardized kept columns.
    Sign: eigvec 1 scores correlate positively with accuracy; for j >= 2 the loadings sum to >= 0."""
    keep = keep_mask(X)
    Z = standardize_columns(X[:, keep].astype(np.float64))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    acc = X.mean(axis=1)
    for j in range(min(N_RANKS, s.size)):
        sg = np.sign(np.corrcoef(U[:, j], acc)[0, 1]) if j == 0 else np.sign(Vt[j].sum())
        sg = sg if sg != 0 else 1.0
        U[:, j] *= sg
        Vt[j] *= sg
    return s ** 2 / X.shape[0], U[:, :N_RANKS], Vt[:N_RANKS].T, keep


def perm_corr(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    r = float(np.corrcoef(x, y)[0, 1])
    null = np.array([np.corrcoef(x[rng.permutation(x.size)], y)[0, 1] for _ in range(n_perm)])
    return {"r": r, "perm_p_two_sided": perm_pvalue_upper(abs(r), np.abs(null))}


def letter_analysis(X: np.ndarray, q_letter: np.ndarray | None, shares: np.ndarray, subj: np.ndarray,
                    fam: np.ndarray, rng: np.random.Generator, q_letter_alt: dict | None = None) -> dict:
    """eta^2 of question loadings by gold letter (and subject); model-score correlations.
    q_letter: (N,) shown gold letter or None; shares: (M, 4) each model's answer share A..D."""
    lam, U, V, keep = eig_vectors(X)
    acc = X.mean(axis=1)
    out = []
    for j in range(N_RANKS):
        rec = {"rank": j + 1, "eigenvalue": float(lam[j])}
        if q_letter is not None:
            ql = q_letter[keep]
            null = np.array([eta_squared(V[:, j], rng.permutation(ql)) for _ in range(N_PERM_ETA)])
            eta = eta_squared(V[:, j], ql)
            rec["eta_sq_by_shown_gold_letter"] = {
                "eta_sq": eta, "perm_p": perm_pvalue_upper(eta, null), "null_p95": float(np.percentile(null, 95)),
                "mean_loading_by_letter": {LETTERS[g]: float(V[ql == g, j].mean()) for g in range(4)}}
        for k, lab in (q_letter_alt or {}).items():
            rec[f"eta_sq_by_{k}"] = label_eta_sq(V[:, j], lab[keep], N_PERM_ETA, rng)
        rec["eta_sq_by_subject"] = label_eta_sq(V[:, j], subj[keep], 2000, rng)
        rec["model_score_corr"] = {f"share_{LETTERS[L]}": perm_corr(U[:, j], shares[:, L], N_PERM_R, rng)
                                   for L in range(4)}
        rec["model_score_corr"]["accuracy"] = perm_corr(U[:, j], acc, N_PERM_R, rng)
        rec["eta_sq_model_scores_by_family"] = label_eta_sq(U[:, j], fam, 5000, rng)
        best = max(range(4), key=lambda L: abs(rec["model_score_corr"][f"share_{LETTERS[L]}"]["r"]))
        rec["letter_with_max_abs_share_corr"] = LETTERS[best]
        rec["model_scores"] = U[:, j].round(4).tolist()
        out.append(rec)
    return {"eigenvectors": out, "n_cols": int(keep.sum())}


def shares_of(letters: np.ndarray) -> np.ndarray:
    """(M, 4) share of answers shown as A..D (unparseable counts in the denominator)."""
    M = letters.shape[0]
    return np.stack([(letters == L).reshape(M, -1).mean(axis=1) for L in range(4)], axis=1)


def kg_fits(d: dict) -> tuple[dict, dict]:
    CR, gold = d["CR"], d["gold"]
    M = CR.shape[0]
    lp = letter_preference(CR, d["shown_choice"])
    S, T = letter_groups_rotations(CR, gold)
    f_letter = fit_guess_mml(S, T, lp["pi_letters"])
    f_unif = fit_guess_mml(S, T, np.full((M, 4), 0.25))
    f_ind = fit_guess_mml(S, T, np.zeros((M, 4)))
    fits = {"kg_letter4": (f_letter, lp["pi_letters"]), "kg_uniform4": (f_unif, np.full((M, 4), 0.25)),
            "indep4_2pl": (f_ind, np.zeros((M, 4)))}
    info = {"letter_preference": {n: {"pi_A_B_C_D": lp["pi_letters"][m].round(4).tolist(),
                                      "pi_unparseable": float(lp["pi_unparseable"][m]),
                                      "kbar": float(lp["kbar"][m]), "n_clipped": int(lp["n_clipped"][m])}
                                  for m, n in enumerate(d["names"])},
            "estimator": "moment estimate over all 4 rotations (v3lib.letter_preference)",
            "fits": {k: {"loglik_composite": f.loglik, "converged": f.converged, "hyper": f.hyper.tolist(),
                         "corr_theta_vs_accuracy_rot0": float(np.corrcoef(f.theta, CR[:, :, 0].mean(1))[0, 1])}
                     for k, (f, _) in fits.items()}}
    return fits, info


def part_b_extra() -> None:
    d = load_rep("A")
    CR, gold, M = d["CR"], d["gold"], d["CR"].shape[0]
    C = CR[:, :, 0]
    fits, kg_info = kg_fits(d)
    store = {}
    ctx = Context(C.astype(np.int8), CR, gold, shown_gold=np.broadcast_to(gold, C.shape), pi=fits["kg_letter4"][1],
                  derive=lambda T: T[:, :, 0], kg_fits=fits)
    suite = run_suite(ctx, ["3pl_post", "letter_post", "ba_a4", "ba_a8", "kg_letter4", "kg_uniform4"],
                      SEED, N_DRAWS, "A_rot0", store)
    # Rank-matched Bernoulli nulls (difficulty-matched = per-question rate; per-model rate),
    # unconditional, split filter re-applied per draw (as d_across_null), 500 draws.
    rng_q, rng_m = np.random.default_rng(d_across_null.SEED).spawn(3)[:2]
    bern = {}
    for key, rng, drawer in (("bern_question", rng_q, lambda r: d_across_null.draw_null_A(C.mean(0), M, r)),
                             ("bern_model", rng_m, lambda r: d_across_null.draw_null_B(C.mean(1), C.shape[1], r))):
        E = np.zeros((d_across_null.N_DRAWS, M))
        ed = np.empty(d_across_null.N_DRAWS)
        for k in range(d_across_null.N_DRAWS):
            S = drawer(rng)
            sm = split_mask(S)
            e = corr_eigs(S[:, sm])
            E[k, :e.size] = e
            ed[k] = mp_edge(int(sm.sum()), M)
        atmn._M_ROWS = M
        bern[key] = atmn.summarize_null(key, E, ed, ctx.obs, ctx.edge, "unconditional; N_split varies per draw")
        bern[key]["lambda2_over_lambda1"] = ratio_stats(E, ctx.obs)
        store[f"A_rot0__{key}"] = E
    rng = np.random.default_rng(SEED + 1)
    letters = letter_analysis(C, gold, shares_of(d["shown_choice"][:, :, 0]), d["subj"], d["fam"], rng)
    np.savez_compressed(OUT / "raw_v3_null_eigs_A_rot0_extra.npz", **store)
    write_json("v3_B_extra", {
        "meta": {"replicate": "A", "matrix": "rotation 0 (= results/mmlu_ensemble_greedy.npz)", "seed": SEED,
                 "n_draws_per_null": N_DRAWS, "n_draws_bernoulli": d_across_null.N_DRAWS,
                 "bernoulli_seed": d_across_null.SEED, "n_perm_eta": N_PERM_ETA, "n_perm_corr": N_PERM_R,
                 "notes": NOTES},
        "kg_fits_4_rotations": kg_info, "suite": suite, "bernoulli_rank_nulls": bern, "letter_analysis": letters})


def random_rotation_matrix(CR: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(X (M, N), rotation index (M, N)): each model draws one rotation per question independently."""
    M, N, R = CR.shape
    rot = rng.integers(0, R, (M, N))
    return np.take_along_axis(CR, rot[..., None], axis=2)[..., 0], rot


def part_c(rep: str, with_rot0_suite: bool) -> None:
    d = load_rep(rep)
    CR, gold, M = d["CR"], d["gold"], d["CR"].shape[0]
    fits, kg_info = kg_fits(d)
    pi = fits["kg_letter4"][1]
    store = {}
    res = {"meta": {"replicate": rep, "seed": SEED, "n_draws_per_null": N_DRAWS, "n_random_rotation_draws": N_RANDOM_ROT,
                    "n_robust_refit_draws": N_ROBUST_DRAWS, "n_robust_null_draws": N_ROBUST_NULL,
                    "n_perm_eta": N_PERM_ETA, "n_perm_corr": N_PERM_R, "notes": NOTES},
           "kg_fits_4_rotations": kg_info}
    rng_letters = np.random.default_rng(SEED + 2)

    # C1: per rotation.
    c1 = {}
    for r in range(4):
        Xr = CR[:, :, r]
        shown = (gold + r) % 4
        c1[str(r)] = {"spectrum": spectrum(Xr), "accuracy_mean": float(Xr.mean()),
                      "letters": letter_analysis(Xr, shown, shares_of(d["shown_choice"][:, :, r]), d["subj"], d["fam"],
                                                 rng_letters, q_letter_alt={"original_gold_letter": gold})}
        print(f"[{rep} C1 r={r}]", np.round(c1[str(r)]["spectrum"]["top_eigs"][:4], 1),
              [round(v["eta_sq_by_shown_gold_letter"]["eta_sq"], 3) for v in c1[str(r)]["letters"]["eigenvectors"]],
              [round(v["eta_sq_by_original_gold_letter"]["eta_sq"], 3) for v in c1[str(r)]["letters"]["eigenvectors"]],
              flush=True)
    # Do the eigenvector-j model scores stay the same across rotations (same models prefer the same letters)?
    sc = {r: np.array([v["model_scores"] for v in c1[str(r)]["letters"]["eigenvectors"]]) for r in range(4)}
    c1["model_score_abs_corr_with_rotation0"] = {
        str(r): [float(abs(np.corrcoef(sc[r][j], sc[0][j])[0, 1])) for j in range(N_RANKS)] for r in range(1, 4)}
    res["C1_per_rotation"] = c1

    # C2: rotation-randomized ensembles.
    rng = np.random.default_rng(SEED + 3)
    specs, rots = [], []
    for _ in range(N_RANDOM_ROT):
        X, rot = random_rotation_matrix(CR, rng)
        specs.append(spectrum(X))
        rots.append(rot.astype(np.int8))
    E = np.array([s["top_eigs"][:6] for s in specs])
    lp_ = np.array([s["lambda_plus"] for s in specs])
    buf = np.array([s["mp_count_buffered"] for s in specs])
    strict = np.array([s["mp_count_strict"] for s in specs])
    med, iqr = np.median(E[:, :4], 0), np.subtract(*np.percentile(E[:, :4], [75, 25], 0))
    rep_idx = int(np.argmin((((E[:, :4] - med) / iqr) ** 2).sum(1)))
    rot = rots[rep_idx].astype(int)
    Xrep = np.take_along_axis(CR, rot[..., None], axis=2)[..., 0]
    shown_rep = (gold[None, :] + rot) % 4
    c2 = {"distribution": {"eig_mean": E.mean(0).tolist(), "eig_median": np.median(E, 0).tolist(),
                           "eig_p01": np.percentile(E, 1, 0).tolist(), "eig_p99": np.percentile(E, 99, 0).tolist(),
                           "lambda_plus_mean": float(lp_.mean()), "lambda_plus_min_max": [float(lp_.min()), float(lp_.max())],
                           "n_cols_mean": float(np.mean([s["n_cols"] for s in specs])),
                           "mp_count_buffered_distribution": {str(k): int(v) for k, v in enumerate(np.bincount(buf)) if v},
                           "mp_count_strict_distribution": {str(k): int(v) for k, v in enumerate(np.bincount(strict)) if v},
                           "lambda2_over_lambda1_median": float(np.median(E[:, 1] / E[:, 0]))},
          "representative_draw": {"index": rep_idx, "rule": "draw minimizing sum_j ((lambda_j - median_j) / IQR_j)^2, j = 1..4",
                                  "spectrum": specs[rep_idx]}}
    ctx = Context(Xrep.astype(np.int8), CR, gold, shown_gold=shown_rep, pi=pi, kg_fits=fits)
    suite_names = ["rasch_jml", "fixed_margin", "2pl_post", "3pl_post", "letter_post", "ba_a4"]
    c2["representative_suite"] = run_suite(ctx, suite_names, SEED, N_DRAWS, f"{rep}_randrot", store)
    # Know-or-guess (4-rotation fit) with a fresh random rotation assignment per null draw.
    rng_kg = np.random.default_rng(SEED).spawn(len(ALL_NULLS))[ALL_NULLS.index("kg_letter4_randrot")]
    fk, pik = fits["kg_letter4"]
    Ekg = np.empty((N_DRAWS, M))
    for k in range(N_DRAWS):
        T = kg_sample_rotations(fk, pik, gold, rng_kg)
        Xs, _ = random_rotation_matrix(T, rng_kg)
        Ekg[k] = corr_eigs(Xs[:, split_mask(Xs)])
    store[f"{rep}_randrot__kg_letter4_randrot"] = Ekg
    p99 = np.percentile(Ekg[:, :N_RANKS], 99, 0)
    c2["kg_letter4_random_rotation_null"] = {
        "notes": "know-or-guess fitted on all 4 rotations; each null draw simulates all 4 rotations and draws a "
                 "fresh random rotation per cell (unconditional; split filter per draw)",
        "rank_eigs_p99": p99.tolist(), "rank_eigs_mean": Ekg[:, :N_RANKS].mean(0).tolist(),
        "frac_observed_draws_above_p99_per_rank": (E[:, :N_RANKS] > p99).mean(0).tolist(),
        "observed_median_above_p99": [int(j + 1) for j in np.flatnonzero(med > p99)],
        "lambda2_over_lambda1": {"observed_median": float(np.median(E[:, 1] / E[:, 0])),
                                 "null_p99": float(np.percentile(Ekg[:, 1] / Ekg[:, 0], 99)),
                                 "frac_observed_draws_above_null_p99": float(np.mean(
                                     E[:, 1] / E[:, 0] > np.percentile(Ekg[:, 1] / Ekg[:, 0], 99)))}}
    # Thresholds of the representative suite applied to all 1000 draws.
    c2["frac_draws_above_representative_p99"] = {
        k: (E[:, :N_RANKS] > np.array(v["rank_eigs_p99"])).mean(0).tolist()
        for k, v in c2["representative_suite"]["nulls"].items()}
    # Robustness: refit 2PL and letter-aware nulls on N_ROBUST_DRAWS other draws.
    rob = []
    rng_pick = np.random.default_rng(SEED + 4)
    for k in rng_pick.choice(N_RANDOM_ROT, N_ROBUST_DRAWS, replace=False):
        rk = rots[int(k)].astype(int)
        Xk = np.take_along_axis(CR, rk[..., None], axis=2)[..., 0]
        ck = Context(Xk.astype(np.int8), CR, gold, shown_gold=(gold[None, :] + rk) % 4, pi=pi, kg_fits=fits)
        s = run_suite(ck, ["2pl_post", "letter_post"], SEED + 100 + int(k), N_ROBUST_NULL, f"{rep}_randrot_{k}")
        rob.append({"draw": int(k), "top_eigs": s["observed"]["top_eigs"][:4],
                    "ranks_above_p99": s["ranks_above_p99_by_null"], "beyond_ability": s["ranks_beyond_ability"]})
    c2["robustness_refit_draws"] = rob
    c2["letters_representative"] = letter_analysis(Xrep, None, shares_of(np.take_along_axis(
        d["shown_choice"], rot[..., None], axis=2)[..., 0]), d["subj"], d["fam"], rng_letters,
        q_letter_alt={"original_gold_letter": gold})
    res["C2_rotation_randomized"] = c2

    # C3: rotation-invariant correctness.
    c3 = {}
    for key, derive, names in (
            ("all4_correct", lambda T: T.all(axis=2).astype(np.int8),
             ["rasch_jml", "fixed_margin", "2pl_post", "2pl_prior", "ba_a4", "ba_a8", "kg_letter4", "kg_uniform4"]),
            ("mean_over_rotations", lambda T: T.mean(axis=2), ["kg_letter4", "kg_uniform4", "indep4_2pl"])):
        X = derive(CR)
        ctx = Context(X, CR, gold, pi=pi, derive=derive, kg_fits=fits)
        c3[key] = {"spectrum": spectrum(X), "accuracy_mean": float(X.mean()),
                   "suite": run_suite(ctx, names, SEED, N_DRAWS, f"{rep}_{key}", store),
                   "letters_subject_family": letter_analysis(X, None, shares_of(d["shown_choice"]), d["subj"], d["fam"],
                                                             rng_letters, q_letter_alt={"original_gold_letter": gold})}
    res["C3_rotation_invariant"] = c3

    if with_rot0_suite:   # C4: the rotation-0 null suite of this replicate (same code and seeds for A and B)
        C = CR[:, :, 0]
        ctx = Context(C.astype(np.int8), CR, gold, shown_gold=np.broadcast_to(gold, C.shape), pi=pi,
                      derive=lambda T: T[:, :, 0], kg_fits=fits)
        res["rot0_suite"] = run_suite(
            ctx, ["rasch_jml", "fixed_margin", "2pl_post", "2pl_prior", "3pl_post", "letter_post", "ba_a4", "ba_a8",
                  "kg_letter4", "kg_uniform4"] + [f"2pl_post_stress_x{f:g}" for f in STRESS],
            SEED, N_DRAWS, f"{rep}_rot0", store)
        res["rot0_letters"] = letter_analysis(C, gold, shares_of(d["shown_choice"][:, :, 0]), d["subj"], d["fam"],
                                              np.random.default_rng(SEED + 1))
    np.savez_compressed(OUT / f"raw_v3_null_eigs_{rep}_rotations.npz", **store,
                        randrot_eigs_1to6=E, randrot_rotations=np.array(rots))
    write_json(f"v3_C_rotations_{rep}", res)


def letter_gaps(lp: np.ndarray) -> np.ndarray:
    """(M, N, 4) gap between the two highest first-token letter logprobs (inf if fewer than 2 finite).
    lp: (M, N, rotation, option); same definition as src/r2_analysis.py::gaps."""
    srt = np.sort(np.where(np.isfinite(lp), lp, -np.inf), axis=3)[..., ::-1]
    with np.errstate(invalid="ignore"):
        g = srt[..., 0] - srt[..., 1]
    return np.where(np.isfinite(g), g, np.inf)


def majority_fill(CR: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Near-tie cells set to the majority correctness over the cell's 4 rotations (2-2 ties: unchanged)."""
    s = CR.sum(axis=2, keepdims=True)
    maj = np.where(s >= 3, 1, np.where(s <= 1, 0, -1))
    out = CR.copy()
    fill = near & (maj >= 0)
    out[fill] = np.broadcast_to(maj, CR.shape)[fill]
    return out


def _kg_fits_with_pi(CR: np.ndarray, gold: np.ndarray, pi: np.ndarray) -> dict:
    """4-rotation know-or-guess fits (letter-aware and uniform) at a given letter preference pi."""
    M = CR.shape[0]
    S, T = letter_groups_rotations(CR, gold)
    return {"kg_letter4": (fit_guess_mml(S, T, pi), pi),
            "kg_uniform4": (fit_guess_mml(S, T, np.full((M, 4), 0.25)), np.full((M, 4), 0.25))}


def _near_variant(args) -> tuple[str, dict]:
    """Run the refitted null suite of one near-tie variant (worker for a process pool)."""
    key, kind, CR, gold, pi = args
    fits = _kg_fits_with_pi(CR, gold, pi)
    if kind == "rot0":
        X = CR[:, :, 0].astype(np.int8)
        ctx = Context(X, CR, gold, shown_gold=np.broadcast_to(gold, X.shape), pi=pi,
                      derive=lambda T: T[:, :, 0], kg_fits=fits)
        names = ["2pl_post", "3pl_post", "letter_post", "kg_letter4"]
    elif kind == "all4":
        X = CR.all(axis=2).astype(np.int8)
        ctx = Context(X, CR, gold, pi=pi, derive=lambda T: T.all(axis=2).astype(np.int8), kg_fits=fits)
        names = ["rasch_jml", "fixed_margin", "2pl_post", "ba_a4", "kg_letter4"]
    else:
        X = CR.mean(axis=2)
        ctx = Context(X, CR, gold, pi=pi, derive=lambda T: T.mean(axis=2), kg_fits=fits)
        names = ["kg_letter4", "kg_uniform4"]
    suite = run_suite(ctx, names, SEED, N_DRAWS, f"near_{key}")
    return key, {"spectrum": spectrum(X), "suite": suite}


def part_near() -> None:
    """C5: near-tie sensitivity, every binary variant with its own fitted nulls (same seeds as the main runs)."""
    from multiprocessing import get_context

    from v2lib import impute_column_mean
    A, B = load_rep("A"), load_rep("B")
    CR, gold = A["CR"], A["gold"]
    pi = letter_preference(CR, A["shown_choice"])["pi_letters"]               # a model property: kept fixed
    gA, gB = letter_gaps(A["lp"]), letter_gaps(B["lp"])
    near = (gA < 0.13) | (gB < 0.13)                                          # (M, N, 4)
    n0 = near[:, :, 0]
    keep_q = ~n0.any(axis=0)
    maj = majority_fill(CR, near)
    # all-4 ignoring near-tie rotations (a cell with 4 near ties keeps its observed value)
    ign = np.where(near, 1, CR)
    ign[near.all(axis=2)] = CR[near.all(axis=2)]
    jobs = [("rot0_majority", "rot0", maj, gold, pi),
            ("rot0_drop_questions_with_rot0_near_tie", "rot0", CR[:, keep_q], gold[keep_q], pi),
            ("all4_majority", "all4", maj, gold, pi),
            ("all4_near_tie_rotations_ignored", "all4", ign.astype(np.int8), gold, pi),
            ("mean_majority", "mean", maj, gold, pi)]
    with get_context("fork").Pool(len(jobs)) as pool:
        res = dict(pool.map(_near_variant, jobs))
    # Fractional variants (no binary parametric null): spectra only.
    w = (~near).sum(axis=2)
    mean_ign = np.where(w > 0, (CR * ~near).sum(axis=2) / np.maximum(w, 1), 0.0)
    frac = {"rot0_near_ties_as_missing_imputed_column_mean": spectrum(impute_column_mean(CR[:, :, 0], n0)),
            "mean_near_tie_rotations_ignored": spectrum(impute_column_mean(mean_ign, w == 0))}
    eA = np.load(OUT / "raw_v3_null_eigs_A_rotations.npz")
    thr_rr = np.percentile(eA["A_randrot__kg_letter4_randrot"][:, :N_RANKS], 99, 0)
    rots = eA["randrot_rotations"].astype(int)[:200]
    rr = {}
    for key, T in (("as_observed", CR), ("majority", maj)):
        E = np.array([corr_eigs(X[:, split_mask(X)])[:N_RANKS] for X in
                      (np.take_along_axis(T, r[..., None], axis=2)[..., 0] for r in rots)])
        rr[key] = {"n_draws": int(E.shape[0]), "eig_median": np.median(E, 0).tolist(),
                   "kg_letter4_randrot_p99_main": thr_rr.tolist(), "frac_draws_above_p99": (E > thr_rr).mean(0).tolist()}
    write_json("v3_C5_near_ties", {
        "definition": "near tie: top-two first-token letter logprob gap < 0.13 nats in replicate A or in B "
                      "(src/r2_analysis.py::gaps); the logprobs lie on a ~0.125-nat grid",
        "n_near_tie_cells_by_rotation": [int(near[:, :, r].sum()) for r in range(4)],
        "frac_near_tie_cells": float(near.mean()),
        "n_questions_with_a_rot0_near_tie": int(n0.any(axis=0).sum()),
        "majority_rule": "near-tie cell set to the majority correctness over its 4 rotations; 2-2 ties unchanged",
        "majority_n_cells_changed": int((maj != CR).sum()),
        "majority_n_cells_changed_rot0": int((maj[:, :, 0] != CR[:, :, 0]).sum()),
        "letter_preference": "pi fixed at the replicate-A estimate (a model property); the IRT part is refitted",
        "seed": SEED, "n_draws_per_null": N_DRAWS,
        "refitted_variants": res, "fractional_variants_spectra_only": frac,
        "rotation_randomized_200_draws_vs_main_kg_null": rr})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["B_extra", "C", "near"])
    ap.add_argument("--rep", default="A", choices=["A", "B"])
    a = ap.parse_args()
    if a.part == "B_extra":
        part_b_extra()
    elif a.part == "C":
        part_c(a.rep, with_rot0_suite=True)
    else:
        part_near()
