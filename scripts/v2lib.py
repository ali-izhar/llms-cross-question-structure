"""Shared paths, run names and helpers for the results/analysis scripts; importing it sets 1 BLAS thread.

Conventions: borderline = eps < f_i < 1 - eps (eps = 0.05); columns centered, divided by the population SD;
R = Z^T Z / K; lambda_+ = (1 + sqrt(N_b / K))^2; buffered count = #{lambda > 1.05 lambda_+}.
"""

from __future__ import annotations

import os

# One BLAS thread per process unless the caller set it (must precede the numpy import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results"
OUT = V2 / "analysis"
MAT = OUT / "matrices"
OUT.mkdir(parents=True, exist_ok=True)
MAT.mkdir(parents=True, exist_ok=True)
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EPS = 0.05
LETTERS = "ABCD"

# (run_id, short label, group). group: iid | shared.
RUNS: list[tuple[str, str, str]] = [
    ("mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct", "MMLU Qwen2.5-7B T1.0", "iid"),
    ("mmlu_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3", "MMLU Mistral-7B-v0.3", "iid"),
    ("mmlu_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct", "MMLU Meta-Llama-3-8B", "iid"),
    ("mmlu_iid-T1.0_HuggingFaceTB_SmolLM2-1.7B-Instruct", "MMLU SmolLM2-1.7B", "iid"),
    ("mmlu_iid-T1.0_microsoft_Phi-3-mini-4k-instruct", "MMLU Phi-3-mini", "iid"),
    ("hellaswag_iid-T1.0_Qwen_Qwen2.5-7B-Instruct", "HellaSwag Qwen2.5-7B", "iid"),
    ("hellaswag_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3", "HellaSwag Mistral-7B-v0.3", "iid"),
    ("hellaswag_iid-T1.0_meta-llama_Meta-Llama-3-8B-Instruct", "HellaSwag Meta-Llama-3-8B", "iid"),
    ("gsm8k_iid-T1.0_Qwen_Qwen2.5-7B-Instruct", "GSM8K Qwen2.5-7B 256 tok", "iid"),
    ("gsm8k_iidlong-T1.0_Qwen_Qwen2.5-7B-Instruct", "GSM8K Qwen2.5-7B 1024 tok", "iid"),
    ("mmlu_iid-T0.5_Qwen_Qwen2.5-7B-Instruct", "MMLU Qwen2.5-7B T0.5", "iid"),
    ("mmlu_iid-T1.5_Qwen_Qwen2.5-7B-Instruct", "MMLU Qwen2.5-7B T1.5", "iid"),
    ("mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct", "MMLU Qwen2.5-7B S1 shared seed", "shared"),
    ("mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct", "MMLU Qwen2.5-7B S2 shared 5-shot", "shared"),
    ("mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct", "MMLU Meta-Llama-3-8B S2 shared 5-shot", "shared"),
]
LABEL = {r: lab for r, lab, _ in RUNS}
QWEN_IID = RUNS[0][0]
LLAMA_IID = RUNS[2][0]
ENSEMBLE_NPZ = V2 / "mmlu_ensemble_greedy.npz"
ENSEMBLE_NPY = MAT / "ensemble_greedy.npy"
ENSEMBLE_NAMES = MAT / "ensemble_models.json"


def load_npz(run_id: str):
    return np.load(V2 / f"{run_id}.npz", allow_pickle=False)


def load_meta(run_id: str) -> dict:
    return json.loads((V2 / f"{run_id}.json").read_text())


def matrix_path(run_id: str) -> Path:
    return MAT / f"{run_id}.npy"


def extract_all() -> None:
    """Write every run's (K, N) int8 correctness matrix and the 24-model matrix as .npy."""
    for run_id, _, _ in RUNS:
        C = load_npz(run_id)["correct"].astype(np.int8)
        np.save(matrix_path(run_id), C)
    z = np.load(ENSEMBLE_NPZ)
    np.save(ENSEMBLE_NPY, z["correct"].astype(np.int8))
    ENSEMBLE_NAMES.write_text(json.dumps(z["models"].tolist(), indent=1))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def write_json(name: str, obj: dict) -> Path:
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=True))
    print(f"[json] {p}")
    return p


def letters_to_codes(parsed: np.ndarray) -> np.ndarray:
    """'A'..'D' -> 0..3; anything else (unparseable '') -> -1. Same shape, int8."""
    out = np.full(parsed.shape, -1, dtype=np.int8)
    for j, L in enumerate(LETTERS):
        out[parsed == L] = j
    return out


def modal_wrong_share(codes: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Per column: count of the most common wrong code / number of wrong passes (NaN if none).

    codes (K, n) in {-1, 0..3}, gold (n,); unparseable (-1) is its own wrong category."""
    K, n = codes.shape
    out = np.full(n, np.nan)
    for i in range(n):
        wrong = codes[:, i][codes[:, i] != gold[i]]
        if wrong.size:
            out[i] = np.bincount(wrong + 1, minlength=5).max() / wrong.size
    return out


def impute_column_mean(C: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Replace missing entries by the observed column mean (0 if none observed: all-wrong column). float64."""
    X = C.astype(np.float64).copy()
    obs = ~missing
    n_obs = obs.sum(axis=0)
    col_mean = np.where(n_obs > 0, (X * obs).sum(axis=0) / np.maximum(n_obs, 1), 0.0)
    X[missing] = np.broadcast_to(col_mean, X.shape)[missing]
    return X


def borderline_mask_float(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    f = X.mean(axis=0)
    return (f > eps) & (f < 1 - eps)


def spectrum_counts(X: np.ndarray) -> dict:
    """N_cols, lambda_+, lambda_1..4, margin, TW z, strict and buffered MP counts of X (rows = samples)."""
    from null_models import MP_BUFFER, corr_eigs, mp_edge, tw_z
    K, n = X.shape
    if n < 2:
        return {"n_cols": int(n), "note": "fewer than 2 columns"}
    e = corr_eigs(X)
    lp = mp_edge(n, K)
    return {
        "n_cols": int(n), "lambda_plus": lp, "lambda_plus_K_minus_1": float((1 + np.sqrt(n / (K - 1))) ** 2),
        "top_eigs": e[:4].tolist(), "lambda_max": float(e[0]), "margin": float(lp - e[0]),
        "tw_z": tw_z(float(e[0]), K, n),
        "mp_count_strict": int((e > lp).sum()), "mp_count_buffered": int((e > MP_BUFFER * lp).sum()),
    }


def ols_r2(y: np.ndarray, F: np.ndarray) -> float:
    """R^2 of the least-squares fit of y on [1, F]; F (K, p)."""
    A = np.column_stack([np.ones(len(y)), F])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    tss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - resid @ resid / tss) if tss > 0 else 0.0


def perm_r2(y: np.ndarray, F: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """R^2 of y on F and its permutation p-value (rows of F permuted), Phipson-Smyth."""
    from null_models import perm_pvalue_upper
    obs = ols_r2(y, F)
    null = np.array([ols_r2(y, F[rng.permutation(len(y))]) for _ in range(n_perm)])
    return {"r2": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "perm_p": perm_pvalue_upper(obs, null), "n_perm": n_perm}


def top_pc_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(pass scores u_1 s_1 (K,), loadings v_1 (n,), lambda_1) of standardized X; sign: sum(v_1) >= 0."""
    from null_models import standardize_columns
    Z = standardize_columns(X.astype(np.float64))
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    v = Vt[0]
    sgn = 1.0 if v.sum() >= 0 else -1.0
    return sgn * U[:, 0] * s[0], sgn * v, float(s[0] ** 2 / X.shape[0])


def letter_share_dispersion(codes: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Sum over letters of the across-pass variance of each pass's letter share; null shuffles each column.

    codes (K, n) in {-1, 0..3}; the null keeps per-question letter frequencies and removes pass coupling."""
    from null_models import perm_pvalue_upper

    def stat(A):
        shares = np.stack([(A == j).mean(axis=1) for j in range(4)], axis=1)  # (K, 4)
        return float(shares.var(axis=0, ddof=1).sum())

    obs = stat(codes)
    null = np.array([stat(rng.permuted(codes, axis=0)) for _ in range(n_perm)])
    return {"stat_sum_var_letter_share": obs, "null_mean": float(null.mean()),
            "null_p99": float(np.percentile(null, 99)), "ratio_obs_over_null_mean": obs / float(null.mean()),
            "perm_p": perm_pvalue_upper(obs, null), "n_perm": n_perm}


def gold_letter_eta_sq(loadings: np.ndarray, gold: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """eta^2 of question loadings by gold letter, label-permutation p-value."""
    from null_models import eta_squared, perm_pvalue_upper
    obs = eta_squared(loadings, gold)
    null = np.array([eta_squared(loadings, rng.permutation(gold)) for _ in range(n_perm)])
    return {"eta_sq": obs, "perm_p": perm_pvalue_upper(obs, null), "null_p95": float(np.percentile(null, 95)),
            "mean_loading_by_gold": {LETTERS[g]: float(loadings[gold == g].mean()) for g in np.unique(gold)},
            "n_by_gold": {LETTERS[g]: int((gold == g).sum()) for g in np.unique(gold)}, "n_perm": n_perm}


def label_eta_sq(values: np.ndarray, labels: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """eta^2 = between-group SS / total SS of `values` by a label, label-permutation p and null mean.

    The null mean is about (G - 1) / (n - 1) for G groups, so many small groups are accounted for."""
    from null_models import eta_squared, perm_pvalue_upper
    labels = np.asarray(labels)
    obs = eta_squared(values, labels)
    null = np.array([eta_squared(values, rng.permutation(labels)) for _ in range(n_perm)])
    return {"eta_sq": obs, "perm_p": perm_pvalue_upper(obs, null), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "n_groups": int(np.unique(labels).size),
            "n": int(values.size), "n_perm": n_perm}


def subject_match_effect(C: np.ndarray, subj: np.ndarray, pass_subjects: list[set],
                         n_perm: int, rng: np.random.Generator) -> dict:
    """Mean over questions of accuracy on subject-matched minus unmatched passes; null permutes exemplar sets.

    C (K, n); subj (n,); pass_subjects: K sets of exemplar subjects."""
    from null_models import perm_pvalue_upper

    names = sorted(set(subj.tolist()) | set().union(*pass_subjects))
    idx = {s: j for j, s in enumerate(names)}
    q_idx = np.array([idx[s] for s in subj])                                   # (n,)
    P = np.zeros((len(pass_subjects), len(names)), dtype=bool)                 # (K, S)
    for k, sets in enumerate(pass_subjects):
        P[k, [idx[s] for s in sets]] = True
    Cf = C.astype(np.float64)

    def stat(Pk):
        match = Pk[:, q_idx]                                                   # (K, n)
        both = match.any(axis=0) & (~match).any(axis=0)
        if not both.any():
            return np.nan, 0, 0
        m1 = (Cf * match).sum(0)[both] / match.sum(0)[both]
        m0 = (Cf * ~match).sum(0)[both] / (~match).sum(0)[both]
        return float(np.mean(m1 - m0)), int(both.sum()), int(match.sum())

    obs, n_q, n_cells = stat(P)
    null = np.array([stat(P[rng.permutation(P.shape[0])])[0] for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    return {"mean_diff_matched_minus_unmatched": obs, "n_questions_with_both": n_q,
            "n_matched_cells": n_cells, "null_sd": float(null.std(ddof=1)),
            "perm_p_two_sided": perm_pvalue_upper(abs(obs), np.abs(null)), "n_perm": int(null.size)}


def exemplar_overlap_similarity(X: np.ndarray, ex_rows: np.ndarray, n_perm: int,
                                rng: np.random.Generator) -> dict:
    """Mean row correlation of pass pairs sharing >= 1 exemplar minus pairs sharing none (Mantel-type null).

    X (K, n); ex_rows (K, m) exemplar ids; the null permutes exemplar sets across passes."""
    from null_models import perm_pvalue_upper, standardize_columns
    Z = standardize_columns(X.astype(np.float64))
    S = np.corrcoef(Z)                                                         # (K, K) row similarity
    K = X.shape[0]
    iu = np.triu_indices(K, 1)
    s = S[iu]
    H = np.zeros((K, int(ex_rows.max()) + 1))
    H[np.arange(K)[:, None], ex_rows] = 1.0                                    # (K, pool) indicator

    def stat(Hk):
        share = (Hk @ Hk.T)[iu] >= 1                                           # pair shares an exemplar
        return float(s[share].mean() - s[~share].mean()) if share.any() and (~share).any() else np.nan, int(share.sum())

    obs, n_share = stat(H)
    null = np.array([stat(H[rng.permutation(K)])[0] for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    return {"mean_sim_sharing_minus_not": obs, "n_pairs_sharing": n_share, "n_pairs": int(s.size),
            "null_sd": float(null.std(ddof=1)), "z_vs_null": float((obs - null.mean()) / null.std(ddof=1)),
            "perm_p_upper": perm_pvalue_upper(obs, null), "n_perm": int(null.size)}


def pair_correlations(C: np.ndarray, families: list[str]) -> dict:
    """Pairwise row Pearson r; same- vs cross-family Cohen's d (pooled SD, ddof=1) and KS test."""
    R = np.corrcoef(C.astype(np.float64))
    M = C.shape[0]
    iu = np.triu_indices(M, 1)
    rs = R[iu]
    fam = np.asarray(families)
    same = fam[iu[0]] == fam[iu[1]]
    rsm, rcr = rs[same], rs[~same]
    ns, nc = rsm.size, rcr.size
    s_pool = np.sqrt(((ns - 1) * rsm.var(ddof=1) + (nc - 1) * rcr.var(ddof=1)) / (ns + nc - 2))
    ks = ks_2samp(rsm, rcr)
    return {"n_pairs": int(rs.size), "mean_r": float(rs.mean()), "min_r": float(rs.min()), "max_r": float(rs.max()),
            "n_same_family": int(ns), "n_cross_family": int(nc),
            "mean_r_same_family": float(rsm.mean()), "mean_r_cross_family": float(rcr.mean()),
            "cohen_d": float((rsm.mean() - rcr.mean()) / s_pool), "ks_D": float(ks.statistic),
            "ks_p": float(ks.pvalue)}


def split_half_r(C: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pearson r of per-question pass rates: first K//2 rows vs the rest."""
    X = C if mask is None else C[:, mask]
    h = C.shape[0] // 2
    return float(np.corrcoef(X[:h].mean(0), X[h:].mean(0))[0, 1])
