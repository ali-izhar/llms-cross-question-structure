"""
Replication of the "locally committed, globally incoherent" result
==================================================================

For each of five within-model runs, test whether answer identity
(or, when answer-identity data is unavailable, the correctness pattern)
is *independent across questions* over the K=100 stochastic passes.

Test statistics per question pair (i, j):
  * Cramer's V  of the K-long categorical vectors
  * Mutual information (nats)

Null: independent columns (each column shuffled independently over K).
Under the null, all pairwise V and MI concentrate near zero up to
finite-K noise; the 99th percentile of the permutation null is the
reference threshold.

Observed statistic: max over all borderline question pairs.
Verdict is "d = 0 at the 99th percentile" iff
    max_observed <= threshold_99.

For Qwen2.5-7B we have the answer-identity matrix
(`semantic_full_answer_choices.npy`, shape (K=100 passes, N=500 questions),
values in {0,1,2,3} encoding A/B/C/D -- this is a single-model within-model
file, NOT a 24-model across-model matrix; see src/run_semantic_full.py)
and reproduce the paper's test exactly.

For the other four models (SmolLM2-1.7B, Phi-3-mini, Meta-Llama-3-8B,
Mistral-7B-Instruct-v0.3) only the binary correctness matrix was
saved by `src/run_within_batch.py`, so we run the *correctness-pattern*
version of the same independence test (labelled accordingly in the JSON).
This is a strictly weaker probe than answer-identity (a letter-identity
dependence that is marginal in correctness could be missed), so a pass
on correctness-pattern is necessary, not sufficient, for the full
answer-identity claim; we state this caveat in the paper.

Inputs:
  results/semantic_full_answer_choices.npy            (Qwen answer-ids)
  results/correctness_matrix_<slug>.npy               (K x N int8)
Output:
  results/answer_identity_replication.json

Run with:
  .venv/bin/python scripts/answer_identity_replication.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Reproducibility: permutation null uses a fixed seed.
SEED = 20260414
N_PERMS = 200
# Borderline defined exactly as in the paper: 0.05 < correctness rate < 0.95.
BORDERLINE_LO = 0.05
BORDERLINE_HI = 0.95


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _contingency(x: np.ndarray, y: np.ndarray, n_cats: int) -> np.ndarray:
    """Row-major n_cats x n_cats contingency table. Both x,y in [0, n_cats)."""
    idx = x * n_cats + y
    counts = np.bincount(idx, minlength=n_cats * n_cats)
    return counts.reshape(n_cats, n_cats)


def cramers_v_pair(x: np.ndarray, y: np.ndarray, n_cats: int) -> float:
    """Cramer's V on two length-K categorical arrays with entries in [0, n_cats).

    Returns 0.0 if either vector is constant (no variation -> no association
    can be defined). Matches the existing src/run_semantic_pilot.py convention.
    """
    K = len(x)
    if K < 5:
        return 0.0
    C = _contingency(x, y, n_cats)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    keep_r = row_sums > 0
    keep_c = col_sums > 0
    if keep_r.sum() < 2 or keep_c.sum() < 2:
        return 0.0
    C = C[np.ix_(keep_r, keep_c)]
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    n = C.sum()
    expected = np.outer(row_sums, col_sums) / n
    # expected > 0 everywhere because we dropped empty rows/cols
    chi2 = ((C - expected) ** 2 / expected).sum()
    k_min = min(C.shape) - 1
    if k_min == 0 or n <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * k_min)))


def mutual_information_pair(x: np.ndarray, y: np.ndarray, n_cats: int) -> float:
    """Mutual information in nats. Matches src/run_semantic_pilot.py."""
    K = len(x)
    if K < 5:
        return 0.0
    C = _contingency(x, y, n_cats).astype(np.float64)
    n = C.sum()
    if n <= 0:
        return 0.0
    joint = C / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((joint > 0) & (denom > 0),
                             np.log(joint / np.where(denom > 0, denom, 1.0)),
                             0.0)
    mi = float((joint * log_ratio).sum())
    return max(mi, 0.0)  # clip tiny negatives from fp rounding


def max_pairwise_stats_binary_vectorized(A: np.ndarray) -> tuple[float, float]:
    """Vectorized max Cramer's V and max MI across all question pairs, *binary* case.

    For binary x, y in {0, 1} with K samples, the 2x2 contingency table is
    fully determined by the joint count n11 = <x, y>, and the marginals
    n1x = sum(x), n1y = sum(y).

    Cramer's V reduces (for a 2x2 table) to |phi| = sqrt(chi2 / K):
        phi = (n11 * n00 - n10 * n01) / sqrt(n1x * n0x * n1y * n0y)
    and this equals the absolute Pearson correlation of x and y viewed as
    0/1 vectors. Using vectorized matrix products gives an O(N_b^2 K) bulk
    computation in BLAS rather than a Python double loop.

    MI in nats from the same 2x2 cell counts, computed with standard
    xlogy trick and clipping tiny negatives from fp rounding.
    """
    K, Nb = A.shape
    X = A.astype(np.float64)
    col_sum = X.sum(axis=0)              # (Nb,) == n1x
    # Joint counts n11[i, j] = sum_k X[k,i] X[k,j]
    n11 = X.T @ X                        # (Nb, Nb)
    # Exclude diagonal from "max" later.
    n10 = col_sum[:, None] - n11
    n01 = col_sum[None, :] - n11
    n00 = K - n11 - n10 - n01

    # Cramer's V for 2x2: |phi| where phi = (n11 n00 - n10 n01) / sqrt(...)
    # Degenerate (constant) columns yield zero denominator -> set V to 0.
    p1x = col_sum
    p0x = K - col_sum
    denom = np.sqrt(np.outer(p1x * p0x, p1x * p0x))
    num = n11 * n00 - n10 * n01
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, num / denom, 0.0)
    V = np.abs(phi)
    np.fill_diagonal(V, 0.0)
    max_v = float(V.max()) if Nb >= 2 else 0.0

    # Mutual information in nats.
    # p11 = n11/K, etc. MI = sum_ab p_ab log(p_ab / (p_a p_b))
    # Use xlogy-ish stable form.
    def _cell(n):
        p = n / K
        return p
    p11 = _cell(n11)
    p10 = _cell(n10)
    p01 = _cell(n01)
    p00 = _cell(n00)
    p_x1 = (col_sum / K)[:, None]
    p_x0 = 1.0 - p_x1
    p_y1 = (col_sum / K)[None, :]
    p_y0 = 1.0 - p_y1

    def _term(p_joint, p_a, p_b):
        # p log(p / (p_a p_b)), 0 if p == 0 or marginals == 0.
        out = np.zeros_like(p_joint)
        denom_ab = p_a * p_b
        mask = (p_joint > 0) & (denom_ab > 0)
        out[mask] = p_joint[mask] * np.log(p_joint[mask] / denom_ab[mask])
        return out

    MI = (
        _term(p11, p_x1, p_y1)
        + _term(p10, p_x1, p_y0)
        + _term(p01, p_x0, p_y1)
        + _term(p00, p_x0, p_y0)
    )
    np.fill_diagonal(MI, 0.0)
    MI = np.maximum(MI, 0.0)  # clip fp-rounding negatives
    max_mi = float(MI.max()) if Nb >= 2 else 0.0
    return max_v, max_mi


def max_pairwise_stats(A: np.ndarray, n_cats: int) -> tuple[float, float]:
    """Max Cramer's V and max MI across all unordered question pairs.

    A has shape (K, N_b) with entries in [0, n_cats). Uses a vectorized
    fast path for n_cats == 2; otherwise falls back to a (loop) general form.
    """
    if n_cats == 2:
        return max_pairwise_stats_binary_vectorized(A)
    K, Nb = A.shape
    max_v = 0.0
    max_mi = 0.0
    for i in range(Nb):
        xi = A[:, i]
        for j in range(i + 1, Nb):
            xj = A[:, j]
            v = cramers_v_pair(xi, xj, n_cats)
            if v > max_v:
                max_v = v
            mi = mutual_information_pair(xi, xj, n_cats)
            if mi > max_mi:
                max_mi = mi
    return max_v, max_mi


def permutation_null(
    A: np.ndarray, n_cats: int, n_perms: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of (max_V, max_MI) over n_perms column-wise shuffles."""
    K, Nb = A.shape
    vs = np.empty(n_perms)
    mis = np.empty(n_perms)
    for p in range(n_perms):
        B = A.copy()
        # Shuffle each column independently (destroys inter-question coupling).
        for j in range(Nb):
            rng.shuffle(B[:, j])
        vs[p], mis[p] = max_pairwise_stats(B, n_cats)
    return vs, mis


# ---------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str                    # pretty name for JSON key
    data_kind: str               # "answer_identity" or "correctness_pattern"
    path: Path                   # .npy to load
    n_cats: int                  # 4 for A/B/C/D, 2 for correctness


def run_model(spec: ModelSpec, n_perms: int, seed: int) -> dict:
    A_full = np.load(spec.path)
    if A_full.dtype != np.int8:
        A_full = A_full.astype(np.int8)
    K, N = A_full.shape

    if spec.data_kind == "answer_identity":
        # Derive per-question correctness: for borderline we need the correctness
        # matrix. We reuse the companion correctness file (paper convention).
        C = np.load(RESULTS / "semantic_full_correctness.npy")
        assert C.shape == A_full.shape, \
            f"semantic_full_correctness shape {C.shape} != {A_full.shape}"
        flip = C.mean(axis=0)
        # Filter out unparseable entries (-1). Keep borderline.
        valid_col = (A_full >= 0).all(axis=0)
        bord = (flip > BORDERLINE_LO) & (flip < BORDERLINE_HI) & valid_col
    elif spec.data_kind == "correctness_pattern":
        flip = A_full.mean(axis=0)
        bord = (flip > BORDERLINE_LO) & (flip < BORDERLINE_HI)
    else:
        raise ValueError(spec.data_kind)

    A = A_full[:, bord]
    Nb = int(bord.sum())
    n_pairs = Nb * (Nb - 1) // 2

    t0 = time.time()
    obs_v, obs_mi = max_pairwise_stats(A, spec.n_cats)

    rng = np.random.default_rng(seed)
    null_v, null_mi = permutation_null(A, spec.n_cats, n_perms, rng)

    thr_v_99 = float(np.percentile(null_v, 99))
    thr_mi_99 = float(np.percentile(null_mi, 99))
    thr_v_95 = float(np.percentile(null_v, 95))
    thr_mi_95 = float(np.percentile(null_mi, 95))

    verdict_v = obs_v <= thr_v_99
    verdict_mi = obs_mi <= thr_mi_99
    replicates = bool(verdict_v and verdict_mi)

    elapsed = time.time() - t0
    print(
        f"[{spec.name:32s}] kind={spec.data_kind:20s} "
        f"Nb={Nb:3d}  pairs={n_pairs:6d}  "
        f"maxV={obs_v:.4f} (p99 null={thr_v_99:.4f})  "
        f"maxMI={obs_mi:.4f} (p99={thr_mi_99:.4f})  "
        f"replicates={replicates}  ({elapsed:.1f}s)"
    )

    return {
        "data_kind": spec.data_kind,
        "K": int(K),
        "N_total": int(N),
        "n_borderline": Nb,
        "n_borderline_pairs": int(n_pairs),
        "n_permutations": int(n_perms),
        "max_cramers_v": {
            "observed": float(obs_v),
            "null_p99": thr_v_99,
            "null_p95": thr_v_95,
            "passes_p99": bool(verdict_v),
        },
        "max_mi_nats": {
            "observed": float(obs_mi),
            "null_p99": thr_mi_99,
            "null_p95": thr_mi_95,
            "passes_p99": bool(verdict_mi),
        },
        "verdict": (
            "d=0 at 99th percentile"
            if replicates
            else "d>0 (observed exceeds 99th percentile null)"
        ),
        "elapsed_sec": round(elapsed, 1),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

MODEL_SPECS = [
    ModelSpec(
        name="Qwen2.5-7B-Instruct",
        data_kind="answer_identity",
        path=RESULTS / "semantic_full_answer_choices.npy",
        n_cats=4,
    ),
    ModelSpec(
        name="Mistral-7B-Instruct-v0.3",
        data_kind="correctness_pattern",
        path=RESULTS / "correctness_matrix_mistralai_Mistral-7B-Instruct-v0.3.npy",
        n_cats=2,
    ),
    ModelSpec(
        name="SmolLM2-1.7B-Instruct",
        data_kind="correctness_pattern",
        path=RESULTS / "correctness_matrix_HuggingFaceTB_SmolLM2-1.7B-Instruct.npy",
        n_cats=2,
    ),
    ModelSpec(
        name="Phi-3-mini-4k-instruct",
        data_kind="correctness_pattern",
        path=RESULTS / "correctness_matrix_microsoft_Phi-3-mini-4k-instruct.npy",
        n_cats=2,
    ),
    ModelSpec(
        name="Meta-Llama-3-8B-Instruct",
        data_kind="correctness_pattern",
        path=RESULTS / "correctness_matrix_meta-llama_Meta-Llama-3-8B-Instruct.npy",
        n_cats=2,
    ),
]


def main() -> None:
    out: dict = {
        "description": (
            "Replication of Qwen2.5-7B answer-identity independence test across "
            "5 within-model runs. Qwen uses answer-identity (A/B/C/D); the other "
            "four models only have correctness matrices saved, so use correctness-"
            "pattern independence (same permutation null, binary labels)."
        ),
        "n_permutations": N_PERMS,
        "seed": SEED,
        "borderline_range": [BORDERLINE_LO, BORDERLINE_HI],
        "per_model": {},
    }

    for spec in MODEL_SPECS:
        if not spec.path.exists():
            print(f"MISSING: {spec.path}")
            out["per_model"][spec.name] = {"status": "missing", "path": str(spec.path)}
            continue
        out["per_model"][spec.name] = run_model(spec, N_PERMS, SEED)

    # Aggregate verdict
    verdicts = [
        m.get("verdict", "") for m in out["per_model"].values() if isinstance(m, dict)
    ]
    n_replicate = sum(v == "d=0 at 99th percentile" for v in verdicts)
    total = len(verdicts)
    out["summary"] = {
        "n_models": total,
        "n_replicate_at_p99": n_replicate,
        "aggregate_n_borderline": int(sum(
            m["n_borderline"] for m in out["per_model"].values() if isinstance(m, dict) and "n_borderline" in m
        )),
        "one_line": (
            f"{n_replicate}/{total} models replicate the Qwen result "
            f"(max Cramer's V and max MI <= 99th-percentile permutation null)."
        ),
    }

    out_path = RESULTS / "answer_identity_replication.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print("SUMMARY:", out["summary"]["one_line"])


if __name__ == "__main__":
    main()
