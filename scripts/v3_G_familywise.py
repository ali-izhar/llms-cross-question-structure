"""
G. Pre-registered family-wise test over ranks 2-4, sibling-model test, one validity rule
(specification: results/analysis/specs/G_familywise.md, written before the G run; quoted in the output).

Parts (python v3_G_familywise.py <part>):
  run      : G1, G2, G3, G4, G6 in one pool (checkpointed per job under results/analysis/logs_G/ckpt).
  assemble : G5 (from stored F outputs) and the summary results/analysis/v3_G_familywise.json + raw files.
  all      : run, then assemble.

Statistic (G1). Null eigenvalue sample E (n draws, ranks 1..4), observed obs.
  pct_r(obs)   = 100 (#{d: E_dr < obs_r} + #{d: E_dr = obs_r} / 2) / n       (= v3_softpref.pct_of)
  T            = max_{r in 2,3,4} pct_r(obs)
  T_d          = max_{r in 2,3,4} pct_r(E_d) against the other n - 1 draws (leave-one-out)
  p_T          = (1 + #{d: T_d >= T}) / (1 + n)
  S            = lambda_2 + lambda_3 + lambda_4,  p_S = (1 + #{d: S_d >= S}) / (1 + n)   (secondary, no decision)
All percentiles are kept as integer half-counts h = 2 #{<} + #{=} (pct = 100 h / (2 n)), and T_d >= T is
tested exactly by integer cross-multiplication: T_d >= T <=> h_d n >= h_obs (n - 1).
Validity (G5, one rule everywhere): p01 <= observed lambda_1 <= p99 of the null lambda_1
(v3_softpref.lambda1_valid). G1 rejects a (score, null) iff the null is valid and p_T < 0.05 (raw), or,
where the G2 false-positive rate exceeds 0.10, p_T < the 5th percentile of p_T under SP truth (calibrated).

Every null, fit and simulation is the existing code: v3_across.run_suite / run_null / Context / kg_fits,
v3_softpref.simulate_choices / sp_null / load_base / variant_data / SCORES / family_nulls /
lambda1_valid / rank_summary, v2lib.impute_column_mean. Seeds: new namespace SEED_G (see rng_g).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from contextlib import contextmanager
from pathlib import Path

import v2lib  # noqa: F401  (threads = 1 per process, sys.path)
import numpy as np
from v2lib import OUT, impute_column_mean, write_json

import v3_across as V3
import v3_softpref as V

SEED_G = 1202610137             # disjoint from every earlier seed (test_v3_G_familywise)
STREAM_G = {"g1": 1, "g1_sp": 2, "g2_data": 3, "g2_null": 4, "g3_data": 5, "g3_null": 6,
            "g4_null": 7, "g4_random_pick": 8, "g6_null": 9}
ALPHA = 0.05
FP_CAL_LIMIT = 0.10
T_RANKS = (2, 3, 4)
N_RANKS = 4
SCORES = tuple(V.SCORES)        # rot0, ge1, ge2, ge3, ge4, mean (the step-1 construction family)
N_G1 = 2000
N_G2 = 200
N_G2_NULL = 500
G2_SCORES = ("rot0", "ge4", "ge1", "ge2", "ge3")
G2_NULLS = ("rasch_jml", "2pl_post")
N_G3 = 200
N_G3_NULL = 500
S_G3 = (1.0, 2.0)
N_G4 = 1000
N_G4_RANDOM = 24
N_G4_REMOVE = 5
G4_NULLS = {"ge4": ["rasch_jml", "2pl_post"],
            # rotation 0: 2PL and Rasch plus the nulls valid on the full rotation-0 data (G1 needs valid nulls)
            "rot0": ["rasch_jml", "2pl_post", "3pl_post", "kg_uniform4", "letter_post"]}
N_G6 = 1000
G6_NULLS = {"rot0": ["letter_post", "3pl_post"], "ge4": ["2pl_post", "rasch_jml"]}
KG_NULLS = {"kg_letter4", "kg_uniform4", "indep4_2pl", "letter_post"}

# Sibling clusters; names as in results/mmlu_ensemble_rotations.npz.
CLUSTERS = (("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct"),
            ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
            ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3",
             "HuggingFaceH4/zephyr-7b-beta"),
            ("Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"))

LOG_DIR = OUT / "logs_G"
CKPT = LOG_DIR / "ckpt"
PREREG = Path(__file__).resolve().parents[1] / "results/analysis/specs/G_familywise.md"


def rng_g(*key: int) -> np.random.Generator:
    """default_rng([SEED_G, *key]); every key component a positive integer (SeedSequence drops trailing 0s)."""
    return np.random.default_rng(seed_g(*key))


def seed_g(*key: int) -> list[int]:
    """[SEED_G, *key], the seed passed to v3_across.run_suite (which spawns one child per null)."""
    if not key or not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k > 0 for k in key):
        raise ValueError(f"stream key components must be positive integers: {key}")
    return [SEED_G, *[int(k) for k in key]]


def score_code(score: str) -> int:
    return SCORES.index(score) + 1


def obs_half_counts(obs: np.ndarray, E: np.ndarray) -> np.ndarray:
    """h_r = 2 #{d: E_dr < obs_r} + #{d: E_dr = obs_r}, r = 1..4 -> (4,) int64."""
    E = np.asarray(E)[:, :N_RANKS]
    o = np.asarray(obs)[:N_RANKS]
    return (2 * (E < o).sum(axis=0) + (E == o).sum(axis=0)).astype(np.int64)


def loo_half_counts(E: np.ndarray) -> np.ndarray:
    """(n, 4) int64: h_dr of draw d against the other n - 1 draws (itself excluded)."""
    E = np.asarray(E)[:, :N_RANKS]
    H = np.empty(E.shape, dtype=np.int64)
    for r in range(E.shape[1]):
        s = np.sort(E[:, r])
        less = np.searchsorted(s, E[:, r], side="left")
        leq = np.searchsorted(s, E[:, r], side="right")
        H[:, r] = 2 * less + (leq - less - 1)
    return H


def half_to_pct(h, n: int):
    return 100.0 * np.asarray(h, dtype=np.float64) / (2.0 * n)


def familywise(obs: np.ndarray, E: np.ndarray) -> dict:
    """G1 statistics (p_T, p_S, validity) of one (observed, null sample) pair."""
    E = np.asarray(E, dtype=np.float64)
    n = int(E.shape[0])
    if n < V.MIN_NULL_DRAWS:
        return {"n_draws": n, "insufficient_draws": True}
    ri = [r - 1 for r in T_RANKS]
    h_obs = obs_half_counts(obs, E)
    t_obs = int(h_obs[ri].max())
    H = loo_half_counts(E)
    t_loo = H[:, ri].max(axis=1)
    k_T = int(np.sum(t_loo * n >= t_obs * (n - 1)))
    s_obs = float(np.sum(obs[1:N_RANKS]))
    s_null = E[:, 1:N_RANKS].sum(axis=1)
    k_S = int(np.sum(s_null >= s_obs))
    t_null_pct = half_to_pct(t_loo, n - 1)
    rs = V.rank_summary(E, np.asarray(obs))
    return {"n_draws": n, "insufficient_draws": False,
            "pct_rank1to4": half_to_pct(h_obs, n).tolist(),
            "T_pct": float(half_to_pct(t_obs, n)), "T_argmax_rank": int(T_RANKS[int(np.argmax(h_obs[ri]))]),
            "p_T": (1 + k_T) / (1 + n), "k_T": k_T,
            "T_null_pct_quantiles_50_95_99": np.percentile(t_null_pct, [50, 95, 99]).tolist(),
            "S": s_obs, "p_S": (1 + k_S) / (1 + n), "S_null_p95_p99": np.percentile(s_null, [95, 99]).tolist(),
            "valid": bool(V.lambda1_valid(E, np.asarray(obs))),
            "ranks_above_p99": rs["observed_ranks_above_p99"], "ranks_beyond_ability": rs["ranks_beyond_ability"],
            "mc_p_upper_rank1to4": rs["mc_p_upper_rank1to4"], "rank_eigs_p99": rs["rank_eigs_p99"],
            "rank_eigs_p01": rs["rank_eigs_p01"], "observed_top_eigs": np.asarray(obs)[:N_RANKS].tolist()}


def g1_reject(fw: dict, threshold: float = ALPHA) -> bool:
    """Valid null and p_T < threshold."""
    return bool(not fw.get("insufficient_draws") and fw["valid"] and fw["p_T"] < threshold)


ROW_KEYS = ("CR", "shown_choice", "fam", "lp", "chosen")
COL_KEYS_2D = ("CR", "shown_choice", "lp", "chosen")


def subset_rows(d: dict, keep_names) -> dict:
    """d restricted to the models keep_names (order of d['names'] kept), as variant_data('no_llama1b')."""
    keep_set = set(keep_names)
    if not keep_set <= set(d["names"]):
        raise ValueError(f"unknown models: {keep_set - set(d['names'])}")
    keep = np.array([n in keep_set for n in d["names"]])
    out = dict(d)
    for k in ROW_KEYS:
        out[k] = d[k][keep]
    out["names"] = [n for n, k in zip(d["names"], keep, strict=True) if k]
    return out


def subset_cols(d: dict, keep: np.ndarray) -> dict:
    """d restricted to the questions keep (bool (N,)), as variant_data('no_optref')."""
    out = dict(d)
    for k in COL_KEYS_2D:
        out[k] = d[k][:, keep]
    out["gold"], out["subj"] = d["gold"][keep], d["subj"][keep]
    out["question_index"] = np.flatnonzero(keep)
    return out


def score_context(d: dict, score: str, nulls) -> V3.Context:
    """The Context of v3_softpref.family_task for one score (4-rotation fits only when a null needs them)."""
    CR, gold = d["CR"], d["gold"]
    fits = V3.kg_fits(d)[0] if (set(nulls) & KG_NULLS) else {}
    derive = V.SCORES[score]
    X = derive(CR)
    shown_gold = np.broadcast_to(gold, X.shape) if score == "rot0" else None
    pi = fits["kg_letter4"][1] if fits else None
    return V3.Context(X, CR, gold, shown_gold=shown_gold, pi=pi, derive=derive, kg_fits=fits)


def run_nulls(ctx: V3.Context, nulls, seed, n_draws: int, tag: str) -> dict:
    """{null: (n_draws, M) eigenvalues} from v3_across.run_suite (child stream by the null's ALL_NULLS index)."""
    store = {}
    V3.run_suite(ctx, list(nulls), seed, n_draws, tag, store)
    return {nm: store[f"{tag}__{nm}"] for nm in nulls}


def sibling_subsets(names: list[str]) -> list[dict]:
    """The 2*2*3*2 = 24 subsets that keep exactly one model per sibling cluster (all other models kept)."""
    for c in CLUSTERS:
        for n in c:
            if n not in names:
                raise ValueError(f"cluster model not in the data: {n}")
    in_cluster = {n for c in CLUSTERS for n in c}
    out = []
    for pick in itertools.product(*CLUSTERS):
        removed = sorted(in_cluster - set(pick))
        out.append({"kept_from_clusters": list(pick), "removed": removed,
                    "keep": [n for n in names if n not in removed]})
    return out


def random_control_subsets(names: list[str], rng: np.random.Generator, n_sets: int = N_G4_RANDOM,
                           k: int = N_G4_REMOVE) -> list[dict]:
    """n_sets distinct random sets of k models removed from the non-sibling models (clusters intact)."""
    in_cluster = {n for c in CLUSTERS for n in c}
    pool = [n for n in names if n not in in_cluster]
    seen, out = set(), []
    while len(out) < n_sets:
        pick = tuple(sorted(rng.choice(len(pool), size=k, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)
        removed = [pool[i] for i in pick]
        out.append({"removed": removed, "keep": [n for n in names if n not in removed]})
    return out


def all4_unknown_mask(CR: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """(M, N) bool: the all-4 entry is decided by unparsed replies (no parsed rotation wrong and at least one
    rotation unparsed). Elsewhere the all-4 value is the same for every completion of the unparsed replies."""
    unp = chosen < 0
    parsed_wrong = (~unp) & (CR == 0)
    return (~parsed_wrong.any(axis=2)) & unp.any(axis=2)


_IMPUTE_DROPS: list[int] = []


def imputed_eigs(Y: np.ndarray, mask_keep: np.ndarray, eig_fn) -> np.ndarray:
    """Eigenvalues of v2lib.impute_column_mean(Y, mask) on its non-constant columns (Y: kept columns)."""
    if Y.shape != mask_keep.shape:
        raise ValueError(f"draw shape {Y.shape} != mask shape {mask_keep.shape}")
    Xi = impute_column_mean(Y, mask_keep)
    nc = Xi.std(axis=0) > 0
    _IMPUTE_DROPS.append(int((~nc).sum()))
    return eig_fn(Xi[:, nc])


@contextmanager
def impute_null_draws(mask_keep: np.ndarray):
    """Inside the block, impute every v3_across null draw (M, n_keep) at the masked cells before its eigenvalues."""
    orig = V3.corr_eigs
    V3.corr_eigs = lambda Y: imputed_eigs(Y, mask_keep, orig)
    try:
        yield
    finally:
        V3.corr_eigs = orig


def g1_task(score: str) -> dict:
    """G1: every step-1 null of the score, N_G1 new draws, full data."""
    d = V.variant_data("full")
    nulls = V.family_nulls(score)
    ctx = score_context(d, score, nulls)
    E = run_nulls(ctx, nulls, seed_g(STREAM_G["g1"], score_code(score)), N_G1, f"G1_{score}")
    return {"score": score, "obs": ctx.obs, "E": E}


def g1_sp_task(_: int = 0) -> dict:
    """G1 under the primary soft-preference null (common tau, base fit), N_G1 new draws, every score."""
    data, _d = V.sp_data_of("full")
    keeps, obs = V.observed_keeps(data)
    par = V.load_base("full")
    E, fail = V.sp_null(par, data.gold, data.chosen < 0, keeps, N_G1, rng_g(STREAM_G["g1_sp"], 1))
    return {"obs": obs, "E": E, "n_failed": fail}


def g2_simulate(rng: np.random.Generator) -> np.ndarray:
    """(M, N, 4) correctness simulated from the SP base fit, observed unparsed mask (as calib_task)."""
    par = V.load_base("full")
    d0 = V.variant_data("full")
    gold = d0["gold"]
    chosen = V.simulate_choices(par, gold, None, rng, unp_mask=d0["chosen"] < 0)
    return (chosen == gold[None, :, None]).astype(np.int8)


def nulls_on_scores(CR: np.ndarray, scores, nulls, seed_of, n_draws: int, tag: str) -> dict:
    """{score: {null: familywise record + top-4 null eigenvalues}} for 2PL/Rasch-type nulls on X = score(CR)."""
    out = {}
    for sc in scores:
        ctx = V3.Context(V.SCORES[sc](CR))
        E = run_nulls(ctx, nulls, seed_of(sc), n_draws, f"{tag}_{sc}")
        out[sc] = {nm: {**familywise(ctx.obs, e), "E_top4": e[:, :N_RANKS].copy()} for nm, e in E.items()}
    return out


def g2_task(rep: int) -> dict:
    CR = g2_simulate(rng_g(STREAM_G["g2_data"], rep + 1))
    res = nulls_on_scores(CR, G2_SCORES, G2_NULLS, lambda sc: seed_g(STREAM_G["g2_null"], rep + 1, score_code(sc)),
                          N_G2_NULL, f"G2r{rep}")
    return {"rep": rep, "res": res}


def planted_correct(par: dict, gold: np.ndarray, stem: np.ndarray, unp_mask: np.ndarray, s: float,
                    rng: np.random.Generator) -> np.ndarray:
    """v3_softpref.power_dataset (same RNG order): gold utility + s theta2_m on STEM, theta2 ~ N(0, 1).
    Returns (M, N, 4) correctness."""
    M = par["theta"].size
    theta2 = rng.standard_normal(M)
    extra = theta2[:, None] * (s * stem.astype(float))[None, :]
    chosen = V.simulate_choices(par, gold, None, rng, extra_gold=extra, unp_mask=unp_mask)
    return (chosen == gold[None, :, None]).astype(np.int8)


def g3_task(args) -> dict:
    s, rep = args
    si = S_G3.index(s) + 1
    d = V.variant_data("full")
    CR = planted_correct(V.load_base("full"), d["gold"], V.stem_mask(d["subj"]), d["chosen"] < 0, s,
                         rng_g(STREAM_G["g3_data"], si, rep + 1))
    res = nulls_on_scores(CR, ("ge4",), G2_NULLS, lambda sc: seed_g(STREAM_G["g3_null"], si, rep + 1), N_G3_NULL,
                          f"G3s{s:g}r{rep}")
    return {"s": s, "rep": rep, "res": res}


def g4_task(args) -> dict:
    """G4: one subset (kind 1 = sibling, 2 = random control), all-4 and rotation 0."""
    kind, idx, keep = args
    d = subset_rows(V.variant_data("full"), keep)
    out = {"kind": kind, "idx": idx, "keep": keep, "M": len(d["names"])}
    for sc, nulls in G4_NULLS.items():
        ctx = score_context(d, sc, nulls)
        E = run_nulls(ctx, nulls, seed_g(STREAM_G["g4_null"], kind, idx + 1, score_code(sc)), N_G4,
                      f"G4k{kind}i{idx}_{sc}")
        out[sc] = {nm: {**familywise(ctx.obs, e), "E_top4": e[:, :N_RANKS].copy()} for nm, e in E.items()}
    return out


G6_VARIANTS = (("rot0", "drop"), ("rot0", "impute"), ("ge4", "drop_affected"), ("ge4", "drop_any"),
               ("ge4", "impute"))


def g6_masks(d: dict) -> dict:
    """Unparsed masks. rot0: (M, N) rotation-0 unparsed; ge4: (M, N) all-4 entries decided by unparsed replies;
    ge4_any: (M, N) any rotation unparsed."""
    unp = d["chosen"] < 0
    return {"rot0": unp[:, :, 0], "ge4": all4_unknown_mask(d["CR"], d["chosen"]), "ge4_any": unp.any(axis=2)}


def g6_task(args) -> dict:
    score, variant = args
    vi = G6_VARIANTS.index((score, variant)) + 1
    d = V.variant_data("full")
    masks = g6_masks(d)
    nulls = G6_NULLS[score]
    seed = seed_g(STREAM_G["g6_null"], vi)
    tag = f"G6_{score}_{variant}"
    if variant.startswith("drop"):
        m = masks["ge4_any"] if variant == "drop_any" else masks[score]
        keep_q = ~m.any(axis=0)
        dd = subset_cols(d, keep_q)
        ctx = score_context(dd, score, nulls)
        E = run_nulls(ctx, nulls, seed, N_G6, tag)
        return {"score": score, "variant": variant, "n_questions_dropped": int((~keep_q).sum()),
                "n_cols": int(ctx.keep.sum()), "obs": ctx.obs, "E": E, "n_impute_dropped_cols": None}
    # impute: observed matrix and every null draw go through v2lib.impute_column_mean at the masked cells;
    # the nulls are fitted to the as-scored binary matrix (the binary fits take no fractional data).
    ctx = score_context(d, score, nulls)
    mask_keep = masks[score][:, ctx.keep]
    Xi = impute_column_mean(ctx.X[:, ctx.keep], mask_keep)
    nc = Xi.std(axis=0) > 0
    ctx.obs = V3.corr_eigs(Xi[:, nc])
    _IMPUTE_DROPS.clear()
    with impute_null_draws(mask_keep):
        E = run_nulls(ctx, nulls, seed, N_G6, tag)
    drops = np.array(_IMPUTE_DROPS)
    return {"score": score, "variant": variant, "n_cells_imputed": int(masks[score].sum()),
            "n_cols_as_scored": int(ctx.keep.sum()), "n_cols": int(nc.sum()), "obs": ctx.obs, "E": E,
            "n_impute_dropped_cols": {"observed": int((~nc).sum()), "draws_mean": float(drops.mean()),
                                      "draws_max": int(drops.max()), "n_draws": int(drops.size)}}


V.KIND_FN.update({"G1": g1_task, "G1sp": g1_sp_task, "G2": g2_task, "G3": g3_task, "G4": g4_task, "G6": g6_task})


def g4_jobs() -> list[tuple]:
    names = V.variant_data("full")["names"]
    sib = sibling_subsets(names)
    rnd = random_control_subsets(names, rng_g(STREAM_G["g4_random_pick"], 1))
    return ([(1, i, s["keep"]) for i, s in enumerate(sib)] + [(2, i, s["keep"]) for i, s in enumerate(rnd)])


def all_items() -> list[tuple[str, object]]:
    """Job list, longest first."""
    items = [("G1", "rot0")] + [("G1", s) for s in SCORES if s != "rot0"] + [("G1sp", 0)]
    items += [("G4", j) for j in g4_jobs()]
    items += [("G6", j) for j in G6_VARIANTS]
    g2 = [("G2", r) for r in range(N_G2)]
    g3 = [("G3", (s, r)) for r in range(N_G3) for s in S_G3]
    for k in range(max(len(g2), len(g3))):
        items += g2[k:k + 1] + g3[2 * k:2 * k + 2]
    items += g3[2 * len(g2):]
    return items


def all_seed_keys() -> list[tuple]:
    """Every (stream, ...) key used by G (for the disjointness test)."""
    keys = [(STREAM_G["g1"], score_code(s)) for s in SCORES] + [(STREAM_G["g1_sp"], 1)]
    keys += [(STREAM_G["g2_data"], r + 1) for r in range(N_G2)]
    keys += [(STREAM_G["g2_null"], r + 1, score_code(s)) for r in range(N_G2) for s in G2_SCORES]
    keys += [(STREAM_G["g3_data"], S_G3.index(s) + 1, r + 1) for s in S_G3 for r in range(N_G3)]
    keys += [(STREAM_G["g3_null"], S_G3.index(s) + 1, r + 1) for s in S_G3 for r in range(N_G3)]
    keys += [(STREAM_G["g4_null"], k, i + 1, score_code(s)) for k in (1, 2) for i in range(24) for s in G4_NULLS]
    keys += [(STREAM_G["g4_random_pick"], 1)]
    keys += [(STREAM_G["g6_null"], i + 1) for i in range(len(G6_VARIANTS))]
    return keys


def part_run(n_proc: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return V.run_checkpointed(all_items(), CKPT, n_proc=n_proc)


def _strip(fw: dict) -> dict:
    return {k: v for k, v in fw.items() if k != "E_top4"}


def calibration_of(pts: np.ndarray) -> dict:
    """Size of the raw G1 test under SP truth and the calibrated threshold (5th percentile of p_T)."""
    pts = np.asarray(pts, dtype=float)
    k = int(np.sum(pts < ALPHA))
    return {"n": int(pts.size), "fp_raw": k / pts.size, "fp_raw_ci95": V.clopper_pearson(k, pts.size),
            "cal_threshold_p_T_q05": float(np.percentile(pts, 5)),
            "p_T_quantiles_1_5_10_50": np.percentile(pts, [1, 5, 10, 50]).tolist(),
            "uses_calibrated_threshold": bool(k / pts.size > FP_CAL_LIMIT)}


def cal_p(p_obs: float, pts: np.ndarray) -> float:
    """Calibrated p of an observed p_T: fraction of SP-truth datasets with p_T <= the observed one."""
    return float(np.mean(np.asarray(pts) <= p_obs))


def assemble_g1(res: dict, g2cal: dict) -> tuple[dict, dict]:
    store = {}
    out = {}
    for r in res["G1"]:
        sc = r["score"]
        out[sc] = {}
        for nm, E in r["E"].items():
            store[f"{sc}__{nm}"] = E
            out[sc][nm] = familywise(r["obs"], E)
    sp = res["G1sp"][0]
    for sc in SCORES:
        store[f"{sc}__sp_common_tau"] = sp["E"][sc]
        out[sc]["sp_common_tau"] = {**familywise(sp["obs"][sc], sp["E"][sc]), "n_failed_draws": sp["n_failed"][sc]}
    for sc in out:
        for nm, fw in out[sc].items():
            if fw.get("insufficient_draws"):
                continue
            fw["reject_raw"] = g1_reject(fw)
            c = g2cal.get(sc, {}).get(nm)
            if c is not None:
                fw["g2_calibration"] = {k: c[k] for k in ("fp_raw", "cal_threshold_p_T_q05", "uses_calibrated_threshold")}
                fw["cal_p_T"] = cal_p(fw["p_T"], c["_pts"])
                fw["reject_calibrated"] = g1_reject(fw, c["cal_threshold_p_T_q05"])
                fw["reject_decision"] = fw["reject_calibrated"] if c["uses_calibrated_threshold"] else fw["reject_raw"]
                fw["decision_rule"] = "calibrated" if c["uses_calibrated_threshold"] else "raw (G2 FP <= 0.10)"
            else:
                fw["reject_decision"] = fw["reject_raw"]
                fw["decision_rule"] = "raw (no G2 calibration for this null)"
    return out, store


def assemble_g2(res: dict) -> tuple[dict, dict, dict]:
    rows = sorted(res["G2"], key=lambda r: r["rep"])
    summ, cal, store = {}, {}, {}
    step2b = json.loads((OUT / "v3_F_softpref.json").read_text())["step2b_step1_nulls_under_soft_preference_truth"]["summary"]
    for sc in G2_SCORES:
        summ[sc], cal[sc] = {}, {}
        for nm in G2_NULLS:
            recs = [r["res"][sc][nm] for r in rows]
            ok = [x for x in recs if not x.get("insufficient_draws")]
            pts = np.array([x["p_T"] for x in ok])
            val = np.array([x["valid"] for x in ok])
            c = calibration_of(pts)
            r2 = np.array([2 in x["ranks_above_p99"] for x in ok])
            kv = int(np.sum(val & (pts < ALPHA)))
            c.update({"frac_valid": float(val.mean()), "fp_valid_and_p_T_lt_alpha": kv / len(ok),
                      "fp_valid_and_p_T_lt_alpha_ci95": V.clopper_pearson(kv, len(ok)),
                      "fp_given_valid": float(np.mean(pts[val] < ALPHA)) if val.any() else None,
                      "p_S_lt_alpha": float(np.mean([x["p_S"] < ALPHA for x in ok])),
                      "per_rank_rule_rank2_above_p99": float(r2.mean()),
                      "per_rank_rule_rank2_above_p99_ci95": V.clopper_pearson(int(r2.sum()), len(ok)),
                      "per_rank_rule_beyond_ge1_and_valid": float(np.mean([x["ranks_beyond_ability"] >= 1 and x["valid"]
                                                                           for x in ok])),
                      "n_insufficient_draws": len(recs) - len(ok)})
            ref = step2b.get(f"{sc}:{nm}")
            if ref is not None:
                c["step2b_reference_rank2_above_p99"] = {"value": ref["frac_rank2_above_p99"], "ci95": ref["ci95"],
                                                         "n_null_draws": 200, "frac_valid": ref["frac_valid"]}
            summ[sc][nm] = c
            cal[sc][nm] = {**c, "_pts": pts}
            store[f"{sc}__{nm}__p_T"] = pts
            store[f"{sc}__{nm}__obs_top4"] = np.array([x["observed_top_eigs"] for x in ok])
            store[f"{sc}__{nm}__null_top4"] = np.stack([x["E_top4"] for x in ok])
    per = [{"rep": r["rep"], **{sc: {nm: _strip(r["res"][sc][nm]) for nm in G2_NULLS} for sc in G2_SCORES}} for r in rows]
    return summ, cal, {"store": store, "per_dataset": per}


def assemble_g3(res: dict, cal: dict) -> tuple[dict, list]:
    rows = sorted(res["G3"], key=lambda r: (r["s"], r["rep"]))
    pw = json.loads((OUT / "v3_F_power.json").read_text())["summary"]
    out = {}
    for s in S_G3:
        out[f"{s:g}"] = {}
        rs = [r for r in rows if r["s"] == s]
        for nm in G2_NULLS:
            recs = [r["res"]["ge4"][nm] for r in rs]
            n = len(recs)
            thr = cal["ge4"][nm]["cal_threshold_p_T_q05"]
            k_raw = sum(g1_reject(x) for x in recs)
            k_cal = sum(g1_reject(x, thr) for x in recs)
            k_r2 = sum(x["valid"] and 2 in x["ranks_above_p99"] for x in recs)
            out[f"{s:g}"][nm] = {
                "n": n, "detect_G1_raw": k_raw / n, "detect_G1_raw_ci95": V.clopper_pearson(k_raw, n),
                "detect_G1_calibrated": k_cal / n, "detect_G1_calibrated_ci95": V.clopper_pearson(k_cal, n),
                "cal_threshold_p_T_q05": thr, "frac_valid": float(np.mean([x["valid"] for x in recs])),
                "detect_p_T_lt_alpha_ignoring_validity": float(np.mean([x["p_T"] < ALPHA for x in recs])),
                "detect_p_S_lt_alpha_and_valid": float(np.mean([x["valid"] and x["p_S"] < ALPHA for x in recs])),
                "per_rank_rule_rank2_above_p99_and_valid": k_r2 / n,
                "per_rank_rule_beyond_ge1_and_valid": float(np.mean([x["valid"] and x["ranks_beyond_ability"] >= 1
                                                                     for x in recs])),
                "F3_reference_2pl_rank2_and_valid": (pw[f"{s:g}"]["by_null"]["2pl:ge4"]["detect_rank2_and_valid"]
                                                     if nm == "2pl_post" else None)}
    per = [{"s": r["s"], "rep": r["rep"], "ge4": {nm: _strip(r["res"]["ge4"][nm]) for nm in G2_NULLS}} for r in rows]
    return out, per


def assemble_g4(res: dict, cal: dict, full_g1: dict) -> tuple[dict, dict]:
    rows = sorted(res["G4"], key=lambda r: (r["kind"], r["idx"]))
    names = V.variant_data("full")["names"]
    sib = sibling_subsets(names)
    rnd = random_control_subsets(names, rng_g(STREAM_G["g4_random_pick"], 1))
    out, store = {"sibling": [], "random_control": []}, {}
    for r in rows:
        kind = "sibling" if r["kind"] == 1 else "random_control"
        meta = (sib if r["kind"] == 1 else rnd)[r["idx"]]
        assert meta["keep"] == r["keep"]
        rec = {"idx": r["idx"], "removed": meta["removed"], "M": r["M"]}
        if r["kind"] == 1:
            rec["kept_from_clusters"] = meta["kept_from_clusters"]
        for sc in G4_NULLS:
            rec[sc] = {}
            for nm in G4_NULLS[sc]:
                fw = _strip(r[sc][nm])
                fw["reject_raw"] = g1_reject(fw)
                c = cal.get(sc, {}).get(nm)
                if c is not None:
                    fw["reject_calibrated_24model_threshold"] = g1_reject(fw, c["cal_threshold_p_T_q05"])
                rec[sc][nm] = fw
                store[f"{kind}{r['idx']}__{sc}__{nm}"] = r[sc][nm]["E_top4"]
        out[kind].append(rec)
    summ = {}
    for kind in ("sibling", "random_control"):
        recs = out[kind]
        summ[kind] = {}
        for sc in G4_NULLS:
            summ[kind][sc] = {}
            for nm in G4_NULLS[sc]:
                fws = [x[sc][nm] for x in recs]
                pts = np.array([f["p_T"] for f in fws])
                d = {"n_subsets": len(fws), "p_T_min_median_max": [float(pts.min()), float(np.median(pts)), float(pts.max())],
                     "p_T_sorted": np.sort(pts).tolist(), "n_valid": int(sum(f["valid"] for f in fws)),
                     "n_reject_raw": int(sum(f["reject_raw"] for f in fws)),
                     "n_p_T_lt_alpha_ignoring_validity": int(np.sum(pts < ALPHA)),
                     "p_S_median": float(np.median([f["p_S"] for f in fws])),
                     "ranks_beyond_ability_counts_0to3": np.bincount([f["ranks_beyond_ability"] for f in fws],
                                                                     minlength=4).tolist(),
                     "ranks_beyond_ability_valid_counts_0to3": np.bincount(
                         [f["ranks_beyond_ability"] for f in fws if f["valid"]], minlength=4).tolist(),
                     "T_argmax_rank_counts_2to4": [int(sum(f["T_argmax_rank"] == j for f in fws)) for j in T_RANKS]}
                if fws and "reject_calibrated_24model_threshold" in fws[0]:
                    d["n_reject_calibrated_24model_threshold"] = int(sum(f["reject_calibrated_24model_threshold"]
                                                                         for f in fws))
                summ[kind][sc][nm] = d
            if sc == "ge4":
                rr = [x["ge4"]["2pl_post"]["reject_raw"] or x["ge4"]["rasch_jml"]["reject_raw"] for x in recs]
                rb = [x["ge4"]["2pl_post"]["reject_raw"] and x["ge4"]["rasch_jml"]["reject_raw"] for x in recs]
                summ[kind][sc]["n_reject_raw_either_null"] = int(sum(rr))
                summ[kind][sc]["n_reject_raw_both_nulls"] = int(sum(rb))
                summ[kind][sc]["n_no_reject_either_null"] = int(len(recs) - sum(rr))
    # per removed model: mean p_T (all-4) over subsets that remove it
    for kind in ("sibling", "random_control"):
        by = {}
        for x in out[kind]:
            for m in x["removed"]:
                by.setdefault(m, []).append(x)
        summ[kind]["ge4_mean_p_T_by_removed_model"] = {
            m.split("/")[-1]: {nm: float(np.mean([x["ge4"][nm]["p_T"] for x in xs])) for nm in G4_NULLS["ge4"]}
            | {"n_subsets": len(xs)} for m, xs in sorted(by.items())}
    summ["full_24_models_G1_reference"] = {sc: {nm: {k: full_g1[sc][nm][k] for k in ("p_T", "valid", "reject_raw")}
                                                for nm in G4_NULLS[sc] if nm in full_g1[sc]} for sc in G4_NULLS}
    return {"summary": summ, "subsets": out}, store


def assemble_g5() -> dict:
    """Plain validity (p01 <= lambda_1 <= p99) for the SP column of Table A2 and of the power study (stored F outputs)."""
    F = json.loads((OUT / "v3_F_softpref.json").read_text())
    calib = F["step2_soft_preference"]["calibration"]["by_variant"]
    fits = F["step2_soft_preference"]["fits"]
    z = np.load(OUT / "raw_v3_F_sp_null_eigs.npz")
    table = {}
    for v in V.CAL_VARIANTS:
        data, _ = V.sp_data_of(v)
        _, obs = V.observed_keeps(data)
        key = V.fit_key(v, True)
        table[v] = {}
        for sc in SCORES:
            E = z[f"{key}__{sc}"]
            valid = V.lambda1_valid(E, obs[sc])
            stored = fits[key]["null"][sc]["null_valid_lambda1_in_p01_p99"]
            if valid != stored:
                raise AssertionError(f"plain validity differs from the stored record: {v} {sc}")
            c = calib[v][sc]
            thr = c["cal_threshold_pct_99_rank1to4"]
            po = c["observed_pct_rank1to4"]
            above_cal = [j + 1 for j in range(1, N_RANKS) if po[j] > thr[j]]
            rs = V.rank_summary(E, obs[sc])
            table[v][sc] = {
                "plain_valid": bool(valid), "lambda1_pct": po[0], "previous_calibrated_valid": c["lambda1_calibrated_valid"],
                "rank2_cal_p": c["cal_p_rank1to4"][1], "rank2_cal_p_mc": c["cal_p_mc_rank1to4"][1],
                "cal_p_rank2to4": c["cal_p_rank1to4"][1:], "rank2_uncalibrated_mc_p": rs["mc_p_upper_rank1to4"][1],
                "pct_rank1to4": po, "ranks_above_cal_threshold_2to4": above_cal,
                "ranks_beyond_ability_calibrated": V.contiguous_from_2(above_cal) if valid else None,
                "ranks_above_p99_uncalibrated": rs["observed_ranks_above_p99"],
                "table_A2_cell": (f"{c['cal_p_rank1to4'][1]:.3f}" if valid else "inv."),
                "table_A2_cell_previous": (f"{c['cal_p_rank1to4'][1]:.3f}" if c["lambda1_calibrated_valid"] else "inv.")}
    # Power under G5: plain-valid AND rank-2 percentile above the calibrated p99 threshold.
    pw = json.loads((OUT / "raw_v3_F_power.json").read_text())["results"]
    pws = json.loads((OUT / "v3_F_power.json").read_text())["summary"]
    thr2 = {sc: calib["full"][sc]["cal_threshold_pct_99_rank1to4"][1] for sc in V.POWER_SCORES}
    power = {}
    for s in V.S_GRID:
        rs_ = [r for r in pw if r["s"] == s]
        power[f"{s:g}"] = {}
        for sc in V.POWER_SCORES:
            ds = [r["sp"][sc] for r in rs_]
            n = len(ds)
            ok = [x for x in ds if not x.get("insufficient_draws")]
            k = sum(x["valid"] and x["pct_rank1to4"][1] > thr2[sc] for x in ok)
            k_any = sum(x["valid"] and any(x["pct_rank1to4"][j] > calib["full"][sc]["cal_threshold_pct_99_rank1to4"][j]
                                           for j in (1, 2, 3)) for x in ok)
            prev = pws[f"{s:g}"]["by_null"][f"sp:{sc}"]
            power[f"{s:g}"][sc] = {"n": n, "detect_plain_valid_and_rank2_above_cal_threshold": k / n,
                                   "ci95": V.clopper_pearson(k, n),
                                   "detect_plain_valid_and_any_rank2to4_above_cal_threshold": k_any / n,
                                   "frac_plain_valid": float(np.mean([x["valid"] for x in ds])),
                                   "previous_detect_rank2_and_valid_p99": prev["detect_rank2_and_valid"],
                                   "previous_detect_rank2_calibrated_no_validity": prev.get("detect_rank2_calibrated"),
                                   "cal_threshold_pct_rank2": thr2[sc]}
    return {"rule": "a null is valid iff p01 <= observed lambda_1 <= p99 of its lambda_1 (lambda1_valid); no "
                    "calibrated validity", "sp_table_A2": table, "power_sp_column": power}


def assemble_g6(res: dict) -> tuple[dict, dict]:
    out, store = {}, {}
    full = json.loads((OUT / "v3_F_softpref.json").read_text())["step1_construction_family"]["table"]["full"]
    for r in res["G6"]:
        key = f"{r['score']}__{r['variant']}"
        rec = {k: r[k] for k in r if k not in ("obs", "E")}
        rec["observed_top_eigs"] = np.asarray(r["obs"])[:N_RANKS].tolist()
        rec["nulls"] = {}
        for nm, E in r["E"].items():
            fw = familywise(r["obs"], E)
            fw["reject_raw"] = g1_reject(fw)
            fw["as_scored_reference"] = {"pct_rank1to4": full[r["score"]]["observed_pct_rank1to4_by_null"][nm],
                                         "valid": full[r["score"]]["null_valid"][nm],
                                         "ranks_beyond_ability": full[r["score"]]["ranks_beyond_ability_all_nulls"][nm]}
            rec["nulls"][nm] = fw
            store[f"{key}__{nm}"] = E
        out[key] = rec
    return out, store


DEVIATIONS = [
    "G1 null set: 'every null that is VALID' is read as the step-1 suite of each score (v3_softpref.family_nulls) "
    "plus the primary soft-preference null (common tau, base fit). The tau = 0, per-model tau, penalty and "
    "bootstrap-averaged SP variants are sensitivity fits and are not included. Validity is decided on the 2000 NEW "
    "draws of each null, so it can differ from step 1 near the 1st/99th percentile.",
    "G1 data: the full 24-model data only (the pre-registration names no data variants; the no-Llama-3.2-1B and "
    "no-option-referencing variants are not rerun).",
    "G1 'against the other draws' is implemented literally: T_d uses the other n - 1 NULL draws (the observed value "
    "is not among them), T_obs uses all n draws.",
    "G2 calibrates the G1 test only for the 2PL and Rasch (as pre-registered). Rotation 0 and all-4 are the "
    "pre-registered scores; >=1, >=2, >=3 were added because the budget allowed (the mean score has no 2PL/Rasch). "
    "G2 p_T uses 500 null draws, the real-data G1 p_T uses 2000: the calibrated threshold (5th percentile of the "
    "500-draw p_T) is applied to the 2000-draw p_T (both estimate the same p; the 500-draw grid is 1/501).",
    "Found after the run (caveat, not a change): where T saturates at 100 (rotation 0, >=1, >=2, >=3 under "
    "most nulls), p_T is set by the number of null draws whose own LOO T also reaches 100, so it sits at a few "
    "grid steps (2000 draws: about 0.002; 500 draws: 0.006-0.008). There, comparing a 2000-draw p_T with a "
    "500-draw calibrated threshold is a grid artifact: the calibrated G1 decisions for >=2 (2PL reject) and "
    ">=3 (2PL, Rasch no reject) are not reliable. The all-4 decisions do not saturate (T = 98.6, 99.1) and "
    "use the raw threshold (G2 FP = 0).",
    "G2 datasets use new seeds (namespace G), not the step-2b datasets; the step-2b pipeline is reproduced exactly "
    "on 2 of its own datasets in the tests.",
    "G3 uses 500 null draws per dataset (the pre-registration gives no number; equal to G2 so the calibrated "
    "threshold applies). s = 0 is not rerun: its size is G2's all-4 false-positive rate (same pipeline).",
    "G4 rotation 0: the 2PL and Rasch are run as named, plus the nulls valid on the full rotation-0 data "
    "(3PL, KG uniform, letter-aware), since G1 applies to valid nulls only; validity is decided per subset. "
    "A subset 'rejects' on all-4 when G1 rejects under at least one of the two nulls (both counts reported). "
    "The G2 threshold (24 models) is reported for the 19-model subsets as a reference only; the subset decisions "
    "use the raw alpha = 0.05.",
    "G5 power column: 'the procedure the paper uses on real data' after G5 = plain validity AND rank-2 percentile "
    "above the calibrated 99% threshold; computed from the stored power datasets (no new simulation).",
    "G6 drop on all-4: two definitions are run, drop_affected (questions with an all-4 entry decided by unparsed "
    "replies: no parsed rotation wrong and >= 1 unparsed; 95 questions) and drop_any (any unparsed reply in any "
    "rotation; 217 questions). Rotation 0 drops the 112 questions with an unparsed rotation-0 reply (as App. B).",
    "G6 impute: the binary nulls cannot be fitted to fractional data, so each null is fitted to the as-scored "
    "binary matrix, and the observed matrix and every null draw are imputed at the same cells by "
    "v2lib.impute_column_mean (column mean of the other entries); columns constant after imputation are dropped "
    "(counts reported). All-4 imputes the entries decided by unparsed replies (124 cells).",
]


def part_assemble() -> None:
    res = V.run_checkpointed(all_items(), CKPT, n_proc=1)   # all done: loads the checkpoints only
    g2_summ, g2_cal, g2_raw = assemble_g2(res)
    g1, g1_store = assemble_g1(res, g2_cal)
    g3, g3_per = assemble_g3(res, g2_cal)
    g4, g4_store = assemble_g4(res, g2_cal, g1)
    g5 = assemble_g5()
    g6, g6_store = assemble_g6(res)
    np.savez_compressed(OUT / "raw_v3_G_g1_null_eigs.npz", **g1_store)
    np.savez_compressed(OUT / "raw_v3_G_g2.npz", **g2_raw["store"])
    np.savez_compressed(OUT / "raw_v3_G_g4_null_top4.npz", **g4_store)
    np.savez_compressed(OUT / "raw_v3_G_g6_null_eigs.npz", **g6_store)
    write_json("raw_v3_G_g2", {"per_dataset": g2_raw["per_dataset"]})
    write_json("raw_v3_G_g3", {"per_dataset": g3_per})
    write_json("raw_v3_G_g4", g4)
    # Decision table and rule A/B/C.
    full_valid = {sc: [nm for nm, fw in g1[sc].items() if not fw.get("insufficient_draws") and fw["valid"]]
                  for sc in SCORES}
    ge4 = {nm: g1["ge4"][nm] for nm in ("2pl_post", "rasch_jml")}
    rej = {nm: bool(ge4[nm]["valid"] and ge4[nm]["reject_decision"]) for nm in ge4}
    sib = g4["summary"]["sibling"]["ge4"]
    rnd = g4["summary"]["random_control"]["ge4"]
    if not any(rej.values()):
        rule, why = "A", "G1 on all-4 does not reject (decision threshold) under the 2PL or the Rasch null"
    elif all(rej.values()):
        if sib["n_no_reject_either_null"] >= 18 and rnd["n_reject_raw_either_null"] > N_G4_RANDOM / 2:
            rule, why = "B", "G1 rejects under both nulls; >= 18 of 24 sibling subsets no longer reject; the control mostly rejects"
        elif sib["n_no_reject_either_null"] >= 18:
            rule, why = "mixed (nearest B)", "sibling removal removes it, but the random control does not keep rejecting in most"
        else:
            rule, why = "C", "G1 rejects under both nulls and sibling removal does not remove it"
    else:
        rule, why = "mixed", f"G1 rejects under one null only: {rej}"
    out = {
        "meta": {"preregistration": PREREG.read_text() if PREREG.exists() else None,
                 "seed_namespace": {"SEED_G": SEED_G, "streams": STREAM_G},
                 "n_draws": {"G1": N_G1, "G2_datasets": N_G2, "G2_null": N_G2_NULL, "G3_datasets_per_s": N_G3,
                             "G3_null": N_G3_NULL, "G4": N_G4, "G6": N_G6},
                 "statistic": __doc__, "alpha": ALPHA, "fp_calibration_limit": FP_CAL_LIMIT},
        "G1_familywise": {"by_score_and_null": g1, "valid_nulls_by_score": full_valid,
                          "decision_table": {sc: {nm: {"valid": fw["valid"], "p_T": fw["p_T"], "T_pct": fw["T_pct"],
                                                      "p_S": fw["p_S"], "reject_raw": fw["reject_raw"],
                                                      "reject_decision": fw["reject_decision"],
                                                      "decision_rule": fw["decision_rule"],
                                                      "ranks_beyond_ability_per_rank_rule": fw["ranks_beyond_ability"],
                                                      "pct_rank1to4": fw["pct_rank1to4"]}
                                                 for nm, fw in g1[sc].items() if not fw.get("insufficient_draws")}
                                             for sc in SCORES}},
        "G2_calibration_under_SP_truth": g2_summ,
        "G3_power": g3,
        "G4_siblings": g4["summary"],
        "G5_single_validity_rule": g5,
        "G6_unparsed_variants": g6,
        "decision": {"rule": rule, "why": why, "all4_reject_by_null": rej},
        "Deviations": DEVIATIONS,
    }
    write_json("v3_G_familywise", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["run", "assemble", "all"])
    ap.add_argument("--nproc", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    if a.part in ("run", "all"):
        part_run(a.nproc)
    if a.part in ("assemble", "all"):
        part_assemble()
    print(f"[G] done ({time.time() - t0:.0f}s)", flush=True)
