"""
Construction family, soft-preference null and power for the 24-model MMLU matrix (4 option rotations).
Run: python v3_softpref.py <part>; outputs raw_v3_F_*.json/npz and v3_F_softpref.json, v3_F_power.json.

Parts:
  family   : step 1, scores {rot0, >=1..>=4, mean of 4} under the v3_across null suite, per data variant.
  base     : primary soft-preference fits (common tau) of full and no_llama1b; the start of every simulation.
  sp       : step 2, soft-preference fits: FD gradient check, recovery, PPC, nulls, penalty/quadrature sensitivity.
  cal      : step 2 calibration, parametric bootstrap of the whole pipeline (simulate, refit, own null).
  power    : step 3, planted second ability on the STEM block.
  calib    : step 2b, the step-1 nulls refitted to data simulated from the soft-preference fit.
  calib_bc : >=2 / 2PL false-positive rate under plug-in and bias-corrected parameters (needs `cal`).
  char     : step 4, characterization of the directions above the soft-preference null.
  assemble : the final JSONs from the raw parts.   all: everything, checkpointed, then assemble.

Rule: rank j rejected when observed lambda_j > null p99_j; a null is valid when p01 <= observed lambda_1 <= p99.
Simulations condition on the observed unparsed mask (unparsed = wrong) and redraw constant kept columns.

Soft-preference model (model m, question i, rotation r, original option j; option j shown at letter (j + r) mod 4):
    P(choose j | parsed, e_mi) = softmax_j(V_mirj),
    V_mirj = b_ij + 1[j = g_i] (a_i theta_m + e_mi) + log pi_m(l_r(j)),   b_{i, g_i} = 0,
    e_mi ~ N(0, tau^2), shared by the 4 rotations of a cell (tau = 0: no cell effect).
Penalized ML (Gauss-Hermite over e):
    log L = sum_{m,i} log int prod_{r parsed} P(c_mir | e) phi_tau(e) de,
    theta ~ N(0, 1), a ~ N(0, SIGMA_A^2), b ~ N(0, SIGMA_B^2), lambda ~ N(0, SIGMA_LAM^2),
    log pi_m = log_softmax([0, lambda_m]).
Reported parameters are standardized to theta mean 0, SD 1 (theta -> c + s theta is absorbed by a, b).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy
from scipy.stats import spearmanr

import v3_across as V3
from null_models import corr_eigs, perm_pvalue_upper
from v2lib import OUT, V2, label_eta_sq, write_json

N_OPT = 4
N_ROT = 4
LETTERS = "ABCD"
# LETTER_OF[r, j] = shown letter of original option j in rotation r.
LETTER_OF = (np.arange(N_OPT)[None, :] + np.arange(N_ROT)[:, None]) % N_OPT

SEED_FAMILY = V3.SEED          # same seed and child streams as the existing suites (reproduces them)
SEED_F = 20261014              # every new random stream of this file is spawned from this seed
N_DRAWS_FAMILY = V3.N_DRAWS    # 1000 per null
N_SP_NULL = 1000               # soft-preference null datasets per fit
N_PPC = 500                    # posterior-predictive datasets
N_CAL = 200                    # calibration: bootstrap datasets per variant (simulate, refit, own null)
N_CAL_NULL = N_SP_NULL         # null draws per bootstrap dataset (= the real data's, so percentiles compare)
CAL_VARIANTS = ("full", "no_llama1b")
N_BOOT_PER = 10                # bootstrap-averaged null: draws per refit on the observed kept columns ...
N_BOOT_MAX_ATTEMPTS = 100      # ... redrawn up to this many attempts; a refit short of N_BOOT_PER is excluded
N_RECOVERY = 3                 # parameter-recovery replicates per model
N_POWER = 150                  # simulated datasets per planted strength
N_POWER_NULL = N_CAL_NULL      # soft-preference null draws per refit in the power study (= calibration's)
N_POWER_2PL_NULL = 200         # 2PL null draws per dataset in the power study
MIN_NULL_DRAWS = 100           # a null with fewer successful draws gives no percentile or decision
S_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
K_GH = 41                      # Gauss-Hermite nodes for e_mi (checked against 81 in `sp`)
SIGMA_A = 4.0
SIGMA_B = 4.0
SIGMA_LAM = 5.0
PENALTY_SENSITIVITY = (2.0, 8.0)   # SIGMA_A = SIGMA_B alternatives
N_PROC = 8
N_TAUM_STARTS = 4
MODES = (False, True, "model")
MODE_TAG = {False: "tau0", True: "tau", "model": "taum"}
MODE_CODE = {False: 0, True: 1, "model": 2}
N_RANKS = 4
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"

# Hendrycks et al. (2021) MMLU categories (categories.py): STEM = physics, chemistry, biology,
# computer science, math, engineering. The 18 of our 50 subjects in that category (180 questions).
STEM_SUBJECTS = (
    "abstract_algebra", "astronomy", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics", "high_school_physics",
    "high_school_statistics", "machine_learning")

# Nominal parameter counts (billions) from the model names / model cards; used only in step 4.
MODEL_SIZE_B = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.7, "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5, "microsoft/phi-2": 2.7, "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "microsoft/Phi-3.5-mini-instruct": 3.8, "Qwen/Qwen2.5-3B-Instruct": 3.1, "Qwen/Qwen2-7B-Instruct": 7.6,
    "Qwen/Qwen2.5-7B-Instruct": 7.6, "mistralai/Mistral-7B-Instruct-v0.2": 7.2,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.2, "HuggingFaceH4/zephyr-7b-beta": 7.2,
    "allenai/OLMo-2-1124-7B-Instruct": 7.3, "01-ai/Yi-1.5-6B-Chat": 6.1,
    "deepseek-ai/deepseek-llm-7b-chat": 6.9, "ibm-granite/granite-3.1-8b-instruct": 8.2,
    "mistralai/Ministral-8B-Instruct-2410": 8.0, "google/gemma-2-2b-it": 2.6,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0, "google/gemma-2-9b-it": 9.2,
    "mistralai/Mistral-Small-Instruct-2409": 22.2, LLAMA_1B: 1.2,
    "meta-llama/Llama-3.2-3B-Instruct": 3.2, "meta-llama/Meta-Llama-3-8B-Instruct": 8.0}


def _ge(k: int):
    return lambda T: (T.sum(axis=2) >= k).astype(np.int8)


SCORES = {"rot0": lambda T: T[:, :, 0].astype(np.int8), "ge1": _ge(1), "ge2": _ge(2), "ge3": _ge(3),
          "ge4": _ge(4), "mean": lambda T: T.mean(axis=2)}
SCORE_LABEL = {"rot0": "rotation 0", "ge1": ">=1 of 4", "ge2": ">=2 of 4", "ge3": ">=3 of 4",
               "ge4": "all 4 (>=4)", "mean": "mean of 4"}
POWER_SCORES = ("rot0", "ge4", "ge2", "mean")
BINARY_POWER_SCORES = ("rot0", "ge4", "ge2")


def keep_mask(X: np.ndarray) -> np.ndarray:
    """Non-constant columns (binary: split columns)."""
    return X.std(axis=0) > 0


def pct_of(obs: float, null: np.ndarray) -> float:
    """Percentile of obs in the null sample: 100 (#{null < obs} + #{null == obs} / 2) / n."""
    null = np.asarray(null)
    return float(100.0 * (np.mean(null < obs) + 0.5 * np.mean(null == obs)))


# Random streams: SeedSequence([a, 0]) == SeedSequence([a]) (trailing zeros are dropped), so every key
# component is a positive integer and the first one names the stream.
STREAM = {"sp": 1, "sp_sens": 2, "taum_starts": 3, "cal": 7, "power": 9, "char": 11, "calib": 13, "calib_bc": 15}


def rng_for(*key: int) -> np.random.Generator:
    """default_rng([SEED_F, *key]); every component of key must be a positive integer."""
    if not key or not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k > 0 for k in key):
        raise ValueError(f"stream key components must be positive integers: {key}")
    return np.random.default_rng([SEED_F, *[int(k) for k in key]])


def contiguous_from_2(ranks: list[int]) -> int:
    return V3.contiguous_from_2(ranks)


def null_bands(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p01, p99) of the null lambda_1..4 from one np.percentile call (validity and rejection share it)."""
    q = np.percentile(E[:, :N_RANKS], [1, 99], axis=0)
    return q[0], q[1]


def lambda1_valid(E: np.ndarray, obs: np.ndarray) -> bool:
    """Null validity: p01 <= observed lambda_1 <= p99 of the null lambda_1."""
    p01, p99 = null_bands(E)
    return bool(p01[0] <= obs[0] <= p99[0])


def null_z(E: np.ndarray, obs: np.ndarray) -> list[float]:
    """(observed lambda_j - null mean) / null SD, j = 1..4 (a tie-free companion of the percentile)."""
    sd = E[:, :N_RANKS].std(axis=0)
    return ((obs[:N_RANKS] - E[:, :N_RANKS].mean(axis=0)) / np.where(sd > 0, sd, np.nan)).tolist()


def _detect(E: np.ndarray, obs: np.ndarray) -> dict:
    """The paper's rule on one null sample E (n, >= 4): rank j rejected when obs_j > p99_j, valid when
    p01 <= obs_1 <= p99. Fewer than MIN_NULL_DRAWS draws: no decision (neither detected nor valid)."""
    n = int(E.shape[0])
    if n < MIN_NULL_DRAWS:
        return {"n_draws": n, "insufficient_draws": True, "rank2_above_p99": False, "valid": False,
                "ranks_above_p99": [], "observed_top_eigs": obs[:N_RANKS].tolist()}
    p01, p99 = null_bands(E)
    above = [int(j + 1) for j in range(N_RANKS) if obs[j] > p99[j]]
    pct = [pct_of(obs[j], E[:, j]) for j in range(N_RANKS)]
    return {"n_draws": n, "insufficient_draws": False, "rank2_above_p99": bool(obs[1] > p99[1]),
            "valid": bool(p01[0] <= obs[0] <= p99[0]), "lambda1_pct": pct[0], "pct_rank1to4": pct,
            "z_rank1to4": null_z(E, obs), "ranks_above_p99": above, "p99": p99.tolist(), "p01": p01.tolist(),
            "observed_top_eigs": obs[:N_RANKS].tolist()}


def rank_summary(E: np.ndarray, obs: np.ndarray) -> dict:
    """Rank-matched p99 rule and percentiles of the observed lambda_1..4 in a null sample E (n, >=4)."""
    d = _detect(E, obs)
    if d["insufficient_draws"]:
        return {"n_draws": d["n_draws"], "insufficient_draws": True, "observed_top_eigs": obs[:N_RANKS].tolist()}
    return {"n_draws": d["n_draws"], "rank_eigs_p99": d["p99"], "rank_eigs_p01": d["p01"],
            "rank_eigs_median": np.median(E[:, :N_RANKS], axis=0).tolist(),
            "observed_top_eigs": obs[:N_RANKS].tolist(), "observed_pct_rank1to4": d["pct_rank1to4"],
            "observed_z_rank1to4": d["z_rank1to4"],
            "mc_p_upper_rank1to4": [perm_pvalue_upper(obs[j], E[:, j]) for j in range(N_RANKS)],
            "observed_ranks_above_p99": d["ranks_above_p99"], "ranks_beyond_ability": contiguous_from_2(d["ranks_above_p99"]),
            "null_valid_lambda1_in_p01_p99": d["valid"]}


_AGG = re.compile(r"\b(?:the above|(?:all|none|both|neither|any|either|each) of (?:these|those|the options))\b", re.I)
_LETTER = (
    re.compile(r"\b(?i:both)\s+\(?[A-D]\)?\s*(?:,|and|&)\s*\(?[A-D]\)?"),         # "Both A and C"
    re.compile(r"\(?\b[A-D]\)?\s+(?:and|or|&)\s+\(?[A-D]\)?\s+(?i:only)\b"),      # "A and B only"
    re.compile(r"\b(?i:only)\s+\(?[A-D]\)?\s+(?:and|or)\s+\(?[A-D]\)?"),          # "only A and B"
    re.compile(r"\([A-D]\)\s*,?\s*\([A-D]\)"),                                    # "(B) (C) and (D)"
    re.compile(r"^\s*\(?[A-D]\)?\s*(?:,|and|or|&)\s*\(?[A-D]\)?\W*$"),           # bare "A and B"
)
_COMBO_WORD = re.compile(r"\b(both|neither)\b", re.I)
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def option_reference_rule(choices: list[str]) -> list[tuple[int, str, str]]:
    """[(option index, rule, matched text)] for the options that refer to the other options.

      aggregate     : "... the above", or "all/none/both/neither/any/either/each of these/those/the options";
      letter        : names other options by capital letter A-D ("Both A and C", "A and B only", "(B) (C)");
      content_combo : "both"/"neither" plus the content words of at least two other options.
    References to stem items ("I and II only") and "either ... or" combinations are not counted.
    """
    out = []
    for k, c in enumerate(choices):
        m = _AGG.search(c)
        if m:
            out.append((k, "aggregate", m.group(0)))
            continue
        hit = next((p.search(c) for p in _LETTER if p.search(c)), None)
        if hit:
            out.append((k, "letter", hit.group(0)))
            continue
        if _COMBO_WORD.search(c):
            tk = _tokens(c)
            n_sub = sum(1 for kk, o in enumerate(choices) if kk != k and _tokens(o) and _tokens(o) <= tk)
            if n_sub >= 2:
                out.append((k, "content_combo", c))
    return out


def find_option_referencing(questions: list[dict]) -> list[dict]:
    res = []
    for i, q in enumerate(questions):
        hits = option_reference_rule(q["choices"])
        if hits:
            res.append({"index": i, "subject": q["subject"], "gold_letter": q["gold_letter"],
                        "choices": q["choices"],
                        "hits": [{"option": LETTERS[k], "rule": r, "match": t} for k, r, t in hits]})
    return res


# Reviewer's pattern, only to compare counts.
REVIEWER_PATTERN = re.compile(r"(of the above|of these|both [A-D]\b|\b[A-D] and [A-D]\b|neither|either)", re.I)


def load_questions() -> list[dict]:
    return json.loads((V2 / "questions_mmlu.json").read_text())


def variant_data(variant: str, rep: str = "A") -> dict:
    """load_rep(rep) restricted to a variant: full, no_llama1b (row removed), no_optref (columns removed)."""
    d = V3.load_rep(rep)
    z = np.load(V3.REPS[rep], allow_pickle=False)
    d["chosen"] = z["chosen_option"].astype(int)                                   # (M, N, 4) original option
    if variant == "full":
        return d
    if variant == "no_llama1b":
        keep = np.array([n != LLAMA_1B for n in d["names"]])
        assert keep.sum() == len(d["names"]) - 1
        for k in ("CR", "shown_choice", "fam", "lp", "chosen"):
            d[k] = d[k][keep]
        d["names"] = [n for n in d["names"] if n != LLAMA_1B]
        return d
    if variant == "no_optref":
        drop = {r["index"] for r in find_option_referencing(load_questions())}
        keep = np.array([i not in drop for i in range(d["CR"].shape[1])])
        for k in ("CR", "shown_choice", "lp", "chosen"):
            d[k] = d[k][:, keep]
        d["gold"], d["subj"] = d["gold"][keep], d["subj"][keep]
        d["question_index"] = np.flatnonzero(keep)
        return d
    raise ValueError(variant)


VARIANTS = ("full", "no_llama1b", "no_optref")


def family_nulls(score: str) -> list[str]:
    if score == "mean":
        return ["kg_letter4", "kg_uniform4", "indep4_2pl"]
    base = ["rasch_jml", "fixed_margin", "2pl_post", "3pl_post", "ba_a4", "kg_letter4", "kg_uniform4"]
    return base + (["letter_post"] if score == "rot0" else [])


def family_task(args) -> tuple[str, str, dict, dict]:
    variant, score = args
    d = variant_data(variant)
    CR, gold = d["CR"], d["gold"]
    fits, kg_info = V3.kg_fits(d)
    derive = SCORES[score]
    X = derive(CR)
    shown_gold = np.broadcast_to(gold, X.shape) if score == "rot0" else None
    ctx = V3.Context(X, CR, gold, shown_gold=shown_gold, pi=fits["kg_letter4"][1], derive=derive, kg_fits=fits)
    store = {}
    suite = V3.run_suite(ctx, family_nulls(score), SEED_FAMILY, N_DRAWS_FAMILY, f"{variant}_{score}", store)
    pct, valid = {}, {}
    for nm in family_nulls(score):
        E = store[f"{variant}_{score}__{nm}"]
        pct[nm] = [pct_of(ctx.obs[j], E[:, j]) for j in range(N_RANKS)]
        valid[nm] = lambda1_valid(E, ctx.obs)
    beyond_valid = {nm: suite["ranks_beyond_ability"][nm] for nm in valid if valid[nm]}
    res = {"variant": variant, "score": score, "spectrum": V3.spectrum(X), "accuracy_mean": float(X.mean()),
           "suite": suite, "observed_pct_rank1to4_by_null": pct, "null_valid_lambda1_in_p01_p99": valid,
           "ranks_beyond_ability_valid_nulls": beyond_valid,
           "ranks_beyond_ability_valid_min_max": ([min(beyond_valid.values()), max(beyond_valid.values())]
                                                  if beyond_valid else None),
           "kg_fit_info": kg_info["fits"]}
    return variant, score, res, {f"{variant}_{score}__{k.split('__')[1]}": v for k, v in store.items()}


def family_jobs() -> list[tuple[str, str]]:
    return [(v, s) for v in VARIANTS for s in SCORES]


def part_family() -> None:
    family_collect(run_pool(family_task, family_jobs()))


def family_collect(out: list) -> None:
    res, store = {}, {}
    for v, s, r, st in out:
        res.setdefault(v, {})[s] = r
        store.update(st)
    optref = find_option_referencing(load_questions())
    q = load_questions()
    rev = [i for i, x in enumerate(q) if any(REVIEWER_PATTERN.search(c) for c in x["choices"])]
    mine = [r["index"] for r in optref]
    np.savez_compressed(OUT / "raw_v3_F_family_eigs.npz", **store)
    write_json("raw_v3_F_family", {
        "meta": {"seed": SEED_FAMILY, "n_draws_per_null": N_DRAWS_FAMILY, "variants": list(VARIANTS),
                 "scores": SCORE_LABEL, "nulls_by_score": {s: family_nulls(s) for s in SCORES},
                 "validity_rule": "a null counts as evidence only if p01 <= observed lambda_1 <= p99, p01 and p99 "
                                  "from one np.percentile call on the null lambda_1 (the call that gives the "
                                  "rank p99s); percentiles reported as 100 (#{null < obs} + #{null == obs}/2) / n",
                 "note_3pl": "3pl_post uses c = 0.25 for every score; for the >=k scores this floor is nominal "
                             "(a uniform guesser is correct in >= k of 4 rotations with another probability)",
                 "note_letter_post": "letter_post (rotation-0 know-or-guess) exists only for rotation 0; the "
                                     "4-rotation know-or-guess nulls kg_letter4 / kg_uniform4 apply to every score"},
        "option_referencing": {"n": len(optref), "indices": mine, "questions": optref,
                               "reviewer_pattern": REVIEWER_PATTERN.pattern, "reviewer_pattern_n": len(rev),
                               "reviewer_only": [{"index": i, "choices": q[i]["choices"]} for i in rev if i not in mine],
                               "rule_only": [i for i in mine if i not in rev]},
        "results": res})


@dataclass
class SPData:
    """Choices as original option index (-1 = unparsed) and derived arrays."""
    chosen: np.ndarray          # (M, N, R) int
    gold: np.ndarray            # (N,) int
    Y: np.ndarray = field(init=False)        # (M, N, R, J) one-hot of parsed choices
    parsed: np.ndarray = field(init=False)   # (M, N, R) 1.0 if parsed
    yg: np.ndarray = field(init=False)       # (M, N) number of parsed rotations that chose gold
    G: np.ndarray = field(init=False)        # (N, J) bool, gold option
    nongold: np.ndarray = field(init=False)  # (N, 3) int, the non-gold options in increasing order
    rho: np.ndarray = field(init=False)      # (M,) ML unparsed rate
    n_unp: np.ndarray = field(init=False)    # (M,)
    n_par: np.ndarray = field(init=False)    # (M,)

    def __post_init__(self):
        ch = np.asarray(self.chosen, dtype=int)
        assert ch.ndim == 3 and ch.shape[2] == N_ROT and ch.min() >= -1 and ch.max() < N_OPT
        self.chosen = ch
        self.gold = np.asarray(self.gold, dtype=int)
        self.Y = (ch[..., None] == np.arange(N_OPT)).astype(np.float64)
        self.parsed = (ch >= 0).astype(np.float64)
        self.G = self.gold[:, None] == np.arange(N_OPT)[None, :]
        self.nongold = np.array([[j for j in range(N_OPT) if j != g] for g in self.gold], dtype=int)
        self.yg = (ch == self.gold[None, :, None]).sum(axis=2).astype(np.float64)
        self.n_unp = (ch < 0).reshape(ch.shape[0], -1).sum(axis=1).astype(np.float64)
        self.n_par = (ch >= 0).reshape(ch.shape[0], -1).sum(axis=1).astype(np.float64)
        self.rho = self.n_unp / (self.n_unp + self.n_par)

    @property
    def M(self) -> int:
        return self.chosen.shape[0]

    @property
    def N(self) -> int:
        return self.chosen.shape[1]

    def correct(self) -> np.ndarray:
        return (self.chosen == self.gold[None, :, None]).astype(np.int8)

    def unparsed_loglik(self) -> float:
        """sum_m n_unp log rho + n_par log(1 - rho) at the ML rho (0 log 0 = 0)."""
        return float(np.sum(xlogy(self.n_unp, self.rho) + xlogy(self.n_par, 1.0 - self.rho)))


@dataclass
class SPSpec:
    M: int
    N: int
    with_tau: bool | str      # False: tau = 0; True: one common tau; "model": tau_m per model
    K: int = K_GH
    sigma_a: float = SIGMA_A
    sigma_b: float = SIGMA_B
    sigma_lam: float = SIGMA_LAM

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Probabilists' Gauss-Hermite nodes z and log weights for E_{z ~ N(0,1)}[f(z)]."""
        if not self.with_tau:
            return np.zeros(1), np.zeros(1)
        assert 1 <= self.K <= 250, "numpy hermegauss weights overflow above ~350 nodes"
        z, w = hermegauss(self.K)
        return z, np.log(w) - np.log(np.sqrt(2.0 * np.pi))

    @property
    def n_tau(self) -> int:
        return {False: 0, True: 1, "model": self.M}[self.with_tau]

    @property
    def n_params(self) -> int:
        return self.M + self.N + 3 * self.N + 3 * self.M + self.n_tau


def unpack(x: np.ndarray, spec: SPSpec, nongold: np.ndarray) -> dict:
    M, N = spec.M, spec.N
    o = 0
    theta = x[o:o + M]; o += M
    a = x[o:o + N]; o += N
    bfree = x[o:o + 3 * N].reshape(N, 3); o += 3 * N
    lam = x[o:o + 3 * M].reshape(M, 3); o += 3 * M
    logtau = x[o:o + spec.n_tau].copy() if spec.n_tau else -np.inf
    if spec.with_tau is True:
        logtau = float(logtau[0])
    b = np.zeros((N, N_OPT))
    b[np.arange(N)[:, None], nongold] = bfree
    lam_full = np.concatenate([np.zeros((M, 1)), lam], axis=1)
    logpi = lam_full - logsumexp(lam_full, axis=1, keepdims=True)
    return {"theta": theta, "a": a, "bfree": bfree, "b": b, "lam": lam, "logpi": logpi,
            "tau": np.exp(logtau) if spec.with_tau else 0.0, "logtau": logtau}


def pack(theta, a, bfree, lam, logtau=None) -> np.ndarray:
    parts = [theta, a, np.asarray(bfree).ravel(), np.asarray(lam).ravel()]
    if logtau is not None:
        parts.append(np.atleast_1d(logtau))
    return np.concatenate(parts).astype(np.float64)


def sp_terms(x: np.ndarray, data: SPData, spec: SPSpec) -> tuple[float, np.ndarray, float]:
    """(penalized log-lik of the parsed choices, its gradient, unpenalized log-lik of the parsed choices).

    Only the gold logit depends on e = tau z_k, so the non-gold part is reduced once:
      logS_ng[m,i,r] = log sum_{j != g_i} exp(b_ij + LP[m,r,j]),     LP[m,r,j] = log pi_m(l_r(j))
      Lg[m,i,k,r]    = a_i theta_m + tau z_k + LP[m,r,g_i],          D = logaddexp(Lg, logS_ng)
      ll_k[m,i,k]    = sum_r parsed (Y . (b + LP)) + yg (a_i theta_m + tau z_k) - sum_r parsed D
      cell[m,i]      = logsumexp_k(ll_k + log w_k).
    Gradient: posterior node weights post = exp(ll_k + log w_k - cell);
      d cell / d g_mik = post (yg - sum_r parsed p_g),  p_g = exp(Lg - D);
      averaged choice probabilities Pbar[m,i,r,j] = sum_k post p_j(k): for j != g_i,
      exp(b_ij + LP[m,r,j] + logsumexp_k(log post - D)); residual Rbar = Y - parsed Pbar gives
      d/d b_ij = sum_{m,r} Rbar and d/d LP[m,r,j] = sum_i Rbar.
    """
    M, N = spec.M, spec.N
    p = unpack(x, spec, data.nongold)
    theta, a, b, logpi, tau = p["theta"], p["a"], p["b"], p["logpi"], p["tau"]
    z, logw = spec.nodes()
    LP = logpi[:, LETTER_OF]                                                   # (M, R, J)
    Lall = b[None, :, None, :] + LP[:, None, :, :]                             # (M, N, R, J)
    Lng = np.where(data.G[None, :, None, :], -np.inf, Lall)
    logSng = logsumexp(Lng, axis=3)                                            # (M, N, R)
    LPg = logpi[:, (data.gold[:, None] + np.arange(N_ROT)[None, :]) % N_OPT]   # (M, N, R)
    tau_c = np.reshape(tau, (-1, 1, 1)) if np.ndim(tau) else tau                # scalar or (M, 1, 1)
    gk = (theta[:, None] * a[None, :])[:, :, None] + tau_c * z[None, None, :]  # (M, N, K)
    Lg = gk[..., None] + LPg[:, :, None, :]                                    # (M, N, K, R)
    D = np.logaddexp(Lg, logSng[:, :, None, :])                                # (M, N, K, R)
    par = data.parsed                                                          # (M, N, R)
    c0 = (data.Y * Lall).sum(axis=(2, 3))                                      # (M, N)
    llk = c0[..., None] + data.yg[..., None] * gk - np.einsum("mir,mikr->mik", par, D)
    jk = llk + logw[None, None, :]
    cell = logsumexp(jk, axis=2)                                               # (M, N)
    ll = float(cell.sum())

    logpost = jk - cell[..., None]                                             # (M, N, K)
    post = np.exp(logpost)
    pg = np.exp(Lg - D)                                                        # (M, N, K, R)
    dg = post * (data.yg[..., None] - np.einsum("mir,mikr->mik", par, pg))     # (M, N, K)
    logW = logsumexp(logpost[..., None] - D, axis=2)                           # (M, N, R)
    Pbar_ng = np.exp(Lng + logW[..., None])                                    # (M, N, R, J), 0 at gold
    Pbar_g = np.einsum("mik,mikr->mir", post, pg)                              # (M, N, R)
    Pbar = np.where(data.G[None, :, None, :], Pbar_g[..., None], Pbar_ng)
    Rbar = data.Y - par[..., None] * Pbar                                      # (M, N, R, J)
    g_b = Rbar.sum(axis=(0, 2))                                                # (N, J)
    g_bfree = g_b[np.arange(N)[:, None], data.nongold]
    gLP = Rbar.sum(axis=1)                                                     # (M, R, J)
    glogpi = np.zeros((M, N_OPT))
    for r in range(N_ROT):
        glogpi[:, LETTER_OF[r]] += gLP[:, r, :]                                # LETTER_OF[r] is a permutation
    pi = np.exp(logpi)
    glam = (glogpi - pi * glogpi.sum(axis=1, keepdims=True))[:, 1:]
    dgs = dg.sum(axis=2)                                                       # (M, N)
    g_theta = dgs @ a
    g_a = dgs.T @ theta
    grads = [g_theta, g_a, g_bfree.ravel(), glam.ravel()]
    if spec.with_tau == "model":
        grads.append(tau * np.sum(dg * z[None, None, :], axis=(1, 2)))
    elif spec.with_tau:
        grads.append([tau * float(np.sum(dg * z[None, None, :]))])
    grad = np.concatenate(grads)

    pen = -0.5 * (np.sum(theta ** 2) + np.sum(a ** 2) / spec.sigma_a ** 2
                  + np.sum(p["bfree"] ** 2) / spec.sigma_b ** 2 + np.sum(p["lam"] ** 2) / spec.sigma_lam ** 2)
    gpen = [-theta, -a / spec.sigma_a ** 2, -p["bfree"].ravel() / spec.sigma_b ** 2,
            -p["lam"].ravel() / spec.sigma_lam ** 2]
    if spec.n_tau:
        gpen.append(np.zeros(spec.n_tau))
    return ll + pen, grad + np.concatenate(gpen), ll


def default_init(data: SPData, spec: SPSpec) -> np.ndarray:
    """theta: standardized logit accuracy; a = 1; b = 0; letter logits from the letter shares; tau = 1."""
    acc = np.clip(data.correct().reshape(data.M, -1).mean(axis=1), 1e-3, 1 - 1e-3)
    th = np.log(acc / (1 - acc))
    th = (th - th.mean()) / th.std()
    shown = np.where(data.chosen >= 0, (data.chosen + np.arange(N_ROT)[None, None, :]) % N_OPT, -1)
    share = np.stack([(shown == L).reshape(data.M, -1).mean(axis=1) for L in range(N_OPT)], axis=1) + 1e-2
    lam = np.log(share[:, 1:] / share[:, :1])
    return pack(th, np.ones(data.N), np.zeros((data.N, 3)), lam, np.zeros(spec.n_tau) if spec.n_tau else None)


@dataclass
class SPFit:
    spec: SPSpec
    x: np.ndarray
    par: dict                 # standardized parameters (theta mean 0, SD 1)
    rho: np.ndarray
    loglik_choices: float     # unpenalized, parsed choices
    loglik_total: float       # + unparsed part
    objective: float          # penalized
    converged: bool
    n_iter: int
    max_abs_grad: float
    message: str


def standardize(par: dict, nongold: np.ndarray) -> dict:
    """theta = c + s t -> (t, a s, b_ij - a_i c for j != g_i); the likelihood is unchanged."""
    c, s = float(par["theta"].mean()), float(par["theta"].std())
    out = dict(par)
    out["theta"] = (par["theta"] - c) / s
    out["a"] = par["a"] * s
    b = par["b"].copy()
    N = b.shape[0]
    b[np.arange(N)[:, None], nongold] -= (par["a"] * c)[:, None]
    out["b"] = b
    out["bfree"] = b[np.arange(N)[:, None], nongold]
    out["gauge_c_s"] = (c, s)
    return out


def fit_sp(data: SPData, spec: SPSpec, x0: np.ndarray | None = None, maxiter: int = 30000) -> SPFit:
    x0 = default_init(data, spec) if x0 is None else x0

    def f(x):
        obj, g, _ = sp_terms(x, data, spec)
        return -obj, -g

    res = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "gtol": 1e-6, "ftol": 1e-14, "maxcor": 30})
    obj, g, ll = sp_terms(res.x, data, spec)
    par = standardize(unpack(res.x, spec, data.nongold), data.nongold)
    return SPFit(spec, res.x, par, data.rho.copy(), ll, ll + data.unparsed_loglik(), obj, bool(res.success),
                 int(res.nit), float(np.max(np.abs(g))), str(res.message))


def simulate_choices(par: dict, gold: np.ndarray, rho: np.ndarray | None, rng: np.random.Generator,
                     idx: np.ndarray | None = None, extra_gold: np.ndarray | None = None,
                     unp_mask: np.ndarray | None = None) -> np.ndarray:
    """(M, n, R) original-option choices (-1 = unparsed) for the questions idx.

    e_mi ~ N(0, tau^2) per cell; per rotation, option j with probability
    softmax_j(b_ij + 1[j = g_i](a_i theta_m + e_mi + extra_gold_mi) + log pi_m(l_r(j))) (Gumbel-max).
    Unparsed cells: unp_mask (M, N, R) bool over ALL questions (the observed mask; the cells of idx are
    used), or else, when rho (M,) is given instead, independent Bernoulli(rho_m). Exactly one of the two.
    extra_gold (M, N): planted utility added to the gold option (power study).
    RNG order: e (M, n), Gumbel (M, n, R, J), then (rho only) unparsed uniforms (M, n, R).
    """
    if (rho is None) == (unp_mask is None):
        raise ValueError("give exactly one of rho (Bernoulli unparsed) and unp_mask (observed unparsed cells)")
    theta, a, b, logpi, tau = par["theta"], par["a"], par["b"], par["logpi"], par["tau"]
    M = theta.size
    idx = np.arange(gold.size) if idx is None else idx
    n = idx.size
    g = gold[idx]
    tau_c = np.reshape(tau, (-1, 1)) if np.ndim(tau) else tau
    gm = theta[:, None] * a[idx][None, :] + tau_c * rng.standard_normal((M, n))
    if extra_gold is not None:
        gm = gm + extra_gold[:, idx]
    LP = logpi[:, LETTER_OF]                                                   # (M, R, J)
    L = b[idx][None, :, None, :] + LP[:, None, :, :]                           # (M, n, R, J)
    L = L + (g[:, None] == np.arange(N_OPT)[None, :])[None, :, None, :] * gm[:, :, None, None]
    U = rng.random(L.shape)
    choice = np.argmax(L - np.log(-np.log(U)), axis=3)
    if unp_mask is not None:
        unp_mask = np.asarray(unp_mask, dtype=bool)
        if unp_mask.shape != (M, gold.size, N_ROT):
            raise ValueError(f"unp_mask shape {unp_mask.shape} != {(M, gold.size, N_ROT)}")
        unp = unp_mask[:, idx]
    else:
        unp = rng.random((M, n, N_ROT)) < rho[:, None, None]
    return np.where(unp, -1, choice)


def choice_probs(par: dict, gold: np.ndarray, e: np.ndarray | None = None) -> np.ndarray:
    """(M, N, R, J) P(choose j | parsed) at a given cell effect e (M, N) (default 0)."""
    theta, a, b, logpi = par["theta"], par["a"], par["b"], par["logpi"]
    M, N = theta.size, a.size
    e = np.zeros((M, N)) if e is None else e
    L = b[None, :, None, :] + logpi[:, LETTER_OF][:, None, :, :]
    L = L + (gold[:, None] == np.arange(N_OPT)[None, :])[None, :, None, :] * (theta[:, None] * a[None, :] + e)[:, :, None, None]
    return np.exp(L - logsumexp(L, axis=3, keepdims=True))


MAX_REDRAWS = 5000


def null_draw_scores(par: dict, gold: np.ndarray, unp_mask: np.ndarray, rng: np.random.Generator,
                     keeps: dict, extra_gold: np.ndarray | None = None, max_redraws: int = MAX_REDRAWS) -> dict:
    """{score: eigenvalues on the kept columns} of one simulated dataset (observed unparsed mask).

    Constant kept columns are redrawn (whole question) for up to max_redraws rounds; else the score gets None."""
    T = (simulate_choices(par, gold, None, rng, extra_gold=extra_gold, unp_mask=unp_mask)
         == gold[None, :, None]).astype(np.int8)
    out = {}
    for s, keep in keeps.items():
        idx = np.flatnonzero(keep)
        X = SCORES[s](T[:, idx]).astype(np.float64)
        for k in range(max_redraws + 1):
            bad = np.flatnonzero(X.std(axis=0) == 0)
            if bad.size == 0 or k == max_redraws:
                break
            Tb = (simulate_choices(par, gold, None, rng, idx=idx[bad], extra_gold=extra_gold, unp_mask=unp_mask)
                  == gold[idx[bad]][None, :, None]).astype(np.int8)
            X[:, bad] = SCORES[s](Tb)
        out[s] = corr_eigs(X) if bad.size == 0 else None
    return out


def sp_null(par: dict, gold: np.ndarray, unp_mask: np.ndarray, keeps: dict, n: int, rng: np.random.Generator,
            extra_gold: np.ndarray | None = None, max_redraws: int = MAX_REDRAWS) -> tuple[dict, dict]:
    """{score: (n_ok, min(M, N_cols)) null eigenvalues} (failed draws dropped) and {score: n_failed}."""
    M = par["theta"].size
    E = {s: [] for s in keeps}
    fail = {s: 0 for s in keeps}
    for _ in range(n):
        d = null_draw_scores(par, gold, unp_mask, rng, keeps, extra_gold, max_redraws)
        for s in keeps:
            if d[s] is None:
                fail[s] += 1
            else:
                E[s].append(d[s])
    return {s: (np.array(v) if v else np.empty((0, min(M, int(keeps[s].sum()))))) for s, v in E.items()}, fail


def sp_null_target(par: dict, gold: np.ndarray, unp_mask: np.ndarray, keeps: dict, target: int,
                   max_attempts: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """Null draws until every score has `target` successes or max_attempts were made.

    Returns ({score: (<= target, ...) eigenvalues}, {score: attempts}); a short score keeps what it has."""
    M = par["theta"].size
    E = {s: [] for s in keeps}
    att = {s: 0 for s in keeps}
    for _ in range(max_attempts):
        todo = {s: k for s, k in keeps.items() if len(E[s]) < target}
        if not todo:
            break
        d = null_draw_scores(par, gold, unp_mask, rng, todo)
        for s in todo:
            att[s] += 1
            if d[s] is not None:
                E[s].append(d[s])
    return {s: (np.array(v) if v else np.empty((0, min(M, int(keeps[s].sum()))))) for s, v in E.items()}, att


def pool_equal_weight(per_refit: list[np.ndarray], target: int) -> tuple[np.ndarray, int]:
    """Equal-weight pool: concatenate refits with exactly `target` draws, exclude the rest. (pooled, n_excluded)."""
    ok = [e for e in per_refit if e.shape[0] == target]
    if any(e.shape[0] > target for e in per_refit):
        raise ValueError("a refit has more than target draws")
    n_cols = per_refit[0].shape[1] if per_refit else 0
    return (np.concatenate(ok) if ok else np.empty((0, n_cols))), len(per_refit) - len(ok)


def _interval_check(obs: np.ndarray, sims: np.ndarray) -> dict:
    """obs (...), sims (n, ...): central 98% interval coverage, mean abs error of the sim mean."""
    lo, hi = np.percentile(sims, 1, axis=0), np.percentile(sims, 99, axis=0)
    inside = (obs >= lo) & (obs <= hi)
    return {"coverage_98": float(inside.mean()), "n": int(inside.size),
            "mean_abs_obs_minus_sim_mean": float(np.mean(np.abs(obs - sims.mean(axis=0)))),
            "corr_obs_sim_mean": float(np.corrcoef(obs.ravel(), sims.mean(axis=0).ravel())[0, 1])
            if obs.size > 2 and np.std(obs) > 0 else None}


def ppc_stats(chosen: np.ndarray, gold: np.ndarray) -> dict:
    """Statistics compared between observed and simulated data."""
    M = chosen.shape[0]
    C = (chosen == gold[None, :, None])
    S = C.sum(axis=2)
    shown = np.where(chosen >= 0, (chosen + np.arange(N_ROT)[None, None, :]) % N_OPT, -1)
    share = np.stack([(shown == L).reshape(M, -1).mean(axis=1) for L in (0, 1, 2, 3, -1)], axis=1)  # (M, 5)
    hist = np.stack([(S == k).mean(axis=1) for k in range(N_ROT + 1)], axis=1)                       # (M, 5)
    ncols = {s: int(keep_mask(SCORES[s](C.astype(np.int8))).sum()) for s in SCORES}
    return {"acc_model": C.reshape(M, -1).mean(axis=1), "acc_model_rot": C.mean(axis=1),
            "share_model": share, "hist_model": hist, "hist_pooled": hist.mean(axis=0),
            "acc_question_rot": C.mean(axis=0), "acc_question": C.mean(axis=(0, 2)),
            "n_all4_question": (S == 4).sum(axis=0), "n_none_question": (S == 0).sum(axis=0),
            "ncols": ncols}


def ppc(fit: SPFit, data: SPData, n: int, rng: np.random.Generator) -> dict:
    obs = ppc_stats(data.chosen, data.gold)
    mask = data.chosen < 0
    sims = [ppc_stats(simulate_choices(fit.par, data.gold, None, rng, unp_mask=mask), data.gold) for _ in range(n)]
    out = {}
    for k in ("acc_model", "acc_model_rot", "share_model", "hist_model", "hist_pooled", "acc_question_rot",
              "acc_question", "n_all4_question", "n_none_question"):
        out[k] = _interval_check(np.asarray(obs[k], dtype=float), np.array([s[k] for s in sims], dtype=float))
    out["hist_pooled"]["observed"] = obs["hist_pooled"].tolist()
    out["hist_pooled"]["sim_mean"] = np.mean([s["hist_pooled"] for s in sims], axis=0).tolist()
    out["hist_model"]["observed_by_model"] = obs["hist_model"].round(4).tolist()
    out["hist_model"]["sim_mean_by_model"] = np.mean([s["hist_model"] for s in sims], axis=0).round(4).tolist()
    out["acc_model"]["observed"] = obs["acc_model"].round(4).tolist()
    out["acc_model"]["sim_mean"] = np.mean([s["acc_model"] for s in sims], axis=0).round(4).tolist()
    out["share_model"]["observed"] = obs["share_model"].round(4).tolist()
    out["share_model"]["sim_mean"] = np.mean([s["share_model"] for s in sims], axis=0).round(4).tolist()
    nc = {s: np.array([x["ncols"][s] for x in sims]) for s in SCORES}
    out["n_cols_by_score"] = {s: {"observed": obs["ncols"][s], "sim_mean": float(nc[s].mean()),
                                  "sim_p01": float(np.percentile(nc[s], 1)), "sim_p99": float(np.percentile(nc[s], 99)),
                                  "observed_pct": pct_of(obs["ncols"][s], nc[s])} for s in SCORES}
    out["n_sims"] = n
    return out


def sp_data_of(variant: str) -> tuple[SPData, dict]:
    d = variant_data(variant)
    data = SPData(d["chosen"], d["gold"])
    assert np.array_equal(data.correct(), d["CR"]), "correctness must equal chosen == gold (unparsed = wrong)"
    return data, d


def observed_keeps(data: SPData) -> tuple[dict, dict]:
    C = data.correct()
    keeps = {s: keep_mask(SCORES[s](C)) for s in SCORES}
    obs = {s: corr_eigs(SCORES[s](C)[:, keeps[s]].astype(np.float64)) for s in SCORES}
    return keeps, obs


def fd_check(data: SPData, spec: SPSpec, x: np.ndarray, n_coords: int, rng: np.random.Generator,
             h: float = 1e-5) -> dict:
    """Central finite differences of the penalized objective vs the analytic gradient (float64)."""
    _, g, _ = sp_terms(x, data, spec)
    M, N = spec.M, spec.N
    blocks = {"theta": (0, M), "a": (M, M + N), "b": (M + N, M + 4 * N), "lambda": (M + 4 * N, M + 4 * N + 3 * M)}
    if spec.n_tau:
        blocks["logtau"] = (spec.n_params - spec.n_tau, spec.n_params)
    out, worst = {}, 0.0
    for name, (lo, hi) in blocks.items():
        idx = rng.choice(np.arange(lo, hi), size=min(n_coords, hi - lo), replace=False)
        errs = []
        for k in idx:
            e = np.zeros_like(x)
            e[k] = h
            fd = (sp_terms(x + e, data, spec)[0] - sp_terms(x - e, data, spec)[0]) / (2 * h)
            errs.append(abs(fd - g[k]) / max(1.0, abs(g[k])))
        out[name] = {"n": int(len(idx)), "max_rel_err": float(max(errs))}
        worst = max(worst, max(errs))
    return {"blocks": out, "max_rel_err": worst, "h": h,
            "criterion": "max |FD - analytic| / max(1, |analytic|) < 1e-5"}


def recovery(fit: SPFit, data: SPData, rng: np.random.Generator) -> dict:
    """Simulate from the fitted parameters at this size, refit, compare (standardized gauge)."""
    sim = SPData(simulate_choices(fit.par, data.gold, None, rng, unp_mask=data.chosen < 0), data.gold)
    rf = fit_sp(sim, fit.spec)
    t, e = fit.par, rf.par
    pc_t = _p_correct(t, data.gold)
    pc_e = _p_correct(e, data.gold)
    return {"converged": rf.converged, "n_iter": rf.n_iter,
            "corr_theta": float(np.corrcoef(t["theta"], e["theta"])[0, 1]),
            "corr_a": float(np.corrcoef(t["a"], e["a"])[0, 1]),
            "corr_b_nongold": float(np.corrcoef(t["bfree"].ravel(), e["bfree"].ravel())[0, 1]),
            "corr_logpi": float(np.corrcoef(t["logpi"].ravel(), e["logpi"].ravel())[0, 1]),
            "max_abs_err_pi": float(np.max(np.abs(np.exp(t["logpi"]) - np.exp(e["logpi"])))),
            "tau_true_est": [np.round(t["tau"], 4).tolist(), np.round(e["tau"], 4).tolist()],
            "corr_tau_per_model": (float(np.corrcoef(t["tau"], e["tau"])[0, 1]) if np.ndim(t["tau"]) else None),
            "corr_p_correct_cells": float(np.corrcoef(pc_t.ravel(), pc_e.ravel())[0, 1]),
            "mean_abs_err_p_correct": float(np.mean(np.abs(pc_t - pc_e))),
            "slope_a_est_on_true": float(np.polyfit(t["a"], e["a"], 1)[0])}


def _p_correct(par: dict, gold: np.ndarray) -> np.ndarray:
    """(M, N, R) P(correct | parsed), integrated over e by Gauss-Hermite (41 nodes) when tau > 0."""
    if np.all(np.asarray(par["tau"]) == 0):
        P = choice_probs(par, gold)
        return np.take_along_axis(P, np.broadcast_to(gold[None, :, None, None], P.shape[:3] + (1,)), axis=3)[..., 0]
    z, w = hermegauss(K_GH)
    w = w / w.sum()
    acc = 0.0
    for zk, wk in zip(z, w, strict=True):
        e = np.broadcast_to(np.reshape(par["tau"], (-1, 1)) * zk, (par["theta"].size, gold.size))
        P = choice_probs(par, gold, e=e)
        acc = acc + wk * np.take_along_axis(P, np.broadcast_to(gold[None, :, None, None], P.shape[:3] + (1,)), axis=3)[..., 0]
    return acc


def _fit_info(f: SPFit) -> dict:
    p = f.par
    return {"with_tau": f.spec.with_tau, "tau": np.round(p["tau"], 4).tolist() if np.ndim(p["tau"]) else p["tau"],
            "loglik_choices": f.loglik_choices,
            "loglik_total": f.loglik_total, "objective_penalized": f.objective, "converged": f.converged,
            "n_iter": f.n_iter, "max_abs_grad": f.max_abs_grad, "message": f.message,
            "n_params": f.spec.n_params, "K_gauss_hermite": f.spec.K if f.spec.with_tau else 1,
            "sigma_a_b_lam": [f.spec.sigma_a, f.spec.sigma_b, f.spec.sigma_lam],
            "a_quantiles_0_10_50_90_100": np.quantile(p["a"], [0, .1, .5, .9, 1]).tolist(),
            "frac_a_negative": float(np.mean(p["a"] < 0)),
            "b_nongold_quantiles_0_10_50_90_100": np.quantile(p["bfree"], [0, .1, .5, .9, 1]).tolist(),
            "pi_A_B_C_D_by_model": np.exp(p["logpi"]).round(4).tolist(), "rho_unparsed_by_model": f.rho.round(4).tolist(),
            "theta": p["theta"].round(4).tolist()}


def pipeline_null(par: dict, data: SPData, n: int, rng: np.random.Generator) -> tuple[dict, dict, dict, dict]:
    """(keeps, obs, E, n_failed): a dataset's kept columns, eigenvalues and n soft-preference null draws."""
    keeps, obs = observed_keeps(data)
    E, fail = sp_null(par, data.gold, data.chosen < 0, keeps, n, rng)
    return keeps, obs, E, fail


def taum_key(seed_key: tuple[int, ...]) -> tuple[int, ...]:
    return STREAM["taum_starts"], *seed_key


def sp_task(args) -> tuple:
    """One (variant, with_tau, sigma) fit + its null (+ PPC, FD check, recovery when asked)."""
    variant, with_tau, sigma, do_ppc, do_fd, n_rec, n_null, seed_key = args
    rng = rng_for(*seed_key)
    data, _ = sp_data_of(variant)
    spec = SPSpec(data.M, data.N, with_tau, sigma_a=sigma, sigma_b=sigma)
    t0 = time.time()
    fit = fit_sp(data, spec)
    starts = [{"start": "default", "objective": fit.objective, "converged": fit.converged}]
    if with_tau == "model":
        # The per-model tau likelihood is multimodal (restarts reach different optima, some with tau_m -> inf):
        # best penalized objective over the default start and N_TAUM_STARTS perturbed starts.
        rs0 = rng_for(*taum_key(seed_key))
        for k in range(N_TAUM_STARTS):
            f2 = fit_sp(data, spec, x0=default_init(data, spec) + 0.5 * rs0.standard_normal(spec.n_params))
            starts.append({"start": f"perturbed_{k}", "objective": f2.objective, "converged": f2.converged,
                           "max_tau": float(np.max(f2.par["tau"]))})
            if f2.objective > fit.objective:
                fit = f2
    out = {"variant": variant, "fit": _fit_info(fit), "fit_seconds": time.time() - t0, "seed": [SEED_F, *seed_key],
           "starts": starts}
    _keeps, obs, E, fail = pipeline_null(fit.par, data, n_null, rng)
    out["null"] = {s: {**rank_summary(E[s], obs[s]), "n_failed_draws": fail[s]} for s in SCORES}
    if do_ppc:
        out["ppc"] = ppc(fit, data, N_PPC, rng)
    if do_fd:
        out["fd_at_fit"] = fd_check(data, spec, fit.x, 6, rng)
        out["fd_at_init"] = fd_check(data, spec, default_init(data, spec) + 0.3 * rng.standard_normal(spec.n_params),
                                     6, rng)
    if n_rec:
        out["recovery"] = [recovery(fit, data, rng) for _ in range(n_rec)]
    if with_tau and do_fd:
        # Quadrature check: log-lik of the parsed choices at the fit with K = 21, 41, 81, 161 nodes.
        out["gauss_hermite_check"] = {str(K): sp_terms(fit.x, data, SPSpec(data.M, data.N, with_tau, K=K,
                                                                           sigma_a=sigma, sigma_b=sigma))[2]
                                      for K in (21, 41, 81, 161)}
    if do_fd:
        rs = []
        for k in range(2):
            x0 = default_init(data, spec) + 0.5 * rng.standard_normal(spec.n_params)
            f2 = fit_sp(data, spec, x0=x0)
            rs.append({"loglik_choices": f2.loglik_choices, "objective": f2.objective, "converged": f2.converged,
                       "corr_theta_with_main": float(np.corrcoef(f2.par["theta"], fit.par["theta"])[0, 1]),
                       "tau": np.round(f2.par["tau"], 4).tolist() if np.ndim(f2.par["tau"]) else f2.par["tau"]})
        out["restarts"] = rs
    return (variant, with_tau, sigma), out, fit, E


def sp_jobs() -> list[tuple]:
    """Longest first (per-model tau: 5 starts)."""
    jobs = []
    for vi, v in enumerate(VARIANTS):
        for wt in MODES:
            main = v in ("full", "no_llama1b")
            jobs.append((v, wt, SIGMA_A, main, v == "full", N_RECOVERY if v == "full" else 0, N_SP_NULL,
                         (STREAM["sp"], vi + 1, MODE_CODE[wt] + 1)))
    for si, sg in enumerate(PENALTY_SENSITIVITY):
        for wt in MODES:
            jobs.append(("full", wt, sg, False, False, 0, 500, (STREAM["sp_sens"], si + 1, MODE_CODE[wt] + 1)))
    return sorted(jobs, key=lambda j: (j[1] != "model", not j[4], j[1] is not True))


def part_sp() -> None:
    sp_collect(run_pool(sp_task, sp_jobs()))


def fit_key(variant: str, with_tau, sigma: float = SIGMA_A) -> str:
    return f"{variant}__{MODE_TAG[with_tau]}__sigma{sigma:g}"


def save_fits(path, fits: dict) -> None:
    np.savez_compressed(path, **{
        f"{k}__{name}": np.asarray(f.par[name]) for k, f in fits.items()
        for name in ("theta", "a", "b", "bfree", "lam", "logpi", "tau")},
        **{f"{k}__rho": f.rho for k, f in fits.items()}, **{f"{k}__x": f.x for k, f in fits.items()})


def sp_collect(out: list) -> None:
    res, store, fits = {}, {}, {}
    for (v, wt, sg), r, f, E in out:
        key = fit_key(v, wt, sg)
        res[key] = r
        fits[key] = f
        for s, e in E.items():
            store[f"{key}__{s}"] = e
    np.savez_compressed(OUT / "raw_v3_F_sp_null_eigs.npz", **store)
    save_fits(OUT / "raw_v3_F_sp_fits.npz", fits)      # fitted parameters (for char and the checks)
    write_json("raw_v3_F_sp", {"meta": {
        "seed": SEED_F, "n_null": N_SP_NULL, "n_ppc": N_PPC, "K_gauss_hermite": K_GH,
        "sigma_a_b_lam": [SIGMA_A, SIGMA_B, SIGMA_LAM], "penalty_sensitivity_sigma_a_b": list(PENALTY_SENSITIVITY),
        "unparsed": "conditional on the observed unparsed mask: every simulated dataset has the observed unparsed "
                    "cells unparsed (scored wrong) and the other cells parsed. The choice likelihood conditions on "
                    "the parsed status, so the fit is unchanged; rho_m (the per-model unparsed share) enters "
                    "loglik_total only",
        "null_sampling": "complete 4-rotation datasets from the fitted parameters (plug-in); per score, each "
                         "observed kept column redrawn (whole question) until non-constant, at most "
                         f"{MAX_REDRAWS} redraw rounds (else the draw fails for that score; counted)"},
        "fits": res})


def load_sp_fit(key: str, path=None) -> tuple[dict, np.ndarray]:
    z = np.load(OUT / "raw_v3_F_sp_fits.npz" if path is None else path)
    par = {n: z[f"{key}__{n}"] for n in ("theta", "a", "b", "bfree", "lam", "logpi")}
    t = np.asarray(z[f"{key}__tau"], dtype=float)
    par["tau"] = float(t) if t.ndim == 0 else t
    par["logtau"] = np.log(par["tau"]) if np.all(t > 0) else -np.inf
    return par, z[f"{key}__rho"]


BASE_FILE = OUT / "raw_v3_F_base_fits.npz"


def base_task(variant: str) -> tuple[str, SPFit]:
    data, _ = sp_data_of(variant)
    return variant, fit_sp(data, SPSpec(data.M, data.N, True))


def part_base() -> None:
    base_collect(run_pool(base_task, list(CAL_VARIANTS)))


def base_collect(out: list) -> None:
    fits = {fit_key(v, True): f for v, f in out}
    save_fits(BASE_FILE, fits)
    write_json("raw_v3_F_base", {"meta": {"what": "primary soft-preference fits (common tau, default start); the "
                                                  "generating parameters of cal, power, calib, calib_bc"},
                                 "fits": {k: _fit_info(f) for k, f in fits.items()}})


def load_base(variant: str) -> dict:
    return load_sp_fit(fit_key(variant, True), BASE_FILE)[0]


def cal_key(variant: str, b: int) -> tuple[int, ...]:
    return STREAM["cal"], CAL_VARIANTS.index(variant) + 1, b + 1


def cal_task(args) -> dict:
    """Simulate from the primary fit, refit, own null; plus N_BOOT_PER refit draws on the observed kept columns."""
    variant, b = args
    rng = rng_for(*cal_key(variant, b))
    par = load_base(variant)
    data, _ = sp_data_of(variant)
    mask = data.chosen < 0
    keeps_real, _ = observed_keeps(data)
    sim = SPData(simulate_choices(par, data.gold, None, rng, unp_mask=mask), data.gold)
    t0 = time.time()
    rf = fit_sp(sim, SPSpec(sim.M, sim.N, True))
    keeps, obs, E, fail = pipeline_null(rf.par, sim, N_CAL_NULL, rng)
    det = {s: {**_detect(E[s], obs[s]), "n_failed": fail[s], "n_cols": int(keeps[s].sum())} for s in SCORES}
    Eb, att = sp_null_target(rf.par, data.gold, mask, keeps_real, N_BOOT_PER, N_BOOT_MAX_ATTEMPTS, rng)
    return {"variant": variant, "b": b, "detect": det, "boot_E": Eb, "boot_attempts": att,
            "refit": {n: np.asarray(rf.par[n]) for n in ("theta", "a", "bfree", "lam", "tau")},
            "converged": rf.converged, "objective": rf.objective, "seconds": time.time() - t0}


def cal_jobs() -> list[tuple[str, int]]:
    return [(v, b) for b in range(N_CAL) for v in CAL_VARIANTS]


def part_cal() -> None:
    cal_collect(run_pool(cal_task, cal_jobs()))


def cal_collect(out: list) -> None:
    per, refits, store, boot = {}, {}, {}, {}
    for v in CAL_VARIANTS:
        rows = sorted([r for r in out if r["variant"] == v], key=lambda r: r["b"])
        per[v] = [{"b": r["b"], "converged": r["converged"], "objective": r["objective"], "seconds": r["seconds"],
                   "refit_tau": float(r["refit"]["tau"]), "detect": r["detect"],
                   "boot_attempts": r["boot_attempts"]} for r in rows]
        for n in ("theta", "a", "bfree", "lam", "tau"):
            refits[f"{v}__{n}"] = np.stack([r["refit"][n] for r in rows])
        data, _ = sp_data_of(v)
        _, obs = observed_keeps(data)
        boot[v] = {}
        for s in SCORES:
            pooled, n_excl = pool_equal_weight([r["boot_E"][s] for r in rows], N_BOOT_PER)
            store[f"boot__{fit_key(v, True)}__{s}"] = pooled
            boot[v][s] = {**rank_summary(pooled, obs[s]), "n_refits": len(rows), "n_refits_excluded": n_excl,
                          "n_attempts_total": int(sum(r["boot_attempts"][s] for r in rows))}
    np.savez_compressed(OUT / "raw_v3_F_cal_refits.npz", **refits)
    np.savez_compressed(OUT / "raw_v3_F_sp_boot_eigs.npz", **store)
    write_json("raw_v3_F_cal", {"meta": {
        "seed": [SEED_F, STREAM["cal"]], "n_datasets_per_variant": N_CAL, "n_null_per_dataset": N_CAL_NULL,
        "model": "common tau", "variants": list(CAL_VARIANTS),
        "pipeline": "simulate from the primary fit conditional on the observed unparsed mask; refit (default start, "
                    "as the real data); the dataset's own kept columns and eigenvalues; N_CAL_NULL null draws from "
                    "the refit (conditional on the mask and on those columns); percentiles of lambda_1..4",
        "bootstrap_averaged_null": f"each refit also gives exactly {N_BOOT_PER} successful draws on the OBSERVED kept "
                                   f"columns (at most {N_BOOT_MAX_ATTEMPTS} attempts per score); refits short of "
                                   f"{N_BOOT_PER} are excluded (counted), so every refit has equal weight"},
        "per_dataset": per, "bootstrap_averaged_null": boot})


def calibrate(obs_det: dict, boot_dets: list[dict]) -> dict:
    """Calibrate an observed decision against bootstrap datasets run through the same pipeline.

    cal_p_j = #{b: pct_b_j >= pct_obs_j} / B (also with z, which does not saturate); threshold = p99 of the
    bootstrap percentiles; lambda_1 valid when its z is within the bootstrap z p01-p99; FPR of the paper's rule."""
    ok = [d for d in boot_dets if not d["insufficient_draws"]]
    B = len(ok)
    out = {"n_boot": B, "n_boot_insufficient_draws": len(boot_dets) - B}
    if obs_det.get("insufficient_draws") or B == 0:
        return {**out, "calibrated": False}
    P = np.array([d["pct_rank1to4"] for d in ok])
    Z = np.array([d["z_rank1to4"] for d in ok])
    po = np.array(obs_det["pct_rank1to4"])
    zo = np.array(obs_det["z_rank1to4"])
    k = (P >= po).sum(axis=0)
    kz = (Z >= zo).sum(axis=0)
    thr = np.percentile(P, 99, axis=0)
    thr_z = np.percentile(Z, 99, axis=0)
    above = [int(j + 1) for j in range(N_RANKS) if po[j] > thr[j]]
    r2 = np.array([d["rank2_above_p99"] for d in ok])
    val = np.array([d["valid"] for d in ok])
    k_rule = int((r2 & val).sum())
    p1 = 2 * min(float(np.mean(P[:, 0] >= po[0])), float(np.mean(P[:, 0] <= po[0])))
    p1z = 2 * min(float(np.mean(Z[:, 0] >= zo[0])), float(np.mean(Z[:, 0] <= zo[0])))
    zq01, zq99 = np.percentile(Z[:, 0], [1, 99])
    cal_valid = bool(zq01 <= zo[0] <= zq99)
    return {**out, "calibrated": True, "lambda1_cal_p_z_two_sided": min(1.0, p1z),
            "lambda1_boot_z_p01_p99": [float(zq01), float(zq99)], "lambda1_calibrated_valid": cal_valid,
            "ranks_beyond_ability_calibrated_valid": contiguous_from_2(above) if cal_valid else None, "observed_pct_rank1to4": po.tolist(), "observed_z_rank1to4": zo.tolist(),
            "cal_p_rank1to4": (k / B).tolist(), "cal_p_mc_rank1to4": ((k + 1) / (B + 1)).tolist(),
            "cal_p_z_rank1to4": (kz / B).tolist(),
            "cal_threshold_pct_99_rank1to4": thr.tolist(), "cal_threshold_z_99_rank1to4": thr_z.tolist(),
            "ranks_above_cal_threshold": above, "ranks_beyond_ability_calibrated": contiguous_from_2(above),
            "lambda1_cal_p_two_sided": min(1.0, p1),
            "boot_pct_quantiles_1_50_99_rank1to4": np.percentile(P, [1, 50, 99], axis=0).tolist(),
            "boot_frac_valid": float(val.mean()), "boot_frac_rank2_above_p99": float(r2.mean()),
            "fpr_rank2_above_p99_and_valid": k_rule / B, "fpr_rank2_above_p99_and_valid_ci95": clopper_pearson(k_rule, B)}


def stem_mask(subj: np.ndarray) -> np.ndarray:
    return np.isin(subj, STEM_SUBJECTS)


def mean_pair_corr(X: np.ndarray, cols: np.ndarray) -> float:
    """Mean Pearson r over pairs of the given (non-constant) columns (r across models); nan if < 2."""
    Xc = X[:, cols]
    Xc = Xc[:, Xc.std(axis=0) > 0]
    if Xc.shape[1] < 2:
        return float("nan")
    R = np.corrcoef(Xc.T)
    iu = np.triu_indices(R.shape[0], 1)
    return float(R[iu].mean())


def power_dataset(par: dict, gold: np.ndarray, stem: np.ndarray, unp_mask: np.ndarray, s: float,
                  rng: np.random.Generator, n_null: int = N_POWER_NULL, with_tau=True,
                  twopl_scores: tuple = BINARY_POWER_SCORES, n_2pl: int = N_POWER_2PL_NULL) -> dict:
    """One power dataset: gold utility + s theta2_m on the STEM block; refitted soft-preference and 2PL nulls."""
    M, N = par["theta"].size, gold.size
    theta2 = rng.standard_normal(M)
    a2 = s * stem.astype(float)
    extra = theta2[:, None] * a2[None, :]
    data = SPData(simulate_choices(par, gold, None, rng, extra_gold=extra, unp_mask=unp_mask), gold)
    C = data.correct()
    keeps = {sc: keep_mask(SCORES[sc](C)) for sc in POWER_SCORES}
    obs = {sc: corr_eigs(SCORES[sc](C)[:, keeps[sc]].astype(np.float64)) for sc in POWER_SCORES}
    out = {"s": s, "obs_top_eigs": {sc: obs[sc][:N_RANKS].tolist() for sc in POWER_SCORES},
           "n_cols": {sc: int(keeps[sc].sum()) for sc in POWER_SCORES},
           "mean_pair_r_rot0": {"stem": mean_pair_corr(C[:, :, 0].astype(float), np.flatnonzero(stem)),
                                "non_stem": mean_pair_corr(C[:, :, 0].astype(float), np.flatnonzero(~stem))}}
    t0 = time.time()
    rf = fit_sp(data, SPSpec(M, N, with_tau))
    E, fail = sp_null(rf.par, gold, unp_mask, keeps, n_null, rng)       # the null has no planted term
    out["sp"] = {sc: {**_detect(E[sc], obs[sc]), "n_failed": fail[sc]} for sc in POWER_SCORES}
    out["sp_fit"] = {"converged": rf.converged, "tau_mean": float(np.mean(rf.par["tau"])), "seconds": time.time() - t0}
    out["2pl"] = {}
    for sc in twopl_scores:
        ctx = V3.Context(SCORES[sc](C))
        out["2pl"][sc] = _detect(V3.run_null("2pl_post", ctx, rng, n_2pl, {}), ctx.obs)
    return out


def power_key(s: float, rep: int) -> tuple[int, ...]:
    return STREAM["power"], S_GRID.index(s) + 1, rep + 1


def power_task(args) -> dict:
    s, rep = args
    rng = rng_for(*power_key(s, rep))
    d = variant_data("full")
    out = power_dataset(load_base("full"), d["gold"], stem_mask(d["subj"]), d["chosen"] < 0, s, rng)
    return {**out, "rep": rep}


def power_jobs() -> list[tuple[float, int]]:
    return [(s, rep) for rep in range(N_POWER) for s in S_GRID]


def part_power() -> None:
    power_collect(run_pool(power_task, power_jobs()))


def power_collect(results: list) -> None:
    write_json("raw_v3_F_power", {"meta": {"seed": [SEED_F, STREAM["power"]], "base_key": fit_key("full", True),
                                           "s_grid": list(S_GRID), "n_datasets_per_s": N_POWER,
                                           "n_sp_null_draws_per_refit": N_POWER_NULL,
                                           "n_2pl_null_draws": N_POWER_2PL_NULL,
                                           "stem_subjects": list(STEM_SUBJECTS)},
                                  "results": sorted(results, key=lambda r: (r["s"], r["rep"]))})


N_CALIB = 200            # datasets simulated from the fitted soft-preference model (no second factor)
N_CALIB_NULL = 200       # draws per null per dataset
# Step-1 nulls examined per score; 3pl_post for >=3 and all 4 is omitted (invalid lambda_1 on the real data).
CALIB_NULLS = {"rot0": ["3pl_post", "kg_uniform4", "letter_post", "kg_letter4"], "ge2": ["2pl_post", "3pl_post"],
               "ge3": ["rasch_jml", "2pl_post", "kg_letter4", "kg_uniform4"],
               "ge4": ["rasch_jml", "2pl_post", "kg_letter4", "kg_uniform4"]}
CALIB_SCORES = tuple(CALIB_NULLS)


def calib_key(rep: int) -> tuple[int, ...]:
    return STREAM["calib"], rep + 1


def calib_task(rep: int) -> dict:
    """One dataset from the soft-preference fit (observed unparsed mask); the step-1 nulls refitted."""
    rng = rng_for(*calib_key(rep))
    par = load_base("full")
    d0 = variant_data("full")
    gold = d0["gold"]
    chosen = simulate_choices(par, gold, None, rng, unp_mask=d0["chosen"] < 0)
    CR = (chosen == gold[None, :, None]).astype(np.int8)
    shown = np.where(chosen >= 0, (chosen + np.arange(N_ROT)[None, None, :]) % N_OPT, -1)
    d = {"CR": CR, "gold": gold, "shown_choice": shown, "names": d0["names"]}
    fits, _ = V3.kg_fits(d)
    out = {"rep": rep}
    for sc, nulls in CALIB_NULLS.items():
        derive = SCORES[sc]
        X = derive(CR)
        shown_gold = np.broadcast_to(gold, X.shape) if sc == "rot0" else None
        ctx = V3.Context(X, CR, gold, shown_gold=shown_gold, pi=fits["kg_letter4"][1], derive=derive, kg_fits=fits)
        store = {}
        suite = V3.run_suite(ctx, nulls, SEED_F + 1000 * (rep + 1), N_CALIB_NULL, f"calib{rep}_{sc}", store)
        res = {}
        for nm, v in suite["nulls"].items():
            E = store[f"calib{rep}_{sc}__{nm}"]
            res[nm] = {"ranks_above_p99": v["observed_ranks_above_p99"],
                       "beyond_ability": suite["ranks_beyond_ability"][nm],
                       "lambda1_pct": pct_of(ctx.obs[0], E[:, 0]), "valid": lambda1_valid(E, ctx.obs)}
        out[sc] = {"obs_top_eigs": ctx.obs[:N_RANKS].tolist(), "nulls": res}
    return out


def calib_jobs() -> list[int]:
    return list(range(N_CALIB))


def part_calib() -> None:
    calib_collect(run_pool(calib_task, calib_jobs()))


def calib_real_validity() -> dict:
    """Validity of every step-1 null on the real data, per calib score (from raw_v3_F_family.json)."""
    fam = json.loads((OUT / "raw_v3_F_family.json").read_text())["results"]["full"]
    return {sc: fam[sc]["null_valid_lambda1_in_p01_p99"] for sc in CALIB_SCORES}


def calib_collect(res: list) -> None:
    res = sorted(res, key=lambda r: r["rep"])
    summ = {}
    for sc in CALIB_SCORES:
        for nm in CALIB_NULLS[sc]:
            rows = [r[sc]["nulls"][nm] for r in res]
            n = len(rows)
            r2 = np.array([2 in x["ranks_above_p99"] for x in rows])
            bey = np.array([x["beyond_ability"] for x in rows])
            val = np.array([x["valid"] for x in rows])
            k2v = int((r2 & val).sum())
            summ[f"{sc}:{nm}"] = {
                "n": n, "frac_rank2_above_p99": float(r2.mean()), "ci95": clopper_pearson(int(r2.sum()), n),
                "frac_beyond_ge1": float((bey >= 1).mean()), "beyond_distribution": np.bincount(bey, minlength=4).tolist(),
                "frac_valid": float(val.mean()), "frac_rank2_above_and_valid": k2v / n,
                "frac_rank2_above_and_valid_ci95": clopper_pearson(k2v, n),
                "frac_beyond_ge1_given_valid": float((bey[val] >= 1).mean()) if val.any() else None}
    write_json("raw_v3_F_calib", {"meta": {"seed": [SEED_F, STREAM["calib"]], "base_key": fit_key("full", True),
                                           "n_datasets": N_CALIB, "n_null_draws": N_CALIB_NULL,
                                           "nulls_by_score": CALIB_NULLS,
                                           "real_data_validity_of_step1_nulls": calib_real_validity(),
                                           "what": "datasets simulated from the fitted soft-preference model (ability "
                                                   "+ soft letter preference + option attractiveness + cell effect; no "
                                                   "second factor; observed unparsed mask); the listed step-1 nulls "
                                                   "refitted to each dataset; false-positive rates of the rank-matched "
                                                   "p99 rule"},
                                  "summary": summ, "per_dataset": res})


BC_SETS = ("plug_in", "boot_bias_corrected", "mean_square_rescaled")
N_CALIB_BC = 200


def nongold_of(gold: np.ndarray) -> np.ndarray:
    return np.array([[j for j in range(N_OPT) if j != g] for g in gold], dtype=int)


def make_par(theta, a, bfree, lam, tau, gold: np.ndarray) -> dict:
    """Parameter dict (b with b_{i,gold} = 0, log pi = log_softmax([0, lam]))."""
    N = gold.size
    b = np.zeros((N, N_OPT))
    b[np.arange(N)[:, None], nongold_of(gold)] = bfree
    lam_full = np.concatenate([np.zeros((lam.shape[0], 1)), lam], axis=1)
    return {"theta": np.asarray(theta, float), "a": np.asarray(a, float), "bfree": np.asarray(bfree, float),
            "b": b, "lam": np.asarray(lam, float), "logpi": lam_full - logsumexp(lam_full, axis=1, keepdims=True),
            "tau": float(tau), "logtau": float(np.log(tau))}


def bias_corrected_sets(hat: dict, refits: dict, gold: np.ndarray) -> tuple[dict, dict]:
    """Parameter sets for the >=2 / 2PL check. refits: {name: (B, ...)} from the calibration bootstrap
    (data simulated from `hat`, refitted; both in the standardized gauge, theta mean 0 and SD 1).

      plug_in              : hat.
      boot_bias_corrected  : x_bc = 2 x_hat - mean_b x_b, elementwise, for a, b (non-gold), lambda and tau;
                             theta = theta_hat (its scale is fixed by the gauge).
      mean_square_rescaled : a, b, lambda each multiplied by sqrt(mean x_hat^2 / mean_b mean x_b^2) (the
                             refit inflation of the mean square, applied in reverse); tau * tau_hat / mean tau_b.
    Returns (sets, diagnostics)."""
    out = {"plug_in": make_par(hat["theta"], hat["a"], hat["bfree"], hat["lam"], hat["tau"], gold)}
    m = {n: refits[n].mean(axis=0) for n in ("a", "bfree", "lam", "tau")}
    tau_bc = 2 * float(hat["tau"]) - float(m["tau"])
    if tau_bc <= 0:
        raise ValueError(f"bias-corrected tau {tau_bc} <= 0")
    out["boot_bias_corrected"] = make_par(hat["theta"], 2 * hat["a"] - m["a"], 2 * hat["bfree"] - m["bfree"],
                                          2 * hat["lam"] - m["lam"], tau_bc, gold)
    k = {n: float(np.sqrt(np.mean(hat[n] ** 2) / np.mean(refits[n] ** 2))) for n in ("a", "bfree", "lam")}
    tau_rs = float(hat["tau"]) ** 2 / float(m["tau"])
    out["mean_square_rescaled"] = make_par(hat["theta"], k["a"] * hat["a"], k["bfree"] * hat["bfree"],
                                           k["lam"] * hat["lam"], tau_rs, gold)
    diag = {"mean_sq_hat": {n: float(np.mean(hat[n] ** 2)) for n in ("a", "bfree", "lam")},
            "mean_sq_refit": {n: float(np.mean(refits[n] ** 2)) for n in ("a", "bfree", "lam")},
            "refit_inflation_of_mean_sq": {n: float(np.mean(refits[n] ** 2) / np.mean(hat[n] ** 2))
                                           for n in ("a", "bfree", "lam")},
            "rescale_factor": k, "tau": {"hat": float(hat["tau"]), "refit_mean": float(m["tau"]),
                                         "boot_bias_corrected": tau_bc, "mean_square_rescaled": tau_rs},
            "mean_sq_a_by_set": {s: float(np.mean(p["a"] ** 2)) for s, p in out.items()},
            "n_refits": int(refits["a"].shape[0])}
    return out, diag


BC_FILE = OUT / "raw_v3_F_calib_bc_sets.npz"


def prepare_bc_sets() -> dict:
    z = np.load(OUT / "raw_v3_F_cal_refits.npz")
    refits = {n: z[f"full__{n}"] for n in ("theta", "a", "bfree", "lam", "tau")}
    hat = load_base("full")
    gold = variant_data("full")["gold"]
    sets, diag = bias_corrected_sets(hat, refits, gold)
    np.savez_compressed(BC_FILE, **{f"{s}__{n}": np.asarray(p[n]) for s, p in sets.items()
                                    for n in ("theta", "a", "bfree", "lam", "tau")})
    return diag


def load_bc_set(name: str) -> dict:
    z = np.load(BC_FILE)
    gold = variant_data("full")["gold"]
    return make_par(*(z[f"{name}__{n}"] for n in ("theta", "a", "bfree", "lam")), float(z[f"{name}__tau"]), gold)


def calib_bc_key(name: str, rep: int) -> tuple[int, ...]:
    return STREAM["calib_bc"], BC_SETS.index(name) + 1, rep + 1


def calib_bc_task(args) -> dict:
    name, rep = args
    rng = rng_for(*calib_bc_key(name, rep))
    d = variant_data("full")
    par = load_bc_set(name)
    C = (simulate_choices(par, d["gold"], None, rng, unp_mask=d["chosen"] < 0) == d["gold"][None, :, None]).astype(np.int8)
    ctx = V3.Context(SCORES["ge2"](C))
    det = _detect(V3.run_null("2pl_post", ctx, rng, N_CALIB_NULL, {}), ctx.obs)
    return {"set": name, "rep": rep, **det, "beyond_ability": contiguous_from_2(det["ranks_above_p99"])}


def calib_bc_jobs() -> list[tuple[str, int]]:
    return [(s, r) for r in range(N_CALIB_BC) for s in BC_SETS]


def part_calib_bc() -> None:
    diag = prepare_bc_sets()
    calib_bc_collect(run_pool(calib_bc_task, calib_bc_jobs()), diag)


def calib_bc_collect(res: list, diag: dict) -> None:
    summ = {}
    for name in BC_SETS:
        rows = [r for r in res if r["set"] == name]
        n = len(rows)
        r2 = np.array([r["rank2_above_p99"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        bey = np.array([r["beyond_ability"] for r in rows])
        summ[name] = {"n": n, "frac_rank2_above_p99": float(r2.mean()), "ci95": clopper_pearson(int(r2.sum()), n),
                      "frac_rank2_above_and_valid": float((r2 & val).mean()),
                      "ci95_and_valid": clopper_pearson(int((r2 & val).sum()), n), "frac_valid": float(val.mean()),
                      "beyond_distribution": np.bincount(bey, minlength=4).tolist(),
                      "mean_obs_top_eigs": np.mean([r["observed_top_eigs"] for r in rows], axis=0).tolist()}
    write_json("raw_v3_F_calib_bc", {"meta": {"seed": [SEED_F, STREAM["calib_bc"]], "n_datasets_per_set": N_CALIB_BC,
                                              "n_null_draws": N_CALIB_NULL, "score": "ge2", "null": "2pl_post",
                                              "sets": bias_corrected_sets.__doc__},
                                     "diagnostics": diag, "summary": summ,
                                     "per_dataset": sorted(res, key=lambda r: (r["set"], r["rep"]))})


def cubic_residual(v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """v minus its least-squares fit on [1, p, p^2, p^3]."""
    A = np.column_stack([np.ones_like(p), p, p ** 2, p ** 3])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def characterize(X: np.ndarray, XB: np.ndarray, ranks: list[int], d: dict, rng: np.random.Generator,
                 n_perm: int = 5000) -> dict:
    lam, U, Vq, keep = V3.eig_vectors(X)
    _, UB, VB, keepB = V3.eig_vectors(XB)
    p = X[:, keep].mean(axis=0)
    subj = d["subj"][keep]
    names = d["names"]
    acc = X.mean(axis=1)
    logsize = np.log([MODEL_SIZE_B[n] for n in names])
    fam = d["fam"]
    # Letter preference (know-or-guess moment estimate over all 4 rotations, v3lib.letter_preference).
    from v3lib import letter_preference
    pi = letter_preference(d["CR"], d["shown_choice"])["pi_letters"]
    pn = pi / pi.sum(axis=1, keepdims=True)
    pi_max = pn.max(axis=1)
    pi_ent = -np.sum(np.where(pn > 0, pn * np.log(np.where(pn > 0, pn, 1.0)), 0.0), axis=1)
    out = {}
    for j in ranks:
        v = Vq[:, j - 1]
        u = U[:, j - 1]
        vr = cubic_residual(v, p)
        rec = {"eigenvalue": float(lam[j - 1]),
               "r2_loading_on_cubic_difficulty": float(1 - np.var(vr) / np.var(v)),
               "eta_sq_subject_after_cubic": label_eta_sq(vr, subj, n_perm, rng),
               "eta_sq_stem_vs_rest_after_cubic": label_eta_sq(vr, np.isin(subj, STEM_SUBJECTS), n_perm, rng),
               "eta_sq_model_scores_by_family": label_eta_sq(u, fam, n_perm, rng),
               "corr_model_score_accuracy": V3.perm_corr(u, acc, 10000, rng),
               "corr_model_score_log_size": V3.perm_corr(u, logsize, 10000, rng),
               "spearman_model_score_log_size": float(spearmanr(u, logsize)[0]),
               "corr_model_score_letter_pref_max": V3.perm_corr(u, pi_max, 10000, rng),
               "corr_model_score_letter_pref_entropy": V3.perm_corr(u, pi_ent, 10000, rng),
               "eta_sq_loadings_by_original_gold_letter": label_eta_sq(v, d["gold"][keep], n_perm, rng)}
        sm = {sb: float(vr[subj == sb].mean()) for sb in np.unique(subj)}
        srt = sorted(sm, key=sm.get)
        rec["subjects_most_negative_after_cubic"] = [(k, round(sm[k], 4)) for k in srt[:5]]
        rec["subjects_most_positive_after_cubic"] = [(k, round(sm[k], 4)) for k in srt[-5:][::-1]]
        qidx = np.flatnonzero(keep)
        top = np.argsort(-np.abs(v))[:12]
        rec["top_loading_questions"] = [{"index": int(qidx[t]), "subject": str(subj[t]), "loading": round(float(v[t]), 4),
                                         "p_correct": round(float(p[t]), 3)} for t in top]
        u2 = u ** 2 / np.sum(u ** 2)
        rec["top_model_share_of_squared_scores"] = {"model": names[int(np.argmax(u2))].split("/")[-1],
                                                    "share": float(u2.max())}
        rec["participation_ratio_models"] = float(1.0 / np.sum(u2 ** 2))   # effective number of models
        order = np.argsort(u)
        rec["model_scores_low_to_high"] = [(names[k].split("/")[-1], round(float(u[k]), 3)) for k in order]
        # Replicate B: same score from replicate B, eigenvector j (and best match among 1-4).
        common = keep & keepB
        vA = np.zeros(X.shape[1]); vA[keep] = v
        rB = []
        for jj in range(N_RANKS):
            vb = np.zeros(X.shape[1]); vb[keepB] = VB[:, jj]
            rB.append(abs(float(np.corrcoef(vA[common], vb[common])[0, 1])))
        rec["replicate_B"] = {"abs_r_loadings_with_B_eigvec_1to4": rB,
                              "abs_r_model_scores_same_rank": abs(float(np.corrcoef(u, UB[:, j - 1])[0, 1]))}
        out[str(j)] = rec
    return out


CHAR_TARGETS = [("full", "ge2", [2, 3, 4]), ("full", "ge3", [2, 3]), ("no_llama1b", "ge2", [2]),
                ("full", "rot0", [2]), ("full", "ge4", [2])]      # order fixes the shared RNG stream


def replicate_ab_difference() -> dict:
    """Cells (model, question, rotation) whose chosen option, and whose correctness, differ between replicates."""
    A, B = V3.load_rep("A"), V3.load_rep("B")
    chA = np.load(V3.REPS["A"])["chosen_option"]
    chB = np.load(V3.REPS["B"])["chosen_option"]
    diff = A["CR"] != B["CR"]
    dch = chA != chB
    by_model = {A["names"][m]: int(dch[m].sum()) for m in range(dch.shape[0]) if dch[m].any()}
    return {"n_cells_choice_differs": int(dch.sum()), "n_cells_correctness_differs": int(diff.sum()),
            "n_cells": int(diff.size), "choice_differs_by_model": by_model}


def part_char(targets: list[tuple[str, str, list[int]]]) -> None:
    """Characterize (variant, score, ranks) targets."""
    rng = rng_for(STREAM["char"])
    res = {}
    for variant, score, ranks in targets:
        d = variant_data(variant, "A")
        dB = variant_data(variant, "B")
        X = SCORES[score](d["CR"])
        XB = SCORES[score](dB["CR"])
        res[f"{variant}__{score}"] = characterize(X, XB, ranks, d, rng)
    ab = replicate_ab_difference()
    write_json("raw_v3_F_char", {"meta": {"seed": [SEED_F, STREAM["char"]], "n_perm": 5000,
                                          "model_size": "nominal parameter count (billions) from the model name/card",
                                          "replicate_B_difference": ab,
                                          "replicate_B_note": f"B differs from A in the chosen option in "
                                                              f"{ab['n_cells_choice_differs']} of {ab['n_cells']} cells "
                                                              f"({', '.join(ab['choice_differs_by_model'])}), in "
                                                              f"correctness in {ab['n_cells_correctness_differs']}; it "
                                                              "tests GPU nondeterminism only, not resampling"},
                                 "targets": res})


PPC_COVERAGE_MIN = 0.90    # min central-98% interval coverage per statistic family


def reproduction_check(fam: dict) -> dict:
    """Max |p99 difference| of the full-data suites vs v3_C_rotations_A.json (same seeds -> 0)."""
    C = json.loads((OUT / "v3_C_rotations_A.json").read_text())
    ref = {"ge4": C["C3_rotation_invariant"]["all4_correct"]["suite"]["nulls"],
           "mean": C["C3_rotation_invariant"]["mean_over_rotations"]["suite"]["nulls"],
           "rot0": C["rot0_suite"]["nulls"]}
    out = {}
    for s, nulls in ref.items():
        mine = fam["full"][s]["suite"]["nulls"]
        for nm in nulls:
            if nm in mine:
                out[f"{s}:{nm}"] = float(np.max(np.abs(np.array(mine[nm]["rank_eigs_p99"])
                                                        - np.array(nulls[nm]["rank_eigs_p99"]))))
    return {"max_abs_p99_diff_by_suite": out, "max_abs_p99_diff": max(out.values()), "n_compared": len(out),
            "reference": "v3_C_rotations_A.json (C3 all4_correct, mean_over_rotations, rot0_suite)"}


def ppc_verdict(pp: dict) -> dict:
    fams = ("acc_model", "acc_model_rot", "share_model", "hist_model", "acc_question_rot", "n_all4_question",
            "n_none_question")
    cov = {k: pp[k]["coverage_98"] for k in fams}
    ncol = {s: pp["n_cols_by_score"][s]["observed_pct"] for s in SCORES}
    passed = {k: v >= PPC_COVERAGE_MIN for k, v in cov.items()}
    passed_ncol = {s: 1.0 <= v <= 99.0 for s, v in ncol.items()}
    return {"coverage_98": cov, "pass_by_family": passed, "n_cols_observed_pct": ncol, "n_cols_pass": passed_ncol,
            "all_pass": bool(all(passed.values()) and all(passed_ncol.values())),
            "rule": f"coverage of the central 98% simulation interval >= {PPC_COVERAGE_MIN} for every statistic "
                    "family, and the observed number of kept columns of every score within the 1st-99th "
                    "simulation percentile"}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


PLANTED_SHARE_NOTE = ("per STEM question i, s^2 / (a_i^2 + s^2 + tau^2): the planted term's share of the across-model "
                      "variance of the gold-option utility shift a_i theta_m + s theta2_m + e_mi (theta, theta2 of unit "
                      "variance; a_i the fitted slope of the base fit; the option-attractiveness and letter terms are "
                      "excluded), averaged over the STEM questions")


def power_summary(raw: dict, base_par: dict, subj: np.ndarray, cal_thr: dict | None = None) -> dict:
    """Detection rates per planted strength s (primary: rank 2 above p99 AND null valid).

    cal_thr {score: (pct, z) rank-2 thresholds from the s = 0 calibration bootstrap}: detect_rank2_calibrated."""
    res = raw["results"]
    stem = stem_mask(subj)
    a_stem = base_par["a"][stem]
    tau2 = float(np.mean(np.asarray(base_par["tau"]) ** 2))
    out = {}
    r0 = {"stem": np.nanmean([r["mean_pair_r_rot0"]["stem"] for r in res if r["s"] == 0.0]),
          "non_stem": np.nanmean([r["mean_pair_r_rot0"]["non_stem"] for r in res if r["s"] == 0.0])}
    for s in S_GRID:
        rs = [r for r in res if r["s"] == s]
        n = len(rs)
        if n == 0:
            continue
        rec = {"n_datasets": n,
               "planted_var_share_stem_mean": float(np.mean(s ** 2 / (a_stem ** 2 + s ** 2 + tau2))),
               "mean_pair_r_rot0_stem": float(np.nanmean([r["mean_pair_r_rot0"]["stem"] for r in rs])),
               "mean_pair_r_rot0_non_stem": float(np.nanmean([r["mean_pair_r_rot0"]["non_stem"] for r in rs])),
               "mean_pair_r_rot0_stem_increase_vs_s0":
                   float(np.nanmean([r["mean_pair_r_rot0"]["stem"] for r in rs]) - r0["stem"]),
               "n_sp_refits_converged": int(sum(r["sp_fit"]["converged"] for r in rs)),
               "sp_refit_tau_mean": float(np.mean([r["sp_fit"]["tau_mean"] for r in rs])),
               "by_null": {}}
        for null, scores in (("sp", POWER_SCORES), ("2pl", BINARY_POWER_SCORES)):
            for sc in scores:
                ds = [r[null][sc] for r in rs if sc in r[null]]
                if not ds:
                    continue
                n = len(ds)
                ok = [d for d in ds if not d.get("insufficient_draws", False)]
                det = np.array([d["rank2_above_p99"] for d in ds])
                val = np.array([d["valid"] for d in ds])
                k1, k2 = int(det.sum()), int((det & val).sum())
                any_v = np.array([d["valid"] and any(j in d["ranks_above_p99"] for j in (2, 3, 4)) for d in ds])
                rr = {"detect_rank2_and_valid": k2 / n, "detect_rank2_and_valid_ci95": clopper_pearson(k2, n),
                      "detect_rank2": k1 / n, "detect_rank2_ci95": clopper_pearson(k1, n),
                      "frac_null_valid": float(val.mean()), "n": n, "n_insufficient_draws": n - len(ok),
                      "mean_obs_lambda2": float(np.mean([d["observed_top_eigs"][1] for d in ds])),
                      "mean_p99_lambda2": float(np.mean([d["p99"][1] for d in ok])) if ok else None,
                      "detect_any_rank_2to4_and_valid": float(any_v.mean())}
                if null == "sp" and cal_thr is not None and sc in cal_thr:
                    tp, tz = cal_thr[sc]
                    kc = int(sum(d["pct_rank1to4"][1] > tp for d in ok))
                    kz = int(sum(d["z_rank1to4"][1] > tz for d in ok))
                    rr.update({"detect_rank2_calibrated": kc / n, "detect_rank2_calibrated_ci95": clopper_pearson(kc, n),
                               "detect_rank2_calibrated_z": kz / n, "cal_threshold_pct": tp, "cal_threshold_z": tz})
                rec["by_null"][f"{null}:{sc}"] = rr
        out[f"{s:g}"] = rec
    return out


def primary_rule(fits: dict) -> dict:
    """Facts behind the choice of the primary model (common tau), computed from the fits."""
    k_t, k_0, k_m = fit_key("full", True), fit_key("full", False), fit_key("full", "model")
    t, z, m = fits[k_t], fits[k_0], fits[k_m]
    d_restart = max(abs(q["objective"] - t["fit"]["objective_penalized"]) for q in t["restarts"])
    gain = t["fit"]["loglik_total"] - z["fit"]["loglik_total"]
    cov_t = t["ppc"]["hist_model"]["coverage_98"]
    cov_0 = z["ppc"]["hist_model"]["coverage_98"]
    pass_t, pass_0 = ppc_verdict(t["ppc"])["all_pass"], ppc_verdict(z["ppc"])["all_pass"]
    m_gain = max(q["objective"] - m["fit"]["objective_penalized"] for q in m["restarts"])
    m_inf = sum(st.get("max_tau", 0.0) > 1e3 for st in m["starts"])
    text = (f"common tau: default start and {len(t['restarts'])} perturbed restarts, max |objective difference| "
            f"{d_restart:.2g}; log-likelihood gain over tau = 0: {gain:.1f} for 1 parameter; PPC coverage of the per-model "
            f"#correct-of-4 histogram {cov_t:.2f} (tau = 0: {cov_0:.2f}); pre-set PPC rule passed: common tau {pass_t}, "
            f"tau = 0 {pass_0}. Per-model tau_m: best perturbed restart minus the chosen fit {m_gain:+.1f} in the "
            f"penalized objective; {m_inf} of {len(m['starts'])} starts reached max tau_m > 1e3.")
    return {"key": k_t, "rule": text, "restart_max_abs_objective_diff": d_restart, "loglik_gain_over_tau0": gain,
            "ppc_hist_model_coverage": {"tau": cov_t, "tau0": cov_0}, "ppc_all_pass": {"tau": pass_t, "tau0": pass_0},
            "taum_best_restart_minus_fit": m_gain, "taum_starts_tau_gt_1e3": int(m_inf),
            "taum_n_starts": len(m["starts"])}


def obs_detect_of(null_rec: dict) -> dict:
    """The observed decision record (as _detect) from a rank_summary record."""
    if null_rec.get("insufficient_draws"):
        return {"insufficient_draws": True}
    return {"insufficient_draws": False, "pct_rank1to4": null_rec["observed_pct_rank1to4"],
            "z_rank1to4": null_rec["observed_z_rank1to4"]}


def _verdict(r: dict) -> dict:
    if r.get("insufficient_draws"):
        return {"insufficient_draws": True}
    return {"ranks_above_p99": r["observed_ranks_above_p99"], "beyond_ability": r["ranks_beyond_ability"],
            "lambda1_pct": r["observed_pct_rank1to4"][0], "valid": r["null_valid_lambda1_in_p01_p99"],
            "pct_rank1to4": r["observed_pct_rank1to4"], "p99": r["rank_eigs_p99"], "observed": r["observed_top_eigs"]}


def part_assemble() -> None:
    fam = json.loads((OUT / "raw_v3_F_family.json").read_text())
    sp = json.loads((OUT / "raw_v3_F_sp.json").read_text())
    cal = json.loads((OUT / "raw_v3_F_cal.json").read_text())
    fres = fam["results"]
    step1 = {"reproduction_check": reproduction_check(fres), "option_referencing": fam["option_referencing"],
             "table": {v: {s: {"n_cols": r["spectrum"]["n_cols"], "lambda_plus": r["spectrum"]["lambda_plus"],
                               "top_eigs": r["spectrum"]["top_eigs"][:4],
                               "rank_p99_by_null": {k: x["rank_eigs_p99"] for k, x in r["suite"]["nulls"].items()},
                               "observed_pct_rank1to4_by_null": r["observed_pct_rank1to4_by_null"],
                               "ranks_above_p99_by_null": r["suite"]["ranks_above_p99_by_null"],
                               "ranks_beyond_ability_all_nulls": r["suite"]["ranks_beyond_ability"],
                               "null_valid": r["null_valid_lambda1_in_p01_p99"],
                               "ranks_beyond_ability_valid_nulls": r["ranks_beyond_ability_valid_nulls"]}
                           for s, r in fres[v].items()} for v in fres},
             "details": fres, "meta": fam["meta"]}
    fits = sp["fits"]
    step2 = {"meta": sp["meta"], "fits": {}, "verdict_table": {}}
    for k, r in fits.items():
        rec = {"fit": r["fit"], "null": r["null"], "fit_seconds": r["fit_seconds"], "seed": r["seed"],
               "starts": r["starts"]}
        for extra in ("fd_at_fit", "fd_at_init", "recovery", "gauss_hermite_check", "restarts"):
            if extra in r:
                rec[extra] = r[extra]
        if "ppc" in r:
            rec["ppc"] = r["ppc"]
            rec["ppc_verdict"] = ppc_verdict(r["ppc"])
        step2["fits"][k] = rec
        step2["verdict_table"][k] = {s: _verdict(r["null"][s]) for s in SCORES}
    for v, bb in cal["bootstrap_averaged_null"].items():
        step2["verdict_table"][f"boot__{fit_key(v, True)}"] = {s: _verdict(bb[s]) for s in SCORES}
    step2["bootstrap_averaged_null"] = {"meta": cal["meta"]["bootstrap_averaged_null"], "by_variant": cal["bootstrap_averaged_null"]}
    import scipy
    step2["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__}
    # The real-data fit of `sp` and the generating fit of `base` are the same call: check they are equal.
    zs, zb = np.load(OUT / "raw_v3_F_sp_fits.npz"), np.load(BASE_FILE)
    base_eq = {v: float(np.max(np.abs(zs[f"{fit_key(v, True)}__x"] - zb[f"{fit_key(v, True)}__x"]))) for v in CAL_VARIANTS}
    step2["checks"] = {
        "base_fit_equals_sp_fit_max_abs_diff_x": base_eq,
        "gauss_hermite_K41_minus_fit_loglik": {k: r["gauss_hermite_check"]["41"] - r["fit"]["loglik_choices"]
                                               for k, r in fits.items() if "gauss_hermite_check" in r},
        "gauss_hermite_K161_minus_K41": {k: r["gauss_hermite_check"]["161"] - r["gauss_hermite_check"]["41"]
                                         for k, r in fits.items() if "gauss_hermite_check" in r},
        "fd_max_rel_err": {k: [r["fd_at_fit"]["max_rel_err"], r["fd_at_init"]["max_rel_err"]]
                           for k, r in fits.items() if "fd_at_fit" in r},
        "restart_objectives_minus_main": {k: [q["objective"] - r["fit"]["objective_penalized"] for q in r["restarts"]]
                                          for k, r in fits.items() if "restarts" in r}}
    step2["primary"] = primary_rule(fits)
    # Calibration of every soft-preference percentile against the same pipeline.
    calib = {}
    for v in CAL_VARIANTS:
        rows = cal["per_dataset"][v]
        calib[v] = {s: {**calibrate(obs_detect_of(fits[fit_key(v, True)]["null"][s]), [r["detect"][s] for r in rows]),
                        "n_failed_draws_total": int(sum(r["detect"][s]["n_failed"] for r in rows))}
                    for s in SCORES}
        calib[v]["_refits"] = {"n": len(rows), "n_converged": int(sum(r["converged"] for r in rows)),
                               "refit_tau_mean": float(np.mean([r["refit_tau"] for r in rows])),
                               "refit_tau_sd": float(np.std([r["refit_tau"] for r in rows])),
                               "generating_tau": float(fits[fit_key(v, True)]["fit"]["tau"])}
    step2["calibration"] = {"meta": cal["meta"], "by_variant": calib}
    out = {"step1_construction_family": step1, "step2_soft_preference": step2}
    if (OUT / "raw_v3_F_calib.json").exists():
        cb = json.loads((OUT / "raw_v3_F_calib.json").read_text())
        out["step2b_step1_nulls_under_soft_preference_truth"] = {"meta": cb["meta"], "summary": cb["summary"]}
    if (OUT / "raw_v3_F_calib_bc.json").exists():
        bc = json.loads((OUT / "raw_v3_F_calib_bc.json").read_text())
        out["step2c_ge2_2pl_under_bias_corrected_parameters"] = {k: bc[k] for k in ("meta", "diagnostics", "summary")}
    if (OUT / "raw_v3_F_char.json").exists():
        out["step4_characterization"] = json.loads((OUT / "raw_v3_F_char.json").read_text())
    write_json("v3_F_softpref", out)
    if (OUT / "raw_v3_F_power.json").exists():
        pw = json.loads((OUT / "raw_v3_F_power.json").read_text())
        base_par = load_base("full")
        d = variant_data("full")
        thr = {sc: (calib["full"][sc]["cal_threshold_pct_99_rank1to4"][1], calib["full"][sc]["cal_threshold_z_99_rank1to4"][1])
               for sc in POWER_SCORES if calib["full"][sc].get("calibrated")}
        write_json("v3_F_power", {"meta": {**pw["meta"], "block": f"STEM ({int(stem_mask(d['subj']).sum())} questions, "
                                                                  f"{len(STEM_SUBJECTS)} subjects; Hendrycks et al. categories)",
                                           "planted": "gold utility + a2_i theta2_m, theta2 ~ N(0,1) iid per dataset, "
                                                      "independent of theta; a2_i = s on the block, 0 elsewhere; "
                                                      "observed unparsed mask",
                                           "rule": "primary: detect_rank2_and_valid = observed lambda_2 > p99 of the null "
                                                   "lambda_2 AND p01 <= observed lambda_1 <= p99 (the paper's rule); each "
                                                   "null refitted to each simulated dataset (soft-preference with the base "
                                                   "model's form; project 2PL posterior draws for binary scores). "
                                                   "detect_rank2_calibrated (soft-preference only): the rank-2 percentile "
                                                   "exceeds the 99th percentile of the rank-2 percentiles of the full-data "
                                                   "calibration bootstrap (s = 0, same pipeline, independent datasets)",
                                           "planted_var_share_stem_mean": PLANTED_SHARE_NOTE,
                                           "base_tau": np.asarray(base_par["tau"]).tolist()},
                                  "summary": power_summary(pw, base_par, d["subj"], thr)})


def run_pool(fn, jobs: list, n_proc: int = N_PROC) -> list:
    from multiprocessing import get_context
    with get_context("fork").Pool(n_proc) as pool:
        return pool.map(fn, jobs, chunksize=1)


KIND_FN = {"family": family_task, "sp": sp_task, "base": base_task, "cal": cal_task, "power": power_task,
           "calib": calib_task, "calib_bc": calib_bc_task}


def _run_item(item):
    kind, job = item
    return kind, job, KIND_FN[kind](job)


def _ckpt_file(ckpt: Path, kind: str, job) -> Path:
    import hashlib
    return ckpt / f"{kind}__{hashlib.sha1(repr(job).encode()).hexdigest()[:20]}.pkl"


def run_checkpointed(items: list[tuple[str, object]], ckpt: Path, n_proc: int = N_PROC) -> dict:
    """Run (kind, job) items in one pool; each result is pickled on arrival, so a rerun skips finished items.
    Returns {kind: [results in listed job order]}."""
    import pickle
    from multiprocessing import get_context
    ckpt.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not _ckpt_file(ckpt, *it).exists()]
    print(f"[all] {len(items)} jobs, {len(items) - len(todo)} done before, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        with get_context("fork").Pool(n_proc) as pool:
            for k, (kind, job, res) in enumerate(pool.imap_unordered(_run_item, todo, chunksize=1)):
                f = _ckpt_file(ckpt, kind, job)
                tmp = f.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump(res, fh)
                tmp.rename(f)
                print(f"[all] {k + 1}/{len(todo)} {kind} {job!r:.60} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for kind, job in items:
        with open(_ckpt_file(ckpt, kind, job), "rb") as fh:
            out.setdefault(kind, []).append(pickle.load(fh))
    return out


def all_items() -> list[tuple[str, object]]:
    """Main-phase (kind, job) items, longest first."""
    items = [("sp", j) for j in sp_jobs()]
    items += [("calib", j) for j in calib_jobs()]
    cal, pw = cal_jobs(), power_jobs()
    for k in range(max(len(cal), len(pw))):
        items += [("cal", cal[k])] if k < len(cal) else []
        items += [("power", pw[k])] if k < len(pw) else []
    items += [("family", j) for j in family_jobs()]
    return items


def part_all(ckpt: Path) -> None:
    t0 = time.time()
    base_collect(run_checkpointed([("base", v) for v in CAL_VARIANTS], ckpt)["base"])
    res = run_checkpointed(all_items(), ckpt)
    family_collect(res["family"])
    sp_collect(res["sp"])
    cal_collect(res["cal"])
    power_collect(res["power"])
    calib_collect(res["calib"])
    diag = prepare_bc_sets()
    calib_bc_collect(run_checkpointed([("calib_bc", j) for j in calib_bc_jobs()], ckpt)["calib_bc"], diag)
    part_char(CHAR_TARGETS)
    part_assemble()
    print(f"[all] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["family", "base", "sp", "cal", "power", "calib", "calib_bc", "char", "assemble",
                                     "all"])
    ap.add_argument("--ckpt", default=str(OUT / "logs_F" / "ckpt"))
    a = ap.parse_args()
    {"family": part_family, "base": part_base, "sp": part_sp, "cal": part_cal, "power": part_power,
     "calib": part_calib, "calib_bc": part_calib_bc, "char": lambda: part_char(CHAR_TARGETS),
     "assemble": part_assemble, "all": lambda: part_all(Path(a.ckpt))}[a.part]()
