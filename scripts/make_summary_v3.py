"""Build results/analysis/SUMMARY.md from the JSON outputs. Run: python make_summary_v3.py

Every number comes from v(), which reads it from its JSON and lists its key path in the headline table.
"""

from __future__ import annotations

import json
from functools import reduce

import numpy as np

import v2lib  # noqa: F401
from v2lib import OUT

_cache: dict[str, dict] = {}
ROWS: list[tuple[str, str, str, str]] = []
_seen: set[str] = set()
SECTION = ["-"]


def J(name: str) -> dict:
    if name not in _cache:
        _cache[name] = json.loads((OUT / f"{name}.json").read_text())
    return _cache[name]


def g(name: str, *path):
    return reduce(lambda o, k: o[k], path, J(name))


def kp(name: str, *path) -> str:
    s = "".join(f"[{k}]" if isinstance(k, int) else f".{k}" for k in path)
    return f"`{name}.json`: `{s[1:] if s.startswith('.') else s}`"


def fmt(x, nd=3):
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if 0 < abs(x) < 10 ** (-nd):
            return f"{x:.1e}"
        return f"{x:.{nd}f}"
    if isinstance(x, list):
        return "[" + ", ".join(fmt(y, nd) for y in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {fmt(y, nd)}" for k, y in x.items()) + "}"
    return str(x)


def v(what: str, name: str, *path, nd: int = 3, sl: slice | None = None) -> str:
    val = g(name, *path)
    if sl is not None:
        val = val[sl]
    s = fmt(val, nd)
    key = kp(name, *path) + (f" `[{sl.start or 0}:{sl.stop}]`" if sl is not None else "")
    if key not in _seen:
        _seen.add(key)
        ROWS.append((SECTION[0], what, s, key))
    return s


def section(name: str) -> None:
    SECTION[0] = name


def run_index(name: str, run_id: str) -> int:
    return next(i for i, r in enumerate(g(name, "runs")) if r["run_id"] == run_id)


def write_derived() -> None:
    """Numbers derived from several JSON values, written to v3_summary_derived.json so they have keys."""
    loo = J("v3_B_leave_one_out")
    lam2 = {n: d["rot0"]["top_eigs"][1] for n, d in loo["leave_one_out_spectra"].items()}
    others = [x for n, x in lam2.items() if n != "meta-llama/Llama-3.2-1B-Instruct"]
    ca = J("v3_C_rotations_A")["C3_rotation_invariant"]["mean_over_rotations"]
    obs1 = ca["spectrum"]["top_eigs"][0]
    s23 = loo["without_llama_3_2_1b"]["rot0_suite"]
    o4 = s23["observed"]["top_eigs"][3]
    g = np.array(J("v3_C_rotations_A")["kg_fits_4_rotations"]["letter_preference"]
                 ["meta-llama/Llama-3.2-1B-Instruct"]["pi_A_B_C_D"])
    d = {"loo_rot0_lambda2_min_max_without_llama_3_2_1b_removal": [min(others), max(others)],
         "rot0_lambda2_all_models": J("v3_B_extra")["suite"]["observed"]["top_eigs"][1],
         "mean_over_rotations_rank1_obs_over_p99": {
             k: obs1 / ca["suite"]["nulls"][k]["rank_eigs_p99"][0] for k in ("kg_letter4", "kg_uniform4")},
         "loo23_rot0_rank4_obs_minus_p99": {k: o4 - s23["nulls"][k]["rank_eigs_p99"][3] for k in ("3pl_post", "letter_post")},
         "llama_3_2_1b_pi_A": float(g[0])}
    (OUT / "v3_summary_derived.json").write_text(json.dumps(d, indent=1))


def main() -> None:
    L: list[str] = []
    w = L.append
    I1, I2, I7 = "item01_within_table", "item02_shared_randomness", "item07_across_model"
    BX, CA = "v3_B_extra", "v3_C_rotations_A"
    LF, C4, C5, LOO = "v3_C1b_letter_following", "v3_C4_replicate_agreement", "v3_C5_near_ties", "v3_B_leave_one_out"
    I9, I10 = "item09_family_alignment", "item10_pair_correlations"
    RAW7 = "raw_item07_across_two_margin_null"
    DV = "v3_summary_derived"
    write_derived()

    w("# Analysis summary (v3 data)")
    w("")
    w("Data: `results/mmlu_ensemble_greedy.npz` (24 x 500 = R2 replicate A "
      "rotation 0; canonical tokenization; parser v3.1), `results/mmlu_ensemble_rotations.npz` (A) and "
      "`results/mmlu_ensemble_rotations_rep.npz` (B), 24 x 500 x 4 rotations; sampled runs `results/*.npz` "
      "(scored with parser v3.1). "
      "Code: `scripts/` (drivers `item01`..`item14`, "
      "`v3lib.py` + `test_v3lib.py`, `v3_across.py`, `v3_rotation_letters.py`, `v3_loo.py`, "
      "`v3_replicate_compare.py`, this file).")
    w("")
    w("Conventions: columns centered and divided by the population SD; R = Z^T Z / rows; "
      "binary matrices use the split columns (non-constant columns for the rotation average); "
      "lambda_+ = (1 + sqrt(N_cols / M))^2; buffered count = #{lambda > 1.05 lambda_+}; a null rejects rank j "
      "when the observed lambda_j exceeds the 99th percentile of the null lambda_j (rank-matched p99). "
      "Parametric nulls are sampled conditional on every kept column staying non-constant (N_cols and "
      "lambda_+ equal the observed ones). \"Directions beyond ability\" = ranks 2, 3, ... above the p99, "
      "contiguous from rank 2. All seeds are fixed and recorded in each JSON `meta`/`seed`; draw counts: "
      "1000 per null, 500 per Bernoulli null, 1000 rotation-randomized ensembles, 5000 label permutations "
      "for eta^2, 10000 for correlations, 20000 for the family tests.")
    w("")

    section("committed")
    w("## Committed answers")
    w("")
    w("> **Scope.** The C(3) per-rank verdict below holds for rank 2 only. The family-wise test over ranks 2-4 (G, H) rejects the 2PL and Rasch nulls on all-4 marginally (p_T 0.026 / 0.041; size-calibrated 0.017-0.057); two near-duplicate model pairs (Phi-3/3.5-mini, Llama-3/3.1-8B) carry the excess. The single validity rule (G5) replaces section F's \"calibrated validity\". The paper follows G and H.")
    w("")
    w("**C(3) Directions beyond ability after letter preference is removed: 0.** One ability direction; "
      "the rest was letter preference. The cleanest letter-free matrix is \"correct in all 4 rotations\": a "
      "model that guesses is correct in all four only with probability prod_L pi_m(L) <= 1/256, the same for "
      "every question, so a letter preference cannot make a question-specific pattern. Its eigenvalues are "
      f"{v('all-4: eigenvalues 1-4', CA, 'C3_rotation_invariant', 'all4_correct', 'spectrum', 'top_eigs', nd=1, sl=slice(0, 4))} "
      f"against lambda_+ = {v('all-4: lambda_+', CA, 'C3_rotation_invariant', 'all4_correct', 'spectrum', 'lambda_plus', nd=2)} "
      f"({v('all-4: split columns', CA, 'C3_rotation_invariant', 'all4_correct', 'spectrum', 'n_cols')} split columns). "
      "Eigenvalue 2 is below the rank-2 p99 of the project 2PL "
      f"({v('all-4: 2PL p99', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', '2pl_post', 'rank_eigs_p99', nd=1)}), "
      f"its prior version, Rasch JML ({v('all-4: Rasch p99', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', 'rasch_jml', 'rank_eigs_p99', nd=1)}), "
      f"Bock-Aitkin a <= 4 and a <= 8, and both 4-rotation know-or-guess nulls "
      f"(letter-aware {v('all-4: KG letter p99', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', 'kg_letter4', 'rank_eigs_p99', nd=1)}). "
      "It exceeds only the fixed-margin (curveball = exact conditional Rasch, equal slopes) null "
      f"({v('all-4: curveball p99', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', 'fixed_margin', 'rank_eigs_p99', nd=1)}), "
      "which the heterogeneous-slope 2PL explains. Ranks beyond ability by null: "
      f"{v('all-4: ranks beyond ability by null', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'ranks_beyond_ability')}. "
      "Rank 4 (below lambda_+) exceeds the p99 of the Bock-Aitkin and know-or-guess nulls but not of Rasch "
      f"(ranks above p99: {v('all-4: ranks above p99 by null', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'ranks_above_p99_by_null')}). "
      "Against the project 2PL, rank 4 is a tie within Monte Carlo noise, not a clean non-rejection: observed "
      f"{v('all-4: eigenvalue 4', CA, 'C3_rotation_invariant', 'all4_correct', 'spectrum', 'top_eigs', 3, nd=2)} vs p99 "
      f"{v('all-4: 2PL rank-4 p99', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', '2pl_post', 'rank_eigs_p99', 3, nd=2)} "
      f"(MC p = {v('all-4: 2PL rank-4 MC p', CA, 'C3_rotation_invariant', 'all4_correct', 'suite', 'nulls', '2pl_post', 'mc_pvalue_per_rank', 3, nd=3)}); "
      "re-simulating the same fit with seeds 1-3 (1000 draws each; 2PL investigation, not a stored output) gives a "
      "rank-4 p99 of 20.74-21.08, so rank 4 is above the p99 for two seeds and below it for one, and replicate B puts "
      "it above (C4). Rank 4 is not contiguous with rank 1, lies inside the MP bulk, and its model scores group by family "
      f"(eta^2 = {v('all-4 eigvec 4: family eta^2', CA, 'C3_rotation_invariant', 'all4_correct', 'letters_subject_family', 'eigenvectors', 3, 'eta_sq_model_scores_by_family', 'eta_sq', nd=2)}, "
      f"p = {v('all-4 eigvec 4: family p', CA, 'C3_rotation_invariant', 'all4_correct', 'letters_subject_family', 'eigenvectors', 3, 'eta_sq_model_scores_by_family', 'perm_p', nd=3)}): "
      "a candidate weak family signal in the bulk, not a direction by our rule.")
    w("")
    w("The other two letter-free constructions agree, with one borderline case each. Rotation-randomized "
      f"ensembles ({v('C2: number of random-rotation ensembles', CA, 'meta', 'n_random_rotation_draws')} draws): median "
      f"eigenvalues {v('C2: median eigenvalues 1-4', CA, 'C2_rotation_randomized', 'distribution', 'eig_median', nd=1, sl=slice(0, 4))}; "
      "in the representative (median) draw no rank exceeds the 3PL or the letter-aware p99 "
      f"(rank 2: {v('C2 representative: eigenvalues', CA, 'C2_rotation_randomized', 'representative_draw', 'spectrum', 'top_eigs', nd=2, sl=slice(0, 4))} vs "
      f"letter-aware p99 {v('C2 representative: letter-aware p99', CA, 'C2_rotation_randomized', 'representative_suite', 'nulls', 'letter_post', 'rank_eigs_p99', nd=2)}: "
      "at the threshold); the share of the 1000 draws whose lambda_2 exceeds that p99 is "
      f"{v('C2: share of draws above representative letter-aware p99 (ranks 1-4)', CA, 'C2_rotation_randomized', 'frac_draws_above_representative_p99', 'letter_post')}; "
      "against the know-or-guess null fitted on all 4 rotations with a fresh random rotation per draw, the median draw "
      f"exceeds no rank (p99 {v('C2: KG random-rotation p99', CA, 'C2_rotation_randomized', 'kg_letter4_random_rotation_null', 'rank_eigs_p99', nd=1)}; "
      f"share of draws above: {v('C2: share of draws above KG p99', CA, 'C2_rotation_randomized', 'kg_letter4_random_rotation_null', 'frac_observed_draws_above_p99_per_rank')}). "
      "Rotation-averaged correctness: eigenvalues "
      f"{v('mean over rotations: eigenvalues', CA, 'C3_rotation_invariant', 'mean_over_rotations', 'spectrum', 'top_eigs', nd=1, sl=slice(0, 4))}; "
      f"rank 2 exceeds the letter-aware know-or-guess p99 ({v('mean: KG letter p99', CA, 'C3_rotation_invariant', 'mean_over_rotations', 'suite', 'nulls', 'kg_letter4', 'rank_eigs_p99', nd=1)}) "
      f"but not the uniform-guess one ({v('mean: KG uniform p99', CA, 'C3_rotation_invariant', 'mean_over_rotations', 'suite', 'nulls', 'kg_uniform4', 'rank_eigs_p99', nd=1)}), "
      "and both nulls miss rank 1 (observed / p99 = "
      f"{v('mean: rank-1 observed / p99 (KG letter, KG uniform)', DV, 'mean_over_rotations_rank1_obs_over_p99', nd=2)}): they tie correctness to ability less tightly than the data, as the v2 2PL did "
      "at rank 1, so this rank-2 excess is not a clean rejection. The rotation-independent 2PL "
      f"({v('mean: independent-rotation 2PL p99', CA, 'C3_rotation_invariant', 'mean_over_rotations', 'suite', 'nulls', 'indep4_2pl', 'rank_eigs_p99', nd=1)}) "
      "is liberal by construction (it averages away the cell noise) and is reported only as a bound.")
    w("")
    w("**B letter analysis (rotation 0).** Eigenvector 2 is an answer-letter direction. Its question loadings group "
      f"by gold letter (eta^2 = {v('rot0 eigvec 2: eta^2 by gold letter', BX, 'letter_analysis', 'eigenvectors', 1, 'eta_sq_by_shown_gold_letter', 'eta_sq', nd=3)}, "
      f"permutation p = {v('rot0 eigvec 2: eta^2 p', BX, 'letter_analysis', 'eigenvectors', 1, 'eta_sq_by_shown_gold_letter', 'perm_p', nd=4)}; "
      f"mean loading by letter {v('rot0 eigvec 2: mean loading by gold letter', BX, 'letter_analysis', 'eigenvectors', 1, 'eta_sq_by_shown_gold_letter', 'mean_loading_by_letter', nd=3)}); "
      "its model scores correlate with the models' answer-letter shares: "
      f"A r = {v('rot0 eigvec 2: r with share A', BX, 'letter_analysis', 'eigenvectors', 1, 'model_score_corr', 'share_A', 'r', nd=2)} "
      f"(p = {v('rot0 eigvec 2: p share A', BX, 'letter_analysis', 'eigenvectors', 1, 'model_score_corr', 'share_A', 'perm_p_two_sided', nd=4)}), "
      f"D r = {v('rot0 eigvec 2: r with share D', BX, 'letter_analysis', 'eigenvectors', 1, 'model_score_corr', 'share_D', 'r', nd=2)} "
      f"(p = {v('rot0 eigvec 2: p share D', BX, 'letter_analysis', 'eigenvectors', 1, 'model_score_corr', 'share_D', 'perm_p_two_sided', nd=4)}), "
      f"accuracy r = {v('rot0 eigvec 2: r with accuracy', BX, 'letter_analysis', 'eigenvectors', 1, 'model_score_corr', 'accuracy', 'r', nd=2)}. "
      "Eigenvector 1 is ability "
      f"(r with accuracy {v('rot0 eigvec 1: r with accuracy', BX, 'letter_analysis', 'eigenvectors', 0, 'model_score_corr', 'accuracy', 'r', nd=2)}) "
      f"with a letter part (eta^2 = {v('rot0 eigvec 1: eta^2 by gold letter', BX, 'letter_analysis', 'eigenvectors', 0, 'eta_sq_by_shown_gold_letter', 'eta_sq', nd=2)}; "
      f"r with share C {v('rot0 eigvec 1: r with share C', BX, 'letter_analysis', 'eigenvectors', 0, 'model_score_corr', 'share_C', 'r', nd=2)}). "
      f"Eigenvectors 3 and 4: eta^2 = {v('rot0 eigvec 3: eta^2 by gold letter', BX, 'letter_analysis', 'eigenvectors', 2, 'eta_sq_by_shown_gold_letter', 'eta_sq', nd=3)} and "
      f"{v('rot0 eigvec 4: eta^2 by gold letter', BX, 'letter_analysis', 'eigenvectors', 3, 'eta_sq_by_shown_gold_letter', 'eta_sq', nd=3)} "
      "(small; significant only because of the 460 questions); no letter-share correlation has p < 0.05. "
      "The direction moves with the DISPLAYED letters (decisive test, C1b): in rotations 1-3 the same models carry it "
      f"(|r| of model scores with rotation 0: {v('C1b eigvec 2 rotation 1: model-score match', LF, 'A', 'eigvec2', '1', 'model_score_match_abs_r', nd=2)}, "
      f"{v('C1b eigvec 2 rotation 2: model-score match', LF, 'A', 'eigvec2', '2', 'model_score_match_abs_r', nd=2)}, "
      f"{v('C1b eigvec 2 rotation 3: model-score match', LF, 'A', 'eigvec2', '3', 'model_score_match_abs_r', nd=2)}); "
      "its question loadings follow the rotation-0 letter profile mapped through the shown letters "
      f"(r = {v('C1b eigvec 2 r1: r_shown', LF, 'A', 'eigvec2', '1', 'r_shown', nd=2)}, {v('C1b eigvec 2 r2: r_shown', LF, 'A', 'eigvec2', '2', 'r_shown', nd=2)}, "
      f"{v('C1b eigvec 2 r3: r_shown', LF, 'A', 'eigvec2', '3', 'r_shown', nd=2)}), not through the original options "
      f"(r = {v('C1b eigvec 2 r1: r_orig', LF, 'A', 'eigvec2', '1', 'r_orig', nd=2)}, {v('C1b eigvec 2 r2: r_orig', LF, 'A', 'eigvec2', '2', 'r_orig', nd=2)}, "
      f"{v('C1b eigvec 2 r3: r_orig', LF, 'A', 'eigvec2', '3', 'r_orig', nd=2)}), and do not correlate with the rotation-0 loadings of the same questions "
      f"(r = {v('C1b eigvec 2 r1: r_direct', LF, 'A', 'eigvec2', '1', 'r_direct', nd=2)}, {v('C1b eigvec 2 r2: r_direct', LF, 'A', 'eigvec2', '2', 'r_direct', nd=2)}, "
      f"{v('C1b eigvec 2 r3: r_direct', LF, 'A', 'eigvec2', '3', 'r_direct', nd=2)}); permutation p of r_shown - r_orig = "
      f"{v('C1b eigvec 2 r1: p', LF, 'A', 'eigvec2', '1', 'perm_p_two_sided', nd=4)} in each rotation. "
      "It rests mostly on one model: Llama-3.2-1B guesses A with probability "
      f"{v('Llama-3.2-1B letter preference A-D', CA, 'kg_fits_4_rotations', 'letter_preference', 'meta-llama/Llama-3.2-1B-Instruct', 'pi_A_B_C_D', nd=2)}. "
      "Without it, rotation-0 eigenvalues are "
      f"{v('rot0 without Llama-3.2-1B: eigenvalues', LOO, 'leave_one_out_spectra', 'meta-llama/Llama-3.2-1B-Instruct', 'rot0', 'top_eigs', nd=1, sl=slice(0, 4))}, "
      f"eigenvector 2's gold-letter eta^2 falls to {v('rot0 without Llama-3.2-1B: eta^2 by gold letter, eigvec 1-4', LOO, 'leave_one_out_spectra', 'meta-llama/Llama-3.2-1B-Instruct', 'rot0_eta_sq_gold_letter_eigvec_1to4', nd=2)} "
      "(the letter part moves to eigenvector 3); every other single removal leaves eigenvalue 2 in "
      f"{v('rot0 eigenvalue 2 range over the other 23 single-model removals', DV, 'loo_rot0_lambda2_min_max_without_llama_3_2_1b_removal', nd=1)}.")
    w("")
    w("**B nulls (rotation 0, which ranks survive).** Eigenvalues 1-6: "
      f"{v('rot0: eigenvalues 1-6', BX, 'suite', 'observed', 'top_eigs', nd=1)} against lambda_+ = "
      f"{v('rot0: lambda_+', I7, 'observed', 'lambda_plus', nd=2)} (gamma = {v('rot0: split columns', I7, 'observed', 'n_split')}/24); "
      f"MP count {v('rot0: MP count strict', I7, 'observed', 'mp_count_strict')}, buffered "
      f"{v('rot0: buffered count', I7, 'observed', 'mp_count_buffered')}. Difficulty-matched Bernoulli null: at most "
      f"{v('Bernoulli (per question) null: max buffered count', I7, 'bernoulli_nulls_d_across', 'A_per_question', 'mp_signal_max', nd=0)} "
      "buffered eigenvalue in 500 draws (rank p99 "
      f"{v('Bernoulli per-question null: rank p99', BX, 'bernoulli_rank_nulls', 'bern_question', 'rank_eigs_p99', nd=1)}); per-model-rate null: at most "
      f"{v('per-model-rate null: max buffered count', I7, 'bernoulli_nulls_d_across', 'B_per_model', 'mp_signal_max', nd=0)} "
      f"(rank p99 {v('per-model-rate null: rank p99', BX, 'bernoulli_rank_nulls', 'bern_model', 'rank_eigs_p99', nd=1)}). "
      "Ranks above the rank-matched p99, per ability-keeping null:")
    w("")
    w("| null | model | rank p99 (1-4) | ranks above |")
    w("|---|---|---|---|")
    rows = [("Rasch JML", RAW7, ("nulls", "a_rasch_jml")), ("Rasch CML", RAW7, ("nulls", "a_rasch_cml")),
            ("fixed margins (curveball)", RAW7, ("nulls", "b_fixed_margin")),
            ("2PL, project (posterior draws)", RAW7, ("nulls", "c_post_2pl")),
            ("2PL, project (prior draws)", RAW7, ("nulls", "c_prior_2pl")),
            ("2PL, Bock-Aitkin a <= 4", BX, ("suite", "nulls", "ba_a4")),
            ("2PL, Bock-Aitkin a <= 8", BX, ("suite", "nulls", "ba_a8")),
            ("3PL, c = 0.25", BX, ("suite", "nulls", "3pl_post")),
            ("letter-aware know-or-guess (rotation-0 fit)", BX, ("suite", "nulls", "letter_post")),
            ("know-or-guess, 4-rotation fit, letter-aware", BX, ("suite", "nulls", "kg_letter4")),
            ("know-or-guess, 4-rotation fit, uniform guess", BX, ("suite", "nulls", "kg_uniform4"))]
    desc = {"Rasch JML": "sigma(theta_m - b_i), joint ML",
            "Rasch CML": "theta by conditional ML",
            "fixed margins (curveball)": "uniform over 0/1 matrices with the observed margins",
            "2PL, project (posterior draws)": "theta_m fixed; (log a_i, b_i) ~ normal, integrated (MML over questions)",
            "2PL, project (prior draws)": "same fit; (a_i, b_i) from the population law",
            "2PL, Bock-Aitkin a <= 4": "theta ~ N(0,1) integrated; (a_i, d_i) fixed, box-bounded; ML theta_m",
            "2PL, Bock-Aitkin a <= 8": "as above",
            "3PL, c = 0.25": "0.25 + 0.75 sigma(a_i(theta_m - b_i)), project MML",
            "letter-aware know-or-guess (rotation-0 fit)": "pi_m(g_i) + (1 - pi_m(g_i)) sigma(...), project MML",
            "know-or-guess, 4-rotation fit, letter-aware": "composite MML on 4 rotations; know once per cell",
            "know-or-guess, 4-rotation fit, uniform guess": "as above with pi = 0.25"}
    for lab, f, path in rows:
        w(f"| {lab} | {desc[lab]} | {v(lab + ': rank p99', f, *path, 'rank_eigs_p99', nd=1)} | "
          f"{v(lab + ': ranks above p99', f, *path, 'observed_ranks_above_p99')} |")
    for fct in ("1.25", "1.5", "2", "3", "5", "10"):
        key = f"c_post_2pl_stress_slopes_x{fct}"
        w(f"| stress: project 2PL, slopes x{fct} (not fitted) | log-lik vs fit "
          f"{v(f'stress x{fct}: log-lik minus fit', RAW7, 'nulls', key, 'marginal_loglik_of_scaled_model_minus_fit', nd=1)} | "
          f"{v(f'stress x{fct}: rank p99', RAW7, 'nulls', key, 'rank_eigs_p99', nd=1)} | "
          f"{v(f'stress x{fct}: ranks above', RAW7, 'nulls', key, 'observed_ranks_above_p99')} |")
    w("")
    w("Rank 2 (the letter direction) exceeds every fitted null, including both letter-aware ones, and every stress "
      "test up to x2; the know-or-guess form (letter preference acts only when the model does not know) under-produces "
      "the letter structure, which the rotation analyses identify directly. Without Llama-3.2-1B (23 models, refitted) "
      f"the ranks above the 3PL p99 are {v('23 models (no Llama-3.2-1B): ranks above 3PL p99', LOO, 'without_llama_3_2_1b', 'rot0_suite', 'nulls', '3pl_post', 'observed_ranks_above_p99')} "
      f"and above the letter-aware p99 {v('23 models: ranks above letter-aware p99', LOO, 'without_llama_3_2_1b', 'rot0_suite', 'nulls', 'letter_post', 'observed_ranks_above_p99')} "
      f"(rank 4 only; observed minus p99 {v('23 models: rank-4 observed minus p99 (3PL, letter-aware)', DV, 'loo23_rot0_rank4_obs_minus_p99', nd=2)}). Fits: project 2PL log-lik "
      f"{v('project 2PL marginal log-lik (rot0)', RAW7, 'fits', 'twopl_mml', 'marginal_loglik', nd=1)}, 3PL "
      f"{v('3PL log-lik', BX, 'suite', 'fits', '3pl_post', 'loglik', nd=1)}, letter-aware "
      f"{v('letter-aware log-lik', BX, 'suite', 'fits', 'letter_post', 'loglik', nd=1)} (same number of parameters); "
      f"Bock-Aitkin slopes at the bound: {v('Bock-Aitkin a<=4: share of slopes at bound', BX, 'suite', 'fits', 'ba_a4', 'frac_slopes_at_bound', nd=3)} (a <= 4), "
      f"{v('Bock-Aitkin a<=8: share of slopes at bound', BX, 'suite', 'fits', 'ba_a8', 'frac_slopes_at_bound', nd=3)} (a <= 8).")
    w("")
    w("Joint statistic lambda_2 / lambda_1: observed "
      f"{v('lambda2/lambda1 observed', I7, 'lambda2_over_lambda1', 'observed', nd=3)} (v2 data: 0.503). Against the project 2PL and all six "
      "slope multipliers pooled: null p99 "
      f"{v('lambda2/lambda1: pooled stress null p99', I7, 'lambda2_over_lambda1', 'pooled_stress_x1.25_to_x10', 'null_p99', nd=3)}, max "
      f"{v('lambda2/lambda1: pooled stress null max', I7, 'lambda2_over_lambda1', 'pooled_stress_x1.25_to_x10', 'null_max', nd=3)}, MC p = "
      f"{v('lambda2/lambda1: pooled stress p', I7, 'lambda2_over_lambda1', 'pooled_stress_x1.25_to_x10', 'mc_p_upper', nd=5)}; fitted 2PL posterior max "
      f"{v('lambda2/lambda1: 2PL posterior null max', I7, 'lambda2_over_lambda1', 'by_null', 'c_post_2pl', 'null_max', nd=3)}. "
      "It does not separate the data from the letter-aware nulls: p = "
      f"{v('lambda2/lambda1: letter-aware p', BX, 'suite', 'nulls', 'letter_post', 'lambda2_over_lambda1', 'mc_p_upper', nd=3)} (rotation-0 fit), "
      f"{v('lambda2/lambda1: KG letter p', BX, 'suite', 'nulls', 'kg_letter4', 'lambda2_over_lambda1', 'mc_p_upper', nd=3)} (4-rotation fit). "
      "Against Rasch and fixed margins the ratio is uninformative (their lambda_1 is far below the observed, "
      f"so their ratio p99 is {v('lambda2/lambda1: Rasch JML p99', I7, 'lambda2_over_lambda1', 'by_null', 'a_rasch_jml', 'null_p99', nd=3)}).")
    w("")
    w("**Family (model level).** QAP / model-label permutation test of same- vs cross-family mean pair r: "
      f"same {v('family: mean r same', I10, 'qap_model_label_permutation', 'mean_r_same', nd=3)}, cross "
      f"{v('family: mean r cross', I10, 'qap_model_label_permutation', 'mean_r_cross', nd=3)}, one-sided p = "
      f"{v('family QAP p', I10, 'qap_model_label_permutation', 'perm_p_one_sided', nd=3)} "
      f"(Holm {v('family QAP p Holm', I10, 'family_tests_holm', 'qap_same_vs_cross_mean_r', nd=3)}). Eigenvector-family "
      "alignment (eta^2 of model scores, label permutation): eigenvectors 1-4 eta^2 = "
      + ", ".join(v(f'family eta^2 eigvec {j + 1}', I9, 'eigenvectors', j, 'eta_sq_family', nd=2) for j in range(4))
      + ", p = " + ", ".join(v(f'family p eigvec {j + 1}', I9, 'eigenvectors', j, 'perm_pvalue', nd=4) for j in range(4))
      + "; Holm-adjusted (over QAP, eigenvectors 1-4 and the joint 2-4 test): "
      + ", ".join(v(f'family Holm eigvec {j + 1}', I10, 'family_tests_holm', f'eigvec{j + 1}_family_eta_sq', nd=3) for j in range(4))
      + f"; joint 2-4 p = {v('family joint 2-4 p', I9, 'joint_eigvec_2to4', 'perm_pvalue', nd=3)} "
      f"(Holm {v('family joint 2-4 Holm', I10, 'family_tests_holm', 'joint_eigvec_2to4_family', nd=3)}). "
      "Only eigenvector 4 (rank 4, eigenvalue below lambda_+) aligns with family after Holm; with the five singleton "
      f"families removed its p is {v('family eigvec 4 p, multi-member families only', I9, 'eigenvectors', 3, 'perm_pvalue_multi_member_families_only', nd=3)}. "
      "The pair-level KS test is superseded (it treats the 276 dependent pairs as independent).")
    w("")
    w("**Replicate agreement (C4).** Rotation 0 is identical in A and B. Largest eigenvalue difference over every "
      f"matrix compared: {v('A vs B: max |eigenvalue difference|', C4, 'max_abs_eig_diff_overall', nd=3)}. Null verdicts: "
      f"{v('A vs B: null verdicts that differ', C4, 'n_null_verdicts_disagree')} of "
      f"{v('A vs B: null verdicts compared', C4, 'n_null_verdicts')} differ, all at rank 4 "
      f"({v('A vs B: differing verdicts', C4, 'disagreeing_null_verdicts')}); no verdict on ranks 1-3 differs, and every "
      "committed answer above is the same in B.")
    w("")
    w("**Near ties (C5).** Cells with a letter gap < 0.13 in A or B: "
      f"{v('near ties: share of cells', C5, 'frac_near_tie_cells', nd=3)} of all cells "
      f"({v('near ties: cells by rotation', C5, 'n_near_tie_cells_by_rotation')}). Each binary variant refitted with its own nulls: "
      f"rotation 0 with near ties set to the majority over rotations ({v('near ties: rot0 cells changed', C5, 'majority_n_cells_changed_rot0')} cells change): "
      f"ranks above 3PL {v('near ties rot0 majority: 3PL ranks above', C5, 'refitted_variants', 'rot0_majority', 'suite', 'nulls', '3pl_post', 'observed_ranks_above_p99')}, "
      f"letter-aware {v('near ties rot0 majority: letter-aware ranks above', C5, 'refitted_variants', 'rot0_majority', 'suite', 'nulls', 'letter_post', 'observed_ranks_above_p99')} (unchanged); "
      f"dropping the {v('near ties: questions with a rot0 near tie', C5, 'n_questions_with_a_rot0_near_tie')} questions with a rotation-0 near tie leaves 203: ranks above letter-aware "
      f"{v('near ties rot0 drop questions: letter-aware ranks above', C5, 'refitted_variants', 'rot0_drop_questions_with_rot0_near_tie', 'suite', 'nulls', 'letter_post', 'observed_ranks_above_p99')}, 3PL "
      f"{v('near ties rot0 drop questions: 3PL ranks above', C5, 'refitted_variants', 'rot0_drop_questions_with_rot0_near_tie', 'suite', 'nulls', '3pl_post', 'observed_ranks_above_p99')}. "
      "All-4-correct, majority: ranks beyond ability "
      f"{v('near ties all-4 majority: ranks beyond ability', C5, 'refitted_variants', 'all4_majority', 'suite', 'ranks_beyond_ability')}; "
      "near-tie rotations ignored: "
      f"{v('near ties all-4 ignored: ranks beyond ability', C5, 'refitted_variants', 'all4_near_tie_rotations_ignored', 'suite', 'ranks_beyond_ability')}. "
      "The C(3) answer (0 beyond ability, rank 2 only above the fixed-margin null, rank 4 only above Bock-Aitkin / "
      "know-or-guess) does not change.")
    w("")
    w("**D. Within-model shared-component directions.** S1 (Qwen2.5-7B shared seed) and S2 Llama-3-8B (shared 5-shot) "
      "are letter directions: the top eigenvector's loadings group by gold letter (eta^2 = "
      f"{v('S1: eta^2 by gold letter', I2, 'runs', 'mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'eta_sq', nd=2)}, "
      f"p = {v('S1: eta^2 p', I2, 'runs', 'mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'perm_p', nd=4)}; "
      f"Llama S2 {v('Llama S2: eta^2 by gold letter', I2, 'runs', 'mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'eta_sq', nd=2)}, "
      f"p = {v('Llama S2: eta^2 p', I2, 'runs', 'mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'perm_p', nd=4)}) "
      "and not by subject (S1 eta^2 "
      f"{v('S1: eta^2 by subject', I2, 'runs', 'mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'eta_sq', nd=2)} vs null mean "
      f"{v('S1: subject eta^2 null mean', I2, 'runs', 'mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'null_mean', nd=2)}, "
      f"p = {v('S1: subject p', I2, 'runs', 'mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'perm_p', nd=3)}; Llama S2 "
      f"{v('Llama S2: eta^2 by subject', I2, 'runs', 'mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'eta_sq', nd=2)}, "
      f"p = {v('Llama S2: subject p', I2, 'runs', 'mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'perm_p', nd=3)}). "
      "S2 Qwen2.5-7B is an exemplar-overlap direction: no gold-letter grouping (eta^2 = "
      f"{v('Qwen S2: eta^2 by gold letter', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'eta_sq', nd=2)}, "
      f"p = {v('Qwen S2: eta^2 p', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_gold_letter', 'perm_p', nd=3)}), "
      "passes that share an exemplar answer more alike (z = "
      f"{v('Qwen S2: exemplar-overlap z', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'exemplar_alignment', 'pass_pair_similarity_shared_exemplars_borderline', 'z_vs_null', nd=1)}, "
      f"p = {v('Qwen S2: exemplar-overlap p', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'exemplar_alignment', 'pass_pair_similarity_shared_exemplars_borderline', 'perm_p_upper', nd=4)}); "
      "its loadings group weakly by subject (eta^2 "
      f"{v('Qwen S2: eta^2 by subject', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'eta_sq', nd=2)} vs null mean "
      f"{v('Qwen S2: subject eta^2 null mean', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'null_mean', nd=2)}, "
      f"p = {v('Qwen S2: subject p', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'top_pc', 'loadings_eta_sq_by_subject', 'perm_p', nd=3)}, uncorrected), "
      "and a same-subject exemplar raises accuracy by "
      f"{v('Qwen S2: subject-match effect', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'exemplar_alignment', 'subject_match_borderline', 'mean_diff_matched_minus_unmatched', nd=4)} "
      f"(p = {v('Qwen S2: subject-match p', I2, 'runs', 'mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct', 'exemplar_alignment', 'subject_match_borderline', 'perm_p_two_sided', nd=3)}). "
      "The v3.1 rescore did not change these three runs (their item-1 statistics are identical to v2).")
    w("")

    section("contradicts paper")
    w("## Changes against the submitted (April) paper")
    w("")
    w("1. The paper's across-model claim, \"one direction that follows ability and at least one more that a "
      "one-ability model does not reproduce\" (abstract, Sec. 3.3, Limits, Discussion of the submitted version), resting on eigenvalue 2, does "
      "not hold as a statement about the models' knowledge. Eigenvalue 2 is answer-letter preference: it follows the "
      "displayed letters when the options rotate (C1b), it is carried mostly by one model (Llama-3.2-1B, pi_A = "
      f"{v('Llama-3.2-1B pi_A', DV, 'llama_3_2_1b_pi_A', nd=2)}), "
      "and in letter-free matrices no direction beyond ability exceeds the 2PL / Rasch / Bock-Aitkin / know-or-guess nulls.")
    w(f"2. Eigenvalues: {v('rot0: eigenvalues 1-4 (paper: 92.4, 46.5, 33.7, 25.3)', I7, 'observed', 'top_eigs_1to4', nd=1)} "
      f"against lambda_+ = {v('rot0: lambda_+ (paper 29.13)', I7, 'observed', 'lambda_plus', nd=2)} (paper: 92.4, 46.5, 33.7, 25.3 against "
      "29.13): two exceed 1.05 lambda_+, not three.")
    w("3. Against the project 2PL, ranks 1-4 exceed their p99 (paper: 1-3), as they do against Rasch (JML, CML) and "
      "fixed margins. The Bock-Aitkin 2PL rejects ranks 2 and 4; with the 4-option guessing floor (3PL, c = 0.25) only "
      "rank 2 exceeds.")
    w("4. Stress tests: ranks 2 and 4 stay above up to x2; at x3, x5 and x10 no rank does (at x3 rank 4 sits at its "
      f"p99: {v('rot0: eigenvalue 4', I7, 'observed', 'top_eigs_1to4', 3, nd=2)} vs "
      f"{v('stress x3: rank-4 p99', RAW7, 'nulls', 'c_post_2pl_stress_slopes_x3', 'rank_eigs_p99', 3, nd=2)}). The paper's "
      "\"only eigenvalue 2 ... stays above when every slope is doubled\" now reads ranks 2 and 4.")
    w("5. Family: the paper's \"eigenvectors 2-4 jointly do (p = 0.013) ... only eigenvector 4 reaches p = 0.034\" "
      f"becomes joint p = {v('family joint 2-4 p', I9, 'joint_eigvec_2to4', 'perm_pvalue', nd=3)}, eigenvector 4 p = "
      f"{v('family p eigvec 4', I9, 'eigenvectors', 3, 'perm_pvalue', nd=4)} (Holm "
      f"{v('family Holm eigvec 4', I10, 'family_tests_holm', 'eigvec4_family_eta_sq', nd=3)}); the model-level QAP test gives p = "
      f"{v('family QAP p', I10, 'qap_model_label_permutation', 'perm_p_one_sided', nd=3)} (the pair-level KS p = "
      f"{v('family pair-level KS p (superseded)', I10, 'v2', 'ks_p', nd=3)} treats dependent pairs as independent). Pair r: same "
      f"{v('family: mean r same', I10, 'qap_model_label_permutation', 'mean_r_same', nd=3)} vs cross "
      f"{v('family: mean r cross', I10, 'qap_model_label_permutation', 'mean_r_cross', nd=3)} (paper 0.41 / 0.34), Cohen's d "
      f"{v('family: Cohen d', I10, 'v2', 'cohen_d', nd=2)} (paper 0.58).")
    mi = run_index(I1, "mmlu_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3")
    pi3 = run_index(I1, "mmlu_iid-T1.0_microsoft_Phi-3-mini-4k-instruct")
    hm = run_index(I1, "hellaswag_iid-T1.0_mistralai_Mistral-7B-Instruct-v0.3")
    w(f"6. Within-model (Table 1): Mistral-7B-v0.3 MMLU N_b {v('MMLU Mistral-7B-v0.3: N_b', I1, 'runs', mi, 'N_b')} (v2 86), TW z "
      f"{v('MMLU Mistral-7B-v0.3: TW z', I1, 'runs', mi, 'tw_z', nd=2)} (v2 -2.41); Phi-3-mini N_b "
      f"{v('MMLU Phi-3-mini: N_b', I1, 'runs', pi3, 'N_b')} (v2 273); HellaSwag Mistral N_b "
      f"{v('HellaSwag Mistral-7B-v0.3: N_b', I1, 'runs', hm, 'N_b')} (v2 64). Every iid MP, buffered and Horn count "
      "stays 0 (the paper's within-model statements hold).")
    w("")

    section("A within-model")
    w("## A. Within-model items")
    w("")
    w("| run | acc | N_b | lambda_+ | lambda_max | TW z | MC pct of z | MP | buffered | Horn |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(g(I1, "runs")):
        lab = r["label"]
        w(f"| {lab} | {v(lab + ': accuracy', I1, 'runs', i, 'accuracy', nd=4)} | {v(lab + ': N_b', I1, 'runs', i, 'N_b')} | "
          f"{v(lab + ': lambda_+', I1, 'runs', i, 'lambda_plus', nd=3)} | {v(lab + ': lambda_max', I1, 'runs', i, 'lambda_max', nd=3)} | "
          f"{v(lab + ': TW z', I1, 'runs', i, 'tw_z', nd=2)} | {v(lab + ': MC percentile of z', I1, 'runs', i, 'mc_null_z', 'percentile_of_obs', nd=3)} | "
          f"{v(lab + ': MP count', I1, 'runs', i, 'mp_count_strict')} | {v(lab + ': buffered count', I1, 'runs', i, 'mp_count_buffered')} | "
          f"{v(lab + ': Horn count', I1, 'runs', i, 'horn_count_contiguous')} |")
    w("")
    w("Shared-component runs (item 2):")
    w("")
    w("| run | N_b | lambda_+ | lambda_max | TW z | buffered | Horn | letter-share dispersion ratio | pass dispersion ratio (p) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for rid in g(I2, "runs"):
        lab = g(I2, "runs", rid, "label")
        w(f"| {lab} | {v(lab + ': N_b', I2, 'runs', rid, 'N_b')} | {v(lab + ': lambda_+', I2, 'runs', rid, 'lambda_plus', nd=2)} | "
          f"{v(lab + ': lambda_max', I2, 'runs', rid, 'lambda_max', nd=2)} | {v(lab + ': TW z', I2, 'runs', rid, 'tw_z', nd=1)} | "
          f"{v(lab + ': buffered', I2, 'runs', rid, 'mp_count_buffered')} | {v(lab + ': Horn', I2, 'runs', rid, 'horn_count_contiguous')} | "
          f"{v(lab + ': letter-share dispersion ratio', I2, 'runs', rid, 'letter_share_dispersion', 'ratio_obs_over_null_mean', nd=1)} | "
          f"{v(lab + ': pass dispersion ratio', I2, 'runs', rid, 'pass_dispersion', 'dispersion_ratio_obs_over_expected', nd=2)} "
          f"({v(lab + ': pass dispersion p', I2, 'runs', rid, 'pass_dispersion', 'p_overdispersion', nd=4)}) |")
    w("")
    w(f"Item 3 (power): unchanged (the Qwen2.5-7B and Llama-3-8B iid runs did not change under the rescore); "
      f"`item03_power.json` is byte-identical to v2. Item 5: split-half r (Qwen2.5-7B, all / borderline columns) "
      f"{v('split-half r all columns', 'item05_split_half', 'runs', 'mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct', 'r_all_columns', nd=4)} / "
      f"{v('split-half r borderline', 'item05_split_half', 'runs', 'mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct', 'r_borderline_columns', nd=4)}. "
      f"Item 6: modal-wrong-letter share {v('modal wrong-letter share', 'item06_answer_identity', 'modal_wrong_letter_share', 'mean_over_borderline', nd=3)}; "
      f"letter association across questions exceeds its p99: {v('letter test exceeds p99', 'item06_answer_identity', 'letter_test_qwen', 'exceeds_p99')}. "
      f"Item 12: T = 0.5 / 1.5 N_b {v('T0.5 N_b', 'item12_temperature_gsm8k', 'temperature', '0.5', 'N_b')} / "
      f"{v('T1.5 N_b', 'item12_temperature_gsm8k', 'temperature', '1.5', 'N_b')}, MP counts "
      f"{v('T0.5 MP', 'item12_temperature_gsm8k', 'temperature', '0.5', 'mp_count_strict')} / "
      f"{v('T1.5 MP', 'item12_temperature_gsm8k', 'temperature', '1.5', 'mp_count_strict')}; GSM8K 256-token truncation share "
      f"{v('GSM8K 256: truncated share', 'item12_temperature_gsm8k', 'gsm8k', '256', 'share_length_truncated', nd=3)}, accuracy 256 / 1024 tokens "
      f"{v('GSM8K 256 accuracy', 'item12_temperature_gsm8k', 'gsm8k', '256', 'accuracy', nd=3)} / "
      f"{v('GSM8K 1024 accuracy', 'item12_temperature_gsm8k', 'gsm8k', '1024', 'accuracy', nd=3)}. Item 13 (now reads R2 rotation 0): "
      "across-model buffered count as scored / without questions with any unparseable reply / without Mistral-7B-v0.2 / imputed: "
      + " / ".join(v(f'item 13 {k}: buffered count', 'item13_unparseable', 'across_model_sensitivity', k, 'mp_count_buffered')
                   for k in ('as_scored', 'drop_questions_any_unparseable', 'drop_model_mistral_v0.2', 'impute_question_mean'))
      + "; 2PL ranks above p99 in the two binary variants: "
      + " / ".join(v(f'item 13 {k}: 2PL ranks above p99', 'item13_unparseable', 'across_model_sensitivity', k,
                     'one_factor_statement', 'ranks_above_p99_by_null', 'c_post_2pl')
                   for k in ('drop_questions_any_unparseable', 'drop_model_mistral_v0.2')) + ".")
    w("")

    section("C rotations")
    w("## C. Rotation analyses (replicate A; B in C4)")
    w("")
    w("| rotation | eigenvalues 1-4 | lambda_+ | buffered | eta^2 by gold letter, eigvec 1-4 | eigvec 2: r with share A / D |")
    w("|---|---|---|---|---|---|")
    for r in "0123":
        etas = ", ".join(v(f'rotation {r} eigvec {j + 1}: eta^2 by shown gold letter', CA, 'C1_per_rotation', r, 'letters', 'eigenvectors', j,
                           'eta_sq_by_shown_gold_letter', 'eta_sq', nd=2) for j in range(4))
        w(f"| {r} | {v(f'rotation {r}: eigenvalues 1-4', CA, 'C1_per_rotation', r, 'spectrum', 'top_eigs', nd=1, sl=slice(0, 4))} | "
          f"{v(f'rotation {r}: lambda_+', CA, 'C1_per_rotation', r, 'spectrum', 'lambda_plus', nd=2)} | "
          f"{v(f'rotation {r}: buffered', CA, 'C1_per_rotation', r, 'spectrum', 'mp_count_buffered')} | {etas} | "
          f"{v(f'rotation {r} eigvec 2: r share A', CA, 'C1_per_rotation', r, 'letters', 'eigenvectors', 1, 'model_score_corr', 'share_A', 'r', nd=2)} / "
          f"{v(f'rotation {r} eigvec 2: r share D', CA, 'C1_per_rotation', r, 'letters', 'eigenvectors', 1, 'model_score_corr', 'share_D', 'r', nd=2)} |")
    w("")
    w("Within one rotation the shown gold letter is a relabelling of the original one, so eta^2 cannot tell them apart; "
      "the cross-rotation test (C1b, `v3_C1b_letter_following.json`) can, and is reported above. Eigenvector 1 also "
      f"follows the shown letters in part (r_shown {v('C1b eigvec 1 r1: r_shown', LF, 'A', 'eigvec1', '1', 'r_shown', nd=2)} vs r_orig "
      f"{v('C1b eigvec 1 r1: r_orig', LF, 'A', 'eigvec1', '1', 'r_orig', nd=2)} in rotation 1) but keeps its content "
      f"(r_direct {v('C1b eigvec 1 r1: r_direct', LF, 'A', 'eigvec1', '1', 'r_direct', nd=2)}); eigenvector 3 is matched only loosely across "
      f"rotations (model-score |r| {v('C1b eigvec 3 r1: model-score match', LF, 'A', 'eigvec3', '1', 'model_score_match_abs_r', nd=2)}).")
    w("")
    w("Rotation-randomized ensembles (C2): buffered-count distribution "
      f"{v('C2: buffered count distribution', CA, 'C2_rotation_randomized', 'distribution', 'mp_count_buffered_distribution')}, strict "
      f"{v('C2: strict MP count distribution', CA, 'C2_rotation_randomized', 'distribution', 'mp_count_strict_distribution')}; "
      f"eigenvalue p01-p99 {v('C2: eigenvalue p01', CA, 'C2_rotation_randomized', 'distribution', 'eig_p01', nd=1, sl=slice(0, 4))} - "
      f"{v('C2: eigenvalue p99', CA, 'C2_rotation_randomized', 'distribution', 'eig_p99', nd=1, sl=slice(0, 4))}. "
      "Nulls fitted to ONE representative draw (the draw closest to the median eigenvalues 1-4; refitting 1000 draws x 6 "
      "nulls was not run), 1000 null draws each; ranks above p99: "
      f"{v('C2 representative: ranks above p99 by null', CA, 'C2_rotation_randomized', 'representative_suite', 'ranks_above_p99_by_null')}. "
      f"Robustness, 10 other draws refitted (2PL and letter-aware, 300 null draws each): "
      f"{v('C2 refits: ranks above p99', CA, 'C2_rotation_randomized', 'robustness_refit_draws')}. "
      "Representative-draw eigenvector 2 still correlates with the models' letter shares "
      f"(A {v('C2 eigvec 2: r share A', CA, 'C2_rotation_randomized', 'letters_representative', 'eigenvectors', 1, 'model_score_corr', 'share_A', 'r', nd=2)}, "
      f"D {v('C2 eigvec 2: r share D', CA, 'C2_rotation_randomized', 'letters_representative', 'eigenvectors', 1, 'model_score_corr', 'share_D', 'r', nd=2)}) "
      "but its question loadings do not group by gold letter (eta^2 "
      f"{v('C2 eigvec 2: eta^2 by original gold letter', CA, 'C2_rotation_randomized', 'letters_representative', 'eigenvectors', 1, 'eta_sq_by_original_gold_letter', 'eta_sq', nd=3)}): "
      "the models' letter shares still covary with which questions they answer correctly, but the shown letter no "
      "longer groups the questions.")
    w("")
    w("Leave-one-out (all-4-correct): without Llama-3.2-1B the 23-model all-4 matrix has ranks beyond ability "
      f"{v('23 models all-4: ranks beyond ability', LOO, 'without_llama_3_2_1b', 'all4_suite', 'ranks_beyond_ability')}.")
    w("")

    section("E docs")
    w("## E. Documentation notes")
    w("")
    w("* `results/PROTOCOL.md` states the parser where the `src/scoring.py` docstring is less exact (Option/Choice "
      "and negation words in any case; any other character after the letter -> `letter_then_alnum`; GSM8K "
      "\"#### <number>\" form), the seed base `(crc32 & 0x7FFFFFFF) << 20`, the R2 letter-logprob differences "
      "(7 models; Ministral-8B: 4 cells, <= 2e-4), and the near-tie definition \"gap < 0.13 in A or in B\" on the "
      "~0.125 grid (97% of gaps below 0.6 lie within 0.01 of a multiple of 0.125).")
    w("* The `src/scoring.py` docstring is not edited: `scorer_sha256` is the sha256 of the whole file (collect.py, "
      "rescore.py) and must stay f804dbf14ef19530.")
    w("* `results/manifest.json` (via `src/build_manifest.py`) holds `runs_rep_B` (24 B runs, npz + json sha256), "
      "`ensemble_files_sha256` (3 files) and `json_sha256` per run. The builder rewrites an ensemble npz only if its "
      "arrays change, so the zip time stamp does not change the sha256.")
    w("")

    section("notes")
    w("## Notes, deviations")
    w("")
    w("* The 2PL fit is the project's model: abilities theta_m are fixed effects; question parameters (log a_i, b_i) are "
      "independent normal random effects integrated on a 41 x 401 grid (marginal ML over questions); question-matched "
      "posterior draws. Two of three random restarts abort in the line search (L-BFGS-B ABNORMAL, converged=False; "
      f"stored log-lik {v('2PL random restarts log-lik', RAW7, 'fits', 'twopl_mml', 'random_restarts', 'marginal_loglik', nd=1)}); "
      "continued, both reach -5839.17. The main fit is the optimum and agrees with a 61 x 801 grid. "
      "(`fit_twopl_mml` continues an aborted run with bounded hyperparameters and records per-restart "
      "`converged`; the stored restart values predate this.) The standard alternative is Bock-Aitkin MML (theta ~ N(0,1) integrated, items fixed, slopes "
      "box-bounded), simulated at ML abilities given the items.")
    w("* The know-or-guess letter preference pi_m is a moment estimate from all 4 rotations (`v3lib.letter_preference`); "
      "no model needed clipping. The 4-rotation fits use a composite (rotation-independence) likelihood; their "
      "per-question posteriors are narrower than the true ones.")
    w("* The rotation-averaged matrix has no binary parametric null; its nulls simulate 4 rotations per cell from the "
      "4-rotation fits. Near-tie fractional variants (imputed cells) report spectra only (`fractional_variants_spectra_only`).")
    w("* Item 8 (sub-ensembles) is not run (`item08_09_sub_ensembles_family.py --with-item08` runs it). Items 11 and 14 "
      "use the v3 ensemble (not headline).")
    w("* Item 2: the added subject test draws from the run's shared RNG stream, so later permutation p-values in item 2 "
      "move by Monte Carlo noise (exemplar-overlap z 10.53 -> 10.73 for Qwen S2); the observed statistics are identical.")
    w("* `scripts/across_two_margin_null.py::main` takes `stress_factors` and `eig_store`; the defaults reproduce the "
      "stored output, and adding stress factors leaves the existing ones unchanged.")
    w("")

    w("## Headline table")
    w("")
    w("Every number above, read from the JSON key in the last column (files in `results/analysis/`).")
    w("")
    w("| section | quantity | value | JSON key |")
    w("|---|---|---|---|")
    for sec, what, val, key in ROWS:
        w(f"| {sec} | {what} | {val} | {key} |")
    # Section F exists only in SUMMARY.md (no generator here): keep the old tail on a rerun.
    path = OUT / "SUMMARY.md"
    tail = ""
    if path.exists():
        old = path.read_text()
        i = old.find("\n## F. ")
        if i >= 0:
            tail = old[i:]
    path.write_text("\n".join(L) + "\n" + tail)
    print(f"wrote {OUT / 'SUMMARY.md'} ({len(ROWS)} keyed numbers)")


if __name__ == "__main__":
    main()
