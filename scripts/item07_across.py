"""Item 7: across-model spectrum (24 models, greedy, MMLU 500) against its nulls.

Bernoulli nulls (d_across_null), one-factor and fixed-margin nulls with slope stress tests
(across_two_margin_null), and lambda_2 / lambda_1 against every null.
Output: results/analysis/item07_across_model.json (+ raw_item07_*)
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import ENSEMBLE_NAMES, ENSEMBLE_NPY, OUT, write_json

import across_two_margin_null
import d_across_null
from null_models import MP_BUFFER, perm_pvalue_upper, split_spectrum

STRESS = (1.25, 1.5, 2.0, 3.0, 5.0, 10.0)


def ratio_stats(e: np.ndarray, obs: float) -> dict:
    r = e[:, 1] / e[:, 0]
    return {"null_mean": float(r.mean()), "null_p99": float(np.percentile(r, 99)), "null_max": float(r.max()),
            "mc_p_upper": perm_pvalue_upper(obs, r), "n_draws": int(r.size)}


def main() -> None:
    C = np.load(ENSEMBLE_NPY)
    e, lp, n_split = split_spectrum(C)
    obs = {"M": int(C.shape[0]), "N": int(C.shape[1]), "n_split": n_split, "lambda_plus": lp,
           "lambda_plus_buffered": MP_BUFFER * lp, "top_eigs_1to4": e[:4].tolist(),
           "mp_count_strict": int((e > lp).sum()), "mp_count_buffered": int((e > MP_BUFFER * lp).sum()),
           "accuracy_per_model": dict(zip(json.loads(ENSEMBLE_NAMES.read_text()), C.mean(1).round(4).tolist()))}
    d = d_across_null.main(matrix_path=ENSEMBLE_NPY, out_json=OUT / "raw_item07_d_across_null.json", fig_path=None)
    store = {}
    t = across_two_margin_null.main(matrix_path=ENSEMBLE_NPY, names_path=ENSEMBLE_NAMES,
                                    out_path=OUT / "raw_item07_across_two_margin_null.json",
                                    stress_factors=STRESS, eig_store=store)
    np.savez_compressed(OUT / "raw_item07_null_eigs.npz", **store)
    r_obs = float(e[1] / e[0])
    ratio = {k: ratio_stats(v, r_obs) for k, v in store.items() if k != "observed"}
    stress_keys = [k for k in store if "stress" in k]
    pooled = np.concatenate([store[k] for k in stress_keys])
    lam_ratio = {"observed": r_obs, "by_null": ratio,
                 "pooled_stress_x1.25_to_x10": ratio_stats(pooled, r_obs),
                 "max_over_all_nulls_and_multipliers": float(max(v["null_max"] for v in ratio.values())),
                 "max_p99_over_all_nulls_and_multipliers": float(max(v["null_p99"] for v in ratio.values()))}
    nulls = t["nulls"]
    summary = {k: {"observed_ranks_above_p99": v["observed_ranks_above_p99"],
                   "rank_eigs_p99": v["rank_eigs_p99"],
                   "count_above_1p05_edge_mean": v["count_above_1p05_edge"]["mean"],
                   "count_above_1p05_edge_max": v["count_above_1p05_edge"]["max"]} for k, v in nulls.items()}
    write_json("item07_across_model", {
        "observed": obs,
        "bernoulli_nulls_d_across": {k: {"mp_signal_max": v["mp_signal"]["max"], "mp_signal_mean": v["mp_signal"]["mean"],
                                         "frac_null_ge_obs_mp": v["frac_null_ge_obs_mp"],
                                         "eff_rank_mean": v["effective_rank"]["mean"],
                                         "eff_rank_min": v["effective_rank"]["min"]}
                                     for k, v in d["nulls"].items()},
        "observed_effective_rank": d["observed"]["d_effective_rank"],
        "one_factor_and_fixed_margin_nulls": summary,
        "twopl_fit": {k: t["fits"]["twopl_mml"][k] for k in ("converged", "hyper_mu_log_a", "hyper_sd_log_a",
                                                              "corr_theta_vs_accuracy", "max_posterior_mass_on_grid_boundary",
                                                              "grid_check", "random_restarts")},
        "statement": t["statement"],
        "stress_factors": list(STRESS),
        "lambda2_over_lambda1": lam_ratio,
    })
    print(t["statement"]["supported_sentence"])
    print(t["statement"]["robustness"])


if __name__ == "__main__":
    main()
