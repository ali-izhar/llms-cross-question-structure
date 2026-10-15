"""Item 11: effective rank exp(H) of the split columns and its ensemble-size sweep (M = 4..24).

Per M: random subsets vs a difficulty-matched Bernoulli null; plus within- vs across-model shares.
Output: results/analysis/item11_effective_rank.json
"""

from __future__ import annotations

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import ENSEMBLE_NPY, LABEL, RUNS, matrix_path, write_json

import spectral_metrics
import d_across_null
import ensemble_size_sweep as ess

SEED = 20261011
N_SUB = 500
K_VALUES = list(range(4, 25, 2))


def null_sweep(C: np.ndarray, rng: np.random.Generator) -> dict:
    M = C.shape[0]
    out = {}
    for k in K_VALUES:
        obs, null = [], []
        for _ in range(1 if k == M else N_SUB):
            idx = rng.choice(M, size=k, replace=False)
            S = C[idx]
            d_obs = ess.d_across_of_submatrix(S)
            d_null = ess.d_across_of_submatrix(d_across_null.draw_null_A(S.mean(axis=0), k, rng))
            if d_obs is not None and d_null is not None:
                obs.append(d_obs)
                null.append(d_null)
        obs, null = np.array(obs), np.array(null)
        out[k] = {"obs_median": float(np.median(obs)), "null_median": float(np.median(null)),
                  "null_min": float(null.min()), "null_max": float(null.max()),
                  "obs_share_of_max": float(np.median(obs) / (k - 1)),
                  "null_share_of_max": float(np.median(null) / (k - 1)),
                  "frac_subsets_obs_below_paired_null": float(np.mean(obs < null)), "n": int(obs.size)}
    return out


def main() -> None:
    C = np.load(ENSEMBLE_NPY)
    sw = ess.sweep(C, K_VALUES)
    ns = null_sweep(C, np.random.default_rng(SEED))
    meds = {k: sw[k]["median"] for k in sw}
    ks_fit = [k for k in meds if k < C.shape[0]]
    slope = float(np.polyfit(ks_fit, [meds[k] for k in ks_fit], 1)[0])
    mmlu_iid = [(LABEL[r], str(matrix_path(r))) for r, _, g in RUNS if g == "iid" and r.startswith("mmlu_iid-T1.0")]
    share = spectral_metrics.effective_rank_share(models=mmlu_iid, ensemble_path=ENSEMBLE_NPY)
    write_json("item11_effective_rank", {
        "column_filter": "split columns (0 < column sum < rows) of the matrix measured",
        "sweep_observed": sw, "sweep_slope_of_median_k_lt_M": slope, "null_per_M": ns,
        "effective_rank_share": share, "seed_null": SEED, "seed_sweep": "ensemble_size_sweep.RNG (42)"})
    for k in K_VALUES:
        print(k, round(sw[k]["median"], 2), round(ns[k]["obs_median"], 2), "null", round(ns[k]["null_median"], 2),
              round(ns[k]["null_share_of_max"], 3), ns[k]["frac_subsets_obs_below_paired_null"])
    print("slope", slope, share["range_split"], share["across_24_models_split"])


if __name__ == "__main__":
    main()
