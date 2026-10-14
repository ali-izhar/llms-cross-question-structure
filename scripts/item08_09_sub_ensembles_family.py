"""Items 8 and 9 on the 24-model greedy matrix, via sub_ensemble_null.

Item 9: eta^2 of eigenvectors 1-4 by model family (label permutation). Item 8 (single-family
sub-ensembles vs finite-M nulls) runs only with `--with-item08`.
Outputs: results/analysis/item09_family_alignment.json (and item08_sub_ensembles.json)
"""

from __future__ import annotations

import json
import sys

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import ENSEMBLE_NAMES, ENSEMBLE_NPY, write_json

import sub_ensemble_null as sen


def main() -> None:
    C = np.load(ENSEMBLE_NPY).astype(np.int8)
    names = json.loads(ENSEMBLE_NAMES.read_text())
    r = np.random.default_rng(sen.SEED).spawn(7)        # streams 0-2 qwen, 3-5 llama, 6 family
    meta = {"seed": sen.SEED, "n_draws_A_B": sen.N_DRAWS, "n_draws_fixed_margin": sen.N_DRAWS_FIXED_MARGIN,
            "curveball_burnin_trades": sen.CURVEBALL_BURNIN, "curveball_thin_trades": sen.CURVEBALL_THIN,
            "columns": "split columns of the sub-ensemble, re-filtered per null draw",
            "edge": "lambda_+ = (1 + sqrt(N_split / M))^2", "matrix": str(ENSEMBLE_NPY)}
    with_item08 = "--with-item08" in sys.argv
    if with_item08:
        qwen = sen.sub_ensemble(C, names, "qwen", r[0:3])
        llama = sen.sub_ensemble(C, names, "llama", r[3:6])
        write_json("item08_sub_ensembles", {"meta": meta, "qwen": qwen, "llama": llama})
    fam = sen.eigvec_family(C, names, r[6])
    write_json("item09_family_alignment", {"meta": {"seed": sen.SEED, "n_perm": sen.N_PERM}, **fam})
    for tag, d in ((("qwen", qwen), ("llama", llama)) if with_item08 else ()):
        print(tag, d["models"], d["n_split"], d["rank"], round(d["lambda_plus"], 1), round(d["lambda1"], 1),
              "bound/edge", round(d["trace_bound_over_edge"], 3), "obs/edge", round(d["lambda1_over_edge"], 3),
              {k: (round(d[k]["lam1_over_edge_p99"], 3), d[k]["mc_pvalue_observed"])
               for k in ("null_A_per_question_rate", "null_B_ability_only", "null_F_fixed_margin")})
    for e in fam["eigenvectors"]:
        print(e["rank"], round(e["eigenvalue"], 2), "eta2", round(e["eta_sq_family"], 3), "p", e["perm_pvalue"],
              "multi", round(e["eta_sq_multi_member_families_only"], 3), e["perm_pvalue_multi_member_families_only"],
              "r(acc)", round(e["corr_with_accuracy"], 3))
    print(fam["joint_eigvec_2to4"])


if __name__ == "__main__":
    main()
