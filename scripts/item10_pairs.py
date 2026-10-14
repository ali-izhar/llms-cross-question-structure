"""Item 10: Pearson r of correctness for all 276 pairs of the 24 greedy models, same vs cross family.

Family test: model-label permutation (QAP), Holm with the item-9 eigenvector tests (run after item 9).
The pair-level KS fields treat dependent pairs as independent; kept for reference only.
Output: results/analysis/item10_pair_correlations.json
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import ENSEMBLE_NAMES, ENSEMBLE_NPY, ROOT, pair_correlations, write_json

from null_models import family_of
from v2lib import OUT
from v3lib import holm, qap_family_test

SEED_QAP = 20261013
N_PERM_QAP = 20000


def main() -> None:
    names = json.loads(ENSEMBLE_NAMES.read_text())
    fam = [family_of(n) for n in names]
    new = pair_correlations(np.load(ENSEMBLE_NPY), fam)
    old_names = json.loads((ROOT / "results" / "provenance" / "ensemble_expanded_models.json").read_text())
    old = pair_correlations(np.load(ROOT / "results" / "provenance" / "ensemble_expanded_correctness.npy"),
                            [family_of(n) for n in old_names])
    qap = qap_family_test(np.load(ENSEMBLE_NPY), fam, N_PERM_QAP, np.random.default_rng(SEED_QAP))
    item9 = json.loads((OUT / "item09_family_alignment.json").read_text())
    pv = {"qap_same_vs_cross_mean_r": qap["perm_p_one_sided"]}
    pv.update({f"eigvec{e['rank']}_family_eta_sq": e["perm_pvalue"] for e in item9["eigenvectors"]})
    pv["joint_eigvec_2to4_family"] = item9["joint_eigvec_2to4"]["perm_pvalue"]
    new["ks_note"] = "pair-level KS treats dependent pairs as independent; superseded by the QAP test"
    write_json("item10_pair_correlations", {"v2": new, "qap_model_label_permutation": qap,
                                            "family_tests_raw_p": pv, "family_tests_holm": holm(pv),
                                            "old_data_reproduction": old,
                                            "families": dict(zip(names, fam))})
    print("QAP", qap, "Holm", holm(pv))
    print("v2 ", {k: round(v, 4) for k, v in new.items() if not isinstance(v, str)})
    print("old", {k: round(v, 4) for k, v in old.items()})


if __name__ == "__main__":
    main()
