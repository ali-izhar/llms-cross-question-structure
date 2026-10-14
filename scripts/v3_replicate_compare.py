"""C4: replicate A vs B (separate engine starts) -> results/analysis/v3_C4_replicate_agreement.json.

A - B difference of every eigenvalue in v3_C_rotations_{A,B}.json, and per null whether the ranks above p99 agree.
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401
import numpy as np
from v2lib import OUT, write_json


def main() -> None:
    A = json.loads((OUT / "v3_C_rotations_A.json").read_text())
    B = json.loads((OUT / "v3_C_rotations_B.json").read_text())
    eig, verdict = {}, {}

    def cmp_eigs(key, a, b):
        a, b = np.array(a[:4]), np.array(b[:4])
        eig[key] = {"A": a.tolist(), "B": b.tolist(), "max_abs_diff": float(np.abs(a - b).max())}

    for r in "0123":
        cmp_eigs(f"rotation{r}", A["C1_per_rotation"][r]["spectrum"]["top_eigs"], B["C1_per_rotation"][r]["spectrum"]["top_eigs"])
    cmp_eigs("randomized_median", A["C2_rotation_randomized"]["distribution"]["eig_median"],
             B["C2_rotation_randomized"]["distribution"]["eig_median"])
    for k in ("all4_correct", "mean_over_rotations"):
        cmp_eigs(k, A["C3_rotation_invariant"][k]["spectrum"]["top_eigs"], B["C3_rotation_invariant"][k]["spectrum"]["top_eigs"])

    def cmp_suite(tag, sa, sb):
        for k in sa["nulls"]:
            ra, rb = sa["nulls"][k]["observed_ranks_above_p99"], sb["nulls"][k]["observed_ranks_above_p99"]
            pa, pb = np.array(sa["nulls"][k]["rank_eigs_p99"]), np.array(sb["nulls"][k]["rank_eigs_p99"])
            verdict[f"{tag}:{k}"] = {"A": ra, "B": rb, "agree": ra == rb, "max_abs_diff_p99": float(np.abs(pa - pb).max())}

    cmp_suite("rot0", A["rot0_suite"], B["rot0_suite"])
    cmp_suite("randomized_representative", A["C2_rotation_randomized"]["representative_suite"],
              B["C2_rotation_randomized"]["representative_suite"])
    for k in ("all4_correct", "mean_over_rotations"):
        cmp_suite(k, A["C3_rotation_invariant"][k]["suite"], B["C3_rotation_invariant"][k]["suite"])
    let = {}
    for r in "0123":
        ea = [e["eta_sq_by_shown_gold_letter"]["eta_sq"] for e in A["C1_per_rotation"][r]["letters"]["eigenvectors"]]
        eb = [e["eta_sq_by_shown_gold_letter"]["eta_sq"] for e in B["C1_per_rotation"][r]["letters"]["eigenvectors"]]
        let[f"rotation{r}_eta_sq_gold_letter"] = {"A": ea, "B": eb, "max_abs_diff": float(np.abs(np.subtract(ea, eb)).max())}
    dis = [k for k, v in verdict.items() if not v["agree"]]
    write_json("v3_C4_replicate_agreement", {
        "eigenvalues": eig, "null_verdicts": verdict, "letters": let,
        "max_abs_eig_diff_overall": max(v["max_abs_diff"] for v in eig.values()),
        "n_null_verdicts": len(verdict), "n_null_verdicts_disagree": len(dis), "disagreeing_null_verdicts": dis,
        "note": "rotation 0 is identical in A and B (the 6 A-vs-B letter flips are in rotations 2 and 3 of Qwen2-7B); "
                "B's letter preference pi is estimated from B's 4 rotations"})
    print(len(dis), dis, {k: round(v["max_abs_diff"], 3) for k, v in eig.items()})


if __name__ == "__main__":
    main()
