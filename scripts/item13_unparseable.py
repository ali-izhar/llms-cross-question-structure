"""Item 13: unparseable replies per run, and whether scoring them as missing (drop the question, or
impute its mean) instead of wrong changes any headline count, within and across models.

Across-model rows are R2 replicate A, rotation 0 (results/r2/mmlu_greedyrot_<model>.npz).
Output: results/analysis/item13_unparseable.json (+ raw_item13_*)
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import (ENSEMBLE_NAMES, ENSEMBLE_NPY, LABEL, MAT, OUT, RUNS, V2, borderline_mask_float,
                   impute_column_mean, load_meta, load_npz, spectrum_counts, write_json)

import across_two_margin_null
import d_across_null
import horn_vs_mp
from null_models import split_mask

SEED = 20261012
MISTRAL_V02 = "mistralai/Mistral-7B-Instruct-v0.2"
R2_DIR = V2 / "r2"          # replicate A
ROTATION = 0                # original option order


def r2_rotation0(model: str):
    """(npz, meta) of the R2 replicate-A run of `model`; callers use row ROTATION."""
    rid = f"mmlu_greedyrot_{model.replace('/', '_')}"
    return np.load(R2_DIR / f"{rid}.npz", allow_pickle=False), json.loads((R2_DIR / f"{rid}.json").read_text())


def unparse_mask(z) -> np.ndarray:
    p = z["parsed"]
    return np.isnan(p) if p.dtype.kind == "f" else (p == "")


def within_variant(X: np.ndarray, seed: int) -> dict:
    bm = borderline_mask_float(X)
    rec = {"N_b": int(bm.sum()), **spectrum_counts(X[:, bm])}
    if bm.sum() >= 2:
        _, d_contig, _, _ = horn_vs_mp.horn_pa(X[:, bm], 200, seed)
        rec["horn_count_contiguous"] = int(d_contig)
    return rec


def across_variant(C: np.ndarray, rng: np.random.Generator) -> dict:
    sm = split_mask(C) if set(np.unique(C)) <= {0, 1} else (C.std(axis=0) > 0)
    rec = {"M": int(C.shape[0]), "N": int(C.shape[1]), **spectrum_counts(C[:, sm])}
    ns = d_across_null.run_null("A", lambda r: d_across_null.draw_null_A(C.mean(axis=0), C.shape[0], r), rng, 500)
    rec["bernoulli_null_A_mp_count_max"] = int(max(ns.d_mp))
    rec["bernoulli_null_A_mp_count_mean"] = float(np.mean(ns.d_mp))
    return rec


def main() -> None:
    rng = np.random.default_rng(SEED)
    rates, within = {}, {}
    for r, _, g in RUNS:
        z = load_npz(r)
        U = unparse_mask(z)
        meta = load_meta(r)
        rates[r] = {"label": LABEL[r], "unparseable_rate": float(U.mean()), "meta_unparseable_rate": meta["unparseable_rate"],
                    "n_unparseable": int(U.sum()), "n_questions_with_any": int(U.any(axis=0).sum()),
                    "n_A_word_replies": meta.get("n_A_word_replies")}
        if U.any() and z["parsed"].dtype.kind != "f":
            C = z["correct"].astype(np.float64)
            keep = ~U.any(axis=0)
            within[r] = {"label": LABEL[r], "as_scored": within_variant(C, SEED),
                         "drop_questions_with_any_unparseable": {"n_dropped": int((~keep).sum()),
                                                                 **within_variant(C[:, keep], SEED)},
                         "impute_question_mean": within_variant(impute_column_mean(C, U), SEED)}
    names = json.loads(ENSEMBLE_NAMES.read_text())
    C = np.load(ENSEMBLE_NPY).astype(np.int8)
    r2 = {n: r2_rotation0(n) for n in names}
    U = np.stack([unparse_mask(r2[n][0])[ROTATION] for n in names])                       # (24, 500)
    greedy_rates = {n: {"unparseable_rate": float(U[m].mean()), "n_unparseable": int(U[m].sum()),
                        "meta_unparseable_rate": r2[n][1]["unparseable_by_rotation"][ROTATION],
                        "source": f"results/r2/mmlu_greedyrot_{n.replace('/', '_')}.npz rotation {ROTATION}"}
                    for m, n in enumerate(names)}
    for m, n in enumerate(names):   # the ensemble matrix must equal the R2 rotation-0 rows
        assert np.array_equal(C[m], r2[n][0]["correct"][ROTATION]), n
        assert abs(greedy_rates[n]["unparseable_rate"] - greedy_rates[n]["meta_unparseable_rate"]) < 1e-9, n
    keep_q = ~U.any(axis=0)
    keep_m = np.array([n != MISTRAL_V02 for n in names])
    variants = {"as_scored": C, "drop_questions_any_unparseable": C[:, keep_q],
                "drop_model_mistral_v0.2": C[keep_m]}
    across = {k: across_variant(v, rng) for k, v in variants.items()}
    across["impute_question_mean"] = across_variant(impute_column_mean(C, U), rng)
    across["drop_questions_any_unparseable"]["n_dropped"] = int((~keep_q).sum())
    # Full one-factor analysis on the two binary variants.
    for k in ("drop_questions_any_unparseable", "drop_model_mistral_v0.2"):
        mp = MAT / f"ensemble_greedy_{k}.npy"
        npth = MAT / f"ensemble_models_{k}.json"
        np.save(mp, variants[k])
        npth.write_text(json.dumps([n for n, km in zip(names, keep_m) if km] if "model" in k else names))
        t = across_two_margin_null.main(matrix_path=mp, names_path=npth, out_path=OUT / f"raw_item13_{k}.json")
        across[k]["one_factor_statement"] = t["statement"]
    write_json("item13_unparseable", {"rates_sampled_runs": rates, "rates_greedy_24": greedy_rates,
                                      "within_model_sensitivity": within, "across_model_sensitivity": across})
    for r, d in within.items():
        print(LABEL[r], {k: (v.get("N_b"), v.get("mp_count_strict"), v.get("mp_count_buffered"),
                             v.get("horn_count_contiguous")) for k, v in d.items() if isinstance(v, dict)})
    for k, v in across.items():
        print(k, v.get("n_cols"), [round(x, 1) for x in v["top_eigs"]], round(v["lambda_plus"], 2),
              v["mp_count_buffered"], v["bernoulli_null_A_mp_count_max"],
              v.get("one_factor_statement", {}).get("supported_sentence"))


if __name__ == "__main__":
    main()
