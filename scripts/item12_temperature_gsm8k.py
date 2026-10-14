"""Item 12: temperature check (Qwen2.5-7B MMLU at T = 0.5 / 1.0 / 1.5, and T = 1.0 cut to K = 30)
and GSM8K truncation (256- vs 1024-token runs).

Output: results/analysis/item12_temperature_gsm8k.json
"""

from __future__ import annotations

import json
import re

import v2lib
import numpy as np
from v2lib import EPS, OUT, load_npz, spectrum_counts, write_json

from null_models import borderline_mask

T_RUNS = {"0.5": "mmlu_iid-T0.5_Qwen_Qwen2.5-7B-Instruct", "1.0": "mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct",
          "1.5": "mmlu_iid-T1.5_Qwen_Qwen2.5-7B-Instruct"}
G256, G1024 = "gsm8k_iid-T1.0_Qwen_Qwen2.5-7B-Instruct", "gsm8k_iidlong-T1.0_Qwen_Qwen2.5-7B-Instruct"
KEYS = ("K", "accuracy", "N_b", "lambda_plus", "lambda_max", "margin", "tw_z", "mp_count_strict",
        "mp_count_buffered", "horn_count_contiguous", "mc_null_z")


def gsm8k_stats(run_id: str) -> dict:
    z = load_npz(run_id)
    C = z["correct"].astype(np.int8)
    texts = z["texts"]
    trunc = z["finish_reason"] == "length"
    has_marker = np.vectorize(lambda t: "####" in t)(texts)
    has_number = np.vectorize(lambda t: re.search(r"\d", t) is not None)(texts)
    bm = borderline_mask(C, EPS)
    return {"K": int(C.shape[0]), "N": int(C.shape[1]), "accuracy": float(C.mean()), "N_b": int(bm.sum()),
            "share_length_truncated": float(trunc.mean()),
            "share_without_final_marker": float((~has_marker).mean()),
            "share_without_any_number": float((~has_number).mean()),
            "share_parsed_nan": float(np.isnan(z["parsed"]).mean()),
            "accuracy_truncated": float(C[trunc].mean()) if trunc.any() else None,
            "accuracy_complete": float(C[~trunc].mean()) if (~trunc).any() else None,
            "share_truncated_on_borderline_columns": float(trunc[:, bm].mean()),
            "share_questions_with_any_truncation": float(trunc.any(axis=0).mean())}


def main() -> None:
    item1 = {r["run_id"]: r for r in json.loads((OUT / "item01_within_table.json").read_text())["runs"]}
    temp = {T: {k: item1[r][k] for k in KEYS} for T, r in T_RUNS.items()}
    C10 = np.load(v2lib.matrix_path(T_RUNS["1.0"]))[:30]
    bm = borderline_mask(C10, EPS)
    temp["1.0_first30"] = {"K": 30, "accuracy": float(C10.mean()), "N_b": int(bm.sum()),
                           **spectrum_counts(C10[:, bm].astype(float))}
    g = {"256": gsm8k_stats(G256), "1024": gsm8k_stats(G1024)}
    f256 = np.load(v2lib.matrix_path(G256)).mean(0)
    f1024 = np.load(v2lib.matrix_path(G1024)).mean(0)
    g["pass_rate_corr_256_vs_1024"] = float(np.corrcoef(f256, f1024)[0, 1])
    g["item1_rows"] = {"256": {k: item1[G256][k] for k in KEYS}, "1024": {k: item1[G1024][k] for k in KEYS}}
    write_json("item12_temperature_gsm8k", {"temperature": temp, "gsm8k": g})
    print(json.dumps(v2lib.to_jsonable({"temperature": {T: {k: v for k, v in d.items() if k != "mc_null_z"}
                                                        for T, d in temp.items()}, "gsm8k": g}), indent=1))


if __name__ == "__main__":
    main()
