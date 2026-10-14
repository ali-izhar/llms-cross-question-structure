"""Item 3: power of the within-model test at the Qwen2.5-7B and Llama-3-8B MMLU pass rates (K = 100).

Runs within_power.main per model: MP strict / buffered, Horn, calibrated MC test, within-subject test.
Output: results/analysis/item03_power.json (+ raw_item03_power_<model>.json)
"""

from __future__ import annotations

from v2lib import LABEL, LLAMA_IID, OUT, QWEN_IID, V2, matrix_path, write_json

import within_power

MAX_WORKERS = 10   # single-threaded BLAS each (v2lib); total memory < 2 GB


def main() -> None:
    out = {"meta": {"source": "scripts/within_power.py::main, unchanged settings",
                    "n_sim": within_power.N_SIM, "n_sim_null": within_power.N_SIM_NULL,
                    "n_horn_surrogates": within_power.N_SURR, "seed": within_power.SEED},
           "models": {}}
    for run_id, tag in ((QWEN_IID, "qwen"), (LLAMA_IID, "llama")):
        raw = within_power.main(matrix_path=matrix_path(run_id), questions_path=V2 / "questions_mmlu.json",
                                out_path=OUT / f"raw_item03_power_{tag}.json", max_workers=MAX_WORKERS)
        out["models"][LABEL[run_id]] = raw
    write_json("item03_power", out)


if __name__ == "__main__":
    main()
