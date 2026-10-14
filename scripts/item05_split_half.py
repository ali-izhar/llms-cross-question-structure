"""Item 5: split-half reliability of pass rates (first K/2 passes vs the rest), all and borderline columns.

Output: results/analysis/item05_split_half.json
"""

from __future__ import annotations

import json

import v2lib  # noqa: F401  (threads, sys.path)
import numpy as np
from v2lib import EPS, LABEL, OUT, QWEN_IID, RUNS, matrix_path, split_half_r, write_json

from null_models import borderline_mask


def main() -> None:
    rows = {}
    for r, _, g in RUNS:
        C = np.load(matrix_path(r))
        bm = borderline_mask(C, EPS)
        rows[r] = {"label": LABEL[r], "K": int(C.shape[0]), "N": int(C.shape[1]), "N_b": int(bm.sum()),
                   "r_all_columns": split_half_r(C), "r_borderline_columns": split_half_r(C, bm)}
    q = rows[QWEN_IID]
    wp = OUT / "raw_item03_power_qwen.json"
    if wp.exists():
        sh = json.loads(wp.read_text())["split_half"]
        assert abs(sh["r_all_columns"] - q["r_all_columns"]) < 1e-12
        assert abs(sh["r_borderline_columns"] - q["r_borderline_columns"]) < 1e-12
        q["cross_check_within_power"] = "equal to 1e-12"
    write_json("item05_split_half", {"headline_run": QWEN_IID, "halves": "first K//2 passes vs the rest",
                                     "eps": EPS, "runs": rows})
    print(json.dumps(q, indent=1))


if __name__ == "__main__":
    main()
