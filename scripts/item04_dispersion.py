"""Item 4: per-pass accuracy overdispersion and pass-index trend for every iid run (pass_dispersion.main).

Output: results/analysis/item04_pass_dispersion.json
"""

from __future__ import annotations

from v2lib import LABEL, OUT, RUNS, matrix_path, write_json

import pass_dispersion

ROW_ORDER = ("row k = pass k; passes were generated in index order, floor(5000/N) passes per "
             "vLLM generate call (src/collect.py)")


def main() -> None:
    iid = [r for r, _, g in RUNS if g == "iid"]
    raw = pass_dispersion.main(models=[(LABEL[r], str(matrix_path(r))) for r in iid],
                               out_path=OUT / "raw_item04_pass_dispersion.json", row_order=ROW_ORDER)
    for rec, r in zip(raw["models"], iid):
        rec["run_id"] = r
    write_json("item04_pass_dispersion", raw)


if __name__ == "__main__":
    main()
