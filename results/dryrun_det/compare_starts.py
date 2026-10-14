"""Determinism test: the same 20 questions x 4 rotations, 3 separate engine starts per model.
Requires identical replies (texts and generated ids) and identical first-token logprobs."""
import json, sys
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parent
out = {}
ok = True
for f in sorted((D / "start1").glob("*.npz")):
    Z = [np.load(D / f"start{s}" / f.name) for s in (1, 2, 3)]
    r = {}
    r["texts_identical"] = all((Z[0]["texts"] == z["texts"]).all() for z in Z[1:])
    r["gen_ids_identical"] = all((Z[0]["gen_token_ids"] == z["gen_token_ids"]).all() for z in Z[1:])
    r["parsed_identical"] = all((Z[0]["parsed"] == z["parsed"]).all() for z in Z[1:])
    d = 0.0
    for z in Z[1:]:
        for key in ("top_lp_first", "letter_logprob_first"):
            a, b = Z[0][key], z[key]
            assert (np.isnan(a) == np.isnan(b)).all() or key == "letter_logprob_first"
            d = max(d, float(np.nanmax(np.abs(a - b))))
        r["top_ids_identical"] = r.get("top_ids_identical", True) and bool((Z[0]["top_ids_first"] == z["top_ids_first"]).all())
    r["max_abs_diff_first_token_logprob"] = d
    r["cells"] = int(Z[0]["texts"].size)
    meta = json.loads((D / "start1" / f.name).with_suffix(".json").read_text())
    r["deterministic_flags"] = meta["deterministic"]
    out[f.stem] = r
    ok &= r["texts_identical"] and r["gen_ids_identical"] and r["top_ids_identical"] and d == 0.0
    print(f.stem, {k: v for k, v in r.items() if k != "deterministic_flags"})
(D / "determinism_test.json").write_text(json.dumps({"deterministic": ok, "models": out}, indent=1))
print("DETERMINISTIC" if ok else "NOT DETERMINISTIC")
