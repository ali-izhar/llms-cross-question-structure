"""Checks of the v3.1 GPU dry run (results/dryrun_v31/): canonical + reference asserts recorded,
rotation 0 replies == v2 greedy replies for the R2 models, rotation mapping, parser version.
    .venv/bin/python results/dryrun_v3/check_dry_v31.py"""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "src"))
from plan import rotate
from scoring import PARSER_VERSION, parse_choice
D = R / "results/dryrun_v31"
qs = json.loads((R / "results/questions_mmlu.json").read_text())
ok = True
for js in sorted(D.glob("mmlu_*.json")):
    if js.name.endswith(".tokens.json"):
        continue
    m = json.loads(js.read_text()); z = np.load(js.with_suffix(".npz")); pc = m["tokenization"]["prompt_check"]
    K, N = z["correct"].shape
    c = {"canonical": pc["n_canonical"] == pc["n_prompts_checked"],
         "reference": pc["n_equal_to_reference_hash"] == pc["n_prompts_checked"],
         "parser": m["parser_version"] == PARSER_VERSION,
         "pins_rev": m["tokenization"]["pins"]["revision"] == m["model_revision"]}
    g = np.array([q["gold_idx"] for q in qs[:N]])
    # re-parse every reply independently of the stored arrays
    rot = m["condition"] == "greedyrot"
    rp = all(parse_choice(str(z["texts"][k, i]), rotate(qs[i]["choices"], k) if rot else qs[i]["choices"])[0] or ""
             == str(z["parsed"][k, i]) for k in range(K) for i in range(N))
    c["reparse"] = rp
    if rot:
        c["gold_rot"] = all(str(z["gold_rot"][k, i]) == "ABCD"[(g[i] + k) % 4] for k in range(K) for i in range(N))
        exp = np.array([[("ABCD".index(L) - k) % 4 if L else -1 for L in z["parsed"][k]] for k in range(K)])
        c["chosen"] = bool((z["chosen_option"] == exp).all())
        c["correct"] = bool((z["correct"] == (z["chosen_option"] == g[None])).all())
        v2 = np.load(R / f"results/mmlu_greedy_{m['model'].replace('/', '_')}.npz")
        c["r0_eq_v2"] = f"{sum(str(z['texts'][0, i]) == str(v2['texts'][0, i]) for i in range(N))}/{N}"
        c["acc_by_rot"] = [round(a, 3) for a in m["accuracy_by_rotation"]]
    else:
        c["correct"] = bool((z["correct"][0] == np.array([str(z["parsed"][0, i]) == qs[i]["gold_letter"] for i in range(N)])).all())
        c["acc"] = m["accuracy"]
    c["n_prompts"] = pc["n_prompts_checked"]
    c["host_used_gib"] = m["memory_settings"]["host_used_gib_at_engine_ready"]
    ok &= all(v for k, v in c.items() if isinstance(v, bool)) and (not rot or c["r0_eq_v2"] == f"{N}/{N}")
    print(m["run_id"], c)
print("DRY RUN CHECK", "PASS" if ok else "FAIL")
