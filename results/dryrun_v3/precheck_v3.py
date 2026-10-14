"""CPU-only check of the v3 pre-check path of src/collect.py with the pins (no engine, no GPU).
For each ensemble model: load_pins, tokenizer.json sha256 + revision asserts, then
do_run(llm=None) for every v3 run (R1 greedy + R2 greedyrot) at 500 questions, which
asserts canonical ids, the pinned decode kind and the independent reference hash per prompt."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
import collect
from canonical_tok import CanonicalPrompter
from plan import ENSEMBLE_MODELS, run_id, v3_runs
from transformers import AutoTokenizer
V2 = Path("results")
ctx = {t: json.loads((V2 / f"questions_{t}.json").read_text()) for t in ("mmlu", "mmlu_dev", "hellaswag", "gsm8k")}
out = {}
for m in ENSEMBLE_MODELS:
    b = CanonicalPrompter(m)
    pins = collect.load_pins(m, collect.PINS_DIR)
    assert b.tokenizer_json_sha256 == pins["tokenizer_json_sha256"], m
    assert b.snap.name == pins["revision"] == collect.model_revision(m), m
    hf = AutoTokenizer.from_pretrained(m)
    for r in v3_runs(m):
        rid = run_id(m, r)
        if (V2 / f"{rid}.npz").exists():   # the F1 guard: a v2 file here must raise
            assert not collect.is_v3_output(V2 / f"{rid}.json")
            print(rid, "exists in results/ and is not v3 -> collect would refuse"); continue
        pc = collect.do_run(None, hf, b, m, r, ctx, Path("/nonexistent"), None, None, lambda s: None, pins)
        out[rid] = pc["prompt_check"]
        print(rid, pc["prompt_check"], flush=True)
ok = all(v["n_equal_to_reference_hash"] == v["n_prompts_checked"] for v in out.values())
Path("results/dryrun_v3/precheck_v3.json").write_text(json.dumps({"all_reference_equal": ok, "runs": out}, indent=1))
print("ALL OK" if ok else "FAIL")
