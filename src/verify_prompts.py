"""Check every prompt of every planned run with CanonicalPrompter.check_prompt; writes the pins that collect.py reads.

Also records agreement with transformers' render and ids. Runs in the gb10-vllm container:
    src/docker_run.sh gb10-vllm python src/verify_prompts.py --out results/pins/prompt_check.json [models...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canonical_tok import CanonicalPrompter, ids_sha  # noqa: E402
from collect import GSM8K_COT_PROMPT, exemplar_rows, hellaswag_user_text, mmlu_user_text  # noqa: E402
from plan import ENSEMBLE_MODELS, GREEDYROT, PLAN, rotate, run_id  # noqa: E402

RESULTS = HERE.parent / "results"
LETTERS = "ABCD"


def user_texts(run: dict, ctx: dict):
    task, cond, K = run["task"], run["condition"], run["K"]
    qs = ctx[task]
    if cond == "sharedprompt":
        ex = exemplar_rows(K, len(ctx["mmlu_dev"]))
        for k in range(K):
            for q in qs:
                yield k, q, mmlu_user_text(q, [ctx["mmlu_dev"][r] for r in ex[k]])
        return
    if cond == "greedyrot":
        for k in range(K):
            for q in qs:
                yield k, q, mmlu_user_text({**q, "choices": rotate(q["choices"], k)})
        return
    for q in qs:  # same prompt in every pass
        if task == "mmlu":
            yield 0, q, mmlu_user_text(q)
        elif task == "hellaswag":
            yield 0, q, hellaswag_user_text(q)
        else:
            yield 0, q, GSM8K_COT_PROMPT.format(question=q["question"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    ctx = {t: json.loads((RESULTS / f"questions_{t}.json").read_text()) for t in ("mmlu", "mmlu_dev", "hellaswag", "gsm8k")}
    models = a.models or ENSEMBLE_MODELS
    report = {}
    all_ok = True
    for m in models:
        t0 = time.time()
        cp = CanonicalPrompter(m)
        hf = AutoTokenizer.from_pretrained(m)
        mrep = {"tokenizer_json_sha256": cp.tokenizer_json_sha256, "revision": cp.snap.name, "prompt_mode": cp.mode,
                "chat_template_source": cp.template_source, "versions": cp.versions, "runs": {}}
        for run in PLAN[m] + [GREEDYROT]:
            rid = run_id(m, run)
            n = n_ok = v2_eq = render_eq = 0
            decode: dict[str, int] = {}
            problems = []
            q0_render = None
            for k, q, ut in user_texts(run, ctx):
                text = cp.render(ut)
                ids = cp.encode_text(text)
                chk = cp.check_prompt(text, ids, hf)
                n += 1
                n_ok += chk["ok"]
                v2_eq += chk["v2_pipeline_equal"]
                decode[chk["decode"]] = decode.get(chk["decode"], 0) + 1
                if not chk["ok"] and len(problems) < 3:
                    problems.append({"pass": k, "row": q["row"], "problems": chk["problems"]})
                if n <= 50 or run["condition"] != "sharedprompt" or n % 97 == 0:
                    render_eq += int(cp.render_transformers(hf, ut) == text)
                    mrep.setdefault("_render_checked", 0)
                    mrep["_render_checked"] += 1
                if q0_render is None:
                    q0_render = text
                    q0_ids = ids
            r = {"n_prompts": n, "n_canonical": n_ok, "decode": decode, "n_v2_pipeline_equal": v2_eq,
                 "n_render_equal_transformers": render_eq, "problems": problems,
                 "q0_ids_sha": ids_sha(q0_ids), "q0_n_tokens": len(q0_ids)}
            js = RESULTS / f"{rid}.json"
            if js.exists():
                old = json.loads(js.read_text())
                r["q0_render_equals_v2_saved"] = old.get("rendered_prompt_q0_pass0") == q0_render
                r["q0_n_tokens_v2"] = old.get("prompt_tokens_q0_pass0")
            all_ok &= n_ok == n
            mrep["runs"][rid] = r
            print(f"{rid}: {n_ok}/{n} canonical, decode {decode}, v2-equal {v2_eq}/{n}, "
                  f"q0 render==v2 {r.get('q0_render_equals_v2_saved')}", flush=True)
        mrep["sec"] = round(time.time() - t0, 1)
        report[m] = mrep
    for mrep in report.values():  # collect.py allows only the decode kinds seen here
        kinds = set()
        for r in mrep["runs"].values():
            kinds |= set(r["decode"])
        mrep["allowed_decode_kinds"] = sorted(kinds)
    Path(a.out).write_text(json.dumps({"all_canonical": all_ok, "models": report}, indent=1))
    print("ALL CANONICAL" if all_ok else "FAILURES", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
