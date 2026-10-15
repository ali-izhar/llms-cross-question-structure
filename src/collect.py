"""Run every planned run of one model on one vLLM engine (inside the gb10-vllm container).

Prompt ids come from canonical_tok and are checked before generation; replies are decoded with the same
tokenizer (vLLM's own text is kept as `texts_vllm`). Each run writes <run_id>.npz + .json and resumes
from <run_id>.partial.npz. Protocol: results/PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from canonical_tok import CanonicalPrompter, ids_sha  # noqa: E402
from plan import PLAN, positions_to_options, rotate, run_id as make_run_id, slug, v3_runs  # noqa: E402
from scoring import LETTERS, PARSER_VERSION, parse_choice, score_gsm8k  # noqa: E402

RESULTS = ROOT / "results"
# per-model pins written by verify_prompts.py and reference_hashes.py
PINS_DIR = RESULTS / "pins"


def load_pins(model: str, pins_dir: Path) -> dict:
    """Tokenizer sha256, revision and decode kinds (prompt_check.json) + reference hashes; missing pins raise."""
    pc = json.loads((pins_dir / "prompt_check.json").read_text())["models"][model]
    ref = json.loads((pins_dir / "reference_hashes.json").read_text())["models"][model]
    if ref["revision"] != pc["revision"]:
        raise AssertionError(f"{model}: reference and prompt-check revisions differ")
    return {"dir": str(pins_dir), "tokenizer_json_sha256": pc["tokenizer_json_sha256"], "revision": pc["revision"],
            "allowed_decode_kinds": set(pc["allowed_decode_kinds"]), "reference": ref}


def is_v3_output(js: Path, planned_n: int | None = None, pins: dict | None = None) -> bool:
    """True iff a finished, non-dry, deterministic run with the planned N, the current parser and pins,
    and every prompt equal to its reference hash. Callers raise when False and the .npz exists."""
    if not js.exists():
        return False
    m = json.loads(js.read_text())
    t = m.get("tokenization")
    if not t or m.get("dry_run") is not False or m.get("parser_version") != PARSER_VERSION:
        return False
    if not (m.get("deterministic") or {}).get("on"):
        return False
    if planned_n is not None and m.get("N") != planned_n:
        return False
    if pins is not None and ((t.get("pins") or {}).get("revision") != pins["revision"]
                             or m.get("model_revision") != pins["revision"]):
        return False
    pc = t.get("prompt_check") or {}
    return pc.get("n_equal_to_reference_hash") is not None and \
        pc.get("n_equal_to_reference_hash") == pc.get("n_prompts_checked")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


MAX_MODEL_LEN = 3072      # MC prompts < 1.5k tokens (S2 incl.), GSM8K < 1k
MAX_NUM_SEQS = 128



# Memory budget (hard cap: total system "used" < 100 GB; baseline ~18 GiB).
# used ~= baseline + util * 121 GiB, so util <= 0.5 keeps the peak near 80 GiB.
def mem_util_for(model: str) -> float:
    big = {"mistralai/Mistral-Small-Instruct-2409": 0.48}
    if model in big:
        return big[model]
    m = model.lower()
    if any(t in m for t in ("0.5b", "1.5b", "1.7b", "1b-", "2b-it")):
        return 0.25
    if any(t in m for t in ("3b-", "phi-2", "phi-3")):
        return 0.30
    return 0.40


# Deterministic kernels: no timing-based kernel selection, so greedy replies and logprobs repeat
# across engine starts. The env is set before torch loads.
DETERMINISTIC_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",        # deterministic cuBLAS workspace (no split-K atomics)
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE": "0",    # no inductor max-autotune
    "VLLM_USE_FLASHINFER_SAMPLER": "0",          # PyTorch-native sampler
}
DETERMINISTIC_ENGINE = {
    "enforce_eager": True,                       # no torch.compile/inductor benchmarking, no CUDA graphs
    "kernel_config": {"enable_flashinfer_autotune": False},
    "async_scheduling": False,                   # batch formation does not depend on timing
}
# Fixed KV-cache size (blocks of 16 tokens = 32,768 tokens): the same block pool at every start.
# Fits the largest KV per token (gemma-2-9b ~344 KB/token -> 11 GB; Mistral-Small-22B -> 7.5 GB).
NUM_GPU_BLOCKS = 2048
BATCH_INVARIANT_ENV = {"VLLM_BATCH_INVARIANT": "1"}
MC_MAX_TOKENS = 8
GSM_MAX_TOKENS = 256
GSM_LONG_MAX_TOKENS = 1024  # supplementary: 256 truncates ~2/3 of Qwen CoT replies
N_LOGPROBS = 20
N_SHOTS = 5
ENGINE_SEED = 0

# Same GSM8K prompt as the submitted paper's runs.
GSM8K_COT_PROMPT = (
    'Solve this math problem step by step. Show your reasoning, '
    'then give the final numerical answer on the last line as "#### <number>".\n'
    "\n"
    "Question: {question}\n"
    "\n"
    "Solution:"
)


def mmlu_user_text(q: dict, exemplars: list[dict] | None = None) -> str:
    from mmlu_utils import build_mmlu_prompt

    text = build_mmlu_prompt(q["question"], q["choices"])
    if not exemplars:
        return text
    header = "Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
    assert text.startswith(header)
    shots = "Here are some example questions with their answers.\n\n"
    for ex in exemplars:
        shots += build_mmlu_prompt(ex["question"], ex["choices"])[len(header):] + f" {ex['gold_letter']}\n\n"
    shots += "Now answer this question.\n\n"
    return header + shots + text[len(header):]


def hellaswag_user_text(q: dict) -> str:
    from hellaswag_utils import build_hellaswag_prompt

    return build_hellaswag_prompt(q["context"], q["endings"])


def run_seed_base(rid: str) -> int:
    return (zlib.crc32(rid.encode()) & 0x7FFFFFFF) << 20


def seeds_for(rid: str, condition: str, K: int, N: int) -> np.ndarray:
    base = run_seed_base(rid)
    k = np.arange(K, dtype=np.int64)[:, None]
    i = np.arange(N, dtype=np.int64)[None, :]
    if condition in ("greedy", "greedyrot"):
        return np.full((K, N), -1, dtype=np.int64)
    if condition == "sharedseed":
        return np.broadcast_to(base + k, (K, N)).copy()
    return base + k * N + i  # iid and sharedprompt: one independent seed per request


def exemplar_rows(K: int, n_dev: int) -> np.ndarray:
    # Same exemplar sets for every model (depends on pass index only).
    return np.stack([np.random.default_rng(10_000 + k).choice(n_dev, size=N_SHOTS, replace=False)
                     for k in range(K)])


def _letter_of(tok_text: str | None) -> str | None:
    if tok_text is None:
        return None
    t = tok_text.strip().lstrip("(*").strip()
    return t if t in ("A", "B", "C", "D") else None


def letter_logprobs(pos_dict, tok_text) -> np.ndarray:
    """logsumexp of the top-k logprobs whose canonical token text is a letter A-D."""
    out = np.full(4, np.nan, dtype=np.float32)
    acc: dict[str, list[float]] = {}
    for tid, lp in pos_dict.items():
        L = _letter_of(tok_text(tid))
        if L is not None and math.isfinite(lp.logprob):
            acc.setdefault(L, []).append(lp.logprob)
    for L, vals in acc.items():
        m = max(vals)
        out[LETTERS.index(L)] = m + math.log(sum(math.exp(v - m) for v in vals))
    return out


def save_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def do_run(llm, hf_tok, builder: CanonicalPrompter, model: str, run: dict, ctx: dict, out_dir: Path,
           n_questions: int | None, k_cap: int | None, log, pins: dict | None = None) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    task, cond, K, T = run["task"], run["condition"], run["K"], run["temperature"]
    rid = make_run_id(model, run)
    questions = ctx[task]
    dry = n_questions is not None
    if n_questions is not None:
        questions = questions[:n_questions]
    if k_cap is not None:
        K = min(K, k_cap)
    N = len(questions)
    rot = cond == "greedyrot"
    final = out_dir / f"{rid}.npz"
    partial = out_dir / f"{rid}.partial.npz"
    if final.exists():
        if not is_v3_output(out_dir / f"{rid}.json", N, pins):
            raise RuntimeError(f"[{rid}] {final} exists but is not a complete v3 output of this plan "
                               "(tokenization record, not dry, N, parser version, pins, reference hashes); "
                               "archive it before re-running")
        log(f"[{rid}] final v3 output exists; skip")
        return json.loads((out_dir / f"{rid}.json").read_text())

    mc = task in ("mmlu", "hellaswag")
    max_tokens = MC_MAX_TOKENS if mc else (GSM_LONG_MAX_TOKENS if cond == "iidlong" else GSM_MAX_TOKENS)
    seeds = seeds_for(rid, cond, K, N)
    ex_rows = exemplar_rows(K, len(ctx["mmlu_dev"])) if cond == "sharedprompt" else None

    def qview(k: int, q: dict) -> dict:
        """The question as presented in pass k (rotation k for greedyrot)."""
        if not rot:
            return q
        ch = rotate(q["choices"], k)
        g = (LETTERS.index(q["gold_letter"]) + k) % 4
        return {**q, "choices": ch, "gold_letter": LETTERS[g], "gold_idx": g}

    def options(k: int, q: dict) -> list[str] | None:
        if task == "mmlu":
            return qview(k, q)["choices"]
        if task == "hellaswag":
            return q["endings"]
        return None

    def user_text(k: int, q: dict) -> str:
        if task == "mmlu":
            ex = [ctx["mmlu_dev"][r] for r in ex_rows[k]] if ex_rows is not None else None
            return mmlu_user_text(qview(k, q), ex)
        if task == "hellaswag":
            return hellaswag_user_text(q)
        return GSM8K_COT_PROMPT.format(question=q["question"])

    # Prompts differ per pass only for S2 exemplars and rotations. Every prompt is checked before
    # any generation; one failure aborts the run.
    per_pass = ex_rows is not None or rot
    n_sets = K if per_pass else 1
    prompt_ids: list[list[list[int]]] = []
    decode_kinds: dict[str, int] = {}
    v2_equal = 0
    n_checked = 0
    n_ref = 0
    for k in range(n_sets):
        row = []
        for q in questions:
            text = builder.render(user_text(k, q))
            ids = builder.ids(user_text(k, q))
            chk = builder.check_prompt(text, ids, hf_tok)
            if not chk["ok"]:
                raise AssertionError(f"[{rid}] non-canonical prompt (pass {k}, row {q['row']}): {chk['problems']}")
            if pins is not None:
                if chk["decode"] not in pins["allowed_decode_kinds"]:
                    raise AssertionError(f"[{rid}] decode kind {chk['decode']!r} not pinned for {model} "
                                         f"({sorted(pins['allowed_decode_kinds'])})")
                if task == "mmlu" and cond in ("greedy", "greedyrot"):  # independent reference hash
                    ref = pins["reference"]
                    j = ref["question_rows"].index(q["row"])
                    if ids_sha(ids) != ref["hashes"][f"r{k if rot else 0}"][j]:
                        raise AssertionError(f"[{rid}] prompt ids differ from the independent reference "
                                             f"(pass {k}, row {q['row']})")
                    n_ref += 1
            decode_kinds[chk["decode"]] = decode_kinds.get(chk["decode"], 0) + 1
            v2_equal += int(chk.get("v2_pipeline_equal", False))
            n_checked += 1
            row.append(ids)
        prompt_ids.append(row)
    prompt_check = {"n_prompts_checked": n_checked, "n_canonical": n_checked, "decode": decode_kinds,
                    "n_equal_to_v2_pipeline_ids": v2_equal if hf_tok is not None else None,
                    "n_equal_to_reference_hash": n_ref if pins is not None else None}
    log(f"[{rid}] prompt check: {n_checked}/{n_checked} canonical; decode {decode_kinds}; "
        f"v2-pipeline-equal {v2_equal}/{n_checked}; reference-hash-equal {n_ref}/{n_checked}")
    prompt_hash = ids_sha([t for row in prompt_ids for ids in row for t in ids + [-1]])

    def prompt_of(k: int, i: int) -> list[int]:
        return prompt_ids[k if per_pass else 0][i]

    if llm is None:  # pre-check only (before the engine is loaded)
        return {"run_id": rid, "prompt_check": prompt_check}

    def sp(seed: int) -> SamplingParams:
        return SamplingParams(
            n=1, temperature=T, top_p=1.0, top_k=0, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=max_tokens, seed=None if seed < 0 else int(seed),
            logprobs=N_LOGPROBS if mc else None, skip_special_tokens=True,
        )

    sp0 = sp(int(seeds[0, 0]))
    settings = {k: getattr(sp0, k) for k in ("n", "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                                              "presence_penalty", "frequency_penalty", "max_tokens", "logprobs",
                                              "skip_special_tokens", "stop", "stop_token_ids", "ignore_eos")}
    settings["seed"] = "per-request, see seeds array" if cond not in ("greedy", "greedyrot") else None

    # State arrays (resume from partial).
    A = {
        "correct": np.zeros((K, N), np.int8),
        "parsed": np.full((K, N), "", dtype="<U1") if mc else np.full((K, N), np.nan),
        "texts": np.full((K, N), "", dtype=object),
        "texts_vllm": np.full((K, N), "", dtype=object),
        "finish_reason": np.full((K, N), "", dtype="<U8"),
        "n_gen_tokens": np.zeros((K, N), np.int16),
        "cum_logprob": np.full((K, N), np.nan, np.float32),
    }
    if mc:
        A["letter_logprob_first"] = np.full((K, N, 4), np.nan, np.float32)
        A["letter_logprob_at_letter"] = np.full((K, N, 4), np.nan, np.float32)
        A["letter_token_pos"] = np.full((K, N), -1, np.int16)
        A["top_ids_first"] = np.full((K, N, N_LOGPROBS + 1), -1, np.int32)
        A["top_lp_first"] = np.full((K, N, N_LOGPROBS + 1), np.nan, np.float32)
    if rot:
        A["chosen_option"] = np.full((K, N), -1, np.int8)
        A["gold_rot"] = np.full((K, N), "", dtype="<U1")
        A["letter_logprob_first_by_option"] = np.full((K, N, 4), np.nan, np.float32)
    A["parse_reason"] = np.full((K, N), "", dtype="<U20")
    A["gen_token_ids"] = np.full((K, N, max_tokens), -1, np.int32) if mc else np.zeros((K, N, 0), np.int32)
    k_done = 0
    runtime_prev = 0.0
    scorer_sha = hashlib.sha256((HERE / "scoring.py").read_bytes()).hexdigest()[:16]
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        # resume only with the same scorer and the same prompts
        if "scorer_sha256" not in z or str(z["scorer_sha256"]) != scorer_sha:
            raise RuntimeError(f"[{rid}] {partial} was written with another scorer; remove it to restart")
        if "prompt_hash" not in z or str(z["prompt_hash"]) != prompt_hash:
            raise RuntimeError(f"[{rid}] {partial} was written with other prompt ids; remove it to restart")
        if int(z["K"]) == K and int(z["N"]) == N:
            k_done = int(z["k_done"])
            runtime_prev = float(z["runtime_sec"])
            for key in A:
                if key in ("texts", "texts_vllm"):
                    A[key][:k_done] = z[key][:k_done].astype(object)
                else:
                    A[key][:k_done] = z[key][:k_done]
            log(f"[{rid}] resume at pass {k_done}/{K}")
    token_strings: dict[int, str] = {}
    _tt: dict[int, str] = {}

    def tok_text(tid: int) -> str:
        if tid not in _tt:
            _tt[tid] = builder.token_text(tid)
        return _tt[tid]

    n_text_differs = 0

    block = max(1, min(K, 5000 // N))
    t0 = time.time()
    log(f"[{rid}] start K={K} N={N} T={T} block={block} passes")
    for k0 in range(k_done, K, block):
        k1 = min(K, k0 + block)
        prompts, params, where = [], [], []
        for k in range(k0, k1):
            for i, q in enumerate(questions):
                ids = prompt_of(k, i)
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(sp(int(seeds[k, i])))
                where.append((k, i))
        outs = llm.generate(prompts, params, use_tqdm=False)
        for (k, i), o in zip(where, outs, strict=True):
            if list(o.prompt_token_ids) != prompt_of(k, i):
                raise AssertionError(f"[{rid}] vLLM prompt ids differ from the checked ids (pass {k}, q {i})")
            c = o.outputs[0]
            # canonical decode of the generated ids (vLLM's detokenizer uses the transformers tokenizer)
            text = builder.decode(c.token_ids, skip_special_tokens=True)
            n_text_differs += int(text != c.text)
            A["texts"][k, i] = text
            A["texts_vllm"][k, i] = c.text
            if mc:
                n_ids = min(len(c.token_ids), A["gen_token_ids"].shape[2])
                A["gen_token_ids"][k, i, :n_ids] = list(c.token_ids)[:n_ids]
            A["finish_reason"][k, i] = str(c.finish_reason)
            A["n_gen_tokens"][k, i] = len(c.token_ids)
            if c.cumulative_logprob is not None:
                A["cum_logprob"][k, i] = c.cumulative_logprob
            if mc:
                qv = qview(k, questions[i])
                letter, why = parse_choice(text, options(k, questions[i]))
                letter = letter or ""
                A["correct"][k, i] = int(letter != "" and letter == qv["gold_letter"])
                A["parsed"][k, i] = letter
                A["parse_reason"][k, i] = why
                if rot:
                    A["gold_rot"][k, i] = qv["gold_letter"]
                    if letter:
                        A["chosen_option"][k, i] = (LETTERS.index(letter) - k) % 4
                lps = c.logprobs
                if lps is not None and len(lps) > 0:
                    d0 = lps[0]
                    A["letter_logprob_first"][k, i] = letter_logprobs(d0, tok_text)
                    if rot:  # position p shows original option (p - k) mod 4
                        A["letter_logprob_first_by_option"][k, i] = positions_to_options(
                            A["letter_logprob_first"][k, i], k)
                    items = sorted(d0.items(), key=lambda kv: -kv[1].logprob)[:N_LOGPROBS + 1]
                    for j, (tid, lp) in enumerate(items):
                        A["top_ids_first"][k, i, j] = tid
                        A["top_lp_first"][k, i, j] = lp.logprob
                        token_strings.setdefault(int(tid), tok_text(tid))
                    for p, tid in enumerate(c.token_ids):
                        d = lps[p]
                        if tid in d and _letter_of(tok_text(tid)) is not None:
                            A["letter_token_pos"][k, i] = p
                            A["letter_logprob_at_letter"][k, i] = letter_logprobs(d, tok_text)
                            break
            else:
                corr, val = score_gsm8k(text, questions[i]["gold_answer"])
                A["correct"][k, i] = corr
                A["parsed"][k, i] = np.nan if val is None else val
        acc_blk = A["correct"][k0:k1].mean()
        elapsed = runtime_prev + time.time() - t0
        save_npz(partial, {**{k: (v.astype(str) if k in ("texts", "texts_vllm") else v) for k, v in A.items()},
                           "K": K, "N": N, "k_done": k1, "runtime_sec": elapsed,
                           "scorer_sha256": scorer_sha, "prompt_hash": prompt_hash})
        log(f"[{rid}] passes {k0}-{k1 - 1} done, block acc={acc_blk:.3f}, elapsed={elapsed:.0f}s")

    runtime = runtime_prev + time.time() - t0
    C = A["correct"]
    p = C.mean(axis=0)
    if mc:
        unparse = float((A["parsed"] == "").mean())
    else:
        unparse = float(np.isnan(A["parsed"]).mean())
    nb = int(((p > 0.05) & (p < 0.95)).sum()) if K > 1 else None
    arrays = {k: (v.astype(str) if k in ("texts", "texts_vllm") else v) for k, v in A.items()}
    # prompt ids actually sent (flat + offsets over [n_sets, N]) and a per-prompt hash
    flat = [t for row in prompt_ids for ids in row for t in ids]
    lens = [len(ids) for row in prompt_ids for ids in row]
    arrays["prompt_ids_flat"] = np.array(flat, np.int32)
    arrays["prompt_ids_offsets"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    arrays["prompt_ids_sha"] = np.array([[ids_sha(ids) for ids in row] for row in prompt_ids])
    arrays["seeds"] = seeds
    arrays["question_rows"] = np.array([q["row"] for q in questions], np.int64)
    arrays["gold"] = (np.array([q["gold_letter"] for q in questions]) if mc
                      else np.array([q["gold_answer"] for q in questions], np.float64))
    if ex_rows is not None:
        arrays["exemplar_dev_rows"] = ex_rows

    rendered0 = builder.render(user_text(0, questions[0]))
    meta = {
        "run_id": rid, "task": task, "condition": cond, "model": model,
        "K": K, "N": N, "temperature": T, "dry_run": dry,
        "sampling_params": settings,
        "seed_scheme": {
            "iid": "seed[k,i] = base + k*N + i (independent per request)",
            "sharedprompt": "seed[k,i] = base + k*N + i (independent per request); exemplar set shared within pass k",
            "sharedseed": "seed[k,i] = base + k for every i (common random numbers within pass k)",
            "greedy": "none (temperature 0)",
            "iidlong": "seed[k,i] = base + k*N + i (independent per request); max_tokens=1024",
            "greedyrot": "none (temperature 0); pass k = option rotation k (option j shown at (j+k) mod 4)",
        }[cond],
        "seed_base": run_seed_base(rid) if cond not in ("greedy", "greedyrot") else None,
        "prompt_mode": builder.mode,
        "rendered_prompt_q0_pass0": rendered0,
        "prompt_tokens_q0_pass0": len(prompt_of(0, 0)),
        "prompt_ids_q0_pass0_sha": ids_sha(prompt_of(0, 0)),
        "tokenization": {
            "method": "tokenizers.Tokenizer.from_file(<snapshot>/tokenizer.json); chat template rendered with jinja2",
            "tokenizer_json_sha256": builder.tokenizer_json_sha256,
            "chat_template_source": builder.template_source,
            "pinned_template_date": "2026-10-10",
            **{f"{k}_version": v for k, v in builder.versions.items()},
            "prompt_check": prompt_check,
            "prompt_hash_all": prompt_hash,
            "pins": ({"dir": pins["dir"], "revision": pins["revision"],
                      "tokenizer_json_sha256": pins["tokenizer_json_sha256"],
                      "allowed_decode_kinds": sorted(pins["allowed_decode_kinds"]),
                      "reference_method": "transformers 4.46.3 apply_chat_template(tokenize=True); "
                                          "Yi: slow SentencePiece"} if pins is not None else None),
            "n_texts_differing_from_vllm_detokenizer": n_text_differs,
        },
        "scorer_sha256": scorer_sha,
        "parser_version": PARSER_VERSION,
        "parse_reason_counts": {r: int((A["parse_reason"] == r).sum()) for r in np.unique(A["parse_reason"])} if mc else None,
        "accuracy": float(C.mean()),
        "unparseable_rate": unparse,
        "length_truncated_rate": float((A["finish_reason"] == "length").mean()),
        "N_b_eps0.05": nb,
        "n_always_correct": int((p == 1).sum()), "n_always_wrong": int((p == 0).sum()),
        "runtime_sec": round(runtime, 1),
        "first_token_letter_logprob_coverage": (
            float(np.isfinite(A["letter_logprob_first"]).any(axis=2).mean()) if mc else None),
        **ctx["engine_meta"],
    }
    if mc:
        meta["parsed_letter_counts"] = {L: int((A["parsed"] == L).sum()) for L in ["A", "B", "C", "D", ""]}
        if rot:
            meta["accuracy_by_rotation"] = [float(C[k].mean()) for k in range(K)]
            meta["unparseable_by_rotation"] = [float((A["parsed"][k] == "").mean()) for k in range(K)]
            meta["letter_share_by_rotation"] = [{L: float((A["parsed"][k] == L).mean()) for L in LETTERS}
                                                for k in range(K)]
        write_json_atomic(out_dir / f"{rid}.tokens.json", token_strings)
    save_npz(final, arrays)
    write_json_atomic(out_dir / f"{rid}.json", meta)
    if partial.exists():
        partial.unlink()
    log(f"[{rid}] DONE acc={meta['accuracy']:.4f} unparseable={unparse:.4f} N_b={nb} runtime={runtime:.0f}s")
    return meta


def _host_used_gib() -> float | None:
    try:
        info = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
        return round((info["MemTotal"] - info["MemAvailable"]) / 2**20, 1)
    except Exception:
        return None


def model_revision(model: str) -> str:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True)).name


def max_len_for(model: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model)
    cands = [getattr(cfg, a, None) for a in ("max_position_embeddings", "n_positions", "seq_length")]
    cands = [c for c in cands if isinstance(c, int) and c > 0]
    return min([MAX_MODEL_LEN] + cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", default="", help="comma-separated run ids (default: all planned)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", default="v2", choices=["v2", "v3"], help="v3 = R1 greedy re-run + R2 greedyrot")
    ap.add_argument("--out-dir", default=None, help="output dir (default results/, or results/dryrun/)")
    ap.add_argument("--n-questions", type=int, default=None, help="first N questions only (dry run)")
    ap.add_argument("--k-cap", type=int, default=None, help="cap the passes per run (dry run)")
    ap.add_argument("--pins-dir", default=str(PINS_DIR), help="prompt_check.json + reference_hashes.json")
    ap.add_argument("--tokenizer-mode", default="hf")
    ap.add_argument("--gpu-mem-util", type=float, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--deterministic", default=None, choices=["on", "off"],
                    help="deterministic kernels (default: on for --plan v3, off for v2)")
    ap.add_argument("--batch-invariant", default="on", choices=["on", "off"],
                    help="VLLM_BATCH_INVARIANT=1 (only with --deterministic on)")
    ap.add_argument("--blocks-override", default="on", choices=["on", "off"],
                    help=f"num_gpu_blocks_override={NUM_GPU_BLOCKS} (only with --deterministic on)")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (only if a model cannot fit)")
    args = ap.parse_args()

    plan_runs = PLAN[args.model] if args.plan == "v2" else v3_runs(args.model)
    if args.runs:  # every requested run id must be in the plan (checked before anything is written)
        unknown = set(args.runs.split(",")) - {make_run_id(args.model, r) for r in plan_runs}
        if unknown:
            raise SystemExit(f"--runs ids not in the {args.plan} plan of {args.model}: {sorted(unknown)}")
    if args.dry_run:  # 20 questions, K <= 5
        args.n_questions = args.n_questions or 20
        args.k_cap = args.k_cap or 5
    out_dir = Path(args.out_dir) if args.out_dir else (RESULTS / "dryrun" if args.dry_run else RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    logf = open(out_dir / "logs" / f"collect_{slug(args.model)}{'_dry' if args.dry_run else ''}.log", "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    runs = plan_runs
    if args.runs:
        want = set(args.runs.split(","))
        runs = [r for r in runs if make_run_id(args.model, r) in want]
    ctx = {
        "mmlu": json.loads((RESULTS / "questions_mmlu.json").read_text()),
        "mmlu_dev": json.loads((RESULTS / "questions_mmlu_dev.json").read_text()),
        "hellaswag": json.loads((RESULTS / "questions_hellaswag.json").read_text()),
        "gsm8k": json.loads((RESULTS / "questions_gsm8k.json").read_text()),
    }
    pins = load_pins(args.model, Path(args.pins_dir)) if args.plan == "v3" else None
    if args.n_questions is None:
        keep = []
        for r in runs:
            rid = make_run_id(args.model, r)
            if (out_dir / f"{rid}.npz").exists():
                if not is_v3_output(out_dir / f"{rid}.json", len(ctx[r["task"]]), pins):  # never a silent skip
                    raise SystemExit(f"{rid}: {out_dir / (rid + '.npz')} exists but is not a v3 output; "
                                     "archive it before re-running")
                continue
            keep.append(r)
        runs = keep
    if not runs:
        log(f"{args.model}: nothing to do")
        return

    deterministic = (args.deterministic or ("on" if args.plan == "v3" else "off")) == "on"
    if deterministic:
        os.environ.update(DETERMINISTIC_ENV)
        if args.batch_invariant == "on":
            os.environ.update(BATCH_INVARIANT_ENV)
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    # transformers tokenizer: only to record whether the v2 pipeline gave the same ids
    hf_tok = AutoTokenizer.from_pretrained(args.model)
    builder = CanonicalPrompter(args.model)
    if pins is not None:
        if builder.tokenizer_json_sha256 != pins["tokenizer_json_sha256"]:
            raise SystemExit(f"{args.model}: tokenizer.json sha256 differs from the pinned value")
        if builder.snap.name != pins["revision"] or model_revision(args.model) != pins["revision"]:
            raise SystemExit(f"{args.model}: snapshot revision differs from the pinned value {pins['revision']}")
    # Check every prompt of every run BEFORE the engine takes memory (aborts on failure).
    for run in runs:
        pc = do_run(None, hf_tok, builder, args.model, run, ctx, out_dir, args.n_questions, args.k_cap, log, pins)
        log(f"pre-check {pc.get('run_id')}: {pc.get('prompt_check', 'final output exists')}")
    max_len = max_len_for(args.model)
    engine_kwargs = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util or mem_util_for(args.model), max_model_len=max_len,
        seed=ENGINE_SEED, generation_config="vllm", enable_prefix_caching=True, max_num_seqs=MAX_NUM_SEQS,
        tokenizer_mode=args.tokenizer_mode, trust_remote_code=False, enforce_eager=args.enforce_eager,
    )
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if deterministic:
        engine_kwargs.update(DETERMINISTIC_ENGINE)
        if args.blocks_override == "on":
            engine_kwargs["num_gpu_blocks_override"] = NUM_GPU_BLOCKS
    t_load = time.time()
    log(f"{args.model}: loading vLLM engine {engine_kwargs}")
    try:
        llm = LLM(**engine_kwargs)
    except Exception as e:
        if args.tokenizer_mode != "hf":
            raise
        log(f"engine init failed with tokenizer_mode=hf ({e!r}); retrying tokenizer_mode=auto")
        engine_kwargs["tokenizer_mode"] = "auto"
        llm = LLM(**engine_kwargs)
    load_sec = time.time() - t_load
    ctx["engine_meta"] = {
        "backend": "vllm",
        "model_revision": model_revision(args.model),
        "container_image": os.environ.get("GB10_IMAGE_ID", "gb10-vllm"),
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "engine_kwargs": engine_kwargs, "engine_load_sec": round(load_sec, 1),
        "deterministic": {"on": deterministic,
                          "env": {k: os.environ.get(k) for k in {**DETERMINISTIC_ENV, **BATCH_INVARIANT_ENV}},
                          "engine": ({**DETERMINISTIC_ENGINE, **({"num_gpu_blocks_override": NUM_GPU_BLOCKS}
                                     if args.blocks_override == "on" else {})} if deterministic else None),
                          "batch_invariant": deterministic and args.batch_invariant == "on",
                          "blocks_override": deterministic and args.blocks_override == "on"},
        "logprobs_mode": "raw_logprobs (vLLM default: model log-softmax before temperature)",
        "gpu": torch.cuda.get_device_name(0),
        "memory_settings": {
            "gpu_memory_utilization": engine_kwargs["gpu_memory_utilization"],
            "max_model_len": max_len, "max_num_seqs": MAX_NUM_SEQS,
            "enforce_eager": engine_kwargs["enforce_eager"],
            "quantization": engine_kwargs.get("quantization"),
            "docker_memory_cap": os.environ.get("DOCKER_MEMORY", "none"),
            "host_used_gib_at_engine_ready": _host_used_gib(),
        },
    }
    log(f"engine ready in {load_sec:.0f}s; prompt_mode={builder.mode}; max_model_len={max_len}")

    failed = []
    for run in runs:
        try:
            do_run(llm, hf_tok, builder, args.model, run, ctx, out_dir, args.n_questions, args.k_cap, log, pins)
        except Exception:
            log(f"[{make_run_id(args.model, run)}] FAILED\n{traceback.format_exc()}")
            failed.append(make_run_id(args.model, run))
    log(f"{args.model}: finished; failed runs: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
