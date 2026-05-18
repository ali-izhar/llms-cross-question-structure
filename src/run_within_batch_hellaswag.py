"""
Within-Model Stochastic Sampling on HellaSwag (K=50, N=200, τ=1.0)
==================================================================

Sister script to `run_within_batch.py`. Purpose: extend the
"d_within ≈ 0" claim beyond MMLU + the GSM8K (N=30) pilot by running
full-scale within-model stochastic passes on a commonsense
multiple-choice benchmark.

For each model in MODELS:
  - K=50 stochastic forward passes
  - same N=200 stratified HellaSwag validation questions
  - τ=1.0, top_p=1.0, do_sample=True
  - apply_chat_template() for cross-model fairness
  - per-pass checkpoint of correctness matrix (resumable)

Outputs (per model):
  - results/correctness_matrix_hellaswag_{model_slug}.npy   (K × N int8)
  - results/questions_hellaswag_{model_slug}.json           (per-Q metadata)

Aggregate outputs:
  - results/within_batch_hellaswag_metrics.json
  - results/questions_hellaswag.json   (frozen 200-question sample, shared)
  - results/within_batch_hellaswag_console.log   (progress)

GB10 hygiene: bf16, max_memory={0:'100GiB','cpu':'20GiB'}, inference_mode,
no .cpu()/.to() round-trips in the inner loop.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT.parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent))
from hellaswag_utils import (  # noqa: E402
    build_hellaswag_prompt,
    check_hellaswag_answer,
    load_hellaswag,
    stratification_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("within_batch_hellaswag")

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

K = 50
N = 200
TEMPERATURE = 1.0
SEED = 42

# Three models from distinct families (per TODO #3 plan).
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",            # workhorse, direct comparison to MMLU
    "mistralai/Mistral-7B-Instruct-v0.3",  # different family, same tier
    "meta-llama/Meta-Llama-3-8B-Instruct", # third family, slightly larger
]

QUESTIONS_PATH = RESULTS / "questions_hellaswag.json"
METRICS_PATH = RESULTS / "within_batch_hellaswag_metrics.json"


# ─── Prompt assembly ──────────────────────────────────────────────────────


def build_chat_prompt(tokenizer, context: str, endings: list[str]) -> str:
    user_content = build_hellaswag_prompt(context, endings)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template failed, falling back to raw: {e}")
    return user_content


# ─── Per-model pipeline ───────────────────────────────────────────────────


def run_within_model(
    model_name: str,
    questions: list[dict],
    K_target: int = K,
    sanity_n: int | None = None,
) -> tuple[np.ndarray, dict]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_slug = model_name.replace("/", "_")
    ckpt_path = RESULTS / f"correctness_matrix_hellaswag_{model_slug}.npy"
    progress_path = RESULTS / f"_progress_hellaswag_{model_slug}.json"

    qs = questions if sanity_n is None else questions[:sanity_n]
    N_local = len(qs)

    t_load = time.time()
    logger.info(f"[{model_name}] Loading tokenizer + model (bf16, GB10 budget)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_memory = {0: "100GiB", "cpu": "20GiB"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
        )
    except Exception as e:
        logger.warning(f"Standard load failed, retrying with trust_remote_code=True: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
    model.eval()

    # Sanity-check device map (warn on CPU-offload, which would tank throughput).
    try:
        offloaded = [
            name for name, dev in model.hf_device_map.items()
            if dev in ("cpu", "disk")
        ]
        if offloaded:
            logger.warning(
                f"[{model_name}] {len(offloaded)} modules CPU-offloaded "
                f"(first: {offloaded[:3]}). GB10 budget likely too tight."
            )
        else:
            logger.info(f"[{model_name}] device_map all-GPU (no CPU offload).")
    except Exception:
        pass

    load_time = time.time() - t_load
    logger.info(f"[{model_name}] Loaded in {load_time:.1f}s.")

    # Resume support (only for full runs, not sanity check).
    if sanity_n is None and ckpt_path.exists() and progress_path.exists():
        C = np.load(ckpt_path)
        with open(progress_path) as f:
            progress = json.load(f)
        k_start = int(progress.get("k_done", 0))
        if C.shape != (K_target, N_local):
            logger.warning(
                f"[{model_name}] Shape mismatch ({C.shape} vs {(K_target, N_local)}), restarting."
            )
            C = np.zeros((K_target, N_local), dtype=np.int8)
            k_start = 0
        else:
            logger.info(f"[{model_name}] Resuming at pass {k_start}/{K_target}.")
    else:
        C = np.zeros((K_target, N_local), dtype=np.int8)
        k_start = 0

    t_infer = time.time()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    for k in range(k_start, K_target):
        t_pass = time.time()
        n_correct_pass = 0
        for i, q in enumerate(qs):
            prompt = build_chat_prompt(tokenizer, q["context"], q["endings"])
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=1.0,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            correct = check_hellaswag_answer(answer, q["correct_idx"])
            C[k, i] = int(correct)
            n_correct_pass += int(correct)

        # Per-pass checkpoint + structured progress.
        if sanity_n is None:
            np.save(ckpt_path, C)
            with open(progress_path, "w") as f:
                json.dump(
                    {
                        "k_done": k + 1,
                        "K_target": K_target,
                        "N": N_local,
                        "model": model_name,
                        "last_pass_acc": n_correct_pass / max(N_local, 1),
                    },
                    f,
                )

        elapsed = time.time() - t_infer
        passes_done = (k + 1) - k_start
        per_pass = elapsed / max(passes_done, 1)
        remaining = (K_target - (k + 1)) * per_pass
        eta = datetime.now() + timedelta(seconds=remaining)
        logger.info(
            f"[{model_name}] pass {k+1}/{K_target}  "
            f"acc={n_correct_pass/max(N_local,1):.3f}  "
            f"pass_time={time.time()-t_pass:.1f}s  "
            f"per_pass_avg={per_pass:.1f}s  "
            f"elapsed={elapsed/60:.1f}min  "
            f"ETA={eta.strftime('%H:%M:%S')}"
        )

    infer_time = time.time() - t_infer

    flip_rates = C.mean(axis=0)
    metadata = []
    for i, q in enumerate(qs):
        metadata.append({
            "index": i,
            "ind": q.get("ind"),
            "activity_label": q.get("activity_label", ""),
            "context": q["context"][:200],
            "correct_letter": q["correct_letter"],
            "flip_rate": float(flip_rates[i]),
            "borderline": bool(0.05 < flip_rates[i] < 0.95),
        })
    if sanity_n is None:
        with open(RESULTS / f"questions_hellaswag_{model_slug}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        if progress_path.exists():
            progress_path.unlink()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    metrics = {
        "model": model_name,
        "benchmark": "hellaswag",
        "K": K_target,
        "N": N_local,
        "temperature": TEMPERATURE,
        "load_time_sec": round(load_time, 1),
        "inference_time_sec": round(infer_time, 1),
        "mean_accuracy": float(flip_rates.mean()),
        "n_always_correct": int((flip_rates == 1.0).sum()),
        "n_always_wrong": int((flip_rates == 0.0).sum()),
        "n_borderline": int(((flip_rates > 0.05) & (flip_rates < 0.95)).sum()),
    }
    return C, metrics


# ─── Sanity check ─────────────────────────────────────────────────────────


def sanity_check(model_name: str, questions: list[dict], n: int = 5) -> bool:
    """Run ONE pass over `n` questions, print per-Q correctness, return pass/fail.

    Pass = at least one correct answer in `n` samples (catches catastrophic
    prompt/parse breakage; chance for 4-way MC over 5 questions is ~76%).
    """
    logger.info("=" * 70)
    logger.info(f"SANITY CHECK: {model_name}, K=1, N={n}")
    logger.info("=" * 70)
    C, metrics = run_within_model(
        model_name, questions, K_target=1, sanity_n=n,
    )
    logger.info(f"Per-question correctness: {C[0].tolist()}")
    logger.info(f"Sanity accuracy: {metrics['mean_accuracy']:.3f}")
    if metrics["mean_accuracy"] <= 0.0:
        logger.error("SANITY FAILED: 0% accuracy. Check prompt/tokenization/parse.")
        return False
    logger.info("SANITY PASSED.")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run only the 5-Q sanity check on Qwen, then exit.")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Skip the sanity check (for resumed runs).")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Override the default MODELS list.")
    args = ap.parse_args()

    models = args.models if args.models else list(MODELS)

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("WITHIN-MODEL HELLASWAG BATCH")
    logger.info(f"K={K}, N={N}, τ={TEMPERATURE}, models={models}")
    logger.info("=" * 70)

    questions = load_hellaswag(n=N, seed=SEED, cache_path=QUESTIONS_PATH)
    summary = stratification_summary(questions)
    logger.info(f"Loaded {summary['n_examples']} questions across {summary['n_strata']} strata.")
    logger.info(f"Top strata: {summary['top_strata'][:5]}")

    # Sanity check on Qwen (or whichever is first in `models`).
    if not args.skip_sanity:
        ok = sanity_check(models[0], questions, n=5)
        if not ok:
            logger.error("Aborting: sanity check failed.")
            sys.exit(2)
        if args.sanity_only:
            logger.info("--sanity-only set; exiting after sanity check.")
            sys.exit(0)

    all_metrics = []
    for idx, model_name in enumerate(models):
        logger.info("=" * 70)
        logger.info(f"[{idx+1}/{len(models)}] FULL RUN: {model_name}")
        logger.info("=" * 70)
        try:
            _, metrics = run_within_model(model_name, questions, K_target=K)
            all_metrics.append(metrics)
            logger.info(
                f"  -> acc={metrics['mean_accuracy']:.3f}  "
                f"borderline={metrics['n_borderline']}/{metrics['N']}  "
                f"infer={metrics['inference_time_sec']:.0f}s"
            )
        except Exception as e:
            logger.exception(f"FAILED {model_name}: {e}")
            all_metrics.append({
                "model": model_name, "benchmark": "hellaswag",
                "status": "failed", "error": str(e)[:500],
            })

        with open(METRICS_PATH, "w") as f:
            json.dump({
                "per_model": all_metrics,
                "K": K, "N": N, "temperature": TEMPERATURE,
                "benchmark": "hellaswag",
                "elapsed_min": round((time.time() - t_start) / 60, 1),
                "stratification": summary,
            }, f, indent=2)

    total = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info(f"DONE. Total time: {total:.1f} min")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
