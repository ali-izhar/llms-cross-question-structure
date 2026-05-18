"""
GSM8K Medium-Scale Within-Model Run (K=30, N=100, tau=1.0)
==========================================================

Upgrades the pilot run (N=30, K=15 = 450 generations) to a substantially
larger dataset (N=100, K=30 = 3,000 generations, 6.7x more data).

Single model: Qwen/Qwen2.5-7B-Instruct
Chain-of-thought with max_new_tokens=128.

Outputs:
  - results/questions_gsm8k_medium.json               (100-question sample)
  - results/correctness_matrix_gsm8k_medium_Qwen_Qwen2.5-7B-Instruct.npy  (30x100 int8)
  - results/_progress_gsm8k_medium_Qwen_Qwen2.5-7B-Instruct.json          (checkpoint)
  - results/gsm8k_medium_console.log                   (progress)

GB10 hygiene: bf16, max_memory={0:'100GiB','cpu':'20GiB'}, inference_mode,
no .cpu()/.to() round-trips in the inner loop.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gsm8k_medium")

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ─── Experiment config ───────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
K = 30
N = 100
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256  # 128 is too short for CoT; pilot used 256 and got 43%
SEED = 42

QUESTIONS_PATH = RESULTS / "questions_gsm8k_medium.json"
MODEL_SLUG = MODEL_NAME.replace("/", "_")
CKPT_PATH = RESULTS / f"correctness_matrix_gsm8k_medium_{MODEL_SLUG}.npy"
PROGRESS_PATH = RESULTS / f"_progress_gsm8k_medium_{MODEL_SLUG}.json"


# ─── Answer extraction (from run_gsm8k.py) ──────────────────────────────


def _parse_number(s: str) -> float | None:
    """Parse a number string, handling commas."""
    s = s.replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """Extract the final numerical answer from a CoT response.

    Tries multiple patterns in order of specificity:
    1. #### [number] (GSM8K standard format)
    2. "the answer is [number]"
    3. "= [number]" at end
    4. Last number in the text
    """
    text = text.strip()

    # Pattern 1: #### number
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return _parse_number(match.group(1))

    # Pattern 2: "the answer is X" or "answer: X"
    match = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*\$?(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _parse_number(match.group(1))

    # Pattern 3: "= X" near end
    match = re.search(r"=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$", text)
    if match:
        return _parse_number(match.group(1))

    # Pattern 4: last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def check_gsm8k_answer(extracted: float | None, gold: float) -> bool:
    """Check if extracted answer matches gold (within small tolerance)."""
    if extracted is None:
        return False
    return abs(extracted - gold) < 0.01


# ─── Dataset ─────────────────────────────────────────────────────────────


def load_gsm8k(n: int = N, seed: int = SEED) -> list[dict]:
    """Load first n GSM8K test questions (deterministic order)."""
    from datasets import load_dataset

    logger.info(f"Loading GSM8K test split (first {n} questions, seed={seed})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        answer_text = item["answer"]
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
        if match:
            gold = _parse_number(match.group(1))
        else:
            logger.warning(f"Skipping question {i}: no #### in gold answer")
            continue

        samples.append({
            "index": i,
            "question": item["question"],
            "gold_answer": gold,
            "gold_solution": answer_text,
        })

    logger.info(f"Loaded {len(samples)} GSM8K questions.")
    return samples


# ─── Prompt ──────────────────────────────────────────────────────────────

COT_PROMPT = (
    'Solve this math problem step by step. Show your reasoning, '
    'then give the final numerical answer on the last line as "#### <number>".\n'
    "\n"
    "Question: {question}\n"
    "\n"
    "Solution:"
)


def build_chat_prompt(tokenizer, question: str) -> str:
    """Wrap prompt in chat template."""
    user_content = COT_PROMPT.format(question=question)
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


# ─── Model loading ───────────────────────────────────────────────────────


def load_model_and_tokenizer():
    """Load model with GB10 hygiene settings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t_load = time.time()
    logger.info(f"Loading tokenizer + model: {MODEL_NAME} (bf16, GB10 budget)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_memory = {0: "100GiB", "cpu": "20GiB"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=dtype, device_map="auto", max_memory=max_memory,
        )
    except Exception as e:
        logger.warning(f"Standard load failed, retrying with trust_remote_code=True: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=dtype, device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
    model.eval()

    # Sanity-check device map (warn on CPU-offload).
    try:
        offloaded = [
            name for name, dev in model.hf_device_map.items()
            if dev in ("cpu", "disk")
        ]
        if offloaded:
            logger.warning(
                f"WARNING: {len(offloaded)} modules CPU-offloaded "
                f"(first: {offloaded[:3]}). GB10 budget likely too tight."
            )
        else:
            logger.info("device_map all-GPU (no CPU offload). Good.")
    except Exception:
        pass

    load_time = time.time() - t_load
    logger.info(f"Model loaded in {load_time:.1f}s.")
    return model, tokenizer, load_time


# ─── Single generation ──────────────────────────────────────────────────


def generate_one(model, tokenizer, question: str) -> tuple[str, float | None, bool]:
    """Generate a single CoT response and extract the answer.

    Returns (response_text, extracted_number, None used as placeholder).
    """
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    extracted = extract_number(response)
    return response, extracted


# ─── Sanity check ────────────────────────────────────────────────────────


def sanity_check(model, tokenizer, questions: list[dict], n: int = 3) -> bool:
    """Run K=1 on first n questions. Must get at least 1/n correct to proceed."""
    logger.info("=" * 70)
    logger.info(f"SANITY CHECK: K=1, N={n}")
    logger.info("=" * 70)

    n_correct = 0
    for i, q in enumerate(questions[:n]):
        response, extracted = generate_one(model, tokenizer, q["question"])
        gold = q["gold_answer"]
        correct = check_gsm8k_answer(extracted, gold)
        n_correct += int(correct)
        logger.info(
            f"  Q{i}: gold={gold}  extracted={extracted}  "
            f"correct={'YES' if correct else 'NO'}"
        )
        logger.info(f"       response: {response[:200]}...")

    acc = n_correct / n
    logger.info(f"Sanity accuracy: {n_correct}/{n} = {acc:.3f}")

    if n_correct < 1:
        logger.error(
            "SANITY FAILED: 0 correct answers. "
            "Check prompt/tokenization/answer extraction."
        )
        return False

    logger.info("SANITY PASSED.")
    return True


# ─── Main run ────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("GSM8K MEDIUM-SCALE WITHIN-MODEL RUN")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"K={K}, N={N}, tau={TEMPERATURE}, max_new_tokens={MAX_NEW_TOKENS}")
    logger.info(f"Total generations: {K * N}")
    logger.info("=" * 70)

    # ── Load questions ──
    if QUESTIONS_PATH.exists():
        logger.info(f"Loading cached questions from {QUESTIONS_PATH}")
        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} cached questions.")
    else:
        questions = load_gsm8k(n=N, seed=SEED)
        with open(QUESTIONS_PATH, "w") as f:
            json.dump(questions, f, indent=2)
        logger.info(f"Saved {len(questions)} questions to {QUESTIONS_PATH}")

    N_actual = len(questions)
    if N_actual < N:
        logger.warning(f"Only {N_actual} questions available (requested {N}).")

    # ── Load model ──
    model, tokenizer, load_time = load_model_and_tokenizer()

    # ── Sanity check ──
    ok = sanity_check(model, tokenizer, questions, n=3)
    if not ok:
        logger.error("Aborting: sanity check failed.")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(2)

    # ── Resume support ──
    if CKPT_PATH.exists() and PROGRESS_PATH.exists():
        C = np.load(CKPT_PATH)
        with open(PROGRESS_PATH) as f:
            progress = json.load(f)
        k_start = int(progress.get("k_done", 0))
        if C.shape != (K, N_actual):
            logger.warning(
                f"Shape mismatch ({C.shape} vs {(K, N_actual)}), restarting."
            )
            C = np.zeros((K, N_actual), dtype=np.int8)
            k_start = 0
        else:
            logger.info(f"Resuming at pass {k_start}/{K}.")
    else:
        C = np.zeros((K, N_actual), dtype=np.int8)
        k_start = 0

    # ── Main loop ──
    logger.info("=" * 70)
    logger.info(f"STARTING MAIN LOOP: passes {k_start}..{K-1}")
    logger.info("=" * 70)

    t_infer = time.time()

    for k in range(k_start, K):
        t_pass = time.time()
        n_correct_pass = 0

        for i, q in enumerate(questions):
            _, extracted = generate_one(model, tokenizer, q["question"])
            correct = check_gsm8k_answer(extracted, q["gold_answer"])
            C[k, i] = int(correct)
            n_correct_pass += int(correct)

        # Checkpoint after every pass.
        np.save(CKPT_PATH, C)
        with open(PROGRESS_PATH, "w") as f:
            json.dump(
                {
                    "k_done": k + 1,
                    "K_target": K,
                    "N": N_actual,
                    "model": MODEL_NAME,
                    "last_pass_acc": n_correct_pass / max(N_actual, 1),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )

        # Timing & ETA.
        elapsed = time.time() - t_infer
        passes_done = (k + 1) - k_start
        per_pass = elapsed / max(passes_done, 1)
        remaining = (K - (k + 1)) * per_pass
        eta = datetime.now() + timedelta(seconds=remaining)

        logger.info(
            f"[{MODEL_NAME}] pass {k+1}/{K}  "
            f"acc={n_correct_pass/max(N_actual,1):.3f}  "
            f"pass_time={time.time()-t_pass:.1f}s  "
            f"per_pass_avg={per_pass:.1f}s  "
            f"elapsed={elapsed/60:.1f}min  "
            f"ETA={eta.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ── Summary ──
    infer_time = time.time() - t_infer
    total_time = time.time() - t_start

    flip_rates = C.mean(axis=0)
    mean_acc = float(flip_rates.mean())
    n_borderline = int(((flip_rates > 0.05) & (flip_rates < 0.95)).sum())
    n_always_correct = int((flip_rates == 1.0).sum())
    n_always_wrong = int((flip_rates == 0.0).sum())

    logger.info("=" * 70)
    logger.info("GSM8K MEDIUM RUN COMPLETE")
    logger.info(f"  Model:          {MODEL_NAME}")
    logger.info(f"  K={K}, N={N_actual}, tau={TEMPERATURE}")
    logger.info(f"  Mean accuracy:  {mean_acc:.3f}")
    logger.info(f"  Always correct: {n_always_correct}/{N_actual}")
    logger.info(f"  Always wrong:   {n_always_wrong}/{N_actual}")
    logger.info(f"  Borderline:     {n_borderline}/{N_actual}")
    logger.info(f"  Inference time: {infer_time/60:.1f} min")
    logger.info(f"  Total time:     {total_time/60:.1f} min")
    logger.info(f"  Checkpoint:     {CKPT_PATH}")
    logger.info("=" * 70)

    # Clean up progress file on successful completion.
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Done. Model unloaded, GPU memory freed.")


if __name__ == "__main__":
    main()
