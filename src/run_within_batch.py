"""
Within-Model Stochastic Sampling: Batch Runner (K=100, chat-template methodology)
==================================================================================

For each model in MODELS, runs K=100 stochastic forward passes on the same
500 MMLU questions as the existing Qwen/Mistral-7B within-model experiments.

Uses tokenizer.apply_chat_template() for cross-model compatibility (matches
the ensemble_expanded methodology). Saves per-model correctness matrix so
each can be independently analyzed with the same MP + threshold pipeline.

Output files (per model):
- correctness_matrix_{model_slug}.npy   (K × N int8)
- questions_{model_slug}.json           (flip-rate metadata)

Checkpoints per-pass inside each model so interrupted runs can resume.
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT.parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent))
from mmlu_utils import check_answer, build_mmlu_prompt, load_questions_from_saved

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS = PROJECT_ROOT / "results"

K = 100
TEMPERATURE = 1.0

# Diverse within-model replications to complement existing Qwen-7B + Mistral-7B.
# Picked for family/size diversity to show d_within ≈ 0 is not Qwen/Mistral-specific.
MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",       # HF small, FineWeb-Edu corpus
    "microsoft/Phi-3-mini-4k-instruct",          # Microsoft synthetic-data training
    "meta-llama/Meta-Llama-3-8B-Instruct",       # Meta classical training
]


def build_chat_prompt(tokenizer, question: str, choices: list[str]) -> str:
    user_content = build_mmlu_prompt(question, choices)
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


def run_within_model(model_name: str, questions: list[dict]) -> tuple[np.ndarray, dict]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_slug = model_name.replace("/", "_")
    ckpt_path = RESULTS / f"correctness_matrix_{model_slug}.npy"
    progress_path = RESULTS / f"_progress_{model_slug}.json"

    t_load = time.time()
    logger.info(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # GB10 unified memory: force large GPU budget so device_map doesn't CPU-offload.
    max_memory = {0: "100GiB", "cpu": "20GiB"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
        )
    except Exception as e:
        logger.warning(f"Standard load failed, trying trust_remote_code=True: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
    model.eval()

    load_time = time.time() - t_load
    N = len(questions)

    # Resume support: if ckpt exists AND progress says how many passes done, skip them.
    if ckpt_path.exists() and progress_path.exists():
        C = np.load(ckpt_path)
        with open(progress_path) as f:
            progress = json.load(f)
        k_start = progress["k_done"]
        logger.info(f"Resuming {model_name}: {k_start}/{K} passes done")
        if C.shape != (K, N):
            logger.warning(f"Shape mismatch ({C.shape} vs ({K},{N})), restarting")
            C = np.zeros((K, N), dtype=np.int8)
            k_start = 0
    else:
        C = np.zeros((K, N), dtype=np.int8)
        k_start = 0

    t_infer = time.time()
    for k in tqdm(range(k_start, K), desc=model_name.split("/")[-1], file=sys.stdout):
        for i, q in enumerate(questions):
            prompt = build_chat_prompt(tokenizer, q["question"], q["choices"])
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            correct = (
                answer.upper().startswith(q["correct_letter"])
                or check_answer(answer, q["correct_text"])
            )
            C[k, i] = int(correct)

        # Checkpoint after each pass
        np.save(ckpt_path, C)
        with open(progress_path, "w") as f:
            json.dump({"k_done": k + 1, "K_target": K, "model": model_name}, f)

    infer_time = time.time() - t_infer

    flip_rates = C.mean(axis=0)
    metadata = []
    for i, q in enumerate(questions):
        metadata.append({
            "index": i,
            "question": q["question"][:100],
            "correct_letter": q["correct_letter"],
            "flip_rate": float(flip_rates[i]),
            "borderline": bool(0.05 < flip_rates[i] < 0.95),
        })
    with open(RESULTS / f"questions_{model_slug}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean progress marker on success
    if progress_path.exists():
        progress_path.unlink()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    metrics = {
        "model": model_name,
        "K": K,
        "N": N,
        "temperature": TEMPERATURE,
        "load_time_sec": round(load_time, 1),
        "inference_time_sec": round(infer_time, 1),
        "mean_accuracy": float(flip_rates.mean()),
        "n_always_correct": int((flip_rates == 1.0).sum()),
        "n_always_wrong": int((flip_rates == 0.0).sum()),
        "n_borderline": int(((flip_rates > 0.05) & (flip_rates < 0.95)).sum()),
    }
    return C, metrics


def main():
    t_start = time.time()

    questions = load_questions_from_saved(
        RESULTS / "questions_Qwen_Qwen2.5-7B-Instruct.json"
    )
    N = len(questions)
    logger.info(f"Within-model batch: {len(MODELS)} models, K={K}, N={N}, T={TEMPERATURE}")

    all_metrics = []
    for idx, model_name in enumerate(MODELS):
        logger.info("=" * 70)
        logger.info(f"[{idx+1}/{len(MODELS)}] {model_name}")
        logger.info("=" * 70)

        try:
            _, metrics = run_within_model(model_name, questions)
            all_metrics.append(metrics)
            logger.info(
                f"  -> acc={metrics['mean_accuracy']:.3f}, "
                f"borderline={metrics['n_borderline']}/{N}, "
                f"infer={metrics['inference_time_sec']:.0f}s"
            )
        except Exception as e:
            logger.error(f"FAILED {model_name}: {e}", exc_info=True)
            all_metrics.append({"model": model_name, "status": "failed", "error": str(e)[:500]})

        # Checkpoint metrics
        with open(RESULTS / "within_batch_metrics.json", "w") as f:
            json.dump({
                "per_model": all_metrics,
                "K": K, "N": N, "temperature": TEMPERATURE,
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }, f, indent=2)

    total = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info(f"DONE. Total time: {total:.1f} min")


if __name__ == "__main__":
    main()
