"""
Mistral Replication: Does d_within ≈ 0 replicate on a different model family?
==============================================================================
Same protocol as primary experiment but on Mistral-7B-Instruct-v0.3.
Uses the SAME 500 MMLU questions for direct comparison.
"""

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmlu_utils import normalize_answer, check_answer, build_mmlu_prompt, load_questions_from_saved
from stats import marchenko_pastur_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_stochastic(model_name, questions, K=100, temperature=1.0, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    logger.info("Model loaded.")

    N = len(questions)
    C = np.zeros((K, N), dtype=np.int8)

    for k in tqdm(range(K), desc=f"Pass"):
        for i, q in enumerate(questions):
            prompt = build_mmlu_prompt(q["question"], q["choices"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=True,
                    temperature=temperature,
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

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return C


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model_slug = model_name.replace("/", "_")

    questions = load_questions_from_saved(RESULTS_DIR / "questions_Qwen_Qwen2.5-7B-Instruct.json")
    N = len(questions)
    logger.info(f"Loaded {N} questions (same as Qwen experiment)")

    # Run K=100 stochastic passes
    C = run_stochastic(model_name, questions, K=100, temperature=1.0)

    # Save
    np.save(RESULTS_DIR / f"correctness_matrix_{model_slug}.npy", C)

    # Compute stats
    flip_rates = C.mean(axis=0)
    accuracy = flip_rates.mean()
    n_borderline = ((flip_rates > 0.05) & (flip_rates < 0.95)).sum()
    n_always_correct = (flip_rates == 1.0).sum()
    n_always_wrong = (flip_rates == 0.0).sum()

    # Save question metadata
    meta = []
    for i, q in enumerate(questions):
        meta.append({
            "index": i,
            "question": q["question"][:100],
            "subject": q["subject"],
            "correct_letter": q["correct_letter"],
            "flip_rate": float(flip_rates[i]),
            "always_correct": bool(flip_rates[i] == 1.0),
            "always_wrong": bool(flip_rates[i] == 0.0),
            "borderline": bool(0.05 < flip_rates[i] < 0.95),
        })
    with open(RESULTS_DIR / f"questions_{model_slug}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # MP test
    mask = (flip_rates > 0.05) & (flip_rates < 0.95)
    if mask.sum() >= 20:
        C_b = C[:, mask]
        mp_result = marchenko_pastur_test(C_b, label="Mistral")
        n_signal = mp_result["n_signal"]
        lam_max = mp_result["top_10_eigenvalues"][0]
        lam_plus = mp_result["lambda_plus"]
    else:
        n_signal, lam_max, lam_plus = -1, 0, 0

    logger.info(f"\n{'='*60}")
    logger.info(f"MISTRAL REPLICATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Model:           {model_name}")
    logger.info(f"Accuracy:        {accuracy:.3f}")
    logger.info(f"Always correct:  {n_always_correct} ({n_always_correct/N:.1%})")
    logger.info(f"Always wrong:    {n_always_wrong} ({n_always_wrong/N:.1%})")
    logger.info(f"Borderline:      {n_borderline} ({n_borderline/N:.1%})")
    logger.info(f"MP test:         λ_max={lam_max:.2f}, λ+={lam_plus:.2f}, signal={n_signal}")
    logger.info(f"{'='*60}")

    if n_signal == 0:
        logger.info(">>> d_within ≈ 0 REPLICATES on Mistral!")
        logger.info("    Temperature sampling is epistemically shallow across model families.")
    elif n_signal > 0:
        logger.info(f">>> d_within ≈ {n_signal} on Mistral — DIFFERENT from Qwen (d≈0)!")
        logger.info("    The finding may be model-specific.")
    else:
        logger.info(">>> Too few borderline questions for MP test.")

    # Comparison
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON: Qwen vs Mistral")
    logger.info(f"{'='*60}")
    logger.info(f"  {'':>20} {'Qwen2.5-7B':>15} {'Mistral-7B':>15}")
    logger.info(f"  {'Accuracy':>20} {'0.712':>15} {accuracy:>15.3f}")
    logger.info(f"  {'Borderline':>20} {'121':>15} {n_borderline:>15}")
    logger.info(f"  {'MP signal':>20} {'0':>15} {n_signal:>15}")
    logger.info(f"  {'d_within':>20} {'≈ 0':>15} {'≈ 0' if n_signal == 0 else f'≈ {n_signal}':>15}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
