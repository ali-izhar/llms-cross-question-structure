"""
Temperature Sweep: Does within-model dimensionality change with temperature?
==============================================================================
Run K=100 stochastic passes at multiple temperatures on the same 500 MMLU
questions. Compare d_within across temperatures.

If d_within stays ~0 at all temperatures: temperature doesn't matter,
the variation is always unstructured.

If d_within increases with temperature: more noise reveals more structure
(interesting — means structure exists but is hidden at low temperatures).

If d_within peaks at intermediate temperature: there's an optimal
temperature for probing epistemic structure.
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


def run_at_temperature(model, tokenizer, questions, K, temperature, device="cuda"):
    """Run K passes at a given temperature, return K×N correctness matrix."""
    N = len(questions)
    C = np.zeros((K, N), dtype=np.int8)

    for k in tqdm(range(K), desc=f"τ={temperature}"):
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

    return C


def run_sweep():
    """Run temperature sweep."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    questions = load_questions_from_saved(RESULTS_DIR / "questions_Qwen_Qwen2.5-7B-Instruct.json")
    N = len(questions)
    logger.info(f"Loaded {N} questions")

    # Temperatures to test
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    K = 50  # fewer passes per temperature for speed (50 × 6 = 300 total)

    # Load model once
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    logger.info("Model loaded.")

    results = []

    for tau in temperatures:
        logger.info(f"\n{'='*60}")
        logger.info(f"Temperature τ = {tau}")
        logger.info(f"{'='*60}")

        C = run_at_temperature(model, tokenizer, questions, K=K, temperature=tau)

        # Stats
        flip_rates = C.mean(axis=0)
        accuracy = flip_rates.mean()
        n_borderline = ((flip_rates > 0.05) & (flip_rates < 0.95)).sum()
        n_always_correct = (flip_rates == 1.0).sum()
        n_always_wrong = (flip_rates == 0.0).sum()

        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Always correct: {n_always_correct}")
        logger.info(f"Always wrong: {n_always_wrong}")
        logger.info(f"Borderline: {n_borderline}")

        # MP test on borderline questions
        mask = (flip_rates > 0.05) & (flip_rates < 0.95)
        if mask.sum() >= 20:
            C_b = C[:, mask]
            mp_result = marchenko_pastur_test(C_b, label=f"tau={tau}")
            n_signal = mp_result["n_signal"]
            lam_max = mp_result["top_10_eigenvalues"][0]
            lam_plus = mp_result["lambda_plus"]
            logger.info(f"MP test: λ_max={lam_max:.2f}, λ+={lam_plus:.2f}, signal={n_signal}")
        else:
            n_signal, lam_max, lam_plus = 0, 0, 0
            logger.info(f"Too few borderline questions ({mask.sum()}) for MP test")

        result = {
            "temperature": tau,
            "K": K,
            "accuracy": float(accuracy),
            "n_borderline": int(n_borderline),
            "n_always_correct": int(n_always_correct),
            "n_always_wrong": int(n_always_wrong),
            "mp_n_signal": n_signal,
            "mp_lambda_max": lam_max,
            "mp_lambda_plus": lam_plus,
        }
        results.append(result)

        # Save per-temperature matrix
        np.save(RESULTS_DIR / f"correctness_matrix_tau{tau}.npy", C)

    # Unload
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Save combined results
    with open(RESULTS_DIR / "temperature_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("TEMPERATURE SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'τ':>6} {'Acc':>6} {'Borderline':>10} {'λ_max':>8} {'λ+':>8} {'Signal':>8}")
    print(f"{'─'*52}")
    for r in results:
        print(f"{r['temperature']:>6.1f} {r['accuracy']:>6.3f} {r['n_borderline']:>10} "
              f"{r['mp_lambda_max']:>8.2f} {r['mp_lambda_plus']:>8.2f} {r['mp_n_signal']:>8}")
    print(f"{'='*80}")

    # Key finding
    any_signal = any(r["mp_n_signal"] > 0 for r in results)
    if any_signal:
        signal_temps = [r["temperature"] for r in results if r["mp_n_signal"] > 0]
        print(f"\n>>> SIGNAL DETECTED at temperatures: {signal_temps}")
        print("    Structure EXISTS but only at certain temperatures!")
    else:
        print(f"\n>>> NO SIGNAL at any temperature.")
        print("    Temperature variation is ALWAYS unstructured noise.")


if __name__ == "__main__":
    run_sweep()
