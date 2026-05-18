"""
GSM8K Experiment: Does reasoning produce structured stochastic variation?
==========================================================================
Self-consistency was designed for chain-of-thought reasoning on tasks like
GSM8K. Our MMLU finding (d_within ≈ 0) might be specific to factual recall.
Does CoT reasoning produce structured variation where direct answering doesn't?

If d_within ≈ 0 on GSM8K too: temperature is universally shallow.
If d_within > 0 on GSM8K: temperature probes reasoning structure but not factual structure.
Either outcome strengthens the paper.
"""

import gc
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stats import marchenko_pastur_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Answer Extraction ────────────────────────────────────────────────────

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
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return _parse_number(match.group(1))

    # Pattern 2: "the answer is X" or "answer: X"
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?(-?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return _parse_number(match.group(1))

    # Pattern 3: "= X" near end
    match = re.search(r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$', text)
    if match:
        return _parse_number(match.group(1))

    # Pattern 4: last number in text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def _parse_number(s: str) -> float | None:
    """Parse a number string, handling commas."""
    s = s.replace(',', '').strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def check_gsm8k_answer(extracted: float | None, gold: float) -> bool:
    """Check if extracted answer matches gold (within small tolerance)."""
    if extracted is None:
        return False
    return abs(extracted - gold) < 0.01


# ─── Dataset ───────────────────────────────────────────────────────────────

def load_gsm8k(n: int = 200) -> list[dict]:
    """Load GSM8K test questions."""
    from datasets import load_dataset
    logger.info(f"Loading GSM8K (n={n})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        # Extract gold answer from the "#### NUMBER" format
        answer_text = item["answer"]
        match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
        if match:
            gold = _parse_number(match.group(1))
        else:
            continue

        samples.append({
            "question": item["question"],
            "gold_answer": gold,
            "gold_solution": answer_text,
        })

    logger.info(f"Loaded {len(samples)} GSM8K questions")
    return samples


COT_PROMPT = """Solve the following math problem step by step. Show your work, then give the final answer after ####.

Question: {question}

Solution: Let's think step by step.
"""


# ─── Stochastic Passes ────────────────────────────────────────────────────

def run_gsm8k_passes(
    model_name: str,
    samples: list[dict],
    K: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    device: str = "cuda",
):
    """Run K stochastic CoT passes on GSM8K questions."""
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

    N = len(samples)
    correctness = np.zeros((K, N), dtype=np.int8)
    extracted_answers = [[None] * N for _ in range(K)]
    raw_outputs = [[None] * N for _ in range(K)]

    for k in tqdm(range(K), desc="CoT passes"):
        for i, sample in enumerate(samples):
            prompt = COT_PROMPT.format(question=sample["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            extracted = extract_number(response)
            correct = check_gsm8k_answer(extracted, sample["gold_answer"])

            correctness[k, i] = int(correct)
            extracted_answers[k][i] = extracted
            raw_outputs[k][i] = response[:500]  # truncate for storage

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return correctness, extracted_answers, raw_outputs


# ─── Analysis ──────────────────────────────────────────────────────────────

def answer_diversity_stats(extracted_answers, correctness, samples):
    """Compute answer diversity statistics for GSM8K."""
    K = len(extracted_answers)
    N = len(samples)

    unique_counts = []
    wrong_answer_concentrations = []

    for i in range(N):
        answers = [extracted_answers[k][i] for k in range(K)]
        answers_valid = [a for a in answers if a is not None]

        if answers_valid:
            unique_counts.append(len(set(answers_valid)))
        else:
            unique_counts.append(0)

        # Wrong answer concentration (like MMLU analysis)
        wrong_answers = [
            extracted_answers[k][i] for k in range(K)
            if correctness[k, i] == 0 and extracted_answers[k][i] is not None
        ]
        if wrong_answers:
            from collections import Counter
            counts = Counter(wrong_answers)
            most_common_count = counts.most_common(1)[0][1]
            wrong_answer_concentrations.append(most_common_count / len(wrong_answers))

    return {
        "mean_unique_answers": float(np.mean(unique_counts)),
        "median_unique_answers": float(np.median(unique_counts)),
        "max_unique_answers": int(np.max(unique_counts)),
        "mean_wrong_answer_concentration": float(np.mean(wrong_answer_concentrations)) if wrong_answer_concentrations else 0,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def run_gsm8k_experiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    K: int = 50,
    N: int = 200,
    temperature: float = 1.0,
    pilot: bool = False,
):
    """Run GSM8K stochastic probing experiment."""
    if pilot:
        K, N = 15, 30
        logger.info("=== PILOT MODE ===")

    tag = "pilot" if pilot else "full"

    print(f"\n{'#'*70}")
    print(f"# GSM8K EXPERIMENT ({tag})")
    print(f"# K={K}, N={N}, τ={temperature}")
    print(f"{'#'*70}\n")

    t_start = time.time()

    # Load data
    samples = load_gsm8k(N)
    N = len(samples)

    # Run stochastic passes
    correctness, extracted_answers, raw_outputs = run_gsm8k_passes(
        model_name, samples, K=K, temperature=temperature,
    )

    # Save raw data
    slug = f"gsm8k_{tag}"
    np.save(RESULTS_DIR / f"{slug}_correctness.npy", correctness)
    with open(RESULTS_DIR / f"{slug}_answers.json", "w") as f:
        json.dump({
            "extracted": extracted_answers,
            "raw": raw_outputs,
        }, f)

    # Statistics
    flip_rates = correctness.mean(axis=0)
    accuracy = flip_rates.mean()
    n_borderline = ((flip_rates > 0.05) & (flip_rates < 0.95)).sum()
    n_always_correct = (flip_rates == 1.0).sum()
    n_always_wrong = (flip_rates == 0.0).sum()

    # Answer extraction success rate
    n_extracted = sum(
        1 for k in range(K) for i in range(N)
        if extracted_answers[k][i] is not None
    )
    extract_rate = n_extracted / (K * N)

    logger.info(f"\n{'='*60}")
    logger.info(f"GSM8K RESULTS ({tag})")
    logger.info(f"{'='*60}")
    logger.info(f"K={K}, N={N}, τ={temperature}")
    logger.info(f"Answer extraction rate: {extract_rate:.1%}")
    logger.info(f"Mean accuracy: {accuracy:.3f}")
    logger.info(f"Always correct: {n_always_correct} ({n_always_correct/N:.1%})")
    logger.info(f"Always wrong: {n_always_wrong} ({n_always_wrong/N:.1%})")
    logger.info(f"Borderline: {n_borderline} ({n_borderline/N:.1%})")

    # Answer diversity
    div_stats = answer_diversity_stats(extracted_answers, correctness, samples)
    logger.info(f"Mean unique answers per question: {div_stats['mean_unique_answers']:.1f}")
    logger.info(f"Wrong-answer concentration: {div_stats['mean_wrong_answer_concentration']:.3f}")

    # MP test (if enough borderline questions)
    if n_borderline >= 15:
        mask = (flip_rates > 0.05) & (flip_rates < 0.95)
        C_b = correctness[:, mask]
        mp_result = marchenko_pastur_test(C_b, label="GSM8K")
        n_signal = mp_result["n_signal"]
        lam_max = mp_result["top_10_eigenvalues"][0]
        lam_plus = mp_result["lambda_plus"]
        logger.info(f"MP test: λ_max={lam_max:.2f}, λ+={lam_plus:.2f}, signal={n_signal}")

        if n_signal == 0:
            logger.info(">>> d_within ≈ 0 on GSM8K! Temperature is shallow EVEN for reasoning.")
        else:
            logger.info(f">>> d_within ≈ {n_signal} on GSM8K! Reasoning DOES produce structure!")
    else:
        n_signal = -1
        lam_max = lam_plus = 0
        logger.info(f"Too few borderline questions ({n_borderline}) for MP test")

    elapsed = time.time() - t_start
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info(f"{'='*60}")

    # Comparison
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON: MMLU vs GSM8K")
    logger.info(f"{'='*60}")
    logger.info(f"  {'':>25} {'MMLU':>10} {'GSM8K':>10}")
    logger.info(f"  {'Task type':>25} {'Recall':>10} {'Reasoning':>10}")
    logger.info(f"  {'Accuracy':>25} {'0.712':>10} {accuracy:>10.3f}")
    logger.info(f"  {'Borderline':>25} {'121':>10} {n_borderline:>10}")
    logger.info(f"  {'Unique answers/Q':>25} {'4 (A-D)':>10} {div_stats['mean_unique_answers']:>10.1f}")
    logger.info(f"  {'Wrong-ans concentration':>25} {'0.874':>10} {div_stats['mean_wrong_answer_concentration']:>10.3f}")
    logger.info(f"  {'MP signal':>25} {'0':>10} {n_signal:>10}")
    logger.info(f"{'='*60}")

    # Save results
    results = {
        "tag": tag,
        "model": model_name,
        "K": K, "N": N, "temperature": temperature,
        "extract_rate": extract_rate,
        "accuracy": float(accuracy),
        "n_borderline": int(n_borderline),
        "n_always_correct": int(n_always_correct),
        "n_always_wrong": int(n_always_wrong),
        "mp_n_signal": n_signal,
        "mp_lambda_max": lam_max,
        "mp_lambda_plus": lam_plus,
        "answer_diversity": div_stats,
        "elapsed_minutes": elapsed / 60,
    }
    with open(RESULTS_DIR / f"{slug}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    pilot = "--pilot" in sys.argv
    run_gsm8k_experiment(pilot=pilot)
