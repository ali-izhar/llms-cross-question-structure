"""
Within-Model Epistemic Dimensionality: Main Experiment
========================================================
Probes a single LLM's epistemic structure through stochastic sampling.
Measures how many independent factors determine which questions flip
between correct and incorrect across stochastic forward passes.
"""

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmlu_utils import normalize_answer, check_answer, build_mmlu_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Dataset ───────────────────────────────────────────────────────────────

def load_mmlu(n_per_subject: int = 10, max_total: int = 500) -> list[dict]:
    """Load MMLU questions, stratified across subjects."""
    from datasets import load_dataset
    logger.info("Loading MMLU...")

    ds = load_dataset("cais/mmlu", "all", split="test")

    # Group by subject
    by_subject: dict[str, list] = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    # Sample n_per_subject from each, up to max_total
    samples = []
    choices_map = ["A", "B", "C", "D"]
    for subj in sorted(by_subject.keys()):
        items = by_subject[subj]
        np.random.seed(42)
        selected = np.random.choice(len(items), size=min(n_per_subject, len(items)), replace=False)
        for idx in selected:
            item = items[idx]
            correct_idx = item["answer"]
            correct_letter = choices_map[correct_idx]
            correct_text = item["choices"][correct_idx]
            samples.append({
                "question": item["question"],
                "choices": item["choices"],
                "correct_letter": correct_letter,
                "correct_text": correct_text,
                "subject": subj,
            })
            if len(samples) >= max_total:
                break
        if len(samples) >= max_total:
            break

    logger.info(f"Loaded {len(samples)} MMLU questions across {len(by_subject)} subjects")
    return samples


# ─── Stochastic Sampling ──────────────────────────────────────────────────

def run_stochastic_passes(
    model_name: str,
    samples: list[dict],
    K: int = 100,
    temperature: float = 1.0,
    batch_size: int = 1,
    device: str = "cuda",
) -> np.ndarray:
    """Run K stochastic forward passes, return K×N correctness matrix."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    logger.info(f"Model loaded. Running K={K} passes on N={len(samples)} questions...")

    N = len(samples)
    correctness = np.zeros((K, N), dtype=np.int8)

    for k in tqdm(range(K), desc="Stochastic passes"):
        for i, sample in enumerate(samples):
            prompt = build_mmlu_prompt(sample["question"], sample["choices"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,  # no nucleus filtering — pure temperature
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Check correctness: match letter or full text
            correct = (
                answer.upper().startswith(sample["correct_letter"])
                or check_answer(answer, sample["correct_text"])
            )
            correctness[k, i] = int(correct)

    # Unload
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return correctness


# ─── Main ──────────────────────────────────────────────────────────────────

def run_primary_experiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    K: int = 100,
    N: int = 500,
    temperature: float = 1.0,
):
    """Run the full primary experiment."""

    # Step 1: Load data
    samples = load_mmlu(n_per_subject=10, max_total=N)
    N = len(samples)

    # Step 2: Run stochastic passes
    C = run_stochastic_passes(model_name, samples, K=K, temperature=temperature)

    # Step 3: Save raw data
    model_slug = model_name.replace("/", "_")
    np.save(RESULTS_DIR / f"correctness_matrix_{model_slug}.npy", C)

    # Save question metadata
    metadata = []
    for i, s in enumerate(samples):
        flip_rate = C[:, i].mean()
        metadata.append({
            "index": i,
            "question": s["question"][:100],
            "subject": s["subject"],
            "correct_letter": s["correct_letter"],
            "flip_rate": float(flip_rate),
            "always_correct": bool(flip_rate == 1.0),
            "always_wrong": bool(flip_rate == 0.0),
            "borderline": bool(0.05 < flip_rate < 0.95),
        })

    with open(RESULTS_DIR / f"questions_{model_slug}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 4: Quick summary
    flip_rates = np.array([m["flip_rate"] for m in metadata])
    n_always_correct = (flip_rates == 1.0).sum()
    n_always_wrong = (flip_rates == 0.0).sum()
    n_borderline = ((flip_rates > 0.05) & (flip_rates < 0.95)).sum()
    accuracy = flip_rates.mean()

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"N questions:       {N}")
    logger.info(f"K passes:          {K}")
    logger.info(f"Temperature:       {temperature}")
    logger.info(f"Mean accuracy:     {accuracy:.3f}")
    logger.info(f"Always correct:    {n_always_correct} ({n_always_correct/N:.1%})")
    logger.info(f"Always wrong:      {n_always_wrong} ({n_always_wrong/N:.1%})")
    logger.info(f"Borderline:        {n_borderline} ({n_borderline/N:.1%})")
    logger.info(f"{'='*60}")

    if n_borderline < 50:
        logger.warning("KILL CONDITION: Fewer than 50 borderline questions!")
        logger.warning(f"Consider increasing temperature (current: {temperature})")
        return

    logger.info(f"Borderline questions sufficient ({n_borderline}). Proceed to analysis.py")


if __name__ == "__main__":
    run_primary_experiment()
