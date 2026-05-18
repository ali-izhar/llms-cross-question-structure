"""
Upgrade A: Across-Model Dimensionality on the Same Questions
==============================================================
Run 5 different models deterministically (temperature=0) on the SAME
500 MMLU questions used in the stochastic experiment. Build a 5×500
correctness matrix. Compare its structure to the 100×500 stochastic matrix.

If d_across > 0 on just 5 models while d_within = 0 on 100 stochastic passes,
temperature sampling ≠ model diversity. They probe different things.
"""

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmlu_utils import normalize_answer, check_answer, build_mmlu_prompt, load_questions_from_saved

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_single_model_deterministic(
    model_name: str,
    questions: list[dict],
    device: str = "cuda",
) -> np.ndarray:
    """Run one model deterministically on all questions. Return N-dim binary vector."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    N = len(questions)
    correctness = np.zeros(N, dtype=np.int8)

    for i, q in enumerate(tqdm(questions, desc=model_name.split("/")[-1])):
        prompt = build_mmlu_prompt(q["question"], q["choices"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,  # deterministic
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        correct = (
            answer.upper().startswith(q["correct_letter"])
            or check_answer(answer, q["correct_text"])
        )
        correctness[i] = int(correct)

    acc = correctness.mean()
    logger.info(f"{model_name}: accuracy = {acc:.3f}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return correctness


def run_ensemble_comparison():
    """Run 5 models deterministically on the same questions."""

    # Load the SAME questions from the stochastic experiment
    questions = load_questions_from_saved(RESULTS_DIR / "questions_Qwen_Qwen2.5-7B-Instruct.json")
    logger.info(f"Matched {len(questions)} questions from stochastic experiment")

    # Models to compare (diverse families, all ungated)
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    N = len(questions)
    M = len(models)
    C_ensemble = np.zeros((M, N), dtype=np.int8)

    for j, model_name in enumerate(models):
        C_ensemble[j] = run_single_model_deterministic(model_name, questions)

    # Save
    np.save(RESULTS_DIR / "ensemble_correctness_matrix.npy", C_ensemble)
    with open(RESULTS_DIR / "ensemble_models.json", "w") as f:
        json.dump(models, f, indent=2)

    # Quick analysis: across-model covariance
    logger.info(f"\n{'='*60}")
    logger.info("ACROSS-MODEL ANALYSIS")
    logger.info(f"{'='*60}")

    accs = C_ensemble.mean(axis=1)
    for j, m in enumerate(models):
        logger.info(f"  {m:<45} acc={accs[j]:.3f}")

    # Agreement analysis
    all_agree_correct = (C_ensemble.sum(axis=0) == M).sum()
    all_agree_wrong = (C_ensemble.sum(axis=0) == 0).sum()
    any_disagree = N - all_agree_correct - all_agree_wrong

    logger.info(f"\n  All {M} models correct: {all_agree_correct} ({all_agree_correct/N:.1%})")
    logger.info(f"  All {M} models wrong:   {all_agree_wrong} ({all_agree_wrong/N:.1%})")
    logger.info(f"  Disagreement:          {any_disagree} ({any_disagree/N:.1%})")

    # Pairwise agreement (Pearson correlation between model correctness vectors)
    R_models = np.corrcoef(C_ensemble)
    logger.info(f"\n  Pairwise model correlations:")
    for i in range(M):
        for j in range(i+1, M):
            logger.info(f"    {models[i].split('/')[-1]:>30} × {models[j].split('/')[-1]:<30} r={R_models[i,j]:.3f}")

    # "Across-model dimensionality" on disagreement questions
    disagree_mask = (C_ensemble.sum(axis=0) > 0) & (C_ensemble.sum(axis=0) < M)
    C_disagree = C_ensemble[:, disagree_mask]
    n_disagree = disagree_mask.sum()

    logger.info(f"\n  Disagreement questions: {n_disagree}")

    if n_disagree > 0 and M > 2:
        # Correlation between QUESTIONS across models (transpose)
        R_questions = np.corrcoef(C_disagree.T)
        R_questions = np.nan_to_num(R_questions, nan=0.0)
        eig = np.sort(np.linalg.eigvalsh(R_questions))[::-1]

        logger.info(f"  Top 5 eigenvalues (question covariance across models): {[f'{v:.2f}' for v in eig[:5]]}")

        # With only M=5 models, the rank is at most 4
        # So we look at whether the top eigenvalue dominates (d_across≈1)
        # or variance is spread (d_across>1)
        if eig[0] > 0:
            concentration = eig[0] / eig.sum()
            logger.info(f"  Top eigenvalue concentration: {concentration:.1%}")
            if concentration > 0.5:
                logger.info(f"  >>> d_across ≈ 1 (first eigenvalue dominates)")
            else:
                effective_rank = np.exp(-np.sum((eig[eig>0]/eig[eig>0].sum()) * np.log(eig[eig>0]/eig[eig>0].sum())))
                logger.info(f"  >>> d_across ≈ {effective_rank:.1f} (effective rank)")

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON: d_within vs d_across")
    logger.info(f"{'='*60}")
    logger.info(f"  d_within (stochastic, K=100, same model): ~0 (no signal above MP noise)")
    logger.info(f"  d_across (deterministic, M={M} models):   see eigenvalue analysis above")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_ensemble_comparison()
