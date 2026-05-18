"""
Full-Scale Semantic Structure Experiment
=========================================
Same as the pilot but at full scale: K=100 passes, all 500 MMLU questions.
Uses the same infrastructure from run_semantic_pilot.py.

SCOPE: this is a SINGLE-MODEL within-model experiment on
Qwen/Qwen2.5-7B-Instruct. It is NOT the across-model experiment (§3.3/§4),
which uses 24 independent models and lives elsewhere in the pipeline.

Saved files (prefix "semantic_full_") and their shapes:
  semantic_full_correctness.npy      int8, shape (K=100, N=500)
      Per-pass binary correctness for Qwen2.5-7B-Instruct on 500 MMLU qs.
      Conceptually equivalent to correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy
      but from an independent K=100 run that ALSO saved answer letters.
  semantic_full_answer_choices.npy   int8, shape (K=100, N=500), values in {0,1,2,3}
      Per-pass A/B/C/D choice. This is the only within-model file that
      preserves answer identity (the correctness_matrix_*.npy files do not).
      Used by scripts/answer_identity_replication.py for the 4-category
      independence test that the other 4 models cannot run.
  semantic_full_raw_answers.json     list[list[str]], 100 x 500 raw text.
  semantic_full_questions.json       per-question metadata for the N=500 set
      (index, subject, correct_letter, flip_rate, borderline).

So the leading dim is K (passes), NOT "models". The "full" in the filename
refers to the full N=500 MMLU sample vs the 114-question pilot, not to a
multi-model sweep.
"""

import sys
import numpy as np
import time
import json
import logging
from pathlib import Path

from mmlu_utils import load_questions_from_saved

# Import everything from the pilot
from run_semantic_pilot import (
    load_pilot_questions,
    run_semantic_passes,
    marchenko_pastur_test,
    parallel_analysis,
    answer_statistics,
    permutation_null_test,
    one_hot_encode,
    build_cramers_v_matrix,
    build_mi_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"


def run_full(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    K: int = 100,
    N: int = 500,
    temperature: float = 1.0,
):
    """Run the full-scale semantic structure experiment."""
    print(f"\n{'#'*70}")
    print(f"# FULL-SCALE SEMANTIC EXPERIMENT")
    print(f"# Model: {model_name}")
    print(f"# K={K} stochastic passes, N={N} questions, T={temperature}")
    print(f"{'#'*70}\n")

    t_start = time.time()

    # Step 1: Load ALL questions (not just borderline subset)
    logger.info("Step 1: Loading questions...")
    samples = load_questions_from_saved(RESULTS_DIR / "questions_Qwen_Qwen2.5-7B-Instruct.json")

    if N < len(samples):
        samples = samples[:N]
    N = len(samples)
    logger.info(f"Loaded {N} questions")

    # Step 2: Run stochastic passes
    logger.info(f"Step 2: Running K={K} stochastic passes...")
    correctness, answer_choices, raw_answers = run_semantic_passes(
        model_name, samples, K=K, temperature=temperature,
    )

    # Save raw data
    np.save(RESULTS_DIR / "semantic_full_correctness.npy", correctness)
    np.save(RESULTS_DIR / "semantic_full_answer_choices.npy", answer_choices)
    with open(RESULTS_DIR / "semantic_full_raw_answers.json", "w") as f:
        json.dump(raw_answers, f)

    # Save question metadata
    q_meta = []
    for i, s in enumerate(samples):
        flip_rate = float(correctness[:, i].mean())
        q_meta.append({
            "index": i,
            "question": s["question"][:100],
            "subject": s["subject"],
            "correct_letter": s["correct_letter"],
            "flip_rate": flip_rate,
            "borderline": bool(0.05 < flip_rate < 0.95),
        })
    with open(RESULTS_DIR / "semantic_full_questions.json", "w") as f:
        json.dump(q_meta, f, indent=2)

    elapsed_inference = time.time() - t_start
    logger.info(f"Inference complete in {elapsed_inference/3600:.1f} hours")

    # Step 3: Analysis (on borderline questions only)
    logger.info("Step 3: Running analysis...")
    t_analysis = time.time()

    flip_rates = correctness.mean(axis=0)
    borderline_mask = (flip_rates > 0.05) & (flip_rates < 0.95)
    N_b = borderline_mask.sum()
    logger.info(f"Borderline questions: {N_b} out of {N}")

    if N_b < 20:
        logger.warning("Too few borderline questions for analysis!")
        return

    C_b = correctness[:, borderline_mask]
    A_b = answer_choices[:, borderline_mask]

    # 3a: Binary MP test
    logger.info("  Binary MP test...")
    binary_mp = marchenko_pastur_test(C_b, label="Binary")

    # 3b: Binary parallel analysis
    logger.info("  Binary parallel analysis...")
    binary_pa = parallel_analysis(C_b, label="Binary", n_iter=100)

    # 3c: Descriptive statistics
    logger.info("  Descriptive statistics...")
    borderline_meta = [q for q in q_meta if q["borderline"]]
    desc = answer_statistics(A_b, C_b, borderline_meta)

    # 3d: Cramér's V permutation test
    logger.info("  Cramér's V permutation test...")
    cramers = permutation_null_test(A_b, metric="cramers_v", n_permutations=200)

    # 3e: MI permutation test
    logger.info("  MI permutation test...")
    mi = permutation_null_test(A_b, metric="mi", n_permutations=200)

    # 3f: One-hot parallel analysis
    logger.info("  One-hot parallel analysis...")
    A_onehot = one_hot_encode(A_b)
    onehot = parallel_analysis(A_onehot, label="One-hot", n_iter=100)

    elapsed_analysis = time.time() - t_analysis
    logger.info(f"Analysis complete in {elapsed_analysis:.0f} seconds")

    # Step 4: Summary
    print(f"\n{'='*70}")
    print(f"FULL-SCALE SEMANTIC EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"Model:           {model_name}")
    print(f"K:               {K}")
    print(f"N:               {N}")
    print(f"N_borderline:    {N_b}")
    print(f"Mean accuracy:   {flip_rates.mean():.3f}")
    print(f"Wrong-answer concentration: {desc['mean_wrong_answer_concentration']:.3f}")
    print(f"Mean answer entropy: {desc['mean_answer_entropy']:.3f} bits")
    print(f"")
    print(f"{'Method':<35} {'d':<10}")
    print(f"{'-'*45}")
    print(f"{'Binary MP test':<35} {binary_mp['n_signal']:<10}")
    print(f"{'Binary parallel analysis':<35} {binary_pa['d_within']:<10}")
    print(f"{'One-hot parallel analysis':<35} {onehot['d_within']:<10}")
    print(f"{'Cramér V (95th pctl)':<35} {cramers['d_95']:<10}")
    print(f"{'Cramér V (99th pctl)':<35} {cramers['d_99']:<10}")
    print(f"{'MI (95th pctl)':<35} {mi['d_95']:<10}")
    print(f"{'MI (99th pctl)':<35} {mi['d_99']:<10}")
    print(f"{'='*70}")
    print(f"Total time: {(time.time()-t_start)/3600:.1f} hours")

    # Save results
    results = {
        "model": model_name,
        "K": K,
        "N": N,
        "N_borderline": int(N_b),
        "temperature": temperature,
        "mean_accuracy": float(flip_rates.mean()),
        "inference_time_hours": elapsed_inference / 3600,
        "binary_mp": binary_mp,
        "binary_parallel": binary_pa,
        "onehot_parallel": onehot,
        "cramers_v": cramers,
        "mi": mi,
        "descriptive": {k: v for k, v in desc.items() if k != "per_question"},
    }

    with open(RESULTS_DIR / "semantic_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_DIR / 'semantic_full_results.json'}")


if __name__ == "__main__":
    run_full()
