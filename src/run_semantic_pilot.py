"""
Semantic Pilot Experiment: Does answer CONTENT have structure?
================================================================
Our main result: binary correctness (right/wrong) across K stochastic
passes has d_within ≈ 0 (no structure above Marchenko-Pastur noise).

Reviewer question: does the CONTENT of wrong answers have structure
even though binary correctness doesn't?

For MMLU, this is clean: each answer IS a semantic class (A/B/C/D).
We record WHICH letter the model chose (not just correct/incorrect),
then test whether answer-choice patterns across questions show
covariance structure above noise.

Key analysis:
  1. Binary d_within (reproducing main result on this subset)
  2. Semantic d_within using mutual information between answer-choice
     sequences (Cramér's V for categorical data)
  3. One-hot encoded correlation analysis as an alternative approach
"""

import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mmlu_utils import build_mmlu_prompt
from stats import marchenko_pastur_test, parallel_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "results" / "semantic_pilot_log.txt"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

LETTER_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}
INT_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_pilot_questions(n_questions: int = 50) -> list[dict]:
    """
    Load borderline MMLU questions from the existing experiment.
    Select questions with flip rates between 0.1 and 0.9 (maximally borderline).
    We need the FULL question data (with choices), so we reload from MMLU.
    """
    # Load existing metadata to identify borderline questions
    meta_path = RESULTS_DIR / "questions_Qwen_Qwen2.5-7B-Instruct.json"
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Filter to borderline questions (flip_rate between 0.1 and 0.9)
    borderline = [q for q in all_meta if 0.1 <= q["flip_rate"] <= 0.9]
    logger.info(f"Found {len(borderline)} borderline questions (flip_rate 0.1-0.9)")

    # Sort by how close to 0.5 (most interesting borderline)
    borderline.sort(key=lambda q: abs(q["flip_rate"] - 0.5))

    # Take the N most borderline
    selected_meta = borderline[:n_questions]
    selected_indices = {q["index"] for q in selected_meta}

    logger.info(f"Selected {len(selected_meta)} questions closest to 50% flip rate")
    logger.info(f"Flip rate range: {min(q['flip_rate'] for q in selected_meta):.2f} - "
                f"{max(q['flip_rate'] for q in selected_meta):.2f}")

    # Reload MMLU to get full question data (choices needed)
    from datasets import load_dataset
    logger.info("Loading MMLU dataset to get full question data...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Reconstruct the same sampling order as run_experiment.py
    by_subject: dict[str, list] = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    choices_map = ["A", "B", "C", "D"]
    all_samples = []
    for subj in sorted(by_subject.keys()):
        items = by_subject[subj]
        np.random.seed(42)
        sel = np.random.choice(len(items), size=min(10, len(items)), replace=False)
        for idx in sel:
            item = items[idx]
            correct_idx = item["answer"]
            correct_letter = choices_map[correct_idx]
            correct_text = item["choices"][correct_idx]
            all_samples.append({
                "question": item["question"],
                "choices": item["choices"],
                "correct_letter": correct_letter,
                "correct_text": correct_text,
                "subject": subj,
            })
            if len(all_samples) >= 500:
                break
        if len(all_samples) >= 500:
            break

    # Select the questions that match our borderline indices
    pilot_samples = []
    for idx in sorted(selected_indices):
        sample = all_samples[idx]
        meta = next(q for q in selected_meta if q["index"] == idx)
        sample["original_index"] = idx
        sample["original_flip_rate"] = meta["flip_rate"]
        pilot_samples.append(sample)

    logger.info(f"Loaded {len(pilot_samples)} pilot questions with full choice data")
    return pilot_samples


def extract_answer_letter(text: str) -> str | None:
    """Extract the answer letter from model output. Returns A/B/C/D or None."""
    text = text.strip().upper()
    # Direct letter match
    if text and text[0] in "ABCD":
        return text[0]
    # Look for "The answer is X" pattern
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    return None


# ─── Stochastic Sampling (saving answer letters) ────────────────────────────

def run_semantic_passes(
    model_name: str,
    samples: list[dict],
    K: int = 30,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """
    Run K stochastic passes, recording:
      1. Binary correctness matrix (K x N)
      2. Answer choice matrix (K x N) with integers 0-3 for A-D, -1 for unparseable
      3. Raw answer texts (K x N list of lists)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    N = len(samples)
    correctness = np.zeros((K, N), dtype=np.int8)
    answer_choices = np.full((K, N), -1, dtype=np.int8)  # -1 = unparseable
    raw_answers: list[list[str]] = []

    logger.info(f"Running K={K} stochastic passes on N={N} questions...")
    t_start = time.time()

    for k in tqdm(range(K), desc="Stochastic passes"):
        run_answers = []
        for i, sample in enumerate(samples):
            prompt = build_mmlu_prompt(sample["question"], sample["choices"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,  # pure temperature, no nucleus filtering
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            run_answers.append(answer_text)

            # Extract answer letter
            letter = extract_answer_letter(answer_text)
            if letter is not None:
                answer_choices[k, i] = LETTER_TO_INT[letter]
                # Check correctness
                correctness[k, i] = int(letter == sample["correct_letter"])
            else:
                # Unparseable — treat as incorrect
                correctness[k, i] = 0

        raw_answers.append(run_answers)

        # Progress reporting
        elapsed = time.time() - t_start
        rate = (k + 1) * N / elapsed
        eta = (K - k - 1) * N / rate if rate > 0 else 0
        if (k + 1) % 5 == 0:
            logger.info(f"  Pass {k+1}/{K} done. {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining. "
                        f"Rate: {rate:.1f} q/s")

    elapsed = time.time() - t_start
    logger.info(f"Completed {K} passes in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Report unparseable answers
    n_unparseable = (answer_choices == -1).sum()
    total = K * N
    logger.info(f"Unparseable answers: {n_unparseable}/{total} ({n_unparseable/total:.1%})")

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return correctness, answer_choices, raw_answers


# ─── Analysis Functions ──────────────────────────────────────────────────────

def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cramér's V between two categorical arrays."""
    # Remove entries where either is -1 (unparseable)
    mask = (x >= 0) & (y >= 0)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return 0.0

    # Build contingency table
    n_cats = max(x.max(), y.max()) + 1
    contingency = np.zeros((n_cats, n_cats), dtype=int)
    for xi, yi in zip(x, y):
        contingency[xi, yi] += 1

    # Remove zero rows/cols
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    keep_rows = row_sums > 0
    keep_cols = col_sums > 0
    contingency = contingency[np.ix_(keep_rows, keep_cols)]

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0

    n = contingency.sum()
    chi2 = 0.0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (contingency[i, j] - expected) ** 2 / expected

    k = min(contingency.shape) - 1
    if k == 0 or n <= 1:
        return 0.0

    v = np.sqrt(chi2 / (n * k))
    return float(v)


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two categorical arrays (in nats)."""
    mask = (x >= 0) & (y >= 0)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return 0.0

    n = len(x)
    n_cats = max(x.max(), y.max()) + 1

    # Joint and marginal counts
    joint = np.zeros((n_cats, n_cats), dtype=float)
    for xi, yi in zip(x, y):
        joint[xi, yi] += 1
    joint /= n

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_cats):
        for j in range(n_cats):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    return float(mi)


def build_cramers_v_matrix(answer_choices: np.ndarray) -> np.ndarray:
    """Build N×N matrix of Cramér's V between all question pairs."""
    K, N = answer_choices.shape
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i == j:
                V[i, j] = 1.0
            else:
                v = cramers_v(answer_choices[:, i], answer_choices[:, j])
                V[i, j] = v
                V[j, i] = v
    return V


def build_mi_matrix(answer_choices: np.ndarray) -> np.ndarray:
    """Build N×N mutual information matrix between all question pairs."""
    K, N = answer_choices.shape
    MI = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            mi = mutual_information(answer_choices[:, i], answer_choices[:, j])
            MI[i, j] = mi
            MI[j, i] = mi
    return MI


def one_hot_encode(answer_choices: np.ndarray) -> np.ndarray:
    """
    Convert K×N matrix of answer choices (0-3) to K×(4N) one-hot matrix.
    For question i, columns 4i, 4i+1, 4i+2, 4i+3 are indicators for A,B,C,D.
    Unparseable (-1) gets all zeros.
    """
    K, N = answer_choices.shape
    onehot = np.zeros((K, 4 * N), dtype=np.float64)
    for k in range(K):
        for i in range(N):
            c = int(answer_choices[k, i])
            if 0 <= c <= 3:
                onehot[k, 4 * i + c] = 1.0
    return onehot


def permutation_null_test(answer_choices: np.ndarray, metric: str = "cramers_v",
                          n_perm: int = 100) -> dict:
    """
    Permutation-based null test: shuffle each column independently to
    destroy inter-question correlations, compute the similarity matrix,
    and compare eigenvalue spectrum.

    This is more robust than MP for non-Gaussian (categorical) data.
    """
    K, N = answer_choices.shape

    # Observed
    if metric == "cramers_v":
        S_obs = build_cramers_v_matrix(answer_choices)
    else:
        S_obs = build_mi_matrix(answer_choices)

    eig_obs = np.sort(np.linalg.eigvalsh(S_obs))[::-1]

    # Null distribution
    eig_null = np.zeros((n_perm, N))
    for p in tqdm(range(n_perm), desc=f"Permutation null ({metric})"):
        A_perm = answer_choices.copy()
        for j in range(N):
            np.random.shuffle(A_perm[:, j])
        if metric == "cramers_v":
            S_perm = build_cramers_v_matrix(A_perm)
        else:
            S_perm = build_mi_matrix(A_perm)
        eig_null[p] = np.sort(np.linalg.eigvalsh(S_perm))[::-1]

    eig_threshold = np.percentile(eig_null, 95, axis=0)
    eig_threshold_99 = np.percentile(eig_null, 99, axis=0)

    n_check = min(N // 2, len(eig_obs))
    d_95 = int((eig_obs[:n_check] > eig_threshold[:n_check]).sum())
    d_99 = int((eig_obs[:n_check] > eig_threshold_99[:n_check]).sum())

    result = {
        "metric": metric,
        "d_95": d_95,
        "d_99": d_99,
        "n_permutations": n_perm,
        "observed_eigenvalues": eig_obs[:15].tolist(),
        "threshold_95_eigenvalues": eig_threshold[:15].tolist(),
        "threshold_99_eigenvalues": eig_threshold_99[:15].tolist(),
    }

    print(f"\n{'='*60}")
    print(f"PERMUTATION NULL TEST: {metric}")
    print(f"{'='*60}")
    print(f"  d (95th pctl):  {d_95}")
    print(f"  d (99th pctl):  {d_99}")
    print(f"  Top eigenvalues (observed vs null thresholds):")
    print(f"  {'Factor':>8} {'Observed':>10} {'95th pctl':>10} {'99th pctl':>10} {'Signal?':>8}")
    for i in range(min(10, N)):
        sig = "YES" if eig_obs[i] > eig_threshold[i] else "no"
        print(f"  {i+1:>8} {eig_obs[i]:>10.4f} {eig_threshold[i]:>10.4f} {eig_threshold_99[i]:>10.4f} {sig:>8}")
    print(f"{'='*60}\n")

    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_semantic_results(
    binary_pa: dict,
    onehot_pa: dict,
    cramers_perm: dict,
    mi_perm: dict,
    answer_choices: np.ndarray,
    samples: list[dict],
):
    """Generate summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Answer distribution for each question
    ax = axes[0, 0]
    K, N = answer_choices.shape
    for letter_idx in range(4):
        fracs = [(answer_choices[:, i] == letter_idx).sum() / K for i in range(N)]
        ax.bar(np.arange(N) + letter_idx * 0.2, fracs, width=0.2,
               label=INT_TO_LETTER[letter_idx], alpha=0.8)
    ax.set_xlabel("Question index")
    ax.set_ylabel("Fraction of runs")
    ax.set_title("Answer Distribution per Question")
    ax.legend(title="Answer")

    # 2. Binary vs one-hot eigenvalue comparison
    ax = axes[0, 1]
    n_show = min(15, len(binary_pa["observed_eigenvalues"]))
    x = np.arange(1, n_show + 1)
    ax.plot(x, binary_pa["observed_eigenvalues"][:n_show], "o-", color="red",
            label=f"Binary (d={binary_pa['d']})")
    ax.plot(x, binary_pa["threshold_eigenvalues"][:n_show], "s--", color="red",
            alpha=0.5, label="Binary null (95th pctl)")
    # One-hot has 4N columns, only show first N eigenvalues for comparison
    n_oh = min(n_show, len(onehot_pa["observed_eigenvalues"]))
    ax.plot(x[:n_oh], onehot_pa["observed_eigenvalues"][:n_oh], "^-", color="blue",
            label=f"One-hot (d={onehot_pa['d']})")
    ax.plot(x[:n_oh], onehot_pa["threshold_eigenvalues"][:n_oh], "v--", color="blue",
            alpha=0.5, label="One-hot null (95th pctl)")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Parallel Analysis: Binary vs One-Hot")
    ax.legend(fontsize=8)

    # 3. Cramér's V permutation test
    ax = axes[1, 0]
    n_show_c = min(15, len(cramers_perm["observed_eigenvalues"]))
    x = np.arange(1, n_show_c + 1)
    ax.plot(x, cramers_perm["observed_eigenvalues"][:n_show_c], "o-", color="purple",
            label=f"Observed (d={cramers_perm['d_95']})")
    ax.plot(x, cramers_perm["threshold_95_eigenvalues"][:n_show_c], "s--", color="green",
            label="95th pctl null")
    ax.plot(x, cramers_perm["threshold_99_eigenvalues"][:n_show_c], "v--", color="orange",
            label="99th pctl null")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Permutation Test: Cramer's V Matrix")
    ax.legend(fontsize=8)

    # 4. MI permutation test
    ax = axes[1, 1]
    n_show_m = min(15, len(mi_perm["observed_eigenvalues"]))
    x = np.arange(1, n_show_m + 1)
    ax.plot(x, mi_perm["observed_eigenvalues"][:n_show_m], "o-", color="purple",
            label=f"Observed (d={mi_perm['d_95']})")
    ax.plot(x, mi_perm["threshold_95_eigenvalues"][:n_show_m], "s--", color="green",
            label="95th pctl null")
    ax.plot(x, mi_perm["threshold_99_eigenvalues"][:n_show_m], "v--", color="orange",
            label="99th pctl null")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Permutation Test: Mutual Information Matrix")
    ax.legend(fontsize=8)

    plt.suptitle("Semantic Pilot: Does Answer Content Have Structure?", fontsize=14, y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "semantic_pilot_results.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")


# ─── Descriptive Statistics ──────────────────────────────────────────────────

def answer_statistics(answer_choices: np.ndarray, correctness: np.ndarray,
                      samples: list[dict]) -> dict:
    """Compute descriptive statistics about answer choice patterns."""
    K, N = answer_choices.shape

    stats_per_question = []
    for i in range(N):
        choices = answer_choices[:, i]
        valid = choices[choices >= 0]
        if len(valid) == 0:
            continue

        correct_letter = samples[i]["correct_letter"]
        correct_idx = LETTER_TO_INT[correct_letter]

        # Distribution
        counts = np.bincount(valid, minlength=4)
        fracs = counts / len(valid)

        # Entropy of answer distribution
        probs = fracs[fracs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        # When wrong, which letter is most common?
        wrong_choices = valid[valid != correct_idx]
        if len(wrong_choices) > 0:
            wrong_counts = np.bincount(wrong_choices, minlength=4)
            wrong_counts[correct_idx] = 0  # exclude correct
            most_common_wrong = int(np.argmax(wrong_counts))
            wrong_concentration = wrong_counts[most_common_wrong] / len(wrong_choices) if len(wrong_choices) > 0 else 0
        else:
            most_common_wrong = -1
            wrong_concentration = 0

        stats_per_question.append({
            "question_idx": i,
            "original_index": samples[i]["original_index"],
            "subject": samples[i]["subject"],
            "correct_letter": correct_letter,
            "accuracy": float(fracs[correct_idx]),
            "entropy": float(entropy),
            "distribution": {INT_TO_LETTER[j]: float(fracs[j]) for j in range(4)},
            "most_common_wrong_letter": INT_TO_LETTER[most_common_wrong] if most_common_wrong >= 0 else None,
            "wrong_concentration": float(wrong_concentration),
        })

    # Summary statistics
    entropies = [s["entropy"] for s in stats_per_question]
    wrong_concs = [s["wrong_concentration"] for s in stats_per_question if s["wrong_concentration"] > 0]

    summary = {
        "n_questions": N,
        "k_passes": K,
        "mean_entropy": float(np.mean(entropies)),
        "median_entropy": float(np.median(entropies)),
        "max_entropy": float(np.max(entropies)),  # max = log2(4) = 2.0 for uniform
        "mean_wrong_concentration": float(np.mean(wrong_concs)) if wrong_concs else 0,
        "per_question": stats_per_question,
    }

    print(f"\n{'='*60}")
    print("ANSWER CHOICE STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean entropy:              {summary['mean_entropy']:.3f} bits (max = 2.0 for uniform)")
    print(f"  Median entropy:            {summary['median_entropy']:.3f} bits")
    print(f"  Mean wrong concentration:  {summary['mean_wrong_concentration']:.3f}")
    print(f"    (1.0 = always same wrong answer; 0.33 = uniform across wrong options)")
    print()
    print(f"  Example questions:")
    for s in sorted(stats_per_question, key=lambda x: x["entropy"], reverse=True)[:5]:
        dist_str = "  ".join(f"{l}:{v:.0%}" for l, v in s["distribution"].items())
        print(f"    Q{s['question_idx']} [{s['subject'][:20]:>20}] "
              f"correct={s['correct_letter']} entropy={s['entropy']:.2f}  {dist_str}")
    print(f"{'='*60}\n")

    return summary


# ─── Main Pilot ──────────────────────────────────────────────────────────────

def run_pilot(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    K: int = 30,
    N: int = 50,
    temperature: float = 1.0,
):
    """Run the full semantic pilot experiment."""
    print(f"\n{'#'*70}")
    print(f"# SEMANTIC PILOT EXPERIMENT")
    print(f"# Model: {model_name}")
    print(f"# K={K} stochastic passes, N={N} questions, T={temperature}")
    print(f"{'#'*70}\n")

    t_total_start = time.time()

    # ── Step 1: Load questions ──
    logger.info("Step 1: Loading pilot questions...")
    samples = load_pilot_questions(n_questions=N)
    N = len(samples)
    logger.info(f"Loaded {N} pilot questions")

    # ── Step 2: Run stochastic passes ──
    logger.info("Step 2: Running stochastic passes...")
    correctness, answer_choices, raw_answers = run_semantic_passes(
        model_name, samples, K=K, temperature=temperature,
    )

    # Save raw data
    np.save(RESULTS_DIR / "semantic_pilot_correctness.npy", correctness)
    np.save(RESULTS_DIR / "semantic_pilot_answer_choices.npy", answer_choices)
    with open(RESULTS_DIR / "semantic_pilot_raw_answers.json", "w") as f:
        json.dump(raw_answers, f)
    with open(RESULTS_DIR / "semantic_pilot_questions.json", "w") as f:
        json.dump([{
            "index": i,
            "original_index": s["original_index"],
            "question": s["question"][:150],
            "subject": s["subject"],
            "correct_letter": s["correct_letter"],
            "original_flip_rate": s["original_flip_rate"],
        } for i, s in enumerate(samples)], f, indent=2)

    logger.info("Raw data saved.")

    # ── Step 3: Descriptive statistics ──
    logger.info("Step 3: Answer choice statistics...")
    desc_stats = answer_statistics(answer_choices, correctness, samples)

    # ── Step 4: Binary analysis (reproducing main result on this subset) ──
    logger.info("Step 4: Binary correctness analysis...")

    # Filter to truly borderline in THIS experiment
    flip_rates_now = correctness.mean(axis=0)
    borderline_mask = (flip_rates_now > 0.05) & (flip_rates_now < 0.95)
    n_borderline = borderline_mask.sum()
    logger.info(f"  Borderline in this run: {n_borderline}/{N}")

    if n_borderline >= 10:
        C_borderline = correctness[:, borderline_mask]
        binary_mp = marchenko_pastur_test(C_borderline, "Binary Correctness (borderline)")
        binary_pa = parallel_analysis(C_borderline, "Binary Correctness (borderline)")
    else:
        logger.warning(f"Only {n_borderline} borderline questions — using all {N}")
        binary_mp = marchenko_pastur_test(correctness, "Binary Correctness (all)")
        binary_pa = parallel_analysis(correctness, "Binary Correctness (all)")

    # ── Step 5: One-hot encoding analysis ──
    logger.info("Step 5: One-hot encoding analysis...")
    # Use same borderline questions
    if n_borderline >= 10:
        A_borderline = answer_choices[:, borderline_mask]
    else:
        A_borderline = answer_choices

    onehot = one_hot_encode(A_borderline)
    # Remove zero-variance columns (letters never chosen)
    col_var = onehot.var(axis=0)
    onehot_active = onehot[:, col_var > 0]
    logger.info(f"  One-hot matrix: {onehot_active.shape[0]} x {onehot_active.shape[1]} "
                f"(from {onehot.shape[1]} columns, {(col_var == 0).sum()} removed)")

    onehot_mp = marchenko_pastur_test(onehot_active, "One-Hot Answer Encoding")
    onehot_pa = parallel_analysis(onehot_active, "One-Hot Answer Encoding")

    # ── Step 6: Cramér's V analysis ──
    logger.info("Step 6: Cramer's V analysis with permutation test...")
    cramers_perm = permutation_null_test(A_borderline, metric="cramers_v", n_perm=200)

    # ── Step 7: Mutual Information analysis ──
    logger.info("Step 7: Mutual Information analysis with permutation test...")
    mi_perm = permutation_null_test(A_borderline, metric="mi", n_perm=200)

    # ── Step 8: Plot ──
    logger.info("Step 8: Plotting...")
    plot_semantic_results(binary_pa, onehot_pa, cramers_perm, mi_perm, answer_choices, samples)

    # ── Step 9: Final comparison ──
    t_total = time.time() - t_total_start

    print(f"\n{'#'*70}")
    print(f"# SEMANTIC PILOT: FINAL COMPARISON")
    print(f"{'#'*70}")
    print()
    print(f"  Model:          {model_name}")
    print(f"  K passes:       {K}")
    print(f"  N questions:    {N}")
    print(f"  N borderline:   {n_borderline}")
    print(f"  Temperature:    {temperature}")
    print(f"  Total time:     {t_total:.0f}s ({t_total/60:.1f} min)")
    print()
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ BINARY CORRECTNESS (right/wrong):                          │")
    print(f"  │   MP signal eigenvalues:    {binary_mp['n_signal']:>3}                            │")
    print(f"  │   Parallel analysis d:      {binary_pa['d']:>3}                            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ SEMANTIC (one-hot A/B/C/D):                                │")
    print(f"  │   MP signal eigenvalues:    {onehot_mp['n_signal']:>3}                            │")
    print(f"  │   Parallel analysis d:      {onehot_pa['d']:>3}                            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ SEMANTIC (Cramer's V, permutation null):                   │")
    print(f"  │   d (95th percentile):      {cramers_perm['d_95']:>3}                            │")
    print(f"  │   d (99th percentile):      {cramers_perm['d_99']:>3}                            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ SEMANTIC (Mutual Info, permutation null):                   │")
    print(f"  │   d (95th percentile):      {mi_perm['d_95']:>3}                            │")
    print(f"  │   d (99th percentile):      {mi_perm['d_99']:>3}                            │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print()

    # Interpretation
    binary_d = binary_pa['d']
    semantic_onehot_d = onehot_pa['d']
    semantic_cv_d = cramers_perm['d_95']
    semantic_mi_d = mi_perm['d_95']

    if semantic_cv_d <= 2 and semantic_mi_d <= 2 and semantic_onehot_d <= 2:
        print("  INTERPRETATION: Semantic content shows NO significant structure")
        print("  above noise, consistent with binary correctness analysis.")
        print("  The reviewer's concern about hidden semantic structure is")
        print("  empirically addressed: answer CONTENT is as unstructured as")
        print("  answer CORRECTNESS under stochastic sampling.")
        interpretation = "no_semantic_structure"
    elif semantic_cv_d > binary_d + 3 or semantic_mi_d > binary_d + 3:
        print("  INTERPRETATION: Semantic content shows SIGNIFICANT structure")
        print("  invisible to binary analysis! The reviewer's concern is valid.")
        print("  Stochastic sampling produces structured answer-choice patterns")
        print("  even though binary correctness appears random.")
        interpretation = "semantic_structure_found"
    else:
        print("  INTERPRETATION: Weak/marginal semantic structure detected.")
        print("  Some signal above noise but comparable to binary analysis.")
        print("  The difference may not be practically significant.")
        interpretation = "marginal_signal"

    print(f"\n{'#'*70}\n")

    # ── Save results ──
    results = {
        "model": model_name,
        "K": K,
        "N": N,
        "N_borderline": int(n_borderline),
        "temperature": temperature,
        "total_time_seconds": float(t_total),
        "interpretation": interpretation,
        "descriptive_stats": {
            "mean_entropy": desc_stats["mean_entropy"],
            "median_entropy": desc_stats["median_entropy"],
            "mean_wrong_concentration": desc_stats["mean_wrong_concentration"],
        },
        "binary_analysis": {
            "mp_n_signal": binary_mp["n_signal"],
            "mp_lambda_plus": binary_mp["lambda_plus"],
            "mp_top5_eigenvalues": binary_mp["top_10_eigenvalues"][:5],
            "parallel_analysis_d": binary_pa["d"],
            "pa_top10_observed": binary_pa["observed_eigenvalues"][:10],
            "pa_top10_threshold": binary_pa["threshold_eigenvalues"][:10],
        },
        "onehot_analysis": {
            "mp_n_signal": onehot_mp["n_signal"],
            "mp_lambda_plus": onehot_mp["lambda_plus"],
            "mp_top5_eigenvalues": onehot_mp["top_10_eigenvalues"][:5],
            "parallel_analysis_d": onehot_pa["d"],
            "pa_top10_observed": onehot_pa["observed_eigenvalues"][:10],
            "pa_top10_threshold": onehot_pa["threshold_eigenvalues"][:10],
            "n_active_columns": int(onehot_active.shape[1]),
        },
        "cramers_v_analysis": {
            "d_95": cramers_perm["d_95"],
            "d_99": cramers_perm["d_99"],
            "n_permutations": cramers_perm["n_permutations"],
            "top10_observed": cramers_perm["observed_eigenvalues"][:10],
            "top10_threshold_95": cramers_perm["threshold_95_eigenvalues"][:10],
            "top10_threshold_99": cramers_perm["threshold_99_eigenvalues"][:10],
        },
        "mi_analysis": {
            "d_95": mi_perm["d_95"],
            "d_99": mi_perm["d_99"],
            "n_permutations": mi_perm["n_permutations"],
            "top10_observed": mi_perm["observed_eigenvalues"][:10],
            "top10_threshold_95": mi_perm["threshold_95_eigenvalues"][:10],
            "top10_threshold_99": mi_perm["threshold_99_eigenvalues"][:10],
        },
        "answer_statistics": desc_stats["per_question"],
    }

    out_path = RESULTS_DIR / "semantic_pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_pilot()
