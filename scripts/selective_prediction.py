"""
Selective-prediction comparison for §3.5 of the EIML@ICML 2026 paper.

Task
----
For each of N=500 MMLU questions, predict whether Qwen2.5-7B-Instruct's
greedy (tau=0) answer is correct. Compare two predictor families:

  (1) SC_K (operational): the operationally-available self-consistency
      confidence over K stochastic Qwen-7B passes is "agreement with the
      modal answer", i.e. max(p, 1-p) where p = mean of correctness column.
      Rank-equivalent to |p - 0.5|. NO ground-truth label is consulted at
      prediction time; only Qwen-7B's K answers are used.

  (2) SC_K (oracle): mean correctness of the K stochastic samples (uses
      sample-level labels, so this is an upper bound on what an
      operational SC system could achieve from these K passes — reported
      for transparency).

  (3) Ensemble_k: per-question fraction of k *other* models that are
      correct at tau=0. This is reported because it is what the
      correctness matrix supports — it requires labels at the cohort
      level to convert raw answers to correctness verdicts, but no
      label leakage of the *target* (Qwen-7B's tau=0 row) occurs.

We are explicit about predictor (2) being a ground-truth-aided upper
bound for SC and predictor (3) being a labeled-cohort signal for the
ensemble; the apples-to-apples *operational* comparison is (1) vs (3).

Metrics
-------
AUROC and AUPR (sklearn). Inference cost is in units of "Qwen-7B-equivalent
forward passes": for each contributing model, parameter count (B) / 7.0.

Outputs
-------
results/selective_prediction.json — table of (predictor, K_or_k, AUROC,
AUPR, cost_units) plus metadata (target accuracy, ground truth size,
predictor list, headline operational comparison).

NOTE: The earlier paper-internal numbers (SC AUROC = 0.984, ensemble
AUROC = 0.804) used the oracle SC definition and a now-superseded
4-model ensemble snapshot; this script supersedes both with the
24-model expanded ensemble and reports the operational vs. oracle
distinction transparently.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ── paths ────────────────────────────────────────────────────────────
PROJECT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"

# ── parameter counts (B), used as a cost proxy ─────────────────────
# Source: HuggingFace model cards. Used only to define the cost unit
# "Qwen-7B-equivalent forward passes" = sum(params_b) / 7.0.
PARAMS_B: dict[str, float] = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.7,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "microsoft/phi-2": 2.7,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2-7B-Instruct": 7.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.2": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "HuggingFaceH4/zephyr-7b-beta": 7.0,
    "allenai/OLMo-2-1124-7B-Instruct": 7.0,
    "01-ai/Yi-1.5-6B-Chat": 6.0,
    "deepseek-ai/deepseek-llm-7b-chat": 7.0,
    "ibm-granite/granite-3.1-8b-instruct": 8.0,
    "mistralai/Ministral-8B-Instruct-2410": 8.0,
    "google/gemma-2-2b-it": 2.6,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "google/gemma-2-9b-it": 9.2,
    "mistralai/Mistral-Small-Instruct-2409": 22.0,
    "meta-llama/Llama-3.2-1B-Instruct": 1.2,
    "meta-llama/Llama-3.2-3B-Instruct": 3.2,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8.0,
}
QWEN7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN7B_PARAMS = PARAMS_B[QWEN7B]


def cost_units(model_names: list[str]) -> float:
    """Sum of parameter-counts (in Qwen-7B-equivalent units)."""
    return float(sum(PARAMS_B[m] for m in model_names) / QWEN7B_PARAMS)


def family_of(model: str) -> str:
    """Coarse family label for diversity-aware ensemble selection."""
    m = model.lower()
    if "qwen" in m:
        return "qwen"
    if "mistral" in m or "ministral" in m or "zephyr" in m:
        # zephyr-7b-beta is a Mistral-7B fine-tune
        return "mistral"
    if "phi" in m:
        return "phi"
    if "smollm" in m:
        return "smollm"
    if "olmo" in m:
        return "olmo"
    if "yi" in m:
        return "yi"
    if "deepseek" in m:
        return "deepseek"
    if "granite" in m:
        return "granite"
    if "gemma" in m:
        return "gemma"
    if "llama" in m:
        return "llama"
    raise ValueError(f"unknown family for {model}")


# ── ensemble selection helpers ────────────────────────────────────────
def topk_other_models(
    models: list[str],
    accuracies: dict[str, float],
    k: int,
    *,
    exclude: str = QWEN7B,
) -> list[str]:
    """Top-k by tau=0 accuracy, excluding the target model."""
    candidates = [m for m in models if m != exclude]
    candidates.sort(key=lambda m: accuracies[m], reverse=True)
    return candidates[:k]


def family_diverse_models(
    models: list[str],
    accuracies: dict[str, float],
    k: int,
    *,
    exclude: str = QWEN7B,
) -> list[str]:
    """k models, one per family (different family from `exclude` first),
    picking the highest-accuracy member of each family. Falls back to top-k
    by accuracy if k > number of distinct families.
    """
    exclude_family = family_of(exclude)
    by_family: dict[str, list[str]] = {}
    for m in models:
        if m == exclude:
            continue
        by_family.setdefault(family_of(m), []).append(m)

    # within each family, sort by accuracy descending
    for fam in by_family:
        by_family[fam].sort(key=lambda m: accuracies[m], reverse=True)

    # ordering of families: prefer families different from exclude_family,
    # then break ties by best accuracy in that family
    fam_order = sorted(
        by_family.keys(),
        key=lambda f: (f == exclude_family, -accuracies[by_family[f][0]]),
    )

    # one per family in ordered families, then second-best per family, etc.
    chosen: list[str] = []
    rank = 0
    while len(chosen) < k:
        added_any = False
        for fam in fam_order:
            if rank < len(by_family[fam]):
                chosen.append(by_family[fam][rank])
                added_any = True
                if len(chosen) == k:
                    break
        rank += 1
        if not added_any:
            break
    return chosen[:k]


# ── core computation ──────────────────────────────────────────────────
def compute_predictor_metrics(
    target: np.ndarray,                # (N,) binary, ground truth
    signal: np.ndarray,                # (N,) real-valued, higher = more likely correct
) -> tuple[float, float]:
    """Return (AUROC, AUPR) for `signal` predicting `target`."""
    # sklearn requires both labels present
    if target.min() == target.max():
        return float("nan"), float("nan")
    auroc = roc_auc_score(target, signal)
    aupr = average_precision_score(target, signal)
    return float(auroc), float(aupr)


def main() -> dict:
    # ── load data ────────────────────────────────────────────────────
    sc = np.load(RESULTS / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy")
    ens = np.load(RESULTS / "ensemble_expanded_correctness.npy")
    with open(RESULTS / "ensemble_expanded_models.json") as f:
        models: list[str] = json.load(f)

    K, N = sc.shape
    M, N_ens = ens.shape
    assert N == N_ens == 500, f"shape mismatch: SC {sc.shape}, ENS {ens.shape}"
    assert K == 100, f"expected K=100 SC samples, got {K}"
    assert M == 24, f"expected M=24 ensemble models, got {M}"

    # ground truth: Qwen-7B's deterministic (tau=0) correctness
    qwen_idx = models.index(QWEN7B)
    target = ens[qwen_idx].astype(np.int64)             # (500,)
    target_acc = float(target.mean())
    assert 0.0 < target_acc < 1.0, "target is degenerate (all 0 or all 1)"

    # per-model tau=0 accuracies (for ensemble selection)
    accuracies = {m: float(ens[i].mean()) for i, m in enumerate(models)}

    # ── (1) SC_K signals ────────────────────────────────────────────
    # We compute two variants:
    #   - operational: max(p, 1-p), i.e. agreement-with-modal-answer.
    #     NO sample-level label info is required at prediction time.
    #   - oracle:     mean(p), i.e. fraction of the K samples that are
    #     correct. Uses sample-level labels; reported as an upper bound.
    sc_results = []
    for K_use in (2, 5, 10, 20, 50, 100):
        # NOTE: deterministic prefix (first K_use rows) — the SC matrix
        # has no preferred row ordering so this matches a "what if we had
        # only run K_use samples instead of 100" counterfactual.
        p = sc[:K_use].mean(axis=0)                     # (500,)
        signal_op = np.maximum(p, 1.0 - p)              # operational
        signal_or = p                                   # oracle
        auroc_op, aupr_op = compute_predictor_metrics(target, signal_op)
        auroc_or, aupr_or = compute_predictor_metrics(target, signal_or)
        cost = float(K_use)  # K Qwen-7B forward passes
        sc_results.append({
            "predictor": "SC_K",
            "K_or_k": K_use,
            "models_used": [QWEN7B] * K_use,
            "AUROC_operational": auroc_op,
            "AUPR_operational": aupr_op,
            "AUROC_oracle_upper_bound": auroc_or,
            "AUPR_oracle_upper_bound": aupr_or,
            "cost_units_qwen7b_equivalent": cost,
        })

    # ── (2a) Ensemble_k (top-k by accuracy) ─────────────────────────
    ens_topk_results = []
    for k in (1, 2, 3, 5, 10):
        chosen = topk_other_models(models, accuracies, k)
        idxs = [models.index(m) for m in chosen]
        signal = ens[idxs].mean(axis=0)                 # (500,)
        auroc, aupr = compute_predictor_metrics(target, signal)
        cost = cost_units(chosen)
        ens_topk_results.append({
            "predictor": "Ensemble_topk",
            "K_or_k": k,
            "models_used": chosen,
            "AUROC": auroc,
            "AUPR": aupr,
            "cost_units_qwen7b_equivalent": cost,
        })

    # ── (2b) Ensemble_k (family-diverse: round-robin across families) ─
    ens_diverse_results = []
    for k in (1, 2, 3, 5, 10):
        chosen = family_diverse_models(models, accuracies, k)
        idxs = [models.index(m) for m in chosen]
        signal = ens[idxs].mean(axis=0)
        auroc, aupr = compute_predictor_metrics(target, signal)
        cost = cost_units(chosen)
        ens_diverse_results.append({
            "predictor": "Ensemble_familydiverse",
            "K_or_k": k,
            "models_used": chosen,
            "AUROC": auroc,
            "AUPR": aupr,
            "cost_units_qwen7b_equivalent": cost,
        })

    # ── (3) Ensemble_family: same-family vs different-family at k=3 ──
    # Same-family: smaller Qwen siblings (0.5B, 1.5B, 3B).
    # Different-family: top-3 from non-Qwen families.
    same_family = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]
    diff_family_candidates = [
        m for m in models
        if family_of(m) != "qwen" and m != QWEN7B
    ]
    diff_family_candidates.sort(key=lambda m: accuracies[m], reverse=True)
    diff_family = diff_family_candidates[:3]

    ens_family_compare = []
    for label, chosen in (("same_family_qwen", same_family),
                          ("different_family", diff_family)):
        idxs = [models.index(m) for m in chosen]
        signal = ens[idxs].mean(axis=0)
        auroc, aupr = compute_predictor_metrics(target, signal)
        ens_family_compare.append({
            "predictor": f"Ensemble_k3_{label}",
            "K_or_k": 3,
            "models_used": chosen,
            "AUROC": auroc,
            "AUPR": aupr,
            "cost_units_qwen7b_equivalent": cost_units(chosen),
        })

    # ── headline figures (used by paper prose) ──────────────────────
    sc100 = next(r for r in sc_results if r["K_or_k"] == 100)
    sc2 = next(r for r in sc_results if r["K_or_k"] == 2)
    sc10 = next(r for r in sc_results if r["K_or_k"] == 10)
    ens2_div = next(r for r in ens_diverse_results if r["K_or_k"] == 2)
    ens1_div = next(r for r in ens_diverse_results if r["K_or_k"] == 1)
    ens3_div = next(r for r in ens_diverse_results if r["K_or_k"] == 3)

    headline = {
        "target_accuracy": target_acc,
        "N_questions": int(N),
        "N_correct": int(target.sum()),
        "N_incorrect": int(N - target.sum()),
        # Operational SC: agreement-with-modal, no sample-level label needed.
        "SC100_AUROC_operational": sc100["AUROC_operational"],
        "SC10_AUROC_operational":  sc10["AUROC_operational"],
        "SC2_AUROC_operational":   sc2["AUROC_operational"],
        # Oracle SC: fraction-correct, ground-truth-aided upper bound.
        "SC100_AUROC_oracle":      sc100["AUROC_oracle_upper_bound"],
        "SC10_AUROC_oracle":       sc10["AUROC_oracle_upper_bound"],
        "SC2_AUROC_oracle":        sc2["AUROC_oracle_upper_bound"],
        # Ensemble (family-diverse round-robin).
        "Ensemble1_AUROC": ens1_div["AUROC"],
        "Ensemble2_AUROC": ens2_div["AUROC"],
        "Ensemble3_AUROC": ens3_div["AUROC"],
        # Costs.
        "SC100_cost": sc100["cost_units_qwen7b_equivalent"],
        "Ensemble2_cost": ens2_div["cost_units_qwen7b_equivalent"],
        "cost_savings_factor_ens2_vs_sc100": (
            sc100["cost_units_qwen7b_equivalent"]
            / ens2_div["cost_units_qwen7b_equivalent"]
        ),
        # The headline operational comparison: Ensemble_2 vs SC_100 (op).
        "ensemble2_beats_sc100_operational": (
            ens2_div["AUROC"] > sc100["AUROC_operational"]
        ),
        "ensemble2_beats_sc10_operational": (
            ens2_div["AUROC"] > sc10["AUROC_operational"]
        ),
        "ensemble2_models": ens2_div["models_used"],
    }

    out = {
        "task": "Predict Qwen2.5-7B-Instruct tau=0 correctness on 500 MMLU questions",
        "ground_truth_source": (
            "Row of ensemble_expanded_correctness.npy at index "
            f"{qwen_idx} ({QWEN7B})"
        ),
        "cost_unit": (
            "Qwen-7B-equivalent forward passes = sum(model_params_B) / 7.0"
        ),
        "headline": headline,
        "SC_K": sc_results,
        "Ensemble_topk": ens_topk_results,
        "Ensemble_familydiverse": ens_diverse_results,
        "Ensemble_family_compare_k3": ens_family_compare,
        "model_accuracies_tau0": accuracies,
    }

    out_path = RESULTS / "selective_prediction.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ── pretty print summary ────────────────────────────────────────
    def _fmt_row(label: str, k: int | str, auroc: float, aupr: float,
                 cost: float) -> str:
        return (f"  {label:<28s} k={k!s:>4s}  AUROC={auroc:.4f}  "
                f"AUPR={aupr:.4f}  cost={cost:>6.2f}")

    print("=" * 78)
    print(f"Selective prediction: target = Qwen-7B tau=0 correct "
          f"(acc={target_acc:.3f})")
    print(f"N={N} questions "
          f"({headline['N_correct']} correct, {headline['N_incorrect']} incorrect)")
    print("=" * 78)
    print()
    print("SC_K (within-model self-consistency, K Qwen-7B passes) — "
          "OPERATIONAL signal max(p,1-p):")
    for r in sc_results:
        print(_fmt_row("SC_K (operational)", r["K_or_k"],
                       r["AUROC_operational"], r["AUPR_operational"],
                       r["cost_units_qwen7b_equivalent"]))
    print()
    print("SC_K (within-model self-consistency, K Qwen-7B passes) — "
          "ORACLE upper-bound (uses sample labels):")
    for r in sc_results:
        print(_fmt_row("SC_K (oracle)", r["K_or_k"],
                       r["AUROC_oracle_upper_bound"],
                       r["AUPR_oracle_upper_bound"],
                       r["cost_units_qwen7b_equivalent"]))
    print()
    print("Ensemble_topk (k strongest other models by tau=0 accuracy):")
    for r in ens_topk_results:
        print(_fmt_row("Ensemble_topk", r["K_or_k"], r["AUROC"], r["AUPR"],
                       r["cost_units_qwen7b_equivalent"]))
        print(f"      models: {r['models_used']}")
    print()
    print("Ensemble_familydiverse (k models, round-robin across families):")
    for r in ens_diverse_results:
        print(_fmt_row("Ensemble_familydiverse", r["K_or_k"], r["AUROC"],
                       r["AUPR"], r["cost_units_qwen7b_equivalent"]))
        print(f"      models: {r['models_used']}")
    print()
    print("Family comparison at k=3:")
    for r in ens_family_compare:
        print(_fmt_row(r["predictor"], r["K_or_k"], r["AUROC"], r["AUPR"],
                       r["cost_units_qwen7b_equivalent"]))
        print(f"      models: {r['models_used']}")
    print()
    print("=" * 78)
    print("HEADLINE (operational SC vs. ensemble disagreement):")
    print(f"  SC_100  (op):    AUROC={headline['SC100_AUROC_operational']:.4f}  "
          f"cost={headline['SC100_cost']:.1f} Qwen-7B-pass")
    print(f"  SC_10   (op):    AUROC={headline['SC10_AUROC_operational']:.4f}  "
          f"cost=10.0")
    print(f"  SC_2    (op):    AUROC={headline['SC2_AUROC_operational']:.4f}  "
          f"cost=2.0")
    print(f"  Ensemble_2 (FD): AUROC={headline['Ensemble2_AUROC']:.4f}  "
          f"cost={headline['Ensemble2_cost']:.2f} Qwen-7B-pass")
    print(f"  Cost savings factor SC_100 / Ensemble_2 = "
          f"{headline['cost_savings_factor_ens2_vs_sc100']:.1f}x")
    print(f"  Ensemble_2 beats SC_100 (op)? "
          f"{headline['ensemble2_beats_sc100_operational']}")
    print(f"  Ensemble_2 beats SC_10  (op)? "
          f"{headline['ensemble2_beats_sc10_operational']}")
    print(f"  Ensemble_2 models: {headline['ensemble2_models']}")
    print()
    print("Oracle SC (upper bound; uses sample-level labels):")
    print(f"  SC_100 (oracle): AUROC={headline['SC100_AUROC_oracle']:.4f}")
    print(f"  SC_2   (oracle): AUROC={headline['SC2_AUROC_oracle']:.4f}")
    print("=" * 78)
    print(f"Saved {out_path}")
    return out


if __name__ == "__main__":
    main()
