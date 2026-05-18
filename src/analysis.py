"""
Analysis: Within-Model Epistemic Dimensionality
=================================================
Takes the correctness matrix from run_experiment.py and performs:
1. Marchenko-Pastur null test (is there signal above noise?)
2. Factor analysis with parallel analysis (how many factors?)
3. Factor interpretation (what do the factors mean?)
4. Confound checks (topic, difficulty, format)
5. Figures
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stats import marchenko_pastur_test, parallel_analysis

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_data(model_slug: str) -> tuple[np.ndarray, list[dict]]:
    C = np.load(RESULTS_DIR / f"correctness_matrix_{model_slug}.npy")
    with open(RESULTS_DIR / f"questions_{model_slug}.json") as f:
        meta = json.load(f)
    return C, meta


def get_borderline(C: np.ndarray, meta: list[dict], eps: float = 0.05):
    """Extract borderline questions (flip rate between eps and 1-eps)."""
    flip_rates = C.mean(axis=0)
    mask = (flip_rates > eps) & (flip_rates < 1 - eps)
    C_b = C[:, mask]
    meta_b = [m for m, is_b in zip(meta, mask) if is_b]
    return C_b, meta_b, mask



# ─── Phase 3: Factor Analysis ─────────────────────────────────────────────

def run_factor_analysis(C_b: np.ndarray, meta_b: list[dict], n_factors: int) -> dict:
    """Run factor analysis with varimax rotation."""
    from factor_analyzer import FactorAnalyzer

    if n_factors < 1:
        print("No factors to extract (d_within = 0). Skipping factor analysis.")
        return {"n_factors": 0}

    # Use Pearson correlation on binary data (standard approximation)
    fa = FactorAnalyzer(
        n_factors=min(n_factors, C_b.shape[1] - 1),
        rotation="varimax",
        method="principal",
    )

    # Transpose: factor analysis expects observations × variables
    # Our C_b is K × N_b, we want to analyze covariance BETWEEN questions
    # So variables = questions, observations = runs
    try:
        fa.fit(C_b)
    except Exception as e:
        print(f"Factor analysis failed: {e}")
        return {"n_factors": 0, "error": str(e)}

    loadings = fa.loadings_  # N_b × n_factors
    communalities = fa.get_communalities()
    variance_explained = fa.get_factor_variance()

    # Interpret each factor: top-loading questions
    factor_interpretations = []
    for f in range(loadings.shape[1]):
        top_idx = np.argsort(np.abs(loadings[:, f]))[::-1][:10]
        top_questions = []
        subjects_in_factor = []
        for idx in top_idx:
            top_questions.append({
                "question": meta_b[idx]["question"][:80],
                "subject": meta_b[idx]["subject"],
                "loading": float(loadings[idx, f]),
                "flip_rate": meta_b[idx]["flip_rate"],
            })
            subjects_in_factor.append(meta_b[idx]["subject"])

        # Subject diversity: how many unique subjects in top 10?
        unique_subjects = len(set(subjects_in_factor))
        factor_interpretations.append({
            "factor": f + 1,
            "top_questions": top_questions,
            "unique_subjects_in_top10": unique_subjects,
            "cross_topic": unique_subjects >= 3,
        })

    result = {
        "n_factors": int(loadings.shape[1]),
        "variance_explained": [float(v) for v in variance_explained[1]],  # proportional
        "cumulative_variance": [float(v) for v in variance_explained[2]],
        "factor_interpretations": factor_interpretations,
        "loadings": loadings.tolist(),
    }

    print(f"\n{'='*60}")
    print("FACTOR ANALYSIS (Varimax Rotation)")
    print(f"{'='*60}")
    print(f"Factors extracted: {loadings.shape[1]}")
    print(f"\nVariance explained by each factor:")
    for i, v in enumerate(result["variance_explained"]):
        cum = result["cumulative_variance"][i]
        print(f"  Factor {i+1}: {v:.1%} (cumulative: {cum:.1%})")

    print(f"\nFactor interpretation (top-loading questions):")
    for fi in factor_interpretations:
        cross = "CROSS-TOPIC" if fi["cross_topic"] else "within-topic"
        print(f"\n  Factor {fi['factor']} ({fi['unique_subjects_in_top10']} unique subjects → {cross}):")
        for q in fi["top_questions"][:5]:
            print(f"    [{q['subject']:>30}] loading={q['loading']:+.3f}  {q['question']}")

    # Confound check: do factors align with subjects?
    all_subjects = [m["subject"] for m in meta_b]
    unique_subs = list(set(all_subjects))
    cross_topic_factors = sum(1 for fi in factor_interpretations if fi["cross_topic"])
    print(f"\n  Cross-topic factors: {cross_topic_factors}/{loadings.shape[1]}")
    if cross_topic_factors == 0:
        print("  WARNING: All factors are within-topic. Confound risk is HIGH.")
    elif cross_topic_factors == loadings.shape[1]:
        print("  All factors cross topic boundaries. Epistemic structure is genuine.")
    else:
        print(f"  Mixed: {cross_topic_factors} cross-topic, {loadings.shape[1] - cross_topic_factors} within-topic.")

    print(f"{'='*60}\n")
    return result


# ─── Phase 4: Confound Analysis ───────────────────────────────────────────

def confound_analysis(C_b: np.ndarray, meta_b: list[dict]) -> dict:
    """Check whether covariance structure is driven by confounds."""
    subjects = [m["subject"] for m in meta_b]
    flip_rates = np.array([m["flip_rate"] for m in meta_b])

    # Subject-level analysis: average within-subject vs between-subject correlation
    unique_subjects = list(set(subjects))
    R = np.corrcoef(C_b.T)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, np.nan)

    within_corrs = []
    between_corrs = []
    for i in range(len(meta_b)):
        for j in range(i + 1, len(meta_b)):
            if np.isnan(R[i, j]):
                continue
            if subjects[i] == subjects[j]:
                within_corrs.append(R[i, j])
            else:
                between_corrs.append(R[i, j])

    within_mean = np.mean(within_corrs) if within_corrs else 0
    between_mean = np.mean(between_corrs) if between_corrs else 0

    result = {
        "within_subject_mean_corr": float(within_mean),
        "between_subject_mean_corr": float(between_mean),
        "ratio": float(within_mean / between_mean) if between_mean != 0 else float("inf"),
        "n_within_pairs": len(within_corrs),
        "n_between_pairs": len(between_corrs),
    }

    print(f"\n{'='*60}")
    print("CONFOUND ANALYSIS")
    print(f"{'='*60}")
    print(f"Within-subject mean correlation:  {within_mean:.4f} ({len(within_corrs)} pairs)")
    print(f"Between-subject mean correlation: {between_mean:.4f} ({len(between_corrs)} pairs)")
    print(f"Ratio (within/between):           {result['ratio']:.2f}")

    if result["ratio"] > 3:
        print("\n>>> WARNING: Within-subject correlation >> between-subject.")
        print("    Factor structure is likely driven by TOPIC, not epistemic structure.")
    elif result["ratio"] > 1.5:
        print("\n>>> MIXED: Some topic effect, but between-subject correlations exist.")
        print("    Factor structure may reflect both topic and epistemic effects.")
    else:
        print("\n>>> CLEAN: Within ≈ between subject correlations.")
        print("    Factor structure is NOT driven by topic confound.")

    print(f"{'='*60}\n")
    return result


# ─── Phase 5: Figures ──────────────────────────────────────────────────────

def plot_eigenvalue_spectrum(mp_result: dict, pa_result: dict, model_name: str):
    """Plot eigenvalue spectrum with MP null and parallel analysis threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    eig_obs = np.array(pa_result["observed_eigenvalues"])
    eig_thresh = np.array(pa_result["threshold_eigenvalues"])
    n_show = min(30, len(eig_obs))

    # Left: full spectrum with MP edge
    all_eig = np.array(mp_result["eigenvalues"])
    ax1.hist(all_eig, bins=50, density=True, alpha=0.7, color="#1E88E5", label="Observed")
    ax1.axvline(x=mp_result["lambda_plus"], color="red", linestyle="--", linewidth=2,
                label=f"MP edge (λ+ = {mp_result['lambda_plus']:.2f})")
    ax1.set_xlabel("Eigenvalue")
    ax1.set_ylabel("Density")
    ax1.set_title("Eigenvalue Distribution vs Marchenko-Pastur")
    ax1.legend()

    # Right: scree plot with parallel analysis
    x = np.arange(1, n_show + 1)
    ax2.plot(x, eig_obs[:n_show], "o-", color="#E53935", linewidth=2, label="Observed")
    ax2.plot(x, eig_thresh[:n_show], "s--", color="#43A047", linewidth=1.5, label="95th pctl random")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Kaiser criterion")

    d = pa_result["d"]
    if d > 0 and d <= n_show:
        ax2.axvline(x=d + 0.5, color="orange", linestyle="-", linewidth=2, alpha=0.7,
                     label=f"d_within = {d}")

    ax2.set_xlabel("Factor number")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("Scree Plot with Parallel Analysis")
    ax2.legend()
    ax2.set_xlim(0.5, n_show + 0.5)

    slug = model_name.replace("/", "_")
    plt.suptitle(f"Within-Model Epistemic Dimensionality: {model_name.split('/')[-1]}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / f"eigenvalue_spectrum_{slug}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_flip_rate_distribution(meta: list[dict], model_name: str):
    """Plot distribution of flip rates across questions."""
    flip_rates = [m["flip_rate"] for m in meta]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flip_rates, bins=50, color="#1E88E5", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0.05, color="red", linestyle="--", label="Borderline threshold (0.05)")
    ax.axvline(x=0.95, color="red", linestyle="--")
    ax.set_xlabel("Flip Rate (fraction correct across K runs)")
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"Flip Rate Distribution: {model_name.split('/')[-1]}")
    ax.legend()

    n_always_correct = sum(1 for f in flip_rates if f == 1.0)
    n_always_wrong = sum(1 for f in flip_rates if f == 0.0)
    n_borderline = sum(1 for f in flip_rates if 0.05 < f < 0.95)
    ax.text(0.95, 0.95, f"Always correct: {n_always_correct}\nAlways wrong: {n_always_wrong}\nBorderline: {n_borderline}",
            transform=ax.transAxes, va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    slug = model_name.replace("/", "_")
    plt.tight_layout()
    path = FIGURES_DIR / f"flip_rates_{slug}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Main Analysis Pipeline ───────────────────────────────────────────────

def analyze(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Run full analysis pipeline on a completed experiment."""
    model_slug = model_name.replace("/", "_")

    # Load data
    C, meta = load_data(model_slug)
    K, N = C.shape
    print(f"Loaded correctness matrix: {K} runs × {N} questions")

    # Plot flip rates
    plot_flip_rate_distribution(meta, model_name)

    # Get borderline questions
    C_b, meta_b, mask = get_borderline(C, meta)
    K, N_b = C_b.shape
    print(f"Borderline questions: {N_b}")

    if N_b < 50:
        print("KILL: Fewer than 50 borderline questions. Cannot proceed.")
        return

    # Phase 1: Marchenko-Pastur
    mp_result = marchenko_pastur_test(C_b)

    # Phase 2: Parallel analysis
    pa_result = parallel_analysis(C_b)
    d_within = pa_result["d"]

    # Plot eigenvalue spectrum
    plot_eigenvalue_spectrum(mp_result, pa_result, model_name)

    # Phase 3: Factor analysis (if d > 0)
    fa_result = run_factor_analysis(C_b, meta_b, n_factors=d_within)

    # Phase 4: Confound analysis
    confound_result = confound_analysis(C_b, meta_b)

    # Save combined results
    combined = {
        "model": model_name,
        "K": int(K),
        "N_total": int(N),
        "N_borderline": int(N_b),
        "d_within": d_within,
        "mp_test": mp_result,
        "parallel_analysis": {k: v for k, v in pa_result.items() if k != "observed_eigenvalues"},
        "confound": confound_result,
    }
    with open(RESULTS_DIR / f"analysis_{model_slug}.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model:                    {model_name}")
    print(f"Total questions:          {N}")
    print(f"Borderline questions:     {N_b}")
    print(f"MP signal dimensions:     {mp_result['n_signal']}")
    print(f"Parallel analysis d:      {d_within}")
    print(f"Within/between subj corr: {confound_result['ratio']:.2f}")
    print(f"\nd_within = {d_within}")
    print(f"(MetaBench d_across ≈ 1)")
    print(f"{'='*60}")

    if d_within == 0:
        print("\n>>> FINDING: Temperature sampling is epistemically shallow.")
        print("    Self-consistency captures all available structure.")
    elif d_within == 1:
        print("\n>>> FINDING: Single attentiveness axis.")
        print("    All borderline questions covary together.")
    else:
        cross = fa_result.get("n_factors", 0)
        if confound_result["ratio"] > 3:
            print(f"\n>>> FINDING: d_within = {d_within}, but likely topic-driven.")
            print("    The comparison to d_across is still novel.")
        else:
            print(f"\n>>> FINDING: d_within = {d_within} with genuine cross-topic structure!")
            print("    The model has multi-dimensional epistemic uncertainty.")
            print("    Scalar confidence provably loses information.")


if __name__ == "__main__":
    analyze()
