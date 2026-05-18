# TW z-score diagnostic

**Phenomenon.** Five within-model MMLU Tracy–Widom z-scores were flagged as "4 of 5
below the TW F1 mean of -1.77": Qwen2.5-7B -2.01, Mistral-7B-v0.3 -2.37, Phi-3-mini
-1.36, Meta-Llama-3-8B -2.37, SmolLM2-1.7B +0.05. The concern was that this looks
like a systematic downward shift incompatible with the null.

**Setup.** For each model's correctness matrix `results/correctness_matrix_*.npy`
(K=100, N=500 MMLU), I reproduced the pipeline (borderline filter ε=0.05, column
standardization by sample SD, R = C_normᵀ C_norm / K, largest eigenvalue, TW
z-score) and ran N_MC = 1000 Monte-Carlo replicates per model under four nulls
(fixed seed 20260417). Script: `scripts/tw_z_diagnostic.py`; numerical output:
`results/tw_z_diagnostic.json`; figure: `results/figures/tw_z_diagnostic.pdf`.

## Result 1: the observed z-scores are NOT extreme under the correct null.

Mean of simulated z under independent-Bernoulli columns with empirical p_i, filter
applied (H1), at K=100:

| Model | N_b | γ | z_obs | H1 sim mean | H1 sim std | pct(z_obs) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B      | 121 | 1.21 | -2.01 | -1.71 | 1.11 | 0.42 |
| Mistral-7B-v0.3 | 283 | 2.83 | -2.37 | -1.53 | 1.17 | 0.25 |
| Phi-3-mini      | 269 | 2.69 | -1.36 | -1.61 | 1.13 | 0.61 |
| Meta-Llama-3-8B | 173 | 1.73 | -2.38 | -1.66 | 1.12 | 0.28 |
| SmolLM2-1.7B    | 403 | 4.03 | +0.05 | -1.51 | 1.16 | 0.90 |

The observed percentiles span 0.25–0.90 under the correct finite-K null; none is
in the tail of its own null distribution. The unweighted mean of the five observed
z-scores is **-1.61**, essentially identical to the mean of the finite-K null
(**-1.60**).

Fisher-combined left-tail p-value over the five percentiles = **0.60** → the
joint observation is entirely consistent with independent Bernoulli sampling
once finite-K is accounted for. Also note that **only 3 of 5** z_obs lie below
-1.77 (Phi-3 at -1.36 and SmolLM2 at +0.05 are above); under i.i.d. draws
from the null this happens with P ≈ 0.19, not surprising.

## Result 2: hypothesis ranking

All four MC suites give nearly identical distributions:

| Null | mean(z) averaged across 5 models |
|---|---:|
| H1 Bernoulli {p_i} + filter          | **-1.605** |
| H2 Bernoulli {p_i}, no re-filter     | -1.594 |
| H3 IID Gaussian columns              | -1.590 |
| H4 Bernoulli p=0.5 (max variance)    | -1.583 |
| Asymptotic TW F1                     | -1.771 |
| Observed                             | -1.611 |

- **H1 (filter-selection bias): rejected.** H1 vs H2 differ by ≤ 0.04 in z;
  re-applying the borderline filter after simulation does not meaningfully
  shift the edge.
- **H2 (standardization artifact): minor.** Using sample SD (ddof=0) at K=100
  does slightly shrink the top eigenvalue, but the effect is already absorbed
  into the finite-K null: switching to population-SD or ddof=1 changes z by
  O(1/K) ≈ 0.01.
- **H3 (non-universality at moderate γ): rejected.** Gaussian-entry simulations
  produce the same mean shift as Bernoulli at matched (K, N_b), confirming
  edge universality holds in this regime (Pillai–Yin / Bao–Pan–Zhou apply).
- **H4 (finite-K correction): supported and sufficient.** The shift from the
  asymptotic -1.77 to the observed ≈ -1.60 is a finite-K effect on
  `sigma_TW`. All four nulls agree on this shift; it fully explains the
  perceived deficit.

**Bottom line.** The finite-K Tracy–Widom normalization is a known
O(K^{-2/3})-level asymptotic approximation, and at K=100 the true expected z
under the null is **≈ -1.60**, not -1.77. Measured against the correct
finite-K reference, the five observed z-scores are unremarkable.

## Result 3: reporting recommendation

Two reasonable options:

1. **Keep the sample-correlation pipeline and report a finite-K-corrected
   null.** Replace the asymptotic "mean -1.77, std 1.27" in the Table 1
   caption with the empirical finite-K null (mean ≈ -1.60, std ≈ 1.14 at
   K=100 across our γ range; mild γ-dependence). Already computable from
   the existing MC; no code change in the pipeline. Recommended.

2. **Switch to population-correlation MP.** This would require each column's
   true p_i, which we estimate from the same sample. It does not remove the
   O(K^{-2/3}) Tracy–Widom finite-K bias, so it does not meaningfully help.
   Not recommended.

Numerically, a bias-corrected z_bc = (z_obs − μ̂_K) / σ̂_K using the empirical
H1 null gives:

| Model | z_obs | z_bc |
|---|---:|---:|
| Qwen2.5-7B      | -2.01 | -0.27 |
| Mistral-7B-v0.3 | -2.37 | -0.72 |
| Phi-3-mini      | -1.36 | +0.22 |
| Meta-Llama-3-8B | -2.38 | -0.64 |
| SmolLM2-1.7B    | +0.05 | +1.34 |

i.e. none of the models has a signal eigenvalue even after finite-K
correction, and SmolLM2 moves from marginally positive to moderately
positive (still < 2, so still consistent with noise; the signal bound
from ε-sweep in the paper remains the definitive statement on SmolLM2).

## One-sentence Discussion addition (for §4)

> "Under independent-Bernoulli Monte Carlo matching each model's (K, N_b, {p_i}),
> the expected TW z at K=100 is approximately -1.60 (std ≈ 1.14) rather than
> the asymptotic -1.77 (std 1.27), a finite-K correction that eliminates the
> apparent downward deficit in Table 1 and leaves every observed z within the
> central 90% of its own finite-K null."

## Files written

- `results/tw_z_diagnostic.md` (this file)
- `results/tw_z_diagnostic.json` (full numerics)
- `results/figures/tw_z_diagnostic.pdf` (5 per-model histograms + summary panel)
- `scripts/tw_z_diagnostic.py` (reproducible driver; fixed seed = 20260417)
