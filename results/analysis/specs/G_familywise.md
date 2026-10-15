# Pre-registration G: family-wise test over ranks 2-4, sibling-model test, one validity rule
Written 2026-10-13, BEFORE any run of G. Do not change after results exist; record deviations in a
"Deviations" section of the output, with the reason.

Motivation: hostile review round 2, item 1. The paper's per-rank 1% + contiguous-
from-rank-2 rule finds nothing beyond ability on all-4; ranks 3-4 sit at the 97-99th pct of the valid 2PL and
Rasch nulls, and a post-hoc joint statistic rejects. We fix the statistic here, before looking.

## G1. Family-wise statistic (primary)
- T = max over r in {2,3,4} of pct_r, where pct_r = fraction of null draws with lambda_r(null) < lambda_r(obs)
  (ties count half). Null distribution of T: for each null draw d, compute its own max_r pct_r against the
  other draws (leave-one-out). p_T = (1 + #{d: T_d >= T_obs}) / (1 + n_draws).
- Secondary (report, no decision): S = lambda2+lambda3+lambda4, p_S same way.
- Draws: 2000 NEW null draws per (score, null), seeds disjoint from all earlier runs (new seed namespace "G").
- Applied to: rotation 0 and all six rotation scorings (>=1, >=2, >=3, all-4, mean ... whatever the construction
  family in v3_softpref.py step1 has), under every null that is VALID by the single rule below.
- Decision level: alpha = 0.05 family-wise (chosen to be strict AGAINST our current "nothing beyond ability" claim).

## G2. Calibration of the G1 test under soft-preference (SP) truth
- Datasets simulated from the SP fit (common tau, no second factor), conditioned on the observed unparsed mask,
  exactly as in step2b. 200 datasets per score for rotation 0 and all-4 (others if budget allows).
- For each: refit the 2PL and Rasch on the simulated data, draw 500 null draws each, compute p_T.
- Report the false-positive rate FP = P(p_T < 0.05). If FP > 0.10, the G1 decision uses the calibrated threshold
  (the 5th percentile of p_T under SP truth) and both raw and calibrated p are reported.

## G3. Power of the G1 test
- Same planted subject-block factor as the existing power study (STEM block, s in {1, 2}), 200 datasets each,
  all-4 score, 2PL and Rasch nulls. Report detection rate of G1 at alpha 0.05 (raw and calibrated).

## G4. Sibling-model test (lineage)
- Sibling rule, fixed from model metadata, NOT from correlations: two models are siblings if they are fine-tunes
  or successive versions of the same base model at the same parameter count. Clusters:
  {Phi-3-mini-4k, Phi-3.5-mini}, {Meta-Llama-3-8B, Llama-3.1-8B}, {Mistral-7B-v0.2, Mistral-7B-v0.3, zephyr-7b-beta},
  {Qwen2-7B, Qwen2.5-7B}. (Ministral-8B, Mistral-Small, Llama-3.2-1B/3B, Gemma-2-2b/9b, Qwen2.5 0.5/1.5/3B differ
  in size -> not siblings.)
- Run G1 (2PL and Rasch, all-4; also rotation 0) on EVERY subset that keeps exactly one model per cluster
  (2*2*3*2 = 24 subsets, 24-5 = 19 models each), refitting the nulls per subset; 1000 draws per subset.
- Report the distribution of p_T and of ranks-beyond-ability across the 24 subsets.
- Control: the same number (5) of models removed at random from non-sibling models, 24 random subsets, same seed rule.

## G5. Single validity rule everywhere
- Rule (Sec. 2): a null is valid iff observed lambda1 lies within its 1st-99th pct. No second "calibrated validity".
- Recompute Table A2's SP column by this rule (all-4 SP: invalid; rotation 0 SP: valid -> report its rank-2 p).
- Recompute the power study's SP-rule column so that it applies the procedure the paper uses on real data.

## G6. Unparsed-reply variants (hostile minor 7)
- Rerun the valid nulls (rotation 0: letter-aware rot-0 fit and 3PL; all-4: 2PL and Rasch) with per-rank AND G1
  on the drop and impute variants of the unparsed replies. 1000 draws.

## Decision rules for the claim (fixed now)
A. G1 on all-4 does not reject (calibrated) under 2PL or Rasch -> current claim stands, with G3 power caveat.
B. G1 rejects under both valid nulls, and in >= 18 of 24 sibling subsets it no longer rejects (while the random-
   removal control keeps rejecting in most) -> claim: "no second direction separates at rank 2; a weaker excess at
   ranks 3-4 follows model lineage (sibling models)". Letter preference = the largest structure beyond ability,
   not the only one.
C. G1 rejects and sibling removal does not remove it -> claim: "structure beyond ability remains at ranks 3-4,
   of unknown source"; the paper must say so in the abstract.
Mixed outcomes are reported as they are, with the nearest rule named.
