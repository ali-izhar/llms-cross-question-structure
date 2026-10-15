# Specification I (written 2026-10-14 BEFORE any run of I; post hoc, from hostile review round 4)

## I1. Family-wise test without Llama-3.2-1B (23 models)
- Scores: rotation 0 and all-4. Refit every null of the step-1 suite for that score on the 23-model data
  (rotation 0: 2PL, Rasch, 3PL, letter-aware (rot-0 fit), KG-u, SP common tau if the code supports a refit;
  all-4: 2PL, Rasch, 3PL, KG-l, KG-u, BA), 2000 new draws each, seed namespace I.
- Report for each null: validity (single rule), per-rank percentiles of lambda_1..4, per-rank ranks beyond ability
  (contiguous rule), G1 T and p_T, secondary p_S.
- No decision rule: the paper reports the numbers as they are, under BOTH rules, for both scores.

## I2. Joint removal (all-4)
- Remove Llama-3.2-1B AND one model of each pair P1={Phi-3-mini, Phi-3.5-mini}, P2={Meta-Llama-3-8B, Llama-3.1-8B}
  (4 subsets of 21 models). 2PL and Rasch, 1000 draws, validity + G1 p_T.
- Also, from raw_v3_H_h2.json (no new run): the p_T of the reference removals that include Llama-3.2-1B (19 of 190),
  and of the 15 reference removals whose two removed models have accuracy within the accuracy range of the pair
  models (accuracy-matched control), with the range stated.

## Output
results/analysis/v3_I_llama.json (+ raw_v3_I_*). Tests first: subset has exactly 23 / 21 models with
names matched exactly; same code path reproduces the full-data G1 p_T (0.026 / 0.041); new seed namespace.
