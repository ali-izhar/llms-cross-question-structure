# Specification H (written 2026-10-14 BEFORE any run of H; follow-up to hostile round 3, attack3_hostile.md)
Status: these are POST-HOC follow-ups. The two pairs below were named after the data were seen (round-2 audit and
G4). H is fixed now so that its decisions are not tuned to its own results. Record deviations in the output.

## H1. Size of the G1 family-wise test under one-ability truth (all-4 score)
- Truth A: the all-4 2PL fit (marginal ML, the fit used for the paper's 2PL null). Truth B: the all-4 Rasch fit.
- 300 datasets per truth, simulated with the null's own simulator (same unparsed-mask/column-split handling as the
  paper's null), new seed namespace H.
- For each dataset: refit BOTH the 2PL and the Rasch null on the simulated data, 500 null draws each, apply the
  validity rule and G1 (T = max pct over ranks 2-4, leave-one-out null, alpha 0.05).
- Report FP = P(valid and p_T < 0.05) with a 95% Clopper-Pearson CI, for each (truth, null), and the median p_T.
- Decision: a null's all-4 rejection counts as size-controlled if FP <= 0.075 under BOTH truths. Otherwise report
  the rejection as not size-controlled and give the calibrated p (rank of the real p_T among the truth p_T values).

## H2. Is the pair removal exceptional? (all-4 score)
- Pairs (fixed now): P1 = {Phi-3-mini-4k, Phi-3.5-mini}, P2 = {Meta-Llama-3-8B, Llama-3.1-8B}.
- Pair removals: the 4 subsets that remove one member of P1 and one member of P2 (22 models each).
- Reference: ALL 2-model removals that remove no member of P1 or P2 (C(20,2) = 190 subsets of 22 models).
- For every subset: refit the 2PL and Rasch nulls, 1000 draws each, validity rule, G1 p_T.
- Report: p_T for the 4 pair removals; the distribution of p_T over the 190 reference removals; the fraction of
  reference removals that reject at 0.05; and the empirical rank (percentile) of each pair-removal p_T in the
  reference distribution (per null).
- Decision: "the excess is carried by the two near-duplicate pairs" if all 4 pair-removal p_T values exceed the 95th
  percentile of the reference p_T distribution under both nulls. If only some do, report it as it is. If none do, the
  paper must drop the pair attribution and say the excess is not attributable.

## H3. Simple baselines (descriptive, no decision)
- Per-model letter shares (pi_A..pi_D of answers, rotation 0) and max share; flag max > 0.5.
- Residual correlation of every model pair after the all-4 2PL fit (observed minus 2PL-expected correctness,
  Pearson over split columns); top 10 pairs with rank, and the rank of every sibling pair from prereg_G.
- Purpose: show what the spectrum adds beyond these tables (hostile round 3 item 4).

## Report
JSON results/analysis/v3_H_followup.json (+ raw files raw_v3_H_*), S/H_report.md with each decision and any
deviation.
