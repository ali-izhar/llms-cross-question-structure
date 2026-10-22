# K. Selective prediction on the corrected data: do two peer models beat 100-sample self-consistency?

Written before the K run (2026-10-22), after reading the original script (`546a25c:scripts/selective_prediction.py`)
and checking that the data align (the gold letters of the five K = 100 sampled MMLU runs equal
`mmlu_ensemble_rotations.npz:gold_idx` at 500/500 questions). No K result was computed before this file.
Post hoc: this follow-up was chosen after seeing the original result (AUROC 0.807 vs 0.712).

## Original claim (546a25c, Sec. "Robustness and Downstream Impact", App. "Selective Prediction")

Predict whether Qwen2.5-7B's greedy MMLU answer is correct, without labels at inference. Paper text: operational
SC_100 (fraction of 100 T = 1 samples that agree with the modal answer) AUROC 0.712; Ensemble_2 (Llama-3.1-8B +
Gemma-2-9B greedy, confidence = fraction of peers whose answer equals Qwen's greedy answer) 0.807 at ~1/40 the cost;
one peer 0.749 (DeLong p ~ 0.2); two-peer gap p < 0.005.

Note found while reading the code: the 546a25c script did NOT compute the scores the paper describes. Its "SC
operational" was max(p, 1 - p) with p = the fraction of the K samples that are CORRECT, and its "Ensemble_k" was
the fraction of the k peers that are CORRECT. Both use gold labels at inference. No DeLong code exists in the
repository history. K therefore (a) reproduces the 546a25c numbers from the 546a25c artifacts with the 546a25c
code, (b) recomputes the code's label-using definitions on the current data (flagged `label_using`), and (c)
computes the paper's stated, label-free definitions on the current data (primary).

## Data (current, v3)

- Ensemble: `results/mmlu_ensemble_rotations.npz` (replicate A): `chosen_option` [24, 500, 4] (original option
  index, -1 = unparseable), `correct`, `gold_idx`. Rotation 0 is the original option order. Replicate B
  (`mmlu_ensemble_rotations_rep.npz`) is used only for a robustness row of the headline.
- Sampled runs (T = 1, K = 100, iid, rescored with parser v3.1): `results/mmlu_iid-T1.0_<model>.npz`, field
  `parsed` [100, 500] ('' = unparseable). The prompt order is rotation 0.

## Targets

- Primary: Qwen2.5-7B-Instruct. Label y_i = `correct[Qwen, i, 0]` (replicate A, rotation 0; unparseable = wrong).
- Secondary (each has a K = 100 run and a row in the ensemble): Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3,
  SmolLM2-1.7B-Instruct, Phi-3-mini-4k-instruct. Descriptive only; they do not enter the verdict.

## Confidence scores (higher = more likely correct)

Let a_i = the target's rotation-0 greedy chosen option (-1 if unparseable).

1. SC_K (operational, label-free), K in {2, 5, 10, 20, 50, 100}: max over options A-D of (count among the first K
   samples) / K. Unparseable samples count in K but never form the mode. First K passes (prefix), as in 546a25c;
   also the mean AUROC over 200 random K-subsets (K < 100) as a robustness row.
2. SC_K-greedy (label-free, secondary): the fraction of the K samples equal to a_i (agreement with the greedy
   answer that is being judged; the like-for-like analogue of Ensemble_k).
3. Ens_k (label-free), peers greedy, rotation 0: (1/k) sum_j 1[chosen_option[j, i, 0] = a_i]; an unparseable
   peer or an unparseable target counts as no agreement.
4. Rotation-robust ensemble variants (the letter-preference confound):
   - Ens_k-rot{r}, r in {1, 2, 3}: the peer's greedy answer in rotation r (content index) against a_i. Same cost
     as Ens_k (one pass per peer). A shared letter preference cannot raise the agreement, because the same
     content shows under a different letter. Reported per r and as the mean AUROC over r = 1..3.
   - Ens_k-rot4: (1/(4k)) sum_j sum_r 1[chosen_option[j, i, r] = a_i] (content agreement over all 4 rotations).
     Cost 4 passes per peer.
   - Diagnostic: on the questions the target gets wrong, the mean peer agreement in rotation 0 vs rotations 1-3.
     A large drop indicates that rotation-0 agreement on wrong answers is partly letter preference.
5. Label-using (flagged, not deployable), current data: the 546a25c code definitions (SC: max(p, 1 - p) on
   sample correctness; Ens_k: fraction of peers correct) and oracle SC_K (mean sample correctness).

## Peer-selection rules (peers = the 23 other models)

Label-free, fixed here:
- P1 "largest, family-diverse": sort the peers outside the target's family by parameter count (descending; ties by
  model id, alphabetical), take one model per family in that order. k in {1, 2, 3, 5, 10} (if k exceeds the number
  of other families, continue with the second-largest member per family, same order). For Qwen2.5-7B, k = 2 is
  Mistral-Small-22B + Gemma-2-9B. Primary label-free rule.
- P2 "all 23 peers" (k = 23).
- P3 "every pair / random k-subsets": the AUROC distribution over all 253 peer pairs (exhaustive), all 23 single
  peers, and 2000 random subsets for k in {3, 5, 10}. Reported: mean, 5th/50th/95th percentiles, the fraction of
  subsets whose AUROC exceeds SC_100, the fraction significantly above SC_100 (DeLong p < 0.05, one-sided
  direction from the sign), and the paired-bootstrap CI of (mean pair AUROC - SC_100).
- P3' as P3 for k = 2, restricted to pairs whose two families differ from each other and from the target's.
Label-using (flagged): the 546a25c rules on the current rotation-0 accuracies on these 500 questions: L1
"family-diverse by accuracy" and L2 "top-k by accuracy"; and L0 the original named pair (Llama-3.1-8B + Gemma-2-9B)
and single peer (Llama-3.1-8B). L0 is fixed by the old result, so it is also post hoc.

## Cost

Qwen-7B-equivalent forward passes: a greedy pass of a B-parameter model costs B/7 (parameter counts as in 546a25c).
SC_K costs K. Ens_k-rot4 costs 4 x the Ens_k cost.

## Statistics

- AUROC (primary), AUPR (average precision, secondary).
- 95% CIs: paired bootstrap over questions, 4000 resamples, percentile intervals, one seed (20261022), the same
  resamples for every score (so differences are paired).
- Paired comparisons with SC_100 (and with SC_10): DeLong test for correlated AUROCs (two-sided) and the
  paired-bootstrap CI of the AUROC difference.
- The headline is also recomputed on replicate B (robustness row).

## Decision rule (primary target Qwen2.5-7B, k = 2, against operational SC_100)

A comparison "passes" if Delta AUROC = AUROC(Ens) - AUROC(SC_100) > 0, its paired-bootstrap 95% CI excludes 0, and
DeLong p < 0.05.
- HOLDS: (i) P1 Ens_2 rotation 0 passes, (ii) the P3 mean over all 253 pairs passes the bootstrap part, and (iii) the
  rotation-robust P1 Ens_2-rot{1,2,3} (mean over r) passes the bootstrap part, with cost < SC_100 / 10.
- HOLDS ONLY UNDER CONDITIONS: some but not all of (i)-(iii) pass, or only the label-using rules pass. The
  conditions are stated.
- DOES NOT HOLD: none of (i)-(iii) passes.
The "~1/40 the cost" part is reported as the measured cost factor of each rule; it is not a gate.
Secondary targets are reported descriptively (how many of 4 meet the same three conditions).

## Output

`scripts/v3_selective.py` -> `results/analysis/v3_selective.json`.
