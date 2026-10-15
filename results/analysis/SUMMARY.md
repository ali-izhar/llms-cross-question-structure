# Analysis summary (v3 data)

Data: `results/mmlu_ensemble_greedy.npz` (24 x 500 = R2 replicate A rotation 0; canonical tokenization; parser v3.1), `results/mmlu_ensemble_rotations.npz` (A) and `results/mmlu_ensemble_rotations_rep.npz` (B), 24 x 500 x 4 rotations; sampled runs `results/*.npz` (scored with parser v3.1). Code: `scripts/` (drivers `item01`..`item14`, `v3lib.py` + `test_v3lib.py`, `v3_across.py`, `v3_rotation_letters.py`, `v3_loo.py`, `v3_replicate_compare.py`, this file).

Conventions: columns centered and divided by the population SD; R = Z^T Z / rows; binary matrices use the split columns (non-constant columns for the rotation average); lambda_+ = (1 + sqrt(N_cols / M))^2; buffered count = #{lambda > 1.05 lambda_+}; a null rejects rank j when the observed lambda_j exceeds the 99th percentile of the null lambda_j (rank-matched p99). Parametric nulls are sampled conditional on every kept column staying non-constant (N_cols and lambda_+ equal the observed ones). "Directions beyond ability" = ranks 2, 3, ... above the p99, contiguous from rank 2. All seeds are fixed and recorded in each JSON `meta`/`seed`; draw counts: 1000 per null, 500 per Bernoulli null, 1000 rotation-randomized ensembles, 5000 label permutations for eta^2, 10000 for correlations, 20000 for the family tests.

## Committed answers

> **Scope.** The C(3) per-rank verdict below holds for rank 2 only. The family-wise test over ranks 2-4 (G, H) rejects the 2PL and Rasch nulls on all-4 marginally (p_T 0.026 / 0.041; size-calibrated 0.017-0.057); two near-duplicate model pairs (Phi-3/3.5-mini, Llama-3/3.1-8B) carry the excess. The single validity rule (G5) replaces section F's "calibrated validity". The paper follows G and H.

**C(3) Directions beyond ability after letter preference is removed: 0.** One ability direction; the rest was letter preference. The cleanest letter-free matrix is "correct in all 4 rotations": a model that guesses is correct in all four only with probability prod_L pi_m(L) <= 1/256, the same for every question, so a letter preference cannot make a question-specific pattern. Its eigenvalues are [92.1, 29.0, 23.3, 20.8] against lambda_+ = 25.36 (391 split columns). Eigenvalue 2 is below the rank-2 p99 of the project 2PL ([99.5, 31.7, 23.7, 20.8]), its prior version, Rasch JML ([97.3, 30.5, 24.0, 21.0]), Bock-Aitkin a <= 4 and a <= 8, and both 4-rotation know-or-guess nulls (letter-aware [108.3, 39.7, 24.1, 19.9]). It exceeds only the fixed-margin (curveball = exact conditional Rasch, equal slopes) null ([88.3, 27.9, 23.5, 20.9]), which the heterogeneous-slope 2PL explains. Ranks beyond ability by null: {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, 2pl_prior: 0, ba_a4: 0, ba_a8: 0, kg_letter4: 0, kg_uniform4: 0}. Rank 4 (below lambda_+) exceeds the p99 of the Bock-Aitkin and know-or-guess nulls but not of Rasch (ranks above p99: {rasch_jml: [], fixed_margin: [1, 2], 2pl_post: [], 2pl_prior: [], ba_a4: [4], ba_a8: [4], kg_letter4: [4], kg_uniform4: [4]}). Against the project 2PL, rank 4 is a tie within Monte Carlo noise, not a clean non-rejection: observed 20.83 vs p99 20.85 (MC p = 0.012); re-simulating the same fit with seeds 1-3 (1000 draws each; 2PL investigation, not a stored output) gives a rank-4 p99 of 20.74-21.08, so rank 4 is above the p99 for two seeds and below it for one, and replicate B puts it above (C4). Rank 4 is not contiguous with rank 1, lies inside the MP bulk, and its model scores group by family (eta^2 = 0.74, p = 0.006): a candidate weak family signal in the bulk, not a direction by our rule.

The other two letter-free constructions agree, with one borderline case each. Rotation-randomized ensembles (1000 draws): median eigenvalues [79.5, 40.9, 32.1, 26.0]; in the representative (median) draw no rank exceeds the 3PL or the letter-aware p99 (rank 2: [78.87, 41.05, 32.34, 25.86] vs letter-aware p99 [86.26, 41.09, 33.62, 28.78]: at the threshold); the share of the 1000 draws whose lambda_2 exceeds that p99 is [0.010, 0.462, 0.177, 0.017]; against the know-or-guess null fitted on all 4 rotations with a fresh random rotation per draw, the median draw exceeds no rank (p99 [87.7, 43.1, 34.0, 29.4]; share of draws above: [0.006, 0.138, 0.118, 0.008]). Rotation-averaged correctness: eigenvalues [127.5, 43.1, 28.5, 25.7]; rank 2 exceeds the letter-aware know-or-guess p99 ([104.6, 41.8, 29.2, 25.6]) but not the uniform-guess one ([107.6, 44.3, 32.1, 26.9]), and both nulls miss rank 1 (observed / p99 = {kg_letter4: 1.22, kg_uniform4: 1.18}): they tie correctness to ability less tightly than the data, as the v2 2PL did at rank 1, so this rank-2 excess is not a clean rejection. The rotation-independent 2PL ([150.0, 37.6, 27.3, 23.9]) is liberal by construction (it averages away the cell noise) and is reported only as a bound.

**B letter analysis (rotation 0).** Eigenvector 2 is an answer-letter direction. Its question loadings group by gold letter (eta^2 = 0.482, permutation p = 0.0002; mean loading by letter {A: 0.052, B: 0.003, C: 0.005, D: -0.039}); its model scores correlate with the models' answer-letter shares: A r = 0.76 (p = 0.0013), D r = -0.90 (p = 1.0e-04), accuracy r = 0.16. Eigenvector 1 is ability (r with accuracy 0.94) with a letter part (eta^2 = 0.22; r with share C 0.81). Eigenvectors 3 and 4: eta^2 = 0.087 and 0.061 (small; significant only because of the 460 questions); no letter-share correlation has p < 0.05. The direction moves with the DISPLAYED letters (decisive test, C1b): in rotations 1-3 the same models carry it (|r| of model scores with rotation 0: 0.98, 0.96, 0.95); its question loadings follow the rotation-0 letter profile mapped through the shown letters (r = 0.64, 0.60, 0.61), not through the original options (r = -0.36, 0.07, -0.29), and do not correlate with the rotation-0 loadings of the same questions (r = -0.06, 0.16, -3.0e-03); permutation p of r_shown - r_orig = 0.0002 in each rotation. It rests mostly on one model: Llama-3.2-1B guesses A with probability [0.90, 0.02, 0.04, 0.02]. Without it, rotation-0 eigenvalues are [76.0, 35.0, 32.5, 29.0], eigenvector 2's gold-letter eta^2 falls to [0.18, 0.08, 0.21, 0.08] (the letter part moves to eigenvector 3); every other single removal leaves eigenvalue 2 in [47.1, 52.4].

**B nulls (rotation 0, which ranks survive).** Eigenvalues 1-6: [85.9, 49.9, 29.7, 27.3, 25.9, 21.7] against lambda_+ = 28.92 (gamma = 460/24); MP count 3, buffered 2. Difficulty-matched Bernoulli null: at most 1 buffered eigenvalue in 500 draws (rank p99 [29.6, 27.1, 25.8, 25.0]); per-model-rate null: at most 1 (rank p99 [46.8, 31.3, 29.7, 28.3]). Ranks above the rank-matched p99, per ability-keeping null:

| null | model | rank p99 (1-4) | ranks above |
|---|---|---|---|
| Rasch JML | sigma(theta_m - b_i), joint ML | [59.0, 30.7, 28.0, 25.8] | [1, 2, 3, 4] |
| Rasch CML | theta by conditional ML | [56.4, 30.9, 27.7, 25.8] | [1, 2, 3, 4] |
| fixed margins (curveball) | uniform over 0/1 matrices with the observed margins | [53.3, 30.7, 27.6, 26.0] | [1, 2, 3, 4] |
| 2PL, project (posterior draws) | theta_m fixed; (log a_i, b_i) ~ normal, integrated (MML over questions) | [78.2, 34.1, 28.8, 26.0] | [1, 2, 3, 4] |
| 2PL, project (prior draws) | same fit; (a_i, b_i) from the population law | [71.5, 31.9, 27.4, 25.3] | [1, 2, 3, 4] |
| 2PL, Bock-Aitkin a <= 4 | theta ~ N(0,1) integrated; (a_i, d_i) fixed, box-bounded; ML theta_m | [100.7, 41.2, 30.9, 26.1] | [2, 4] |
| 2PL, Bock-Aitkin a <= 8 | as above | [100.0, 42.5, 30.6, 26.1] | [2, 4] |
| 3PL, c = 0.25 | 0.25 + 0.75 sigma(a_i(theta_m - b_i)), project MML | [91.0, 42.6, 34.4, 28.9] | [2] |
| letter-aware know-or-guess (rotation-0 fit) | pi_m(g_i) + (1 - pi_m(g_i)) sigma(...), project MML | [89.0, 47.7, 32.8, 28.2] | [2] |
| know-or-guess, 4-rotation fit, letter-aware | composite MML on 4 rotations; know once per cell | [84.7, 47.2, 32.4, 27.1] | [1, 2, 4] |
| know-or-guess, 4-rotation fit, uniform guess | as above with pi = 0.25 | [86.0, 42.1, 33.6, 28.7] | [2] |
| stress: project 2PL, slopes x1.25 (not fitted) | log-lik vs fit -12.9 | [91.8, 37.0, 30.9, 26.4] | [2, 4] |
| stress: project 2PL, slopes x1.5 (not fitted) | log-lik vs fit -43.1 | [103.0, 41.2, 31.9, 26.7] | [2, 4] |
| stress: project 2PL, slopes x2 (not fitted) | log-lik vs fit -128.2 | [122.5, 46.9, 34.5, 26.8] | [2, 4] |
| stress: project 2PL, slopes x3 (not fitted) | log-lik vs fit -328.9 | [149.5, 56.0, 36.4, 27.3] | [] |
| stress: project 2PL, slopes x5 (not fitted) | log-lik vs fit -721.8 | [175.7, 69.9, 41.3, 29.9] | [] |
| stress: project 2PL, slopes x10 (not fitted) | log-lik vs fit -1515.0 | [196.4, 89.2, 48.7, 31.8] | [] |

Rank 2 (the letter direction) exceeds every fitted null, including both letter-aware ones, and every stress test up to x2; the know-or-guess form (letter preference acts only when the model does not know) under-produces the letter structure, which the rotation analyses identify directly. Without Llama-3.2-1B (23 models, refitted) the ranks above the 3PL p99 are [4] and above the letter-aware p99 [4] (rank 4 only; observed minus p99 {3pl_post: 0.35, letter_post: 0.57}). Fits: project 2PL log-lik -5839.2, 3PL -5774.2, letter-aware -5417.8 (same number of parameters); Bock-Aitkin slopes at the bound: 0.078 (a <= 4), 0.076 (a <= 8).

Joint statistic lambda_2 / lambda_1: observed 0.581 (v2 data: 0.503). Against the project 2PL and all six slope multipliers pooled: null p99 0.463, max 0.498, MC p = 0.00017; fitted 2PL posterior max 0.511. It does not separate the data from the letter-aware nulls: p = 0.023 (rotation-0 fit), 0.095 (4-rotation fit). Against Rasch and fixed margins the ratio is uninformative (their lambda_1 is far below the observed, so their ratio p99 is 0.597).

**Family (model level).** QAP / model-label permutation test of same- vs cross-family mean pair r: same 0.414, cross 0.375, one-sided p = 0.085 (Holm 0.342). Eigenvector-family alignment (eta^2 of model scores, label permutation): eigenvectors 1-4 eta^2 = 0.19, 0.49, 0.63, 0.85, p = 0.8829, 0.3308, 0.1192, 0.0028; Holm-adjusted (over QAP, eigenvectors 1-4 and the joint 2-4 test): 0.883, 0.662, 0.358, 0.017; joint 2-4 p = 0.031 (Holm 0.154). Only eigenvector 4 (rank 4, eigenvalue below lambda_+) aligns with family after Holm; with the five singleton families removed its p is 0.048. The pair-level KS test is superseded (it treats the 276 dependent pairs as independent).

**Replicate agreement (C4).** Rotation 0 is identical in A and B. Largest eigenvalue difference over every matrix compared: 0.081. Null verdicts: 4 of 33 differ, all at rank 4 ([randomized_representative:rasch_jml, all4_correct:fixed_margin, all4_correct:2pl_post, all4_correct:ba_a8]); no verdict on ranks 1-3 differs, and every committed answer above is the same in B.

**Near ties (C5).** Cells with a letter gap < 0.13 in A or B: 0.050 of all cells ([595, 565, 641, 610]). Each binary variant refitted with its own nulls: rotation 0 with near ties set to the majority over rotations (86 cells change): ranks above 3PL [2], letter-aware [2] (unchanged); dropping the 297 questions with a rotation-0 near tie leaves 203: ranks above letter-aware [], 3PL [2]. All-4-correct, majority: ranks beyond ability {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, ba_a4: 0, kg_letter4: 0}; near-tie rotations ignored: {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, ba_a4: 0, kg_letter4: 0}. The C(3) answer (0 beyond ability, rank 2 only above the fixed-margin null, rank 4 only above Bock-Aitkin / know-or-guess) does not change.

**D. Within-model shared-component directions.** S1 (Qwen2.5-7B shared seed) and S2 Llama-3-8B (shared 5-shot) are letter directions: the top eigenvector's loadings group by gold letter (eta^2 = 0.73, p = 0.0002; Llama S2 0.53, p = 0.0002) and not by subject (S1 eta^2 0.70 vs null mean 0.63, p = 0.229; Llama S2 0.20, p = 0.349). S2 Qwen2.5-7B is an exemplar-overlap direction: no gold-letter grouping (eta^2 = 0.07, p = 0.094), passes that share an exemplar answer more alike (z = 10.7, p = 0.0005); its loadings group weakly by subject (eta^2 0.58 vs null mean 0.46, p = 0.042, uncorrected), and a same-subject exemplar raises accuracy by 0.0324 (p = 0.033). The v3.1 rescore did not change these three runs (their item-1 statistics are identical to v2).

## Changes against the submitted (April) paper

1. The paper's across-model claim, "one direction that follows ability and at least one more that a one-ability model does not reproduce" (abstract, Sec. 3.3, Limits, Discussion of the submitted version), resting on eigenvalue 2, does not hold as a statement about the models' knowledge. Eigenvalue 2 is answer-letter preference: it follows the displayed letters when the options rotate (C1b), it is carried mostly by one model (Llama-3.2-1B, pi_A = 0.90), and in letter-free matrices no direction beyond ability exceeds the 2PL / Rasch / Bock-Aitkin / know-or-guess nulls.
2. Eigenvalues: [85.9, 49.9, 29.7, 27.3] against lambda_+ = 28.92 (paper: 92.4, 46.5, 33.7, 25.3 against 29.13): two exceed 1.05 lambda_+, not three.
3. Against the project 2PL, ranks 1-4 exceed their p99 (paper: 1-3), as they do against Rasch (JML, CML) and fixed margins. The Bock-Aitkin 2PL rejects ranks 2 and 4; with the 4-option guessing floor (3PL, c = 0.25) only rank 2 exceeds.
4. Stress tests: ranks 2 and 4 stay above up to x2; at x3, x5 and x10 no rank does (at x3 rank 4 sits at its p99: 27.27 vs 27.33). The paper's "only eigenvalue 2 ... stays above when every slope is doubled" now reads ranks 2 and 4.
5. Family: the paper's "eigenvectors 2-4 jointly do (p = 0.013) ... only eigenvector 4 reaches p = 0.034" becomes joint p = 0.031, eigenvector 4 p = 0.0028 (Holm 0.017); the model-level QAP test gives p = 0.085 (the pair-level KS p = 0.024 treats dependent pairs as independent). Pair r: same 0.414 vs cross 0.375 (paper 0.41 / 0.34), Cohen's d 0.30 (paper 0.58).
6. Within-model (Table 1): Mistral-7B-v0.3 MMLU N_b 83 (v2 86), TW z -1.79 (v2 -2.41); Phi-3-mini N_b 271 (v2 273); HellaSwag Mistral N_b 65 (v2 64). Every iid MP, buffered and Horn count stays 0 (the paper's within-model statements hold).

## A. Within-model items

| run | acc | N_b | lambda_+ | lambda_max | TW z | MC pct of z | MP | buffered | Horn |
|---|---|---|---|---|---|---|---|---|---|
| MMLU Qwen2.5-7B T1.0 | 0.7042 | 43 | 2.741 | 2.562 | -1.72 | 0.601 | 0 | 0 | 0 |
| MMLU Mistral-7B-v0.3 | 0.5708 | 83 | 3.652 | 3.449 | -1.79 | 0.513 | 0 | 0 | 0 |
| MMLU Meta-Llama-3-8B | 0.6318 | 173 | 5.361 | 4.965 | -3.05 | 0.102 | 0 | 0 | 0 |
| MMLU SmolLM2-1.7B | 0.3949 | 406 | 9.090 | 8.742 | -2.17 | 0.286 | 0 | 0 | 0 |
| MMLU Phi-3-mini | 0.6425 | 271 | 7.002 | 6.908 | -0.65 | 0.835 | 0 | 0 | 0 |
| HellaSwag Qwen2.5-7B | 0.8519 | 16 | 2.451 | 2.193 | -1.60 | 0.699 | 0 | 0 | 0 |
| HellaSwag Mistral-7B-v0.3 | 0.6892 | 65 | 4.580 | 4.053 | -2.71 | 0.197 | 0 | 0 | 0 |
| HellaSwag Meta-Llama-3-8B | 0.6686 | 86 | 5.343 | 4.746 | -2.90 | 0.113 | 0 | 0 | 0 |
| GSM8K Qwen2.5-7B 256 tok | 0.5097 | 62 | 5.942 | 5.322 | -2.06 | 0.379 | 0 | 0 | 0 |
| GSM8K Qwen2.5-7B 1024 tok | 0.9033 | 19 | 3.225 | 2.502 | -2.97 | 0.120 | 0 | 0 | 0 |
| MMLU Qwen2.5-7B T0.5 | 0.7035 | 25 | 3.659 | 3.385 | -1.08 | 0.785 | 0 | 0 | 0 |
| MMLU Qwen2.5-7B T1.5 | 0.6989 | 85 | 7.200 | 6.529 | -2.07 | 0.327 | 0 | 0 | 0 |

Shared-component runs (item 2):

| run | N_b | lambda_+ | lambda_max | TW z | buffered | Horn | letter-share dispersion ratio | pass dispersion ratio (p) |
|---|---|---|---|---|---|---|---|---|
| MMLU Qwen2.5-7B S1 shared seed | 41 | 2.69 | 10.00 | 70.2 | 4 | 5 | 11.8 | 0.83 (0.8930) |
| MMLU Qwen2.5-7B S2 shared 5-shot | 93 | 3.86 | 4.77 | 7.9 | 2 | 10 | 1.8 | 1.21 (0.0748) |
| MMLU Meta-Llama-3-8B S2 shared 5-shot | 264 | 6.89 | 11.31 | 30.9 | 2 | 5 | 11.3 | 1.51 (0.0010) |

Item 3 (power): unchanged (the Qwen2.5-7B and Llama-3-8B iid runs did not change under the rescore); `item03_power.json` is byte-identical to v2. Item 5: split-half r (Qwen2.5-7B, all / borderline columns) 0.9975 / 0.9403. Item 6: modal-wrong-letter share 0.955; letter association across questions exceeds its p99: False. Item 12: T = 0.5 / 1.5 N_b 25 / 85, MP counts 0 / 0; GSM8K 256-token truncation share 0.550, accuracy 256 / 1024 tokens 0.510 / 0.903. Item 13 (now reads R2 rotation 0): across-model buffered count as scored / without questions with any unparseable reply / without Mistral-7B-v0.2 / imputed: 2 / 3 / 2 / 2; 2PL ranks above p99 in the two binary variants: [1, 2, 3, 4] / [1, 2, 3, 4].

## C. Rotation analyses (replicate A; B in C4)

| rotation | eigenvalues 1-4 | lambda_+ | buffered | eta^2 by gold letter, eigvec 1-4 | eigvec 2: r with share A / D |
|---|---|---|---|---|---|
| 0 | [85.9, 49.9, 29.7, 27.3] | 28.92 | 2 | 0.22, 0.48, 0.09, 0.06 | 0.76 / -0.90 |
| 1 | [83.2, 48.4, 30.8, 28.1] | 28.87 | 3 | 0.22, 0.41, 0.05, 0.15 | 0.78 / -0.86 |
| 2 | [81.5, 44.3, 33.9, 28.1] | 28.56 | 3 | 0.19, 0.36, 0.11, 0.12 | 0.75 / -0.81 |
| 3 | [88.8, 54.7, 31.5, 25.1] | 29.49 | 3 | 0.26, 0.38, 0.05, 0.04 | 0.70 / -0.90 |

Within one rotation the shown gold letter is a relabelling of the original one, so eta^2 cannot tell them apart; the cross-rotation test (C1b, `v3_C1b_letter_following.json`) can, and is reported above. Eigenvector 1 also follows the shown letters in part (r_shown 0.43 vs r_orig 0.10 in rotation 1) but keeps its content (r_direct 0.55); eigenvector 3 is matched only loosely across rotations (model-score |r| 0.75).

Rotation-randomized ensembles (C2): buffered-count distribution {2: 195, 3: 805}, strict {2: 37, 3: 956, 4: 7}; eigenvalue p01-p99 [73.1, 36.4, 28.3, 23.7] - [86.1, 45.5, 36.1, 29.0]. Nulls fitted to ONE representative draw (the draw closest to the median eigenvalues 1-4; refitting 1000 draws x 6 nulls was not run), 1000 null draws each; ranks above p99: {rasch_jml: [1, 2, 3], fixed_margin: [1, 2, 3], 2pl_post: [1, 2, 3], 3pl_post: [], letter_post: [], ba_a4: [2, 3]}. Robustness, 10 other draws refitted (2PL and letter-aware, 300 null draws each): [{draw: 852, top_eigs: [75.179, 41.064, 35.490, 25.797], ranks_above_p99: {2pl_post: [2, 3], letter_post: [2, 3]}, beyond_ability: {2pl_post: 2, letter_post: 2}}, {draw: 547, top_eigs: [78.847, 43.147, 30.724, 26.465], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: [2]}, beyond_ability: {2pl_post: 3, letter_post: 1}}, {draw: 504, top_eigs: [76.373, 37.000, 31.873, 25.489], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: []}, beyond_ability: {2pl_post: 2, letter_post: 0}}, {draw: 953, top_eigs: [82.443, 38.890, 33.112, 26.980], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: [3]}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 820, top_eigs: [75.760, 39.603, 33.271, 26.716], ranks_above_p99: {2pl_post: [2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 768, top_eigs: [78.017, 41.441, 30.283, 27.836], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 546, top_eigs: [87.151, 44.785, 29.227, 25.362], ranks_above_p99: {2pl_post: [1, 2], letter_post: [2]}, beyond_ability: {2pl_post: 1, letter_post: 1}}, {draw: 822, top_eigs: [76.411, 34.923, 32.175, 25.102], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: []}, beyond_ability: {2pl_post: 2, letter_post: 0}}, {draw: 61, top_eigs: [78.013, 37.601, 31.561, 26.390], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 664, top_eigs: [80.861, 42.937, 32.652, 24.634], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: [2]}, beyond_ability: {2pl_post: 2, letter_post: 1}}]. Representative-draw eigenvector 2 still correlates with the models' letter shares (A 0.70, D -0.62) but its question loadings do not group by gold letter (eta^2 0.006): the models' letter shares still covary with which questions they answer correctly, but the shown letter no longer groups the questions.

Leave-one-out (all-4-correct): without Llama-3.2-1B the 23-model all-4 matrix has ranks beyond ability {rasch_jml: 0, fixed_margin: 0, 2pl_post: 0, ba_a4: 0, kg_letter4: 0}.

## E. Documentation notes

* `results/PROTOCOL.md` states the parser where the `src/scoring.py` docstring is less exact (Option/Choice and negation words in any case; any other character after the letter -> `letter_then_alnum`; GSM8K "#### <number>" form), the seed base `(crc32 & 0x7FFFFFFF) << 20`, the R2 letter-logprob differences (7 models; Ministral-8B: 4 cells, <= 2e-4), and the near-tie definition "gap < 0.13 in A or in B" on the ~0.125 grid (97% of gaps below 0.6 lie within 0.01 of a multiple of 0.125).
* The `src/scoring.py` docstring is not edited: `scorer_sha256` is the sha256 of the whole file (collect.py, rescore.py) and must stay f804dbf14ef19530.
* `results/manifest.json` (via `src/build_manifest.py`) holds `runs_rep_B` (24 B runs, npz + json sha256), `ensemble_files_sha256` (3 files) and `json_sha256` per run. The builder rewrites an ensemble npz only if its arrays change, so the zip time stamp does not change the sha256.

## Notes, deviations

* The 2PL fit is the project's model: abilities theta_m are fixed effects; question parameters (log a_i, b_i) are independent normal random effects integrated on a 41 x 401 grid (marginal ML over questions); question-matched posterior draws. Two of three random restarts abort in the line search (L-BFGS-B ABNORMAL, converged=False; stored log-lik [-5880.4, -5868.1, -5839.2]); continued, both reach -5839.17. The main fit is the optimum and agrees with a 61 x 801 grid. (`fit_twopl_mml` continues an aborted run with bounded hyperparameters and records per-restart `converged`; the stored restart values predate this.) The standard alternative is Bock-Aitkin MML (theta ~ N(0,1) integrated, items fixed, slopes box-bounded), simulated at ML abilities given the items.
* The know-or-guess letter preference pi_m is a moment estimate from all 4 rotations (`v3lib.letter_preference`); no model needed clipping. The 4-rotation fits use a composite (rotation-independence) likelihood; their per-question posteriors are narrower than the true ones.
* The rotation-averaged matrix has no binary parametric null; its nulls simulate 4 rotations per cell from the 4-rotation fits. Near-tie fractional variants (imputed cells) report spectra only (`fractional_variants_spectra_only`).
* Item 8 (sub-ensembles) is not run (`item08_09_sub_ensembles_family.py --with-item08` runs it). Items 11 and 14 use the v3 ensemble (not headline).
* Item 2: the added subject test draws from the run's shared RNG stream, so later permutation p-values in item 2 move by Monte Carlo noise (exemplar-overlap z 10.53 -> 10.73 for Qwen S2); the observed statistics are identical.
* `scripts/across_two_margin_null.py::main` takes `stress_factors` and `eig_store`; the defaults reproduce the stored output, and adding stress factors leaves the existing ones unchanged.

## Headline table

Every number above, read from the JSON key in the last column (files in `results/analysis/`).

| section | quantity | value | JSON key |
|---|---|---|---|
| committed | all-4: eigenvalues 1-4 | [92.1, 29.0, 23.3, 20.8] | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.spectrum.top_eigs` `[0:4]` |
| committed | all-4: lambda_+ | 25.36 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.spectrum.lambda_plus` |
| committed | all-4: split columns | 391 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.spectrum.n_cols` |
| committed | all-4: 2PL p99 | [99.5, 31.7, 23.7, 20.8] | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.2pl_post.rank_eigs_p99` |
| committed | all-4: Rasch p99 | [97.3, 30.5, 24.0, 21.0] | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.rasch_jml.rank_eigs_p99` |
| committed | all-4: KG letter p99 | [108.3, 39.7, 24.1, 19.9] | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.kg_letter4.rank_eigs_p99` |
| committed | all-4: curveball p99 | [88.3, 27.9, 23.5, 20.9] | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.fixed_margin.rank_eigs_p99` |
| committed | all-4: ranks beyond ability by null | {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, 2pl_prior: 0, ba_a4: 0, ba_a8: 0, kg_letter4: 0, kg_uniform4: 0} | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.ranks_beyond_ability` |
| committed | all-4: ranks above p99 by null | {rasch_jml: [], fixed_margin: [1, 2], 2pl_post: [], 2pl_prior: [], ba_a4: [4], ba_a8: [4], kg_letter4: [4], kg_uniform4: [4]} | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.ranks_above_p99_by_null` |
| committed | all-4: eigenvalue 4 | 20.83 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.spectrum.top_eigs[3]` |
| committed | all-4: 2PL rank-4 p99 | 20.85 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.2pl_post.rank_eigs_p99[3]` |
| committed | all-4: 2PL rank-4 MC p | 0.012 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.suite.nulls.2pl_post.mc_pvalue_per_rank[3]` |
| committed | all-4 eigvec 4: family eta^2 | 0.74 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.letters_subject_family.eigenvectors[3].eta_sq_model_scores_by_family.eta_sq` |
| committed | all-4 eigvec 4: family p | 0.006 | `v3_C_rotations_A.json`: `C3_rotation_invariant.all4_correct.letters_subject_family.eigenvectors[3].eta_sq_model_scores_by_family.perm_p` |
| committed | C2: number of random-rotation ensembles | 1000 | `v3_C_rotations_A.json`: `meta.n_random_rotation_draws` |
| committed | C2: median eigenvalues 1-4 | [79.5, 40.9, 32.1, 26.0] | `v3_C_rotations_A.json`: `C2_rotation_randomized.distribution.eig_median` `[0:4]` |
| committed | C2 representative: eigenvalues | [78.87, 41.05, 32.34, 25.86] | `v3_C_rotations_A.json`: `C2_rotation_randomized.representative_draw.spectrum.top_eigs` `[0:4]` |
| committed | C2 representative: letter-aware p99 | [86.26, 41.09, 33.62, 28.78] | `v3_C_rotations_A.json`: `C2_rotation_randomized.representative_suite.nulls.letter_post.rank_eigs_p99` |
| committed | C2: share of draws above representative letter-aware p99 (ranks 1-4) | [0.010, 0.462, 0.177, 0.017] | `v3_C_rotations_A.json`: `C2_rotation_randomized.frac_draws_above_representative_p99.letter_post` |
| committed | C2: KG random-rotation p99 | [87.7, 43.1, 34.0, 29.4] | `v3_C_rotations_A.json`: `C2_rotation_randomized.kg_letter4_random_rotation_null.rank_eigs_p99` |
| committed | C2: share of draws above KG p99 | [0.006, 0.138, 0.118, 0.008] | `v3_C_rotations_A.json`: `C2_rotation_randomized.kg_letter4_random_rotation_null.frac_observed_draws_above_p99_per_rank` |
| committed | mean over rotations: eigenvalues | [127.5, 43.1, 28.5, 25.7] | `v3_C_rotations_A.json`: `C3_rotation_invariant.mean_over_rotations.spectrum.top_eigs` `[0:4]` |
| committed | mean: KG letter p99 | [104.6, 41.8, 29.2, 25.6] | `v3_C_rotations_A.json`: `C3_rotation_invariant.mean_over_rotations.suite.nulls.kg_letter4.rank_eigs_p99` |
| committed | mean: KG uniform p99 | [107.6, 44.3, 32.1, 26.9] | `v3_C_rotations_A.json`: `C3_rotation_invariant.mean_over_rotations.suite.nulls.kg_uniform4.rank_eigs_p99` |
| committed | mean: rank-1 observed / p99 (KG letter, KG uniform) | {kg_letter4: 1.22, kg_uniform4: 1.18} | `v3_summary_derived.json`: `mean_over_rotations_rank1_obs_over_p99` |
| committed | mean: independent-rotation 2PL p99 | [150.0, 37.6, 27.3, 23.9] | `v3_C_rotations_A.json`: `C3_rotation_invariant.mean_over_rotations.suite.nulls.indep4_2pl.rank_eigs_p99` |
| committed | rot0 eigvec 2: eta^2 by gold letter | 0.482 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].eta_sq_by_shown_gold_letter.eta_sq` |
| committed | rot0 eigvec 2: eta^2 p | 0.0002 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].eta_sq_by_shown_gold_letter.perm_p` |
| committed | rot0 eigvec 2: mean loading by gold letter | {A: 0.052, B: 0.003, C: 0.005, D: -0.039} | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].eta_sq_by_shown_gold_letter.mean_loading_by_letter` |
| committed | rot0 eigvec 2: r with share A | 0.76 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].model_score_corr.share_A.r` |
| committed | rot0 eigvec 2: p share A | 0.0013 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].model_score_corr.share_A.perm_p_two_sided` |
| committed | rot0 eigvec 2: r with share D | -0.90 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].model_score_corr.share_D.r` |
| committed | rot0 eigvec 2: p share D | 1.0e-04 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].model_score_corr.share_D.perm_p_two_sided` |
| committed | rot0 eigvec 2: r with accuracy | 0.16 | `v3_B_extra.json`: `letter_analysis.eigenvectors[1].model_score_corr.accuracy.r` |
| committed | rot0 eigvec 1: r with accuracy | 0.94 | `v3_B_extra.json`: `letter_analysis.eigenvectors[0].model_score_corr.accuracy.r` |
| committed | rot0 eigvec 1: eta^2 by gold letter | 0.22 | `v3_B_extra.json`: `letter_analysis.eigenvectors[0].eta_sq_by_shown_gold_letter.eta_sq` |
| committed | rot0 eigvec 1: r with share C | 0.81 | `v3_B_extra.json`: `letter_analysis.eigenvectors[0].model_score_corr.share_C.r` |
| committed | rot0 eigvec 3: eta^2 by gold letter | 0.087 | `v3_B_extra.json`: `letter_analysis.eigenvectors[2].eta_sq_by_shown_gold_letter.eta_sq` |
| committed | rot0 eigvec 4: eta^2 by gold letter | 0.061 | `v3_B_extra.json`: `letter_analysis.eigenvectors[3].eta_sq_by_shown_gold_letter.eta_sq` |
| committed | C1b eigvec 2 rotation 1: model-score match | 0.98 | `v3_C1b_letter_following.json`: `A.eigvec2.1.model_score_match_abs_r` |
| committed | C1b eigvec 2 rotation 2: model-score match | 0.96 | `v3_C1b_letter_following.json`: `A.eigvec2.2.model_score_match_abs_r` |
| committed | C1b eigvec 2 rotation 3: model-score match | 0.95 | `v3_C1b_letter_following.json`: `A.eigvec2.3.model_score_match_abs_r` |
| committed | C1b eigvec 2 r1: r_shown | 0.64 | `v3_C1b_letter_following.json`: `A.eigvec2.1.r_shown` |
| committed | C1b eigvec 2 r2: r_shown | 0.60 | `v3_C1b_letter_following.json`: `A.eigvec2.2.r_shown` |
| committed | C1b eigvec 2 r3: r_shown | 0.61 | `v3_C1b_letter_following.json`: `A.eigvec2.3.r_shown` |
| committed | C1b eigvec 2 r1: r_orig | -0.36 | `v3_C1b_letter_following.json`: `A.eigvec2.1.r_orig` |
| committed | C1b eigvec 2 r2: r_orig | 0.07 | `v3_C1b_letter_following.json`: `A.eigvec2.2.r_orig` |
| committed | C1b eigvec 2 r3: r_orig | -0.29 | `v3_C1b_letter_following.json`: `A.eigvec2.3.r_orig` |
| committed | C1b eigvec 2 r1: r_direct | -0.06 | `v3_C1b_letter_following.json`: `A.eigvec2.1.r_direct` |
| committed | C1b eigvec 2 r2: r_direct | 0.16 | `v3_C1b_letter_following.json`: `A.eigvec2.2.r_direct` |
| committed | C1b eigvec 2 r3: r_direct | -3.0e-03 | `v3_C1b_letter_following.json`: `A.eigvec2.3.r_direct` |
| committed | C1b eigvec 2 r1: p | 0.0002 | `v3_C1b_letter_following.json`: `A.eigvec2.1.perm_p_two_sided` |
| committed | Llama-3.2-1B letter preference A-D | [0.90, 0.02, 0.04, 0.02] | `v3_C_rotations_A.json`: `kg_fits_4_rotations.letter_preference.meta-llama/Llama-3.2-1B-Instruct.pi_A_B_C_D` |
| committed | rot0 without Llama-3.2-1B: eigenvalues | [76.0, 35.0, 32.5, 29.0] | `v3_B_leave_one_out.json`: `leave_one_out_spectra.meta-llama/Llama-3.2-1B-Instruct.rot0.top_eigs` `[0:4]` |
| committed | rot0 without Llama-3.2-1B: eta^2 by gold letter, eigvec 1-4 | [0.18, 0.08, 0.21, 0.08] | `v3_B_leave_one_out.json`: `leave_one_out_spectra.meta-llama/Llama-3.2-1B-Instruct.rot0_eta_sq_gold_letter_eigvec_1to4` |
| committed | rot0 eigenvalue 2 range over the other 23 single-model removals | [47.1, 52.4] | `v3_summary_derived.json`: `loo_rot0_lambda2_min_max_without_llama_3_2_1b_removal` |
| committed | rot0: eigenvalues 1-6 | [85.9, 49.9, 29.7, 27.3, 25.9, 21.7] | `v3_B_extra.json`: `suite.observed.top_eigs` |
| committed | rot0: lambda_+ | 28.92 | `item07_across_model.json`: `observed.lambda_plus` |
| committed | rot0: split columns | 460 | `item07_across_model.json`: `observed.n_split` |
| committed | rot0: MP count strict | 3 | `item07_across_model.json`: `observed.mp_count_strict` |
| committed | rot0: buffered count | 2 | `item07_across_model.json`: `observed.mp_count_buffered` |
| committed | Bernoulli (per question) null: max buffered count | 1 | `item07_across_model.json`: `bernoulli_nulls_d_across.A_per_question.mp_signal_max` |
| committed | Bernoulli per-question null: rank p99 | [29.6, 27.1, 25.8, 25.0] | `v3_B_extra.json`: `bernoulli_rank_nulls.bern_question.rank_eigs_p99` |
| committed | per-model-rate null: max buffered count | 1 | `item07_across_model.json`: `bernoulli_nulls_d_across.B_per_model.mp_signal_max` |
| committed | per-model-rate null: rank p99 | [46.8, 31.3, 29.7, 28.3] | `v3_B_extra.json`: `bernoulli_rank_nulls.bern_model.rank_eigs_p99` |
| committed | Rasch JML: rank p99 | [59.0, 30.7, 28.0, 25.8] | `raw_item07_across_two_margin_null.json`: `nulls.a_rasch_jml.rank_eigs_p99` |
| committed | Rasch JML: ranks above p99 | [1, 2, 3, 4] | `raw_item07_across_two_margin_null.json`: `nulls.a_rasch_jml.observed_ranks_above_p99` |
| committed | Rasch CML: rank p99 | [56.4, 30.9, 27.7, 25.8] | `raw_item07_across_two_margin_null.json`: `nulls.a_rasch_cml.rank_eigs_p99` |
| committed | Rasch CML: ranks above p99 | [1, 2, 3, 4] | `raw_item07_across_two_margin_null.json`: `nulls.a_rasch_cml.observed_ranks_above_p99` |
| committed | fixed margins (curveball): rank p99 | [53.3, 30.7, 27.6, 26.0] | `raw_item07_across_two_margin_null.json`: `nulls.b_fixed_margin.rank_eigs_p99` |
| committed | fixed margins (curveball): ranks above p99 | [1, 2, 3, 4] | `raw_item07_across_two_margin_null.json`: `nulls.b_fixed_margin.observed_ranks_above_p99` |
| committed | 2PL, project (posterior draws): rank p99 | [78.2, 34.1, 28.8, 26.0] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl.rank_eigs_p99` |
| committed | 2PL, project (posterior draws): ranks above p99 | [1, 2, 3, 4] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl.observed_ranks_above_p99` |
| committed | 2PL, project (prior draws): rank p99 | [71.5, 31.9, 27.4, 25.3] | `raw_item07_across_two_margin_null.json`: `nulls.c_prior_2pl.rank_eigs_p99` |
| committed | 2PL, project (prior draws): ranks above p99 | [1, 2, 3, 4] | `raw_item07_across_two_margin_null.json`: `nulls.c_prior_2pl.observed_ranks_above_p99` |
| committed | 2PL, Bock-Aitkin a <= 4: rank p99 | [100.7, 41.2, 30.9, 26.1] | `v3_B_extra.json`: `suite.nulls.ba_a4.rank_eigs_p99` |
| committed | 2PL, Bock-Aitkin a <= 4: ranks above p99 | [2, 4] | `v3_B_extra.json`: `suite.nulls.ba_a4.observed_ranks_above_p99` |
| committed | 2PL, Bock-Aitkin a <= 8: rank p99 | [100.0, 42.5, 30.6, 26.1] | `v3_B_extra.json`: `suite.nulls.ba_a8.rank_eigs_p99` |
| committed | 2PL, Bock-Aitkin a <= 8: ranks above p99 | [2, 4] | `v3_B_extra.json`: `suite.nulls.ba_a8.observed_ranks_above_p99` |
| committed | 3PL, c = 0.25: rank p99 | [91.0, 42.6, 34.4, 28.9] | `v3_B_extra.json`: `suite.nulls.3pl_post.rank_eigs_p99` |
| committed | 3PL, c = 0.25: ranks above p99 | [2] | `v3_B_extra.json`: `suite.nulls.3pl_post.observed_ranks_above_p99` |
| committed | letter-aware know-or-guess (rotation-0 fit): rank p99 | [89.0, 47.7, 32.8, 28.2] | `v3_B_extra.json`: `suite.nulls.letter_post.rank_eigs_p99` |
| committed | letter-aware know-or-guess (rotation-0 fit): ranks above p99 | [2] | `v3_B_extra.json`: `suite.nulls.letter_post.observed_ranks_above_p99` |
| committed | know-or-guess, 4-rotation fit, letter-aware: rank p99 | [84.7, 47.2, 32.4, 27.1] | `v3_B_extra.json`: `suite.nulls.kg_letter4.rank_eigs_p99` |
| committed | know-or-guess, 4-rotation fit, letter-aware: ranks above p99 | [1, 2, 4] | `v3_B_extra.json`: `suite.nulls.kg_letter4.observed_ranks_above_p99` |
| committed | know-or-guess, 4-rotation fit, uniform guess: rank p99 | [86.0, 42.1, 33.6, 28.7] | `v3_B_extra.json`: `suite.nulls.kg_uniform4.rank_eigs_p99` |
| committed | know-or-guess, 4-rotation fit, uniform guess: ranks above p99 | [2] | `v3_B_extra.json`: `suite.nulls.kg_uniform4.observed_ranks_above_p99` |
| committed | stress x1.25: log-lik minus fit | -12.9 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.25.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x1.25: rank p99 | [91.8, 37.0, 30.9, 26.4] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.25.rank_eigs_p99` |
| committed | stress x1.25: ranks above | [2, 4] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.25.observed_ranks_above_p99` |
| committed | stress x1.5: log-lik minus fit | -43.1 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.5.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x1.5: rank p99 | [103.0, 41.2, 31.9, 26.7] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.5.rank_eigs_p99` |
| committed | stress x1.5: ranks above | [2, 4] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x1.5.observed_ranks_above_p99` |
| committed | stress x2: log-lik minus fit | -128.2 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x2.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x2: rank p99 | [122.5, 46.9, 34.5, 26.8] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x2.rank_eigs_p99` |
| committed | stress x2: ranks above | [2, 4] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x2.observed_ranks_above_p99` |
| committed | stress x3: log-lik minus fit | -328.9 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x3.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x3: rank p99 | [149.5, 56.0, 36.4, 27.3] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x3.rank_eigs_p99` |
| committed | stress x3: ranks above | [] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x3.observed_ranks_above_p99` |
| committed | stress x5: log-lik minus fit | -721.8 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x5.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x5: rank p99 | [175.7, 69.9, 41.3, 29.9] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x5.rank_eigs_p99` |
| committed | stress x5: ranks above | [] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x5.observed_ranks_above_p99` |
| committed | stress x10: log-lik minus fit | -1515.0 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x10.marginal_loglik_of_scaled_model_minus_fit` |
| committed | stress x10: rank p99 | [196.4, 89.2, 48.7, 31.8] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x10.rank_eigs_p99` |
| committed | stress x10: ranks above | [] | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x10.observed_ranks_above_p99` |
| committed | 23 models (no Llama-3.2-1B): ranks above 3PL p99 | [4] | `v3_B_leave_one_out.json`: `without_llama_3_2_1b.rot0_suite.nulls.3pl_post.observed_ranks_above_p99` |
| committed | 23 models: ranks above letter-aware p99 | [4] | `v3_B_leave_one_out.json`: `without_llama_3_2_1b.rot0_suite.nulls.letter_post.observed_ranks_above_p99` |
| committed | 23 models: rank-4 observed minus p99 (3PL, letter-aware) | {3pl_post: 0.35, letter_post: 0.57} | `v3_summary_derived.json`: `loo23_rot0_rank4_obs_minus_p99` |
| committed | project 2PL marginal log-lik (rot0) | -5839.2 | `raw_item07_across_two_margin_null.json`: `fits.twopl_mml.marginal_loglik` |
| committed | 3PL log-lik | -5774.2 | `v3_B_extra.json`: `suite.fits.3pl_post.loglik` |
| committed | letter-aware log-lik | -5417.8 | `v3_B_extra.json`: `suite.fits.letter_post.loglik` |
| committed | Bock-Aitkin a<=4: share of slopes at bound | 0.078 | `v3_B_extra.json`: `suite.fits.ba_a4.frac_slopes_at_bound` |
| committed | Bock-Aitkin a<=8: share of slopes at bound | 0.076 | `v3_B_extra.json`: `suite.fits.ba_a8.frac_slopes_at_bound` |
| committed | lambda2/lambda1 observed | 0.581 | `item07_across_model.json`: `lambda2_over_lambda1.observed` |
| committed | lambda2/lambda1: pooled stress null p99 | 0.463 | `item07_across_model.json`: `lambda2_over_lambda1.pooled_stress_x1.25_to_x10.null_p99` |
| committed | lambda2/lambda1: pooled stress null max | 0.498 | `item07_across_model.json`: `lambda2_over_lambda1.pooled_stress_x1.25_to_x10.null_max` |
| committed | lambda2/lambda1: pooled stress p | 0.00017 | `item07_across_model.json`: `lambda2_over_lambda1.pooled_stress_x1.25_to_x10.mc_p_upper` |
| committed | lambda2/lambda1: 2PL posterior null max | 0.511 | `item07_across_model.json`: `lambda2_over_lambda1.by_null.c_post_2pl.null_max` |
| committed | lambda2/lambda1: letter-aware p | 0.023 | `v3_B_extra.json`: `suite.nulls.letter_post.lambda2_over_lambda1.mc_p_upper` |
| committed | lambda2/lambda1: KG letter p | 0.095 | `v3_B_extra.json`: `suite.nulls.kg_letter4.lambda2_over_lambda1.mc_p_upper` |
| committed | lambda2/lambda1: Rasch JML p99 | 0.597 | `item07_across_model.json`: `lambda2_over_lambda1.by_null.a_rasch_jml.null_p99` |
| committed | family: mean r same | 0.414 | `item10_pair_correlations.json`: `qap_model_label_permutation.mean_r_same` |
| committed | family: mean r cross | 0.375 | `item10_pair_correlations.json`: `qap_model_label_permutation.mean_r_cross` |
| committed | family QAP p | 0.085 | `item10_pair_correlations.json`: `qap_model_label_permutation.perm_p_one_sided` |
| committed | family QAP p Holm | 0.342 | `item10_pair_correlations.json`: `family_tests_holm.qap_same_vs_cross_mean_r` |
| committed | family eta^2 eigvec 1 | 0.19 | `item09_family_alignment.json`: `eigenvectors[0].eta_sq_family` |
| committed | family eta^2 eigvec 2 | 0.49 | `item09_family_alignment.json`: `eigenvectors[1].eta_sq_family` |
| committed | family eta^2 eigvec 3 | 0.63 | `item09_family_alignment.json`: `eigenvectors[2].eta_sq_family` |
| committed | family eta^2 eigvec 4 | 0.85 | `item09_family_alignment.json`: `eigenvectors[3].eta_sq_family` |
| committed | family p eigvec 1 | 0.8829 | `item09_family_alignment.json`: `eigenvectors[0].perm_pvalue` |
| committed | family p eigvec 2 | 0.3308 | `item09_family_alignment.json`: `eigenvectors[1].perm_pvalue` |
| committed | family p eigvec 3 | 0.1192 | `item09_family_alignment.json`: `eigenvectors[2].perm_pvalue` |
| committed | family p eigvec 4 | 0.0028 | `item09_family_alignment.json`: `eigenvectors[3].perm_pvalue` |
| committed | family Holm eigvec 1 | 0.883 | `item10_pair_correlations.json`: `family_tests_holm.eigvec1_family_eta_sq` |
| committed | family Holm eigvec 2 | 0.662 | `item10_pair_correlations.json`: `family_tests_holm.eigvec2_family_eta_sq` |
| committed | family Holm eigvec 3 | 0.358 | `item10_pair_correlations.json`: `family_tests_holm.eigvec3_family_eta_sq` |
| committed | family Holm eigvec 4 | 0.017 | `item10_pair_correlations.json`: `family_tests_holm.eigvec4_family_eta_sq` |
| committed | family joint 2-4 p | 0.031 | `item09_family_alignment.json`: `joint_eigvec_2to4.perm_pvalue` |
| committed | family joint 2-4 Holm | 0.154 | `item10_pair_correlations.json`: `family_tests_holm.joint_eigvec_2to4_family` |
| committed | family eigvec 4 p, multi-member families only | 0.048 | `item09_family_alignment.json`: `eigenvectors[3].perm_pvalue_multi_member_families_only` |
| committed | A vs B: max |eigenvalue difference| | 0.081 | `v3_C4_replicate_agreement.json`: `max_abs_eig_diff_overall` |
| committed | A vs B: null verdicts that differ | 4 | `v3_C4_replicate_agreement.json`: `n_null_verdicts_disagree` |
| committed | A vs B: null verdicts compared | 33 | `v3_C4_replicate_agreement.json`: `n_null_verdicts` |
| committed | A vs B: differing verdicts | [randomized_representative:rasch_jml, all4_correct:fixed_margin, all4_correct:2pl_post, all4_correct:ba_a8] | `v3_C4_replicate_agreement.json`: `disagreeing_null_verdicts` |
| committed | near ties: share of cells | 0.050 | `v3_C5_near_ties.json`: `frac_near_tie_cells` |
| committed | near ties: cells by rotation | [595, 565, 641, 610] | `v3_C5_near_ties.json`: `n_near_tie_cells_by_rotation` |
| committed | near ties: rot0 cells changed | 86 | `v3_C5_near_ties.json`: `majority_n_cells_changed_rot0` |
| committed | near ties rot0 majority: 3PL ranks above | [2] | `v3_C5_near_ties.json`: `refitted_variants.rot0_majority.suite.nulls.3pl_post.observed_ranks_above_p99` |
| committed | near ties rot0 majority: letter-aware ranks above | [2] | `v3_C5_near_ties.json`: `refitted_variants.rot0_majority.suite.nulls.letter_post.observed_ranks_above_p99` |
| committed | near ties: questions with a rot0 near tie | 297 | `v3_C5_near_ties.json`: `n_questions_with_a_rot0_near_tie` |
| committed | near ties rot0 drop questions: letter-aware ranks above | [] | `v3_C5_near_ties.json`: `refitted_variants.rot0_drop_questions_with_rot0_near_tie.suite.nulls.letter_post.observed_ranks_above_p99` |
| committed | near ties rot0 drop questions: 3PL ranks above | [2] | `v3_C5_near_ties.json`: `refitted_variants.rot0_drop_questions_with_rot0_near_tie.suite.nulls.3pl_post.observed_ranks_above_p99` |
| committed | near ties all-4 majority: ranks beyond ability | {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, ba_a4: 0, kg_letter4: 0} | `v3_C5_near_ties.json`: `refitted_variants.all4_majority.suite.ranks_beyond_ability` |
| committed | near ties all-4 ignored: ranks beyond ability | {rasch_jml: 0, fixed_margin: 1, 2pl_post: 0, ba_a4: 0, kg_letter4: 0} | `v3_C5_near_ties.json`: `refitted_variants.all4_near_tie_rotations_ignored.suite.ranks_beyond_ability` |
| committed | S1: eta^2 by gold letter | 0.73 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.eta_sq` |
| committed | S1: eta^2 p | 0.0002 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.perm_p` |
| committed | Llama S2: eta^2 by gold letter | 0.53 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.eta_sq` |
| committed | Llama S2: eta^2 p | 0.0002 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.perm_p` |
| committed | S1: eta^2 by subject | 0.70 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.eta_sq` |
| committed | S1: subject eta^2 null mean | 0.63 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.null_mean` |
| committed | S1: subject p | 0.229 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.perm_p` |
| committed | Llama S2: eta^2 by subject | 0.20 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.top_pc.loadings_eta_sq_by_subject.eta_sq` |
| committed | Llama S2: subject p | 0.349 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.top_pc.loadings_eta_sq_by_subject.perm_p` |
| committed | Qwen S2: eta^2 by gold letter | 0.07 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.eta_sq` |
| committed | Qwen S2: eta^2 p | 0.094 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_gold_letter.perm_p` |
| committed | Qwen S2: exemplar-overlap z | 10.7 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.exemplar_alignment.pass_pair_similarity_shared_exemplars_borderline.z_vs_null` |
| committed | Qwen S2: exemplar-overlap p | 0.0005 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.exemplar_alignment.pass_pair_similarity_shared_exemplars_borderline.perm_p_upper` |
| committed | Qwen S2: eta^2 by subject | 0.58 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.eta_sq` |
| committed | Qwen S2: subject eta^2 null mean | 0.46 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.null_mean` |
| committed | Qwen S2: subject p | 0.042 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.top_pc.loadings_eta_sq_by_subject.perm_p` |
| committed | Qwen S2: subject-match effect | 0.0324 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.exemplar_alignment.subject_match_borderline.mean_diff_matched_minus_unmatched` |
| committed | Qwen S2: subject-match p | 0.033 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.exemplar_alignment.subject_match_borderline.perm_p_two_sided` |
| contradicts paper | Llama-3.2-1B pi_A | 0.90 | `v3_summary_derived.json`: `llama_3_2_1b_pi_A` |
| contradicts paper | rot0: eigenvalues 1-4 (paper: 92.4, 46.5, 33.7, 25.3) | [85.9, 49.9, 29.7, 27.3] | `item07_across_model.json`: `observed.top_eigs_1to4` |
| contradicts paper | rot0: eigenvalue 4 | 27.27 | `item07_across_model.json`: `observed.top_eigs_1to4[3]` |
| contradicts paper | stress x3: rank-4 p99 | 27.33 | `raw_item07_across_two_margin_null.json`: `nulls.c_post_2pl_stress_slopes_x3.rank_eigs_p99[3]` |
| contradicts paper | family pair-level KS p (superseded) | 0.024 | `item10_pair_correlations.json`: `v2.ks_p` |
| contradicts paper | family: Cohen d | 0.30 | `item10_pair_correlations.json`: `v2.cohen_d` |
| contradicts paper | MMLU Mistral-7B-v0.3: N_b | 83 | `item01_within_table.json`: `runs[1].N_b` |
| contradicts paper | MMLU Mistral-7B-v0.3: TW z | -1.79 | `item01_within_table.json`: `runs[1].tw_z` |
| contradicts paper | MMLU Phi-3-mini: N_b | 271 | `item01_within_table.json`: `runs[4].N_b` |
| contradicts paper | HellaSwag Mistral-7B-v0.3: N_b | 65 | `item01_within_table.json`: `runs[6].N_b` |
| A within-model | MMLU Qwen2.5-7B T1.0: accuracy | 0.7042 | `item01_within_table.json`: `runs[0].accuracy` |
| A within-model | MMLU Qwen2.5-7B T1.0: N_b | 43 | `item01_within_table.json`: `runs[0].N_b` |
| A within-model | MMLU Qwen2.5-7B T1.0: lambda_+ | 2.741 | `item01_within_table.json`: `runs[0].lambda_plus` |
| A within-model | MMLU Qwen2.5-7B T1.0: lambda_max | 2.562 | `item01_within_table.json`: `runs[0].lambda_max` |
| A within-model | MMLU Qwen2.5-7B T1.0: TW z | -1.72 | `item01_within_table.json`: `runs[0].tw_z` |
| A within-model | MMLU Qwen2.5-7B T1.0: MC percentile of z | 0.601 | `item01_within_table.json`: `runs[0].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Qwen2.5-7B T1.0: MP count | 0 | `item01_within_table.json`: `runs[0].mp_count_strict` |
| A within-model | MMLU Qwen2.5-7B T1.0: buffered count | 0 | `item01_within_table.json`: `runs[0].mp_count_buffered` |
| A within-model | MMLU Qwen2.5-7B T1.0: Horn count | 0 | `item01_within_table.json`: `runs[0].horn_count_contiguous` |
| A within-model | MMLU Mistral-7B-v0.3: accuracy | 0.5708 | `item01_within_table.json`: `runs[1].accuracy` |
| A within-model | MMLU Mistral-7B-v0.3: lambda_+ | 3.652 | `item01_within_table.json`: `runs[1].lambda_plus` |
| A within-model | MMLU Mistral-7B-v0.3: lambda_max | 3.449 | `item01_within_table.json`: `runs[1].lambda_max` |
| A within-model | MMLU Mistral-7B-v0.3: MC percentile of z | 0.513 | `item01_within_table.json`: `runs[1].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Mistral-7B-v0.3: MP count | 0 | `item01_within_table.json`: `runs[1].mp_count_strict` |
| A within-model | MMLU Mistral-7B-v0.3: buffered count | 0 | `item01_within_table.json`: `runs[1].mp_count_buffered` |
| A within-model | MMLU Mistral-7B-v0.3: Horn count | 0 | `item01_within_table.json`: `runs[1].horn_count_contiguous` |
| A within-model | MMLU Meta-Llama-3-8B: accuracy | 0.6318 | `item01_within_table.json`: `runs[2].accuracy` |
| A within-model | MMLU Meta-Llama-3-8B: N_b | 173 | `item01_within_table.json`: `runs[2].N_b` |
| A within-model | MMLU Meta-Llama-3-8B: lambda_+ | 5.361 | `item01_within_table.json`: `runs[2].lambda_plus` |
| A within-model | MMLU Meta-Llama-3-8B: lambda_max | 4.965 | `item01_within_table.json`: `runs[2].lambda_max` |
| A within-model | MMLU Meta-Llama-3-8B: TW z | -3.05 | `item01_within_table.json`: `runs[2].tw_z` |
| A within-model | MMLU Meta-Llama-3-8B: MC percentile of z | 0.102 | `item01_within_table.json`: `runs[2].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Meta-Llama-3-8B: MP count | 0 | `item01_within_table.json`: `runs[2].mp_count_strict` |
| A within-model | MMLU Meta-Llama-3-8B: buffered count | 0 | `item01_within_table.json`: `runs[2].mp_count_buffered` |
| A within-model | MMLU Meta-Llama-3-8B: Horn count | 0 | `item01_within_table.json`: `runs[2].horn_count_contiguous` |
| A within-model | MMLU SmolLM2-1.7B: accuracy | 0.3949 | `item01_within_table.json`: `runs[3].accuracy` |
| A within-model | MMLU SmolLM2-1.7B: N_b | 406 | `item01_within_table.json`: `runs[3].N_b` |
| A within-model | MMLU SmolLM2-1.7B: lambda_+ | 9.090 | `item01_within_table.json`: `runs[3].lambda_plus` |
| A within-model | MMLU SmolLM2-1.7B: lambda_max | 8.742 | `item01_within_table.json`: `runs[3].lambda_max` |
| A within-model | MMLU SmolLM2-1.7B: TW z | -2.17 | `item01_within_table.json`: `runs[3].tw_z` |
| A within-model | MMLU SmolLM2-1.7B: MC percentile of z | 0.286 | `item01_within_table.json`: `runs[3].mc_null_z.percentile_of_obs` |
| A within-model | MMLU SmolLM2-1.7B: MP count | 0 | `item01_within_table.json`: `runs[3].mp_count_strict` |
| A within-model | MMLU SmolLM2-1.7B: buffered count | 0 | `item01_within_table.json`: `runs[3].mp_count_buffered` |
| A within-model | MMLU SmolLM2-1.7B: Horn count | 0 | `item01_within_table.json`: `runs[3].horn_count_contiguous` |
| A within-model | MMLU Phi-3-mini: accuracy | 0.6425 | `item01_within_table.json`: `runs[4].accuracy` |
| A within-model | MMLU Phi-3-mini: lambda_+ | 7.002 | `item01_within_table.json`: `runs[4].lambda_plus` |
| A within-model | MMLU Phi-3-mini: lambda_max | 6.908 | `item01_within_table.json`: `runs[4].lambda_max` |
| A within-model | MMLU Phi-3-mini: TW z | -0.65 | `item01_within_table.json`: `runs[4].tw_z` |
| A within-model | MMLU Phi-3-mini: MC percentile of z | 0.835 | `item01_within_table.json`: `runs[4].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Phi-3-mini: MP count | 0 | `item01_within_table.json`: `runs[4].mp_count_strict` |
| A within-model | MMLU Phi-3-mini: buffered count | 0 | `item01_within_table.json`: `runs[4].mp_count_buffered` |
| A within-model | MMLU Phi-3-mini: Horn count | 0 | `item01_within_table.json`: `runs[4].horn_count_contiguous` |
| A within-model | HellaSwag Qwen2.5-7B: accuracy | 0.8519 | `item01_within_table.json`: `runs[5].accuracy` |
| A within-model | HellaSwag Qwen2.5-7B: N_b | 16 | `item01_within_table.json`: `runs[5].N_b` |
| A within-model | HellaSwag Qwen2.5-7B: lambda_+ | 2.451 | `item01_within_table.json`: `runs[5].lambda_plus` |
| A within-model | HellaSwag Qwen2.5-7B: lambda_max | 2.193 | `item01_within_table.json`: `runs[5].lambda_max` |
| A within-model | HellaSwag Qwen2.5-7B: TW z | -1.60 | `item01_within_table.json`: `runs[5].tw_z` |
| A within-model | HellaSwag Qwen2.5-7B: MC percentile of z | 0.699 | `item01_within_table.json`: `runs[5].mc_null_z.percentile_of_obs` |
| A within-model | HellaSwag Qwen2.5-7B: MP count | 0 | `item01_within_table.json`: `runs[5].mp_count_strict` |
| A within-model | HellaSwag Qwen2.5-7B: buffered count | 0 | `item01_within_table.json`: `runs[5].mp_count_buffered` |
| A within-model | HellaSwag Qwen2.5-7B: Horn count | 0 | `item01_within_table.json`: `runs[5].horn_count_contiguous` |
| A within-model | HellaSwag Mistral-7B-v0.3: accuracy | 0.6892 | `item01_within_table.json`: `runs[6].accuracy` |
| A within-model | HellaSwag Mistral-7B-v0.3: lambda_+ | 4.580 | `item01_within_table.json`: `runs[6].lambda_plus` |
| A within-model | HellaSwag Mistral-7B-v0.3: lambda_max | 4.053 | `item01_within_table.json`: `runs[6].lambda_max` |
| A within-model | HellaSwag Mistral-7B-v0.3: TW z | -2.71 | `item01_within_table.json`: `runs[6].tw_z` |
| A within-model | HellaSwag Mistral-7B-v0.3: MC percentile of z | 0.197 | `item01_within_table.json`: `runs[6].mc_null_z.percentile_of_obs` |
| A within-model | HellaSwag Mistral-7B-v0.3: MP count | 0 | `item01_within_table.json`: `runs[6].mp_count_strict` |
| A within-model | HellaSwag Mistral-7B-v0.3: buffered count | 0 | `item01_within_table.json`: `runs[6].mp_count_buffered` |
| A within-model | HellaSwag Mistral-7B-v0.3: Horn count | 0 | `item01_within_table.json`: `runs[6].horn_count_contiguous` |
| A within-model | HellaSwag Meta-Llama-3-8B: accuracy | 0.6686 | `item01_within_table.json`: `runs[7].accuracy` |
| A within-model | HellaSwag Meta-Llama-3-8B: N_b | 86 | `item01_within_table.json`: `runs[7].N_b` |
| A within-model | HellaSwag Meta-Llama-3-8B: lambda_+ | 5.343 | `item01_within_table.json`: `runs[7].lambda_plus` |
| A within-model | HellaSwag Meta-Llama-3-8B: lambda_max | 4.746 | `item01_within_table.json`: `runs[7].lambda_max` |
| A within-model | HellaSwag Meta-Llama-3-8B: TW z | -2.90 | `item01_within_table.json`: `runs[7].tw_z` |
| A within-model | HellaSwag Meta-Llama-3-8B: MC percentile of z | 0.113 | `item01_within_table.json`: `runs[7].mc_null_z.percentile_of_obs` |
| A within-model | HellaSwag Meta-Llama-3-8B: MP count | 0 | `item01_within_table.json`: `runs[7].mp_count_strict` |
| A within-model | HellaSwag Meta-Llama-3-8B: buffered count | 0 | `item01_within_table.json`: `runs[7].mp_count_buffered` |
| A within-model | HellaSwag Meta-Llama-3-8B: Horn count | 0 | `item01_within_table.json`: `runs[7].horn_count_contiguous` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: accuracy | 0.5097 | `item01_within_table.json`: `runs[8].accuracy` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: N_b | 62 | `item01_within_table.json`: `runs[8].N_b` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: lambda_+ | 5.942 | `item01_within_table.json`: `runs[8].lambda_plus` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: lambda_max | 5.322 | `item01_within_table.json`: `runs[8].lambda_max` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: TW z | -2.06 | `item01_within_table.json`: `runs[8].tw_z` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: MC percentile of z | 0.379 | `item01_within_table.json`: `runs[8].mc_null_z.percentile_of_obs` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: MP count | 0 | `item01_within_table.json`: `runs[8].mp_count_strict` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: buffered count | 0 | `item01_within_table.json`: `runs[8].mp_count_buffered` |
| A within-model | GSM8K Qwen2.5-7B 256 tok: Horn count | 0 | `item01_within_table.json`: `runs[8].horn_count_contiguous` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: accuracy | 0.9033 | `item01_within_table.json`: `runs[9].accuracy` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: N_b | 19 | `item01_within_table.json`: `runs[9].N_b` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: lambda_+ | 3.225 | `item01_within_table.json`: `runs[9].lambda_plus` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: lambda_max | 2.502 | `item01_within_table.json`: `runs[9].lambda_max` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: TW z | -2.97 | `item01_within_table.json`: `runs[9].tw_z` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: MC percentile of z | 0.120 | `item01_within_table.json`: `runs[9].mc_null_z.percentile_of_obs` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: MP count | 0 | `item01_within_table.json`: `runs[9].mp_count_strict` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: buffered count | 0 | `item01_within_table.json`: `runs[9].mp_count_buffered` |
| A within-model | GSM8K Qwen2.5-7B 1024 tok: Horn count | 0 | `item01_within_table.json`: `runs[9].horn_count_contiguous` |
| A within-model | MMLU Qwen2.5-7B T0.5: accuracy | 0.7035 | `item01_within_table.json`: `runs[10].accuracy` |
| A within-model | MMLU Qwen2.5-7B T0.5: N_b | 25 | `item01_within_table.json`: `runs[10].N_b` |
| A within-model | MMLU Qwen2.5-7B T0.5: lambda_+ | 3.659 | `item01_within_table.json`: `runs[10].lambda_plus` |
| A within-model | MMLU Qwen2.5-7B T0.5: lambda_max | 3.385 | `item01_within_table.json`: `runs[10].lambda_max` |
| A within-model | MMLU Qwen2.5-7B T0.5: TW z | -1.08 | `item01_within_table.json`: `runs[10].tw_z` |
| A within-model | MMLU Qwen2.5-7B T0.5: MC percentile of z | 0.785 | `item01_within_table.json`: `runs[10].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Qwen2.5-7B T0.5: MP count | 0 | `item01_within_table.json`: `runs[10].mp_count_strict` |
| A within-model | MMLU Qwen2.5-7B T0.5: buffered count | 0 | `item01_within_table.json`: `runs[10].mp_count_buffered` |
| A within-model | MMLU Qwen2.5-7B T0.5: Horn count | 0 | `item01_within_table.json`: `runs[10].horn_count_contiguous` |
| A within-model | MMLU Qwen2.5-7B T1.5: accuracy | 0.6989 | `item01_within_table.json`: `runs[11].accuracy` |
| A within-model | MMLU Qwen2.5-7B T1.5: N_b | 85 | `item01_within_table.json`: `runs[11].N_b` |
| A within-model | MMLU Qwen2.5-7B T1.5: lambda_+ | 7.200 | `item01_within_table.json`: `runs[11].lambda_plus` |
| A within-model | MMLU Qwen2.5-7B T1.5: lambda_max | 6.529 | `item01_within_table.json`: `runs[11].lambda_max` |
| A within-model | MMLU Qwen2.5-7B T1.5: TW z | -2.07 | `item01_within_table.json`: `runs[11].tw_z` |
| A within-model | MMLU Qwen2.5-7B T1.5: MC percentile of z | 0.327 | `item01_within_table.json`: `runs[11].mc_null_z.percentile_of_obs` |
| A within-model | MMLU Qwen2.5-7B T1.5: MP count | 0 | `item01_within_table.json`: `runs[11].mp_count_strict` |
| A within-model | MMLU Qwen2.5-7B T1.5: buffered count | 0 | `item01_within_table.json`: `runs[11].mp_count_buffered` |
| A within-model | MMLU Qwen2.5-7B T1.5: Horn count | 0 | `item01_within_table.json`: `runs[11].horn_count_contiguous` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: N_b | 41 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.N_b` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: lambda_+ | 2.69 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.lambda_plus` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: lambda_max | 10.00 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.lambda_max` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: TW z | 70.2 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.tw_z` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: buffered | 4 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.mp_count_buffered` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: Horn | 5 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.horn_count_contiguous` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: letter-share dispersion ratio | 11.8 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.letter_share_dispersion.ratio_obs_over_null_mean` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: pass dispersion ratio | 0.83 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.pass_dispersion.dispersion_ratio_obs_over_expected` |
| A within-model | MMLU Qwen2.5-7B S1 shared seed: pass dispersion p | 0.8930 | `item02_shared_randomness.json`: `runs.mmlu_sharedseed-T1.0_Qwen_Qwen2.5-7B-Instruct.pass_dispersion.p_overdispersion` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: N_b | 93 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.N_b` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: lambda_+ | 3.86 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.lambda_plus` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: lambda_max | 4.77 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.lambda_max` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: TW z | 7.9 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.tw_z` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: buffered | 2 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.mp_count_buffered` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: Horn | 10 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.horn_count_contiguous` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: letter-share dispersion ratio | 1.8 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.letter_share_dispersion.ratio_obs_over_null_mean` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: pass dispersion ratio | 1.21 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.pass_dispersion.dispersion_ratio_obs_over_expected` |
| A within-model | MMLU Qwen2.5-7B S2 shared 5-shot: pass dispersion p | 0.0748 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_Qwen_Qwen2.5-7B-Instruct.pass_dispersion.p_overdispersion` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: N_b | 264 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.N_b` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: lambda_+ | 6.89 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.lambda_plus` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: lambda_max | 11.31 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.lambda_max` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: TW z | 30.9 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.tw_z` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: buffered | 2 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.mp_count_buffered` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: Horn | 5 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.horn_count_contiguous` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: letter-share dispersion ratio | 11.3 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.letter_share_dispersion.ratio_obs_over_null_mean` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: pass dispersion ratio | 1.51 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.pass_dispersion.dispersion_ratio_obs_over_expected` |
| A within-model | MMLU Meta-Llama-3-8B S2 shared 5-shot: pass dispersion p | 0.0010 | `item02_shared_randomness.json`: `runs.mmlu_sharedprompt-T1.0_meta-llama_Meta-Llama-3-8B-Instruct.pass_dispersion.p_overdispersion` |
| A within-model | split-half r all columns | 0.9975 | `item05_split_half.json`: `runs.mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct.r_all_columns` |
| A within-model | split-half r borderline | 0.9403 | `item05_split_half.json`: `runs.mmlu_iid-T1.0_Qwen_Qwen2.5-7B-Instruct.r_borderline_columns` |
| A within-model | modal wrong-letter share | 0.955 | `item06_answer_identity.json`: `modal_wrong_letter_share.mean_over_borderline` |
| A within-model | letter test exceeds p99 | False | `item06_answer_identity.json`: `letter_test_qwen.exceeds_p99` |
| A within-model | T0.5 N_b | 25 | `item12_temperature_gsm8k.json`: `temperature.0.5.N_b` |
| A within-model | T1.5 N_b | 85 | `item12_temperature_gsm8k.json`: `temperature.1.5.N_b` |
| A within-model | T0.5 MP | 0 | `item12_temperature_gsm8k.json`: `temperature.0.5.mp_count_strict` |
| A within-model | T1.5 MP | 0 | `item12_temperature_gsm8k.json`: `temperature.1.5.mp_count_strict` |
| A within-model | GSM8K 256: truncated share | 0.550 | `item12_temperature_gsm8k.json`: `gsm8k.256.share_length_truncated` |
| A within-model | GSM8K 256 accuracy | 0.510 | `item12_temperature_gsm8k.json`: `gsm8k.256.accuracy` |
| A within-model | GSM8K 1024 accuracy | 0.903 | `item12_temperature_gsm8k.json`: `gsm8k.1024.accuracy` |
| A within-model | item 13 as_scored: buffered count | 2 | `item13_unparseable.json`: `across_model_sensitivity.as_scored.mp_count_buffered` |
| A within-model | item 13 drop_questions_any_unparseable: buffered count | 3 | `item13_unparseable.json`: `across_model_sensitivity.drop_questions_any_unparseable.mp_count_buffered` |
| A within-model | item 13 drop_model_mistral_v0.2: buffered count | 2 | `item13_unparseable.json`: `across_model_sensitivity.drop_model_mistral_v0.2.mp_count_buffered` |
| A within-model | item 13 impute_question_mean: buffered count | 2 | `item13_unparseable.json`: `across_model_sensitivity.impute_question_mean.mp_count_buffered` |
| A within-model | item 13 drop_questions_any_unparseable: 2PL ranks above p99 | [1, 2, 3, 4] | `item13_unparseable.json`: `across_model_sensitivity.drop_questions_any_unparseable.one_factor_statement.ranks_above_p99_by_null.c_post_2pl` |
| A within-model | item 13 drop_model_mistral_v0.2: 2PL ranks above p99 | [1, 2, 3, 4] | `item13_unparseable.json`: `across_model_sensitivity.drop_model_mistral_v0.2.one_factor_statement.ranks_above_p99_by_null.c_post_2pl` |
| C rotations | rotation 0 eigvec 1: eta^2 by shown gold letter | 0.22 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[0].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 0 eigvec 2: eta^2 by shown gold letter | 0.48 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[1].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 0 eigvec 3: eta^2 by shown gold letter | 0.09 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[2].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 0 eigvec 4: eta^2 by shown gold letter | 0.06 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[3].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 0: eigenvalues 1-4 | [85.9, 49.9, 29.7, 27.3] | `v3_C_rotations_A.json`: `C1_per_rotation.0.spectrum.top_eigs` `[0:4]` |
| C rotations | rotation 0: lambda_+ | 28.92 | `v3_C_rotations_A.json`: `C1_per_rotation.0.spectrum.lambda_plus` |
| C rotations | rotation 0: buffered | 2 | `v3_C_rotations_A.json`: `C1_per_rotation.0.spectrum.mp_count_buffered` |
| C rotations | rotation 0 eigvec 2: r share A | 0.76 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[1].model_score_corr.share_A.r` |
| C rotations | rotation 0 eigvec 2: r share D | -0.90 | `v3_C_rotations_A.json`: `C1_per_rotation.0.letters.eigenvectors[1].model_score_corr.share_D.r` |
| C rotations | rotation 1 eigvec 1: eta^2 by shown gold letter | 0.22 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[0].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 1 eigvec 2: eta^2 by shown gold letter | 0.41 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[1].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 1 eigvec 3: eta^2 by shown gold letter | 0.05 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[2].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 1 eigvec 4: eta^2 by shown gold letter | 0.15 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[3].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 1: eigenvalues 1-4 | [83.2, 48.4, 30.8, 28.1] | `v3_C_rotations_A.json`: `C1_per_rotation.1.spectrum.top_eigs` `[0:4]` |
| C rotations | rotation 1: lambda_+ | 28.87 | `v3_C_rotations_A.json`: `C1_per_rotation.1.spectrum.lambda_plus` |
| C rotations | rotation 1: buffered | 3 | `v3_C_rotations_A.json`: `C1_per_rotation.1.spectrum.mp_count_buffered` |
| C rotations | rotation 1 eigvec 2: r share A | 0.78 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[1].model_score_corr.share_A.r` |
| C rotations | rotation 1 eigvec 2: r share D | -0.86 | `v3_C_rotations_A.json`: `C1_per_rotation.1.letters.eigenvectors[1].model_score_corr.share_D.r` |
| C rotations | rotation 2 eigvec 1: eta^2 by shown gold letter | 0.19 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[0].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 2 eigvec 2: eta^2 by shown gold letter | 0.36 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[1].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 2 eigvec 3: eta^2 by shown gold letter | 0.11 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[2].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 2 eigvec 4: eta^2 by shown gold letter | 0.12 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[3].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 2: eigenvalues 1-4 | [81.5, 44.3, 33.9, 28.1] | `v3_C_rotations_A.json`: `C1_per_rotation.2.spectrum.top_eigs` `[0:4]` |
| C rotations | rotation 2: lambda_+ | 28.56 | `v3_C_rotations_A.json`: `C1_per_rotation.2.spectrum.lambda_plus` |
| C rotations | rotation 2: buffered | 3 | `v3_C_rotations_A.json`: `C1_per_rotation.2.spectrum.mp_count_buffered` |
| C rotations | rotation 2 eigvec 2: r share A | 0.75 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[1].model_score_corr.share_A.r` |
| C rotations | rotation 2 eigvec 2: r share D | -0.81 | `v3_C_rotations_A.json`: `C1_per_rotation.2.letters.eigenvectors[1].model_score_corr.share_D.r` |
| C rotations | rotation 3 eigvec 1: eta^2 by shown gold letter | 0.26 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[0].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 3 eigvec 2: eta^2 by shown gold letter | 0.38 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[1].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 3 eigvec 3: eta^2 by shown gold letter | 0.05 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[2].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 3 eigvec 4: eta^2 by shown gold letter | 0.04 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[3].eta_sq_by_shown_gold_letter.eta_sq` |
| C rotations | rotation 3: eigenvalues 1-4 | [88.8, 54.7, 31.5, 25.1] | `v3_C_rotations_A.json`: `C1_per_rotation.3.spectrum.top_eigs` `[0:4]` |
| C rotations | rotation 3: lambda_+ | 29.49 | `v3_C_rotations_A.json`: `C1_per_rotation.3.spectrum.lambda_plus` |
| C rotations | rotation 3: buffered | 3 | `v3_C_rotations_A.json`: `C1_per_rotation.3.spectrum.mp_count_buffered` |
| C rotations | rotation 3 eigvec 2: r share A | 0.70 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[1].model_score_corr.share_A.r` |
| C rotations | rotation 3 eigvec 2: r share D | -0.90 | `v3_C_rotations_A.json`: `C1_per_rotation.3.letters.eigenvectors[1].model_score_corr.share_D.r` |
| C rotations | C1b eigvec 1 r1: r_shown | 0.43 | `v3_C1b_letter_following.json`: `A.eigvec1.1.r_shown` |
| C rotations | C1b eigvec 1 r1: r_orig | 0.10 | `v3_C1b_letter_following.json`: `A.eigvec1.1.r_orig` |
| C rotations | C1b eigvec 1 r1: r_direct | 0.55 | `v3_C1b_letter_following.json`: `A.eigvec1.1.r_direct` |
| C rotations | C1b eigvec 3 r1: model-score match | 0.75 | `v3_C1b_letter_following.json`: `A.eigvec3.1.model_score_match_abs_r` |
| C rotations | C2: buffered count distribution | {2: 195, 3: 805} | `v3_C_rotations_A.json`: `C2_rotation_randomized.distribution.mp_count_buffered_distribution` |
| C rotations | C2: strict MP count distribution | {2: 37, 3: 956, 4: 7} | `v3_C_rotations_A.json`: `C2_rotation_randomized.distribution.mp_count_strict_distribution` |
| C rotations | C2: eigenvalue p01 | [73.1, 36.4, 28.3, 23.7] | `v3_C_rotations_A.json`: `C2_rotation_randomized.distribution.eig_p01` `[0:4]` |
| C rotations | C2: eigenvalue p99 | [86.1, 45.5, 36.1, 29.0] | `v3_C_rotations_A.json`: `C2_rotation_randomized.distribution.eig_p99` `[0:4]` |
| C rotations | C2 representative: ranks above p99 by null | {rasch_jml: [1, 2, 3], fixed_margin: [1, 2, 3], 2pl_post: [1, 2, 3], 3pl_post: [], letter_post: [], ba_a4: [2, 3]} | `v3_C_rotations_A.json`: `C2_rotation_randomized.representative_suite.ranks_above_p99_by_null` |
| C rotations | C2 refits: ranks above p99 | [{draw: 852, top_eigs: [75.179, 41.064, 35.490, 25.797], ranks_above_p99: {2pl_post: [2, 3], letter_post: [2, 3]}, beyond_ability: {2pl_post: 2, letter_post: 2}}, {draw: 547, top_eigs: [78.847, 43.147, 30.724, 26.465], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: [2]}, beyond_ability: {2pl_post: 3, letter_post: 1}}, {draw: 504, top_eigs: [76.373, 37.000, 31.873, 25.489], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: []}, beyond_ability: {2pl_post: 2, letter_post: 0}}, {draw: 953, top_eigs: [82.443, 38.890, 33.112, 26.980], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: [3]}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 820, top_eigs: [75.760, 39.603, 33.271, 26.716], ranks_above_p99: {2pl_post: [2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 768, top_eigs: [78.017, 41.441, 30.283, 27.836], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 546, top_eigs: [87.151, 44.785, 29.227, 25.362], ranks_above_p99: {2pl_post: [1, 2], letter_post: [2]}, beyond_ability: {2pl_post: 1, letter_post: 1}}, {draw: 822, top_eigs: [76.411, 34.923, 32.175, 25.102], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: []}, beyond_ability: {2pl_post: 2, letter_post: 0}}, {draw: 61, top_eigs: [78.013, 37.601, 31.561, 26.390], ranks_above_p99: {2pl_post: [1, 2, 3, 4], letter_post: []}, beyond_ability: {2pl_post: 3, letter_post: 0}}, {draw: 664, top_eigs: [80.861, 42.937, 32.652, 24.634], ranks_above_p99: {2pl_post: [1, 2, 3], letter_post: [2]}, beyond_ability: {2pl_post: 2, letter_post: 1}}] | `v3_C_rotations_A.json`: `C2_rotation_randomized.robustness_refit_draws` |
| C rotations | C2 eigvec 2: r share A | 0.70 | `v3_C_rotations_A.json`: `C2_rotation_randomized.letters_representative.eigenvectors[1].model_score_corr.share_A.r` |
| C rotations | C2 eigvec 2: r share D | -0.62 | `v3_C_rotations_A.json`: `C2_rotation_randomized.letters_representative.eigenvectors[1].model_score_corr.share_D.r` |
| C rotations | C2 eigvec 2: eta^2 by original gold letter | 0.006 | `v3_C_rotations_A.json`: `C2_rotation_randomized.letters_representative.eigenvectors[1].eta_sq_by_original_gold_letter.eta_sq` |
| C rotations | 23 models all-4: ranks beyond ability | {rasch_jml: 0, fixed_margin: 0, 2pl_post: 0, ba_a4: 0, kg_letter4: 0} | `v3_B_leave_one_out.json`: `without_llama_3_2_1b.all4_suite.ranks_beyond_ability` |
| notes | 2PL random restarts log-lik | [-5880.4, -5868.1, -5839.2] | `raw_item07_across_two_margin_null.json`: `fits.twopl_mml.random_restarts.marginal_loglik` |

## F. Construction family, soft-preference null, power

Code: `scripts/v3_softpref.py` (part `all` = `base`, `family`, `sp`, `cal`, `power`, `calib`, `calib_bc`, `char`, `assemble`; 8 processes, checkpointed per job; 24,816 s wall clock) + `scripts/test_v3_softpref.py` (74 tests). Outputs: `v3_F_softpref.json` (steps 1, 2 with its calibration, 2b, 2c, 4), `v3_F_power.json` (step 3); raw parts `raw_v3_F_*.json` / `.npz`. Seeds: step 1 uses `v3_across.SEED` = 20261013 with the same per-null child streams (it reproduces the 17 overlapping suites of section C exactly, max |p99 difference| = 0.0); every new stream is `default_rng([20261014, stream, i + 1, ...])` with all key components positive (`rng_for`; numpy drops trailing zeros, so `[a, 0]` and `[a]` would be the same stream). Conventions as at the top of this file; "valid null" = p01 <= observed lambda_1 <= p99, both from the same `np.percentile` call as the rank p99s; "beyond ability" counts ranks 2, 3, ... above the rank-matched p99, contiguous from rank 2.

Method:
- Calibration. Every soft-preference (SP) percentile is calibrated against the same pipeline: 200 parametric-bootstrap datasets per variant (full, without Llama-3.2-1B), simulated from the real-data common-tau fit, refitted (default start, as the real data), each with its own 1000-draw null on its own kept columns. Calibrated p of rank j = fraction of bootstrap datasets whose lambda_j percentile is >= the observed one; calibrated 99% threshold = the 99th percentile of the bootstrap percentiles. Calibrated validity of a null: the observed lambda_1 z = (obs - null mean) / null SD inside the 1st-99th percentile of the bootstrap lambda_1 z (defined after the run; see Failed tests 3).
- Unparsed mask. Every SP simulation (nulls, PPC, recovery, bootstrap, power, step-2b datasets) is conditional on the observed unparsed mask (unparsed cells unparsed and wrong, the others parsed). The choice likelihood conditions on which replies were parsed, so the mask affects only the simulations, not the fit.
- Power: primary column = rank 2 above p99 AND null valid (the paper's rule); also detection at the calibrated threshold.
- The bootstrap-averaged null gives every refit exactly 10 successful draws (at most 100 attempts per score; a refit short of 10 is excluded and counted). A null with < 100 draws gives no decision (counted). The power field `planted_var_share_stem_mean` is defined in `v3_F_power.json` `meta`.
- Sample sizes: power 150 datasets per s with 1000 SP null draws per refit (as in the calibration); the step-2b check runs only the nulls of its table (3PL on >=3 and all 4 omitted: invalid on the real data). The tau = 0 bootstrap is not run (the calibration is for the primary model only).

### Committed answer

**Under a calibrated soft-preference null, no score shows a direction beyond ability from rank 2 on (contiguous rule) that survives both data variants and the validity check.** This is narrower than "no structure": all-4 ranks 3-4 exceed the calibrated thresholds (calibrated p < 0.005) under a calibrated-valid null.
- **Valid calibrated nulls (>=1 and all 4 in both variants; >=2 on the full data only; >=3 without Llama-3.2-1B only): 0 ranks beyond ability.** >=2: calibrated p of ranks 2/3/4 = 0.245 / 0.250 / 0.865 (without Llama-3.2-1B 0.280 / 0.990 / 0.810, but there the lambda_1 check fails). All 4: rank 2 calibrated p 0.055 (0.080 without Llama-3.2-1B).
- **Rotation 0, rank 2 (the letter direction):** observed at the 95.1th percentile of the SP null (99.8th if the simulation ignores the observed unparsed mask: the clustered unparsed replies account for the difference), calibrated p 0.010 (2 of 200 bootstrap datasets as high), without Llama-3.2-1B 0.115. The calibrated null is not valid: the observed lambda_1 is larger than the pipeline produces (z 1.08 vs the bootstrap 99th percentile 0.78; without Llama-3.2-1B 2.66 vs 0.66). Not evidence either way.
- **>=3 of 4, rank 2:** above the calibrated threshold on the full data (calibrated p < 0.005, 0 of 200), but the null's lambda_1 lies just outside its calibrated band (z -4.15 vs 1st percentile -4.07; 2 of 200 bootstrap datasets lower), and without Llama-3.2-1B the null is valid and rank 2 is at calibrated p 0.285. The direction depends on one model (step 4: Llama-3.2-1B carries 0.43 of its squared model scores). The validity rule was defined after the run; with the Monte Carlo convention (k + 1)/(B + 1) = 3/201 = 0.0149 > 0.01 the full-data null is valid and >=3 gives 1 rank beyond ability.
- **Mean of 4:** ranks 2-3 (full) / rank 2 (without Llama-3.2-1B) above the calibrated thresholds, with a calibrated-invalid null in both variants (lambda_1 z -6.42 vs band [-5.58, -3.20]). This is the score most affected by the within-cell misfit the PPC finds; not evidence.
- **All 4, ranks 3-4:** above the calibrated thresholds (calibrated p < 0.005, both variants) while rank 2 is not: the SP model does not reproduce the lower all-4 spectrum. The 2PL (valid, calibrated: 0/200 false positives in step 2b, 0/150 in the power study) gives 0 ranks beyond ability.
- **>=2 artifact (2PL):** confirmed qualitatively, not in size. Under data from the SP fit the 2PL rejects rank 2 in 67.5% of datasets (plug-in parameters), 39.0% (bootstrap bias-corrected parameters) and 72.0% (mean-square-rescaled parameters); 3 ranks beyond ability (the real-data result) occurs in 52%, 20.5% and 52%. The "~65-68%" figure depends on the plug-in slopes; a large false-positive rate (>= 32%, lower CI bound) does not.
- **Power under the valid rule is low:** a STEM-block second ability is detected in >= 50% of datasets from s = 1.5 on >=2 (0.53), at s = 2 on rotation 0 (0.51 +- 0.04; >=2 0.79 +- 0.03), never on all 4 (0.42 +- 0.04 at s = 2, because its SP null is valid in only 4-47% of datasets) and never on the mean (<= 0.19 +- 0.03; +- = binomial SE). The calibrated threshold detects more (s = 2, rotation 0 / >=2 / all 4 / mean: 0.66 / 0.95 / 0.89 / 1.00) at an achieved size of 0-4% at s = 0 (0.7% / 2.0% / 0.0% / 4.0%).

Limits: the SP model fails its pre-set posterior-predictive check (below); the calibration is a parametric bootstrap at the fitted parameters, whose slopes are inflated (refit inflation of mean a^2 15.5%, tau 15% low) and are inflated again in each refit; the 99% thresholds come from 200 datasets (noisy in the tail).

### F1. Construction family under the existing null suite (1000 draws per null)

Option-referencing questions (programmatic rule, `option_reference_rule`): **23** (reviewer: 26). Indices: [28, 121, 124, 126, 145, 146, 187, 204, 207, 241, 298, 327, 334, 376, 377, 379, 421, 422, 428, 429, 444, 450, 460]. Rules: "... the above" / "all/none/both/neither of these" (17); explicit letter combinations "A and B only", "Both A and C", "(B) (C) and (D)" (4); "both"/"neither" joining the content of >= 2 other options (145, 422). The reviewer's pattern also hits 141 ("Either AC or DC"), 163 (propositional variables "C and D"), 167 (an "either ... or" proposition in formal logic), 214 ("these Carlsbad Resolutions"), 297 and 458 ("either" inside a sentence); it misses 145, 241, 422. Full list with the matched text: `v3_F_softpref.json` `step1_construction_family.option_referencing`.

Observed eigenvalues and ranks beyond ability under the VALID nulls only (nulls: Rasch JML, curveball, project 2PL posterior, 3PL c = 0.25, Bock-Aitkin a <= 4, know-or-guess 4-rotation letter / uniform, letter-aware rotation-0 for rotation 0 only; the mean has only the know-or-guess and independent-rotation 2PL nulls):

| score | N_cols | eigenvalues 1-4 | lambda_+ | valid nulls: ranks beyond ability | no Llama-3.2-1B | no option-referencing |
|---|---|---|---|---|---|---|
| rotation 0 | 460 | [85.9, 49.9, 29.7, 27.3] | 28.92 | 3PL 1, KG uniform 1, letter-aware 1 | 3PL 0, letter-aware 0 | 3PL 1, KG uniform 1, letter-aware 1 |
| >=1 of 4 | 334 | [34.2, 27.2, 23.7, 21.4] | 22.38 | none valid | BA 0 | BA 0 |
| >=2 of 4 | 444 | [108.3, 42.3, 30.2, 23.0] | 28.10 | 2PL 3, 3PL 0 | 2PL 1, 3PL 0 | 2PL 2, 3PL 0 |
| >=3 of 4 | 439 | [103.4, 37.8, 25.5, 22.1] | 27.85 | Rasch 2, 2PL 0, KG letter 2, KG uniform 0 | 2PL 0, KG letter 0, KG uniform 0 | Rasch 1, 2PL 0, KG letter 0, KG uniform 0 |
| all 4 | 391 | [92.1, 29.0, 23.3, 20.8] | 25.36 | Rasch 0, 2PL 0 | Rasch 0, 2PL 0, KG uniform 0 | Rasch 0, 2PL 0 |
| mean of 4 | 489 | [127.5, 43.1, 28.5, 25.7] | 30.40 | none valid | none valid | none valid |

Percentile of the observed lambda_1 (full data) by null, rotation 0: Rasch 100, curveball 100, 2PL 100, 3PL 76.4, BA 0, KG letter 99.9, KG uniform 98.7, letter-aware 86.7; >=2: 100, 100, 62.5, 91.9, 0, 100, 100; >=3: 93.1, 100, 44.5, 100, 0, 85.2, 97.9; all 4: 48.8, 100, 33.7, 100, 0, 0, 0 (the know-or-guess and Bock-Aitkin nulls cited for the all-4 answer in C(3) are invalid by this rule; Rasch and the 2PL remain and give 0). The >=3 2PL rank 2 is at its 98.9th percentile (37.8 vs p99 37.8). Per-null p99s and percentiles: `step1_construction_family.table`.

### F2. Soft-preference null

Model (per model m, question i, rotation r; rotation r shows option j at letter (j + r) mod 4): the observed unparsed cells are unparsed (wrong, as in the scoring rule); otherwise P(choose j) proportional to exp(b_ij + 1[j = gold_i](a_i theta_m + e_mi)) pi_m(letter of j), b_{i,gold} = 0, pi_m on the simplex (3 free), e_mi ~ N(0, tau^2) one per (model, question) cell, shared by its 4 rotations and independent across models (it adds no across-model structure). tau = 0 is the model exactly as specified. MAP fit over the parsed choices of 24 x 500 x 4 cells: theta ~ N(0, 1), a ~ N(0, 4^2), b ~ N(0, 4^2), letter logits ~ N(0, 5^2); e integrated by 41-node Gauss-Hermite; theta standardized after the fit. Variants: tau = 0; one common tau (fitted 1.166; log-likelihood -32484.0 vs -33100.8 for tau = 0, +616.9 for 1 parameter); per-model tau_m (sensitivity only). Primary null: common tau (`step2_soft_preference.primary`, computed).

Tests (all before use):

| test | result | status |
|---|---|---|
| likelihood vs explicit-loop oracle (tau = 0), 1000 Hypothesis cases | rtol 1e-12 | pass |
| likelihood vs adaptive quad over e (tau, tau_m), 60 cases, 201 nodes | rtol 1e-10 | pass |
| gradient vs central FD, every coordinate, small problems, 3 modes | rtol 1e-6, atol 1e-6 | pass |
| gradient vs central FD at the real fit and at a random point (h = 1e-5) | max rel err 3.1e-7 (tau = 0, tau, tau_m) | pass (FD rounding floor ~ eps |f| / h = 3e-7) |
| Gauss-Hermite at the fit: K = 161 minus K = 41 | 2.1e-9 (common tau), 2.6e-5 (tau_m) | pass |
| simulation frequencies vs analytic probabilities (20000 draws) | within 5 binomial SD | pass |
| simulated unparsed cells = the given mask, every draw and every redraw (static check of every call site too) | exact | pass |
| exchangeability at known parameters with a question-clustered unparsed mask: randomized rank of the observed lambda_1, lambda_2 among 19 null draws, 300 datasets, 3 scores | KS p > 1e-3 for all 6 (a null with unparsed cells at a per-model rate: KS p < 1e-6) | pass |
| parameter recovery at this size (3 replicates) | corr theta 0.995-0.997, a 0.91-0.92, b 0.95-0.97, log pi 0.996, P(correct) per cell 0.98 (mean abs err 0.045-0.047); tau 1.166 -> 0.96 / 1.02 / 1.01 | pass; tau biased low |
| restarts (2 perturbed) | tau = 0 and common tau: same objective (|diff| <= 6.6e-8 and 1.0e-9); tau_m: the 5 starts reach objectives from -30959 to -30330, 2 with max tau_m > 1e3; one second-pass restart equals the chosen fit, one is 348.9 lower | tau_m FAILS (multimodal) |
| posterior-predictive check (500 datasets; pass = 98%-interval coverage >= 0.90 per statistic family, observed N_cols within the 1st-99th percentile) | common tau: per-model accuracy 0.96, letter shares 1.00, per-question accuracy per rotation 0.96, per-model #correct of 4 0.82, N_cols percentile rot0 5.5, >=1 93.1, >=2 100.0, >=3 64.5, all-4 3.1, mean 3.7; tau = 0: #correct of 4 coverage 0.42, N_cols at the 100th percentile for >=1 to all 4 | FAILS (both) |

The PPC failure is in within-cell consistency (pooled share of cells with 0/1/2/3/4 of 4 correct: observed [0.183, 0.196, 0.122, 0.117, 0.382], common tau [0.188, 0.177, 0.134, 0.136, 0.366], tau = 0 [0.151, 0.201, 0.164, 0.152, 0.333]) and in the number of >=2 split columns. Per-model accuracy (max |obs - sim| 0.031) and letter shares (coverage 1.00) fit. Conditioning on the unparsed mask raises the common-tau #correct-of-4 coverage from 0.79 to 0.82, short of a pass.

**Calibration.** Common tau, 200 bootstrap datasets per variant, 1000 null draws each (the real data's null also has 1000 draws). All 400 refits converged; refitted tau 0.994 (SD 0.029) vs the generating 1.166 (without Llama-3.2-1B 0.957 vs 1.136). No conditional-sampling failure in any bootstrap null. `step2_soft_preference.calibration.by_variant.<variant>.<score>`.

| variant | score | observed pct of lambda_1..4 under the SP null fitted to the real data | lambda_1 z (bootstrap 1st-99th pct) | calibrated valid | calibrated p, rank 2 / 3 / 4 | ranks above the calibrated 99% threshold | false-positive rate of "rank 2 > p99 and valid" (95% CI) | plain null valid in |
|---|---|---|---|---|---|---|---|---|
| full | rotation 0 | 85.7, 95.1, 63.1, 97.1 | 1.08 (-3.12, 0.78) | no | 0.010 / 0.140 / 0.025 | 1, 2 | 0.000 (0-0.018) | 0.87 |
| full | >=1 | 2.4, 15.7, 13.2, 14.3 | -1.96 (-2.56, 1.21) | yes | 0.750 / 0.810 / 0.800 | none | 0.000 (0-0.018) | 0.94 |
| full | >=2 | 41.7, 47.8, 52.7, 14.0 | -0.24 (-2.91, 0.17) | yes | 0.245 / 0.250 / 0.865 | none | 0.000 (0-0.018) | 0.91 |
| full | >=3 | 0.0, 94.8, 88.6, 91.1 | -4.15 (-4.07, -0.93) | no | 0.000 / 0.180 / 0.075 | 2 | 0.000 (0-0.018) | 0.42 |
| full | all 4 | 0.0, 24.5, 99.7, 100.0 | -4.55 (-4.87, -1.61) | yes | 0.055 / 0.000 / 0.000 | 3, 4 | 0.000 (0-0.018) | 0.09 |
| full | mean | 0.0, 100.0, 100.0, 100.0 | -6.42 (-5.58, -3.20) | no | 0.000 / 0.000 / 0.025 | 2, 3 | 0.000 (0-0.018) | 0.00 |
| no Llama-3.2-1B | rotation 0 | 99.7, 59.6, 97.4, 98.6 | 2.66 (-3.15, 0.66) | no | 0.115 / 0.025 / 0.015 | 1 | 0.005 (0-0.028) | 0.88 |
| no Llama-3.2-1B | >=1 | 0.5, 0.8, 30.9, 12.9 | -2.38 (-2.85, 0.58) | yes | 0.960 / 0.610 / 0.840 | none | 0.000 (0-0.018) | 0.94 |
| no Llama-3.2-1B | >=2 | 52.2, 51.2, 0.6, 16.8 | 0.04 (-3.32, -0.07) | no | 0.280 / 0.990 / 0.810 | 1 | 0.000 (0-0.018) | 0.81 |
| no Llama-3.2-1B | >=3 | 2.5, 36.3, 22.6, 38.7 | -2.01 (-4.25, -0.88) | yes | 0.285 / 0.795 / 0.685 | none | 0.000 (0-0.018) | 0.41 |
| no Llama-3.2-1B | all 4 | 0.0, 28.8, 99.5, 100.0 | -3.78 (-4.95, -1.95) | yes | 0.080 / 0.000 / 0.000 | 3, 4 | 0.000 (0-0.018) | 0.04 |
| no Llama-3.2-1B | mean | 0.0, 92.4, 67.9, 87.6 | -5.40 (-4.97, -2.71) | no | 0.000 / 0.615 / 0.240 | 2 | 0.000 (0-0.018) | 0.00 |

Calibrated p by the z statistic and the Monte Carlo version (k + 1) / 201: `cal_p_z_rank1to4`, `cal_p_mc_rank1to4` (they differ from the column above by <= 0.025). The uncalibrated SP null is conservative for rank 2: its p99 rule fires in 0-1 of 200 bootstrap datasets for every score, and its validity check fails in most bootstrap datasets of >=3, all 4 and mean although they come from the model itself (last column). Calibrated, the false-positive rate of the paper's rule at s = 0 is 0.000-0.005.

Uncalibrated plug-in nulls, 1000 complete datasets each (bootstrap-averaged: 200 refits x 10 draws), percentiles of the observed lambda_1..4 (`*` = above p99, `inv` = lambda_1 outside p01-p99):

| null | rotation 0 | >=2 | >=3 | all 4 | mean |
|---|---|---|---|---|---|
| tau = 0 (as specified) | 73.9, 78.6, 55.5, 99.5* | inv 0.0, 0.0, 52.9, 37.5 | inv 0.0, 0.0, 87.2, 98.7 | inv 0.0, 0.0, 14.9, 58.0 | inv |
| common tau (primary) | 85.7, 95.1, 63.1, 97.1 | 41.7, 47.8, 52.7, 14.0 | inv 0.0, 94.8, 88.6, 91.1 | inv 0.0, 24.5, 99.7*, 100.0* | inv |
| common tau, bootstrap-averaged | 51.2, 79.2, 40.9, 96.0 | 12.3, 12.4, 31.9, 12.7 | inv 0.0, 63.9, 88.4, 89.2 | inv 0.0, 1.5, 99.7*, 100.0* | inv |
| tau_m (sensitivity) | 83.0, 81.3, 73.4, 99.4* | 20.3, 4.2, 95.1, 82.6 | inv 0.0, 68.2, 94.3, 95.3 | inv 0.0, 46.7, 98.1, 99.0* | inv |
| common tau, penalty sd 2 / 8 | inv / 64.0, 92.8, 72.4, 98.6 | 95.2, 51.0, 22.4, 2.8 / 6.6, 52.2, 89.2, 34.2 | inv / inv | inv / inv | inv |
| no Llama-3.2-1B, common tau | inv 99.7*, 59.6, 97.4, 98.6 | 52.2, 51.2, 0.6, 16.8 | 2.5, 36.3, 22.6, 38.7 | inv 0.0, 28.8, 99.5*, 100.0* | inv |
| no Llama-3.2-1B, bootstrap-averaged | 92.0, 26.8, 93.7, 96.3 | 13.9, 14.8, 0.9, 18.9 | inv 0.0, 9.7, 20.3, 38.4 | inv 0.0, 2.4, 99.6*, 100.0* | inv |
| no option-referencing, common tau | 80.9, 93.6, 37.2, 95.4 | 39.9, 50.1, 53.4, 9.7 | inv 0.0, 94.6, 85.3, 87.7 | inv 0.0, 28.2, 99.9*, 100.0* | inv |

Conditional sampling failures (a kept column constant after 5000 redraws; draw dropped and counted): 0 in every plug-in null except tau = 0 on >=1 (323 full, 18 without Llama-3.2-1B, 378 without the option-referencing questions, 275 penalty sd 8, of 1000). Bootstrap-averaged null: refits excluded for fewer than 10 successful draws: full >=1 2, >=3 1, all 4 2 (of 200); none without Llama-3.2-1B.

### F2b. The step-1 nulls under the soft-preference truth (200 datasets simulated from the common-tau fit, observed unparsed mask, no second factor; each null refitted, 200 draws)

| score: null | false-positive rate of "rank 2 above p99" (95% CI) | and valid | null valid | real data |
|---|---|---|---|---|
| rotation 0: 3PL / KG uniform / letter-aware / KG letter | 0.995 / 0.990 / 0.595 / 0.500 | 0.995 / 0.955 / 0.595 / 0.490 | 1.00 / 0.96 / 1.00 / 0.98 | rank 2 above 3PL, KG uniform, letter-aware |
| >=2: 2PL / 3PL | 0.670 (0.600-0.735) / 0.000 | 0.670 / 0.000 | 0.99 / 0.78 | 2PL: ranks 2-4 above |
| >=3: Rasch / 2PL / KG letter / KG uniform | 0.055 (0.028-0.096) / 0.000 (0-0.018) / 0.145 / 0.000 | 0.000 / 0.000 / 0.010 / 0.000 | 0.14 / 0.99 / 0.13 / 0.00 | Rasch and KG letter: ranks 2-3 above; 2PL 98.9th pct |
| all 4: Rasch / 2PL / KG letter / KG uniform | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 each | 0.91 / 1.00 / 0.35 / 0.48 | none above |

With unparsed cells simulated at a per-model rate instead of the observed mask, three rates differ by more than 0.06: rotation 0 letter-aware 0.365 -> 0.595 and KG letter 0.335 -> 0.500, >=3 KG letter 0.470 -> 0.145; the others moved by <= 0.06.

### F2c. The >=2 / 2PL artifact under bias-corrected parameters (`step2c_ge2_2pl_under_bias_corrected_parameters`)

Inflation measured by the full-data calibration bootstrap (refit of data simulated from the fit vs the fit): mean square of a x 1.155, of the non-gold b x 1.056, of the letter logits x 1.012; tau 0.994 vs 1.166. Parameter sets: plug-in; bootstrap bias-corrected (x_bc = 2 x_hat - mean of the 200 refits, elementwise, for a, b, letter logits and tau = 1.338); mean-square-rescaled (a x 0.931, b x 0.973, letter logits x 0.994, tau = 1.368). 200 datasets per set (observed unparsed mask), project 2PL null on >=2 refitted, 200 draws:

| parameter set | mean a^2 | rank 2 above p99 (95% CI) | and valid (95% CI) | null valid | ranks beyond ability 0 / 1 / 2 / 3 | mean observed lambda_1..4 |
|---|---|---|---|---|---|---|
| plug-in | 4.65 | 0.675 (0.605-0.739) | 0.660 (0.590-0.725) | 0.97 | 65 / 10 / 21 / 104 | 100.1, 38.4, 28.4, 23.7 |
| bootstrap bias-corrected | 4.57 | 0.390 (0.322-0.461) | 0.385 (0.317-0.456) | 0.98 | 122 / 18 / 19 / 41 | 90.3, 35.8, 27.4, 23.5 |
| mean-square-rescaled | 4.03 | 0.720 (0.652-0.781) | 0.690 (0.621-0.753) | 0.97 | 56 / 9 / 31 / 104 | 90.8, 36.8, 28.4, 23.9 |

Real data (>=2): lambda = 108.3, 42.3, 30.2, 23.0; the 2PL gives 3 ranks beyond ability. The 2PL is not calibrated on >=2 under any of the three sets; the size of its false-positive rate depends on the parameter set (the elementwise bias correction also changes b and tau, and lowers it most).

### F3. Power (planted second ability; `v3_F_power.json`)

Base: the common-tau fit. Planted: gold utility + s theta2_m on the STEM block (Hendrycks et al. category: 18 subjects, 180 questions), 0 elsewhere, theta2 ~ N(0, 1) per dataset, independent of theta; observed unparsed mask. 150 datasets per s; per dataset the SP null (1000 draws) and the project 2PL (binary scores, 200 draws) are refitted; all 900 SP refits converged; no null had < 100 draws. Cells: rank 2 above p99 AND null valid (primary) / rank-2 percentile above the calibrated threshold [fraction of datasets with a valid SP null]. Calibrated thresholds (99th percentile of the rank-2 percentiles of the 200 full-data calibration datasets): rotation 0 94.3, all 4 58.4, >=2 84.6, mean 46.1.

| s | planted share (STEM, mean) | mean STEM pair r, rot 0 (increase) | SP rot 0 | SP all 4 | SP >=2 | SP mean | 2PL all 4 | 2PL >=2 | 2PL rot 0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 (size) | 0 | 0.051 (0) | 0.000 / 0.007 [0.90] | 0.000 / 0.000 [0.04] | 0.000 / 0.020 [0.87] | 0.000 / 0.040 [0.00] | 0.000 | 0.693 | 0.000 [valid 0.00] |
| 0.25 | 0.028 | 0.056 (+0.006) | 0.000 / 0.007 [0.89] | 0.000 / 0.000 [0.07] | 0.000 / 0.007 [0.92] | 0.000 / 0.013 [0.00] | 0.000 | 0.693 | 0.000 |
| 0.5 | 0.102 | 0.067 (+0.016) | 0.007 / 0.020 [0.89] | 0.000 / 0.013 [0.10] | 0.000 / 0.047 [0.89] | 0.000 / 0.073 [0.00] | 0.000 | 0.627 | 0.007 |
| 1 | 0.299 | 0.114 (+0.064) | 0.000 / 0.060 [0.90] | 0.007 / 0.147 [0.11] | 0.053 / 0.227 [0.95] | 0.000 / 0.680 [0.00] | 0.047 | 0.587 | 0.000 |
| 1.5 | 0.471 | 0.173 (+0.122) | 0.133 / 0.187 [0.95] | 0.147 / 0.593 [0.31] | 0.527 / 0.760 [0.96] | 0.047 / 0.987 [0.05] | 0.427 | 0.687 | 0.160 |
| 2 | 0.595 | 0.228 (+0.178) | 0.513 / 0.660 [0.98] | 0.420 / 0.893 [0.47] | 0.793 / 0.953 [0.89] | 0.193 / 1.000 [0.19] | 0.767 | 0.867 | 0.340 |

"Planted share" = per STEM question s^2 / (a_i^2 + s^2 + tau^2), the planted term's share of the across-model variance of the gold-utility shift (option-attractiveness and letter terms excluded), averaged over the STEM questions (`meta.planted_var_share_stem_mean`). Size: the valid-null rule fires in 0/150 datasets at s = 0 for every SP score (95% CI upper 0.024). The calibrated threshold has an achieved size of 0.7%, 0.0%, 2.0%, 4.0% at s = 0 (its rank-2 percentiles have the distribution of the calibration datasets: KS p >= 0.22 per score; the 99th percentile of 200 values is noisy). The 2PL rule is calibrated only on all 4 (size 0.000, valid 1.00); on >=2 it fires in 69% of datasets without a second factor; on rotation 0 it rejects rank 2 in every dataset but its null is invalid there. Detection rates with Clopper-Pearson intervals, rank 2 without the validity condition, the z-threshold version and "any of ranks 2-4": `summary.<s>.by_null`.

### F4. Characterization of the borderline directions (descriptive; the step-4 condition "the soft-preference model fits" is not met)

Eigenvectors: >=2 ranks 2-4, >=3 ranks 2-3, >=2 rank 2 without Llama-3.2-1B, all-4 rank 2, rotation-0 rank 2 (`step4_characterization`; 5000 label permutations, 10000 for correlations; seed [20261014, 11]). After removing a cubic in question difficulty, subject eta^2 vs its permutation null: >=2 rank 2 0.165 (null 0.111, p = 0.010), rank 3 0.151 (0.111, p = 0.034), rank 4 0.104 (p = 0.62); >=3 rank 2 0.095 (0.112, p = 0.78), rank 3 0.135 (p = 0.14); STEM vs rest eta^2 <= 0.009 everywhere (p >= 0.055). Model scores by family: eta^2 0.30-0.61, p >= 0.06, except >=3 rank 3 (0.72, p = 0.012; also r with log size -0.44, p = 0.033). Correlation of model scores with accuracy |r| <= 0.19 (p >= 0.36), with letter-preference concentration (max pi) |r| <= 0.38 (p >= 0.066). Loadings by original gold letter: eta^2 <= 0.022 (rotation-0 rank 2: 0.482, p = 0.0002, the known letter direction). Largest share of squared model scores: >=2 rank 2 Llama-3.2-1B 0.30, rank 3 Qwen2.5-0.5B 0.50; >=3 rank 2 Llama-3.2-1B 0.43, rank 3 Qwen2.5-0.5B 0.32 (eigenvector 1 is as concentrated: participation ratio 2.4-7.8 models vs 3.0-10.0 for ranks 2-4 of the same matrices, so this does not separate them). Replicate B: |r| of loadings with the same-rank B eigenvector 0.999-1.000 (B differs from A in 6 of 48,000 cells, so this tests GPU nondeterminism only). None of these directions is a subject, STEM, accuracy or letter-preference direction; the two uncorrected subject p-values (0.010, 0.034) do not survive a correction over the 8 eigenvectors tested. Replicate B differs from A in the chosen option in 6 of 48,000 cells (all Qwen2-7B), in correctness in 5.

### Failed tests and caveats

1. The soft-preference model fails its pre-set posterior-predictive rule in every variant (per-model #correct-of-4 coverage 0.42 for tau = 0, 0.82 common tau, 0.82 tau_m; the >=2 split-column count at the 100th percentile). Its nulls are not fully validated; the calibration below corrects the pipeline's own bias, not the misfit.
2. The calibration is a parametric bootstrap at the fitted parameters. Those parameters carry the incidental-parameter inflation (mean a^2 x 1.155 per refit, tau 15% low), and each bootstrap refit inflates them again; the calibrated p is exact only if this second inflation equals the first.
3. Calibrated validity (the observed lambda_1 z inside the bootstrap 1st-99th percentile) was defined after the run: the percentile version saturates at 0 for >=3, all 4 and mean, where the null never reaches the observed lambda_1. Both versions are in the JSON (`lambda1_cal_p_two_sided`, `lambda1_cal_p_z_two_sided`, `lambda1_calibrated_valid`). The >=3 decision depends on it: its lambda_1 z (-4.15) lies 0.08 below the band.
4. The calibrated thresholds and p-values come from 200 bootstrap datasets per variant: a p of 0.010 is 2 datasets; the 99th percentile of 200 values is noisy (achieved size 0-4% in the power study).
5. The per-model tau_m likelihood is multimodal (start objectives from -30959 to -30330; 2 of 5 starts reach tau_m > 1e3). tau_m results are sensitivity only; it is not bootstrapped and not used for power.
6. Power is for one planted block structure (STEM, 180 questions, a common slope s), 150 datasets per s.
7. The step-2b check runs only the nulls of its table (3PL for >=3 and all 4 omitted; invalid on the real data, `meta.real_data_validity_of_step1_nulls`).
8. 3PL with c = 0.25 is nominal for the >=k scores; the letter-aware rotation-0 null exists only for rotation 0. The option-referencing count (23) depends on the stated rule (141 and 167 not counted).

## G. Family-wise test over ranks 2-4, sibling clusters, one validity rule

Specification fixed before the run (`v3_G_familywise.json` `meta.preregistration`); a similar post-hoc statistic had already been computed on the same data (p 0.033 / 0.048). Code `scripts/v3_G_familywise.py` (+ 23 tests). Output `results/analysis/v3_G_familywise.json`, raw `raw_v3_G_*`. Seed namespace SEED_G = 1202610137. The all-4 p_T was reproduced independently from the raw correctness matrix.

- **Statistic.** T = max over ranks r = 2-4 of the percentile of lambda_r in the null's rank-r draws (ties half); null distribution of T by leave-one-out over 2000 new draws; alpha 0.05. Single validity rule: observed lambda_1 within the null's 1st-99th percentile.
- **G1 all-4.** Both nulls valid. 2PL: percentiles of lambda_2..4 = 71.8 / 98.5 / 99.1, p_T 0.026; Rasch: 94.5 / 97.0 / 98.6, p_T 0.041. Maximum at rank 4. Per-rank rule: 0 ranks beyond ability. Rotation 0: 3PL, KG-u, letter-aware reject (p_T 0.002); SP null valid (lambda_1 at 85.5 pct), p_T 0.080.
- **G2 (SP truth, 200 datasets).** All-4 FP 0/200 for both nulls (SP is invalid on all-4, so this is not a size argument; see H1). Rotation 0, >=1, >=2: FP 1.0; >=3: about 0.6. Decisions on >=2 / >=3 are not reliable (p_T saturates at T = 100).
- **G3 power (all-4, STEM factor, 200 datasets).** s = 1: 2PL 0.335 / Rasch 0.205 (per-rank rule 0.055 / 0.075). s = 2: 0.92 / 0.87 (0.805 / 0.795). Median fitted slope 1.08.
- **G4 siblings** (clusters fixed from metadata: Phi-3/3.5-mini; Llama-3/3.1-8B; Mistral-7B v0.2/v0.3/Zephyr; Qwen2/2.5-7B). Keeping one per cluster (24 subsets of 19): 0/24 reject (p_T 0.09-0.69). Removing 5 non-sibling models at random (24 subsets): 24/24 reject (p_T <= 0.047). Rotation 0: letter direction rejects in 24/24 sibling subsets (3PL, letter-aware).
- **G5 single rule.** Table A2 SP column: rot0 0.010 (MC 0.015); >=1 0.750 / inv.; >=2 0.245 / 0.280; >=3 inv. / 0.285; all-4 inv. / inv.; mean inv. / inv. (full / without Llama-3.2-1B). SP power at s = 2: rot0 0.64, all-4 0.453, >=2 0.847, mean 0.193. This replaces section F's "calibrated validity".
- **G6 unparsed replies.** All-4: impute 0.019 / 0.042; drop 95 affected questions 0.036 / 0.043; drop all 217 with any unparsed reply 0.234 / 0.121 (236 columns, less power). Rotation 0 drop: letter-aware per-rank 0 (rank 2 at 98.3 pct), family-wise 0.013; 3PL keeps 1 rank.
- **Decision: pre-specified rule B** (rejects under both nulls; 0/24 sibling subsets vs 24/24 random).

## H. Follow-ups (post hoc; specification fixed before the run)

Code `scripts/v3_H_followup.py` (+ 8 tests). Output `results/analysis/v3_H_followup.json`, raw `raw_v3_H_*`. SEED_H = 2402610141.

- **H1 size of G1 on all-4 (300 datasets per truth, nulls refitted, 500 draws).** FP under 2PL truth: 2PL 0.067, Rasch 0.067. Under Rasch truth: 2PL 0.087, Rasch 0.007 (Rasch null valid in only 23% of those datasets). Rule (FP <= 0.075 under both): Rasch size-controlled, 2PL not. Calibrated p of the real result: 2PL 0.033 / 0.047, Rasch 0.057 / 0.017 (2PL / Rasch truth). **The all-4 rank 3-4 excess is marginal at the 5% level.** The validity rule does not detect slope misspecification (necessary, not sufficient).
- **H2 pairs.** Removing one of Phi-3/3.5-mini and one of Llama-3/3.1-8B (4 subsets of 22): p_T 0.22-0.69 (2PL), 0.32-0.38 (Rasch), above all 190 removals of two other models (reference max 0.099 / 0.133; 90% / 74% of them reject). **The excess is carried by the two near-duplicate pairs.** The Qwen 7B pair is not part of it.
- **H3 baselines.** Rotation-0 max letter share > 0.5 (unparsed replies in the denominator): Llama-3.2-1B A 0.81, Qwen2.5-0.5B D 0.58, SmolLM2-1.7B D 0.56. Residual pair correlation after the all-4 2PL fit (276 pairs): Phi-3/3.5 0.401 (rank 1), Llama-3/3.1 0.267 (2), Mistral v0.2/v0.3 0.204 (3); Qwen2/2.5-7B rank 37. Simple tables find the same candidates; the spectrum adds the test and the attribution.

## I. Without Llama-3.2-1B, and joint removal (post hoc; specification fixed before the run)

Code `scripts/v3_I_llama.py` (+ 8 tests). Output `results/analysis/v3_I_llama.json`, raw `raw_v3_I_*`. SEED_I = 3602610143. 2000 new draws per null (I2: 1000).

- **Rotation 0, 23 models** (eigenvalues 76.0, 35.0, 32.5, 29.0). Valid nulls: 3PL, letter-aware. Per-rank rule: 0 directions beyond ability (rank 2 at the 48.2 / 72.2 pct). Family-wise: both reject (p_T 0.0035 / 0.015), maximum at rank 4 (ranks 3-4 at 98.5-99.9 pct). Rank 2 rests on Llama-3.2-1B; a weaker excess at ranks 3-4 remains without it.
- **All-4, 23 models** (79.75, 27.18, 23.28, 20.71). Valid: 2PL, Rasch, KG-u. Family-wise p_T 0.060 / 0.089 / 0.152 (no rejection); secondary p_S 0.023 / 0.017 for 2PL / Rasch.
- **I2, all-4, Llama-3.2-1B + one model of each pair removed (21 models):** p_T 0.30-0.59 (2PL), 0.38-0.60 (Rasch). From H2: the 19 two-model removals that include Llama-3.2-1B reject in 12/19 (2PL) and 6/19 (Rasch).
- **Accuracy-matched control (from H2):** no other two models lie in the pair models' rotation-0 accuracy range (0.644-0.684); in the band 0.62-0.71, 15 removals give p_T <= 0.052 (2PL) / 0.113 (Rasch), all below every pair removal.
- **Reading:** the all-4 excess comes mostly from the two near-duplicate pairs, with a smaller part from Llama-3.2-1B.

## J. All-4 power from the valid 2PL fit; the difficulty factor (post hoc; specification fixed before the run)

Code `scripts/v3_J_power2pl.py`. Spec `results/analysis/specs/J_power_2pl.md`. Output `results/analysis/v3_J_power2pl.json`, raw `raw_v3_J_per_dataset.json`. SEED_J = 3502610143. G3 simulated power from the SP fit, which is invalid on all-4; J1 simulates from the valid 2PL fit to all-4.

- **J1 power (200 datasets per s, 2PL and Rasch nulls refitted, 500 draws).** Second ability z ~ N(0, 1) on the 125 kept STEM columns of 391, logit term s * a_med * sd(theta) * z (a_med 1.71, sd(theta) 1). Family-wise (valid and p_T < 0.05), 2PL / Rasch: s = 0: 0.07 / 0.07; s = 0.5: 0.195 / 0.205; s = 1: 0.91 / 0.925; s = 2: 0.99 / 0.995. Rank 2 above p99 (and valid): 0.000 / 0.03; 0.005 / 0.065; 0.685 / 0.78; 0.975 / 0.995. Median lambda_2: 28.2, 28.3, 33.8, 48.3 (observed 29.0). **A second ability with the spread and median slope of the first on the STEM third is usually detected; one half as strong is not.** s = 0.5 was added after a 2-dataset smoke test, before the full run.
- **J2 difficulty factor (200 null draws).** |r| of eigenvector-2 loadings with column pass rate, median (5-95%): all-4 2PL 0.624 (0.575-0.668), Rasch 0.628 (0.559-0.671), observed 0.565; rotation-0 3PL 0.086 (0.029-0.156), observed 0.042. In the all-4 nulls, eigenvector 2 is the difficulty factor of binary items.

