# J. Power of the all-4-correct test from the valid 2PL fit; the second eigenvalue of one-ability nulls

Written before the J run (2026-10-14). Post hoc: both questions arose from a blind read of the paper.

## J1. Power from the 2PL fit to all-4-correct

The G3 power study simulates from the soft-preference (SP) fit, which is invalid on all-4-correct. J1 repeats
it from the 2PL fit to the all-4-correct matrix, which is valid (G1).

- Truth: the 2PL fit of `v3_across.Context(ge4(CR)).fit_2pl()` on the full data. For each column kept in the
  observed matrix (the 391 split columns), (a_i, b_i) are drawn from the column's grid posterior, as the
  `2pl_post` null does. A second ability z_m ~ N(0, 1), independent of theta, enters the STEM columns:
  logit P(c_mi = 1) = a_i (theta_m - b_i) + s * a_med * sd(theta) * z_m * STEM_i,
  with a_med the median posterior-mean slope over the kept columns and sd(theta) the SD of the fitted
  abilities. s = 1: a second ability with the spread of the first and the median slope (the analogue of
  G3 s = 1); s = 2 also; s = 0 gives the size under 2PL truth (compare H1).
- Amendment (after a 2-dataset smoke test, before the full run): s = 0.5 added, because the smoke test
  detected s = 1 in both datasets and the power curve needs a point below s = 1.
- Columns are redrawn until split (as `twopl_sample(condition_on_split=True)`).
- Each dataset: 2PL and Rasch nulls (`2pl_post`, `rasch_jml`) refitted, 500 draws, G1 statistic
  (`v3_G_familywise.familywise`).
- Detection: (a) family-wise: valid and p_M < 0.05; (b) per-rank: valid and rank 2 above the null's p99.
- 200 datasets per s. Seeds: new namespace SEED_J.

## J2. Why one ability lifts the second eigenvalue

For 200 draws of each valid null (all-4-correct: 2PL, Rasch; rotation 0: 3PL), the eigenvector-2 loadings of
the draw are correlated (Pearson, across kept columns) with the column pass rate of the draw. Reported: the
median and the 5th-95th percentiles of |r|, and the same for eigenvector 3. Also reported for the observed
matrices. The "difficulty factor" of binary items (correlations of items of unequal difficulty are capped,
so one trait gives a second component that orders items by difficulty) predicts a large |r| for eigenvector 2.
