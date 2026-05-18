"""Diagnose Horn PA behavior on Qwen MMLU.

Key question: if columns are truly independent Bernoulli (surrogate),
how does observed spectrum differ from surrogate spectrum?
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from horn_vs_mp import borderline_filter, corr_eigs_desc

path = ROOT / "results" / "correctness_matrix_Qwen_Qwen2.5-7B-Instruct.npy"
C = np.load(path).astype(np.float64)
C_B = borderline_filter(C, 0.05)
K, Nb = C_B.shape
gamma = Nb / K
lp = (1 + np.sqrt(gamma)) ** 2
print(f"Qwen MMLU borderline: K={K} N_b={Nb} gamma={gamma:.3f} lambda_+={lp:.3f}")

# per-column flip rates
p = C_B.mean(axis=0)
print(f"per-column flip rate: min={p.min():.3f} median={np.median(p):.3f} max={p.max():.3f}")
print(f"std of flip rate across cols: {p.std():.3f}")

# observed eigenvalues
eig_obs = corr_eigs_desc(C_B)
print(f"Top 10 observed: {eig_obs[:10]}")
print(f"Rank 20/50/100 observed: {eig_obs[19]:.3f}, {eig_obs[49]:.3f}, {eig_obs[99]:.3f}")

# surrogate
rng = np.random.default_rng(20260417)
n_iter = 200
eig_rand = np.empty((n_iter, Nb))
for it in range(n_iter):
    C_r = C_B.copy()
    for j in range(Nb):
        rng.shuffle(C_r[:, j])
    eig_rand[it] = corr_eigs_desc(C_r)

mean_rand = eig_rand.mean(axis=0)
p95 = np.percentile(eig_rand, 95, axis=0)
p50 = np.percentile(eig_rand, 50, axis=0)

print(f"\nRank | obs    mean(rand)  p50(rand)  p95(rand)  obs>p95?")
for r in [0, 1, 2, 4, 9, 14, 19, 29, 49, 99, 120]:
    if r >= Nb: continue
    mark = "YES" if eig_obs[r] > p95[r] else "no"
    print(f"  {r+1:3d} | {eig_obs[r]:.3f}  {mean_rand[r]:.3f}     {p50[r]:.3f}     {p95[r]:.3f}    {mark}")

# How many total "obs > p95"?
total_above = int((eig_obs > p95).sum())
print(f"\nTotal ranks where obs > p95: {total_above}")

# Show where agreement breaks
above_ranks = np.where(eig_obs > p95)[0]
print(f"Ranks above p95: {above_ranks.tolist()[:30]}...")

# Key check: is observed spectrum systematically shifted vs surrogate, or
# are they the same distribution?
print(f"\nSums: obs={eig_obs.sum():.3f} vs mean_rand.sum()={mean_rand.sum():.3f} (should be = Nb = {Nb})")
print(f"Top-half sum obs={eig_obs[:Nb//2].sum():.3f} vs rand mean={mean_rand[:Nb//2].sum():.3f}")

# Contiguous leading factors (classical Horn rule):
# Stop at first rank where obs <= p95. Retain everything before.
d_contig = 0
for r in range(Nb):
    if eig_obs[r] > p95[r]:
        d_contig = r + 1
    else:
        break
print(f"Contiguous leading factors (classical Horn): {d_contig}")
