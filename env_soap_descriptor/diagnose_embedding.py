"""
diagnose_embedding.py
====================
Disambiguate the NO-GO read: is the huge nnz/row genuine chemical crowding, a solvent
artifact, or an embedding-concentration (curse-of-dimensionality) failure? And why did
the skill R^2 go negative? Runs on a CACHED population (no re-fetch).

    python diagnose_embedding.py cache/pop_20000.npz
"""

import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "cache/pop_20000.npz")
Z, mol_of, y = d["Z"], d["mol_of"], d["y"]
n = len(Z)
rng = np.random.default_rng(0)
print(
    f"loaded {sys.argv[1] if len(sys.argv)>1 else 'pop'}: n_env={n:,} n_mol={mol_of.max()+1:,} D={Z.shape[1]}"
)

# 1. CONCENTRATION: CV of pairwise distances. CV<0.3 => distances concentrated
#    (curse of dimensionality) => no radius separates near from far => sparsity impossible.
s = Z[rng.choice(n, 2000, replace=False)]
dd = pdist(s)
print(
    f"\n[concentration] pairwise dist mean={dd.mean():.3f} std={dd.std():.3f} "
    f"CV={dd.std()/dd.mean():.3f}   (CV<0.3 = concentrated, bad)"
)

# 2. NEAR-DUPLICATES / SOLVENT CLUMP: distance to 1st neighbor. Many near-zero =>
#    massive redundancy (e.g. water) that fixed 1e-2 dedup may miss.
tree = cKDTree(Z)
probes = rng.choice(n, 5000, replace=False)
dist2, _ = tree.query(Z[probes], k=2, workers=-1)
nn1 = dist2[:, 1]
print(
    f"\n[near-duplicates] 1st-NN dist median={np.median(nn1):.4f} "
    f"frac<0.01={np.mean(nn1<0.01):.3f} frac<0.1={np.mean(nn1<0.1):.3f} "
    f"frac<0.5={np.mean(nn1<0.5):.3f}"
)

# 3. IS THERE ANY USABLE RADIUS? nnz/row across a FINE grid incl. very small radii.
print("\n[nnz/row vs radius]  (budget for 100M is ~600/row)")
for r in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    c = np.asarray(
        tree.query_ball_point(Z[probes[:2000]], r=r, return_length=True, workers=-1)
    )
    print(
        f"   radius={r:5.2f}  median={np.median(c):8.0f}  mean={c.mean():9.0f}  "
        f"p95={np.percentile(c,95):9.0f}"
    )

# 4. DEDUP AT MULTIPLE TOLERANCES: does redundancy appear at coarser tol than 1e-2?
print("\n[dedup] distinct-environment fraction vs grid tolerance")
samp = Z[rng.choice(n, 50000, replace=False)]
for tol in [1e-2, 0.05, 0.1, 0.3, 0.5]:
    keys = np.round(samp / tol).astype(np.int64)
    frac = len(np.unique(keys, axis=0)) / len(samp)
    print(f"   tol={tol:4.2f}  distinct fraction={frac:.3f}")

# 5. WHY SKILL R^2 EXPLODED: system-size heterogeneity. If |residual| scales with size,
#    a few big solvated systems dominate the scale-sensitive R^2.
sizes = np.bincount(mol_of)
print(
    f"\n[size heterogeneity] atoms/mol median={np.median(sizes):.0f} "
    f"p95={np.percentile(sizes,95):.0f} max={sizes.max()}"
)
print(
    f"   residual y: std={y.std():.2f} eV   corr(|y|, size)={np.corrcoef(np.abs(y), sizes)[0,1]:.2f}"
    f"   (high corr => size drives the residual => per-atom scoring needed)"
)
