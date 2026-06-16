"""
option4_wasserstein.py
======================
Computes the Option-4 molecular representation from the gp2Scale-OMol25
position paper (rotationally invariant pairwise distance profiles) and the
optimal-transport (Wasserstein) distance between molecules, then re-runs the
k-NN sparsity diagnostic under that metric.

Drop-in companion to explore_omol25.py: it reuses `load_dataset` from there
and produces the same style of k-NN histogram, so you can compare the
Wasserstein result directly against the Coulomb-eigenspectrum result.

The representation (Option 4)
-----------------------------
Each atom i is described by:
  - its atomic number Z_i, and
  - its sorted pairwise-distance profile: the sorted vector of distances to
    every other atom in the molecule.
Because the profile is built from interatomic distances and then sorted, it is
invariant to rotation, translation, and atom relabeling -- the three symmetries
a good molecular descriptor must respect, with no spatial cutoff.

To compare profiles across molecules of different sizes, each profile is
summarized at a fixed number of quantiles (n_quantiles), giving every atom a
fixed-length feature vector regardless of how many atoms its molecule has.

A molecule then becomes a *distribution* (a uniform point cloud) over its atoms'
features. The distance between two molecules is the optimal-transport cost of
morphing one cloud into the other, under a ground metric between atoms:
    c(atom_i, atom_j) = alpha * 1[Z_i != Z_j] + beta * ||q_i - q_j||_2
(the type-mismatch + geometric-profile cost, in the spirit of Eq. 10).

Two OT backends
---------------
  --method emd      : exact earth-mover's distance (POT network simplex).
                      No regularization parameter; exact; cheap for <~100 atoms.
                      RECOMMENDED FIRST -- it's the clean baseline.
  --method sinkhorn : entropy-regularized OT (the scalable path the paper uses
                      on Perlmutter). Faster/GPU-friendly at scale but introduces
                      the regularization `reg`, and is no longer a true metric.
                      Run this second to confirm it tracks the exact result.

Requirements
------------
    pip install pot ase scipy scikit-learn matplotlib fairchem-core

Usage
-----
    python option4_wasserstein.py --src /Volumes/LaCie/gpCAM/OMol25/train_4M \
        --n-molecules 800 --method emd

Cost note
---------
This computes a full S x S distance matrix = S(S-1)/2 OT solves. Cost grows
QUADRATICALLY in --n-molecules, so start small (800-1200) on a laptop. Each
solve is on an (N x M) cost matrix with N,M = atom counts (20-60 here), so it's
fast per pair but there are many pairs. Scaling to the paper's ~1e5 diagnostic
is the batched-GPU job (see note at bottom of file).
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import ot  # POT: Python Optimal Transport
from scipy.spatial.distance import cdist, pdist, squareform


def element_filter(mode):
    organic = {"H", "C", "N", "O", "S"}
    organic_ext = organic | {"P", "F", "Cl", "Br", "I"}
    return {
        "organic": lambda s: s <= organic,
        "organic_ext": lambda s: s <= organic_ext,
        "all": lambda s: True,
        "nonorganic": lambda s: not (
            s <= organic
        ),  # enriched: >=1 heteroatom beyond CHNOS
    }[mode]


# ----------------------------------------------------------------------
# Option-4 representation
# ----------------------------------------------------------------------
def distance_profile_representation(atoms, n_quantiles=16):
    """Return (Z, profiles) where Z is (N,) atomic numbers and profiles is
    (N, n_quantiles): each atom's sorted distance profile summarized at
    n_quantiles evenly spaced quantiles (an order-statistics summary that is
    fixed-length regardless of molecule size)."""
    R = atoms.get_positions()
    Z = atoms.get_atomic_numbers().astype(float)
    n = len(Z)
    probs = np.linspace(0.0, 1.0, n_quantiles)

    if n == 1:  # degenerate single atom: no neighbours
        return Z, np.zeros((1, n_quantiles))

    D = squareform(pdist(R))  # (n, n) interatomic distances
    profiles = np.empty((n, n_quantiles))
    for i in range(n):
        d_i = np.delete(D[i], i)  # distances from atom i to all others
        d_i.sort()
        profiles[i] = np.quantile(d_i, probs)
    return Z, profiles


# ----------------------------------------------------------------------
# Wasserstein distance between two molecules
# ----------------------------------------------------------------------
def molecule_wasserstein(repA, repB, alpha=1.0, beta=1.0, method="emd", reg=0.5):
    """Optimal-transport distance between two molecules' atom clouds."""
    ZA, qA = repA
    ZB, qB = repB
    nA, nB = len(ZA), len(ZB)

    geo = cdist(qA, qB, metric="euclidean")  # (nA, nB) profile distances
    typ = (ZA[:, None] != ZB[None, :]).astype(float)
    C = np.ascontiguousarray(alpha * typ + beta * geo)

    a = np.full(nA, 1.0 / nA)  # uniform mass per atom
    b = np.full(nB, 1.0 / nB)

    if method == "emd":
        return float(ot.emd2(a, b, C))  # exact
    # entropy-regularized (stabilized for numerical safety with large costs)
    return float(ot.sinkhorn2(a, b, C, reg, method="sinkhorn_stabilized"))


# ----------------------------------------------------------------------
# Pairwise distance matrix over a sample of molecules
# ----------------------------------------------------------------------
def build_distance_matrix(reps, alpha=1.0, beta=1.0, method="emd", reg=0.5):
    S = len(reps)
    D = np.zeros((S, S))
    total = S * (S - 1) // 2
    done = 0
    for i in range(S):
        for j in range(i + 1, S):
            d = molecule_wasserstein(reps[i], reps[j], alpha, beta, method, reg)
            D[i, j] = D[j, i] = d
            done += 1
        if (i + 1) % 25 == 0:
            print(f"  rows {i + 1}/{S} done ({done}/{total} pairs)", flush=True)
    return D


# ----------------------------------------------------------------------
# k-NN diagnostic from a precomputed distance matrix
# ----------------------------------------------------------------------
def knn_diagnostic_precomputed(D, k=5, fname="omol25_knn_wasserstein.png"):
    S = D.shape[0]
    knn = np.empty((S, k))
    for i in range(S):
        row = np.delete(D[i], i)  # drop self (distance 0)
        knn[i] = np.sort(row)[:k]
    d = knn.ravel()

    print("\n--- k-NN distance diagnostic (Wasserstein / Option 4) ---")
    print(f"k = {k}")
    print(f"distance range : [{d.min():.4f}, {d.max():.4f}]")
    print(f"mean / median  : {d.mean():.4f} / {np.median(d):.4f}")
    cv = d.std() / d.mean() if d.mean() else float("nan")
    print(
        f"coeff. of variation (std/mean): {cv:.3f}  "
        f"(<~0.3 hints at concentration / dense covariance risk)"
    )

    # candidate support radii and the matrix density each would imply
    offdiag = D[np.triu_indices(S, k=1)]
    print("\n  support-radius -> covariance-matrix density (fraction of pairs kept):")
    for q in (50, 75, 90):
        rho = np.percentile(d, q)
        density = float((offdiag <= rho).mean())
        print(
            f"    rho at {q}th pct of kNN dist = {rho:.4f}  ->  density ~ {density:.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./graphs/png", exist_ok=True)
    os.makedirs("./graphs/svg", exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(d, bins=80, color="#dd8452")
    plt.yscale("log")
    plt.title(
        "k-NN distance distribution (Wasserstein, Option 4)\n"
        "bimodal/long-tailed -> sparsity likely | concentrated -> dense risk"
    )
    plt.xlabel("distance to k nearest neighbours")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(f"./graphs/png/{timestamp}_{fname}.png", dpi=300)
    plt.savefig(f"./graphs/svg/{timestamp}_{fname}.svg", dpi=300)
    print(f"\nsaved -> ./graphs/png/{timestamp}_{fname}.png")
    print(f"\nsaved -> ./graphs/svg/{timestamp}_{fname}.svg")
    return d


# ----------------------------------------------------------------------
# Collect an organic subset, computing representations in a single disk pass
# ----------------------------------------------------------------------
def collect_organic_reps(
    dataset, n_molecules, pool, n_quantiles, size_range, seed, mode
):
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(dataset), size=min(pool, len(dataset)), replace=False)
    lo, hi = size_range
    keep = element_filter(mode)  # build once, not per-iteration
    reps = []
    for checked, idx in enumerate(pool_idx, 1):
        if checked % 1000 == 0:
            print(f"  scanned {checked}/{len(pool_idx)}, kept {len(reps)}", flush=True)
        atoms = dataset.get_atoms(int(idx))
        if lo <= len(atoms) <= hi and keep(set(atoms.get_chemical_symbols())):
            reps.append(distance_profile_representation(atoms, n_quantiles))
        if len(reps) >= n_molecules:
            break
    print(f"kept {len(reps)} molecules ({mode}) in [{lo},{hi}] atoms")
    return reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir containing the .aselmdb split")
    p.add_argument(
        "--n-molecules",
        type=int,
        default=800,
        help="molecules to keep (cost grows quadratically!)",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=60000,
        help="random indices to scan to find the organic subset",
    )
    p.add_argument(
        "--quantiles", type=int, default=16, help="profile summary length per atom"
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--method", choices=["emd", "sinkhorn"], default="emd")
    p.add_argument("--reg", type=float, default=0.5, help="Sinkhorn regularization")
    p.add_argument("--alpha", type=float, default=1.0, help="atom-type mismatch weight")
    p.add_argument("--beta", type=float, default=1.0, help="geometric profile weight")
    p.add_argument("--size", type=int, nargs=2, default=[20, 60], metavar=("LO", "HI"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--elements",
        choices=["organic", "organic_ext", "all", "nonorganic"],
        default="organic",
    )
    args = p.parse_args()

    from explore_omol25 import load_dataset  # reuse your loader

    dataset = load_dataset(args.src)

    reps = collect_organic_reps(
        dataset,
        args.n_molecules,
        args.pool,
        args.quantiles,
        tuple(args.size),
        args.seed,
        args.elements,
    )
    if len(reps) < 2:
        raise SystemExit("Not enough molecules kept; widen --size or --pool.")

    print(
        f"\ncomputing {len(reps)*(len(reps)-1)//2:,} pairwise "
        f"{args.method} distances..."
    )
    D = build_distance_matrix(reps, args.alpha, args.beta, args.method, args.reg)

    fname = f"omol25_knn_wasserstein_{args.method}_{args.elements}"
    knn_diagnostic_precomputed(D, k=args.k, fname=fname)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# SCALING NOTE
# ----------------------------------------------------------------------
# This CPU/exact path is for a few-thousand-molecule DIAGNOSTIC, not production.
# For the paper's ~1e5 diagnostic and beyond:
#   * Switch to --method sinkhorn; entropic OT is the parallelizable form.
#   * POT auto-dispatches to GPU if you pass torch tensors on a CUDA device
#     (or MPS on Mac) instead of numpy arrays -- batch many pairs at once with
#     ot.bregman.empirical_sinkhorn or a batched cost-matrix formulation.
#   * The full pairwise matrix is O(S^2) solves; at scale you compute only the
#     blocks you need (the kernel evaluates entries, not the dense matrix).
