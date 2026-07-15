"""
cutoff.py  (wl_gp2scale)
========================
Recompute the compact-support radius (CUTOFF) from the 200k embedding.

Distances shrink as N grows and the embedding is refit on the 200k split, so the
cutoff MUST be recomputed here -- a radius tuned on 10k would be far too large at
200k (near-dense kernel) or too small (mean reversion). We take a percentile of
sampled pairwise distances, then report realised in-support neighbours and the
implied global sparsity so you can confirm memory fits and CG will be
well-conditioned.

Kill rule (matches the constraints): if CG later struggles, TIGHTEN the cutoff
(more diagonal dominance) rather than inflating jitter.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def suggest_percentile(n, target_neighbors=50, frac_same_category=1.0):
    """Percentile to ASK FOR so each point keeps ~target_neighbors in-support peers.

    This is the knob that sets memory, and it does NOT transfer across N. The
    in-support fraction of pairs is ~percentile/100, so nnz ~ (pct/100) * N^2:

        N=10k,  pct=25   -> 2.5e7  nnz   (fine; the validation default)
        N=200k, pct=25   -> 1.0e10 nnz   (~120 GB -- infeasible, near-dense CG)
        N=200k, pct=0.025-> 1.0e7  nnz   (~50 neighbours/point)

    Category block-sparsity divides the realised density by ~1/n_categories, which
    ``frac_same_category`` accounts for. ALWAYS re-derive this at the target N and
    confirm with ``sparsity_report``; a cutoff carried over from 10k will either
    blow up memory or (if too tight) mean-revert.
    """
    pct = 100.0 * float(target_neighbors) / (float(n) * float(frac_same_category))
    print(
        f"[cutoff] for N={n:,}, ~{target_neighbors} neighbours/point and "
        f"P(same category)={frac_same_category:.3g} -> ask for percentile ~{pct:.4g}"
    )
    return pct


def recalibrate(
    Z, percentile: float = 25.0, sample: int = 5000, seed: int = 0, dim=None
):
    """Return CUTOFF = the given percentile of sampled pairwise L2 distances on the
    embedding Z (uses only the first ``dim`` columns if given, i.e. drops the
    category tag). Also returns the sampled distance percentiles for context."""
    Z = np.asarray(Z, dtype=float)
    if dim is not None:
        Z = Z[:, :dim]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    d = pdist(Z[idx])
    pct = np.percentile(d, [5, 25, 50, 75, 95])
    cutoff = float(np.percentile(d, percentile))
    print(
        f"[cutoff] sampled pairwise pctiles [5,25,50,75,95]={np.round(pct,3)}; "
        f"CUTOFF@{percentile:g}pct = {cutoff:.4f}"
    )
    return cutoff, pct


def sparsity_report(Z, cutoff, sample=5000, seed=0, dim=None, data_id=None):
    """Estimate realised sparsity and total non-zeros from a sample, so you can
    confirm the distributed sparse matrix fits worker memory before the full run.

    If ``data_id`` is given, cross-category pairs are treated as zero (category
    block-sparsity), matching the kernel.
    """
    Z = np.asarray(Z, dtype=float)
    coords = Z[:, :dim] if dim is not None else Z
    n = len(coords)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    D = cdist(coords[idx], coords)
    in_supp = D < cutoff
    if data_id is not None:
        cats = np.asarray(data_id)
        same = cats[idx][:, None] == cats[None, :]
        in_supp = in_supp & same
    nbr = in_supp.sum(axis=1)                          # per-row in-support count
    density = float(in_supp.mean())                    # fraction of entries kept
    est_nnz = density * n * n
    # scipy CSR: 8 bytes value + 4 bytes col-index per nnz, + row pointers.
    # NOTE this is DRIVER-side memory: fvgp's gp2Scale distributes the kernel
    # EVALUATION across workers but gathers the COO components and assembles a
    # single scipy CSR on the client (gp_prior.py:294-306), where sparseCG then
    # solves it. So the budget is the driver process's RAM, not n_workers * 30 GiB.
    est_gb = est_nnz * 12 / 1e9
    print(
        f"[cutoff] est. density={density:.3e}  est. nnz={est_nnz:,.0f}  "
        f"(~{est_gb:.1f} GB driver-side CSR for the full {n:,}x{n:,} matrix)"
    )
    print(
        f"[cutoff] in-support neighbours/row: median={np.median(nbr):.0f} "
        f"min={int(nbr.min())} max={int(nbr.max())} frac_zero={np.mean(nbr==0):.1%}"
    )
    if np.mean(nbr == 0) > 0.05:
        print("[cutoff] WARNING: many zero-neighbour rows -> mean reversion; RAISE cutoff.")
    elif density > 0.05:
        print("[cutoff] WARNING: dense-ish kernel -> heavy memory / ill-conditioned; LOWER cutoff.")
    return {
        "density": density,
        "est_nnz": est_nnz,
        "est_gb": est_gb,
        "median_neighbors": float(np.median(nbr)),
        "frac_zero_neighbor": float(np.mean(nbr == 0)),
    }
