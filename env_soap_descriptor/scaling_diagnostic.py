"""
scaling_diagnostic.py
=====================
Part 1 -- THE GO/NO-GO TEST. Everything else is only worth building if this passes.

The question, stated precisely
------------------------------
Environment-level storage is linear (=> 100M reachable) iff nonzeros-per-row of K_env
stays BOUNDED as N grows. It is quadratic (=> 100M dead, fall back to mid-scale) iff
nnz/row grows with N.

The subtlety that makes the naive test lie
-------------------------------------------
At a FIXED support radius rho, nnz/row ALWAYS grows with N: adding molecules raises the
local density of environments, and "environment repetition" makes dense clusters denser,
so a fixed-radius ball catches ever more neighbors. Measuring at fixed rho would give a
falsely pessimistic "always grows" verdict. That is not the real test.

nnz/row is bounded ONLY if the SKILL-PRESERVING radius rho*(N) shrinks fast enough to
offset density growth. Heuristically nnz/row ~ density(N) * rho*(N)^D; linear storage
needs rho*(N)^D ~ 1/density(N). So the honest diagnostic measures nnz/row AT rho*(N),
re-selecting rho* at each N. That is exactly what this script does.

The duplicate-conformer confound (your Stage-1 bottleneck)
----------------------------------------------------------
If N grows mostly by near-identical conformers, then at any *useful* rho* each
environment sees all its duplicate copies, so nnz/row grows ~ (copies/env) ~ N -- but
that growth is an artifact of leakage, not of genuine local diversity. So we run the
whole test twice: on the RAW population and on a DEDUPED population, and report both.
If dedup flattens the curve, your "scale path" is really a "dedup-then-scale" path -- a
different (and weaker) scientific claim that must be stated as such.

Skill without per-environment labels
-------------------------------------
OMol25 labels are molecular totals; f(env) is latent. We define a skill-preserving
radius via a cheap surrogate that respects the aggregate structure:
  1. Partition energy to environments by ridge least squares:  e = argmin ||A e - y||^2.
  2. Smooth e over the radius-rho neighbor graph (Nadaraya-Watson): e_hat_rho.
  3. Re-aggregate: y_hat_M = sum_{i in M} e_hat_rho(env_i).
  4. rho* = smallest rho whose HELD-OUT molecule R^2 is within `skill_tol` of the best.
As rho->0, e_hat->e (overfits training); as rho->inf, e_hat->const (underfits). The
plateau's left edge is the skill-preserving radius. This is a proxy for the aggregate
GP's own skill; it is cheap enough to sweep and uses only observables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


# ------------------------------ dedup ---------------------------------------- #
def dedup_environments(Z, mol_of, tol=1e-2, seed=0):
    """
    Collapse near-identical environments (grid-hash in the whitened embedding).
    Returns a deduped embedding, a representative molecule membership, and the
    multiplicity of each kept environment (how many raw envs it stood in for).
    """
    rng = np.random.default_rng(seed)
    keys = np.round(Z / tol).astype(np.int64)
    # hash rows -> unique
    order = np.lexsort(keys.T[::-1])
    ks = keys[order]
    first = np.ones(len(ks), bool)
    first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    rep_local = order[first]
    # multiplicity per unique key
    grp = np.cumsum(first) - 1
    mult = np.bincount(grp)
    return Z[rep_local], mol_of[rep_local], mult, len(rep_local)


# ---------------------- neighbor / nnz-per-row stats ------------------------- #
def nnz_per_row_stats(
    Z, radius, n_probe=3000, seed=0, explosion_cap=200_000, workers=-1
):
    """
    Explosion-safe estimate of the nnz/row distribution of the radius-`radius` neighbor
    graph, WITHOUT materializing the full graph. Build the tree once and count neighbors
    for a random probe set of rows in ONE batched, parallel query (return_length=True
    returns only counts, so memory is O(n_probe) regardless of density). If the median
    probe count exceeds `explosion_cap`, that alone is a no-go (too dense to store).
    """
    rng = np.random.default_rng(seed)
    tree = cKDTree(Z)
    probes = rng.choice(len(Z), size=min(n_probe, len(Z)), replace=False)
    counts = tree.query_ball_point(
        Z[probes], r=radius, return_length=True, workers=workers
    )
    counts = np.asarray(counts, dtype=np.int64)
    return {
        "radius": float(radius),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "max": int(counts.max()),
        "exploded": bool(np.median(counts) > explosion_cap),
    }


# ----------------------- skill surrogate at a radius ------------------------- #
def _partition_energy(A, y, ridge=1e-3):
    """e = argmin ||A e - y||^2 + ridge||e||^2  via normal equations (A^T A + rI) e = A^T y."""
    n = A.shape[1]
    AtA = (A.T @ A).tocsc() + ridge * sp.eye(n, format="csc")
    Aty = A.T @ y
    from scipy.sparse.linalg import cg

    e, _ = cg(AtA, Aty, rtol=1e-6, maxiter=500)
    return e


def _skill_from_knn(
    Z,
    A,
    y,
    dist,
    idx,
    e_part,
    env_is_train,
    train_mean,
    scored_mol,
    scored_env_idx,
    radius,
    k_smooth_cap=64,
):
    """
    Skill at radius rho from a PRECOMPUTED k-NN neighborhood (dist, idx of shape (S,k)).
    Thresholding cached neighbors by radius is bounded in memory (O(S*k)) regardless of
    density -- the radius-ball query was what OOM-ed. Fully vectorized, no per-env loop.
    """
    valid = (dist < radius) & env_is_train[idx]  # (S,k) train neighbors in support
    if k_smooth_cap < idx.shape[1]:  # keep the nearest k_smooth_cap
        keep = np.zeros_like(valid)
        keep[:, :k_smooth_cap] = True
        valid &= keep
    e_vals = e_part[idx]  # (S,k)
    counts = valid.sum(axis=1)
    sums = (e_vals * valid).sum(axis=1)
    e_scored = np.where(counts > 0, sums / np.maximum(counts, 1), train_mean)
    e_full = np.zeros(len(Z))
    e_full[scored_env_idx] = e_scored
    y_hat = A[scored_mol] @ e_full
    yt = y[scored_mol]
    if len(yt) < 5 or np.var(yt) == 0:
        return -np.inf
    return 1.0 - np.sum((yt - y_hat) ** 2) / np.sum((yt - yt.mean()) ** 2)


def skill_preserving_radius(
    Z,
    A,
    y,
    radius_grid,
    skill_tol=0.02,
    n_scored_mol=2000,
    k_query=200,
    workers=-1,
    seed=0,
    verbose=True,
):
    """
    Sweep radii, return the SMALLEST radius within `skill_tol` R^2 of the best.
    Smallest-within-tolerance is deliberate: it is the sparsest kernel that keeps skill,
    i.e. the honest rho* the sparsity-preferring MCMC prior would target.

    Scales to 10^6: molecule-level 80/20 split; score only a subsample of test molecules;
    build the KD-tree once; do ONE bounded k-NN query (k=k_query) and threshold it per
    radius. Memory is O(n_scored_mol * atoms/mol * k_query), independent of density.
    """
    rng = np.random.default_rng(seed)
    n_mol, n_env = A.shape
    train_mol_mask = rng.random(n_mol) < 0.8
    env_is_train = (A.T @ train_mol_mask.astype(float)) > 0.5  # env in a train molecule
    e_part = _partition_energy(A, y)
    train_mean = e_part[env_is_train].mean()

    test_mol = np.where(~train_mol_mask)[0]
    scored_mol = (
        test_mol
        if len(test_mol) <= n_scored_mol
        else rng.choice(test_mol, n_scored_mol, replace=False)
    )
    scored_env_idx = np.unique(A[scored_mol].indices)

    if verbose:
        print(
            f"    [skill] n_env={n_env} scored_mol={len(scored_mol)} "
            f"scored_env={len(scored_env_idx)} — tree + one k-NN query (k={k_query})...",
            flush=True,
        )
    tree = cKDTree(Z)
    k = min(k_query, n_env)
    dist, idx = tree.query(Z[scored_env_idx], k=k, workers=workers)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)

    skills = np.empty(len(radius_grid))
    for j, r in enumerate(radius_grid):
        skills[j] = _skill_from_knn(
            Z,
            A,
            y,
            dist,
            idx,
            e_part,
            env_is_train,
            train_mean,
            scored_mol,
            scored_env_idx,
            r,
        )
        if verbose:
            print(f"    [skill] radius={r:.2f}  R2={skills[j]:+.4f}", flush=True)

    best = np.nanmax(skills)
    ok = np.where(skills >= best - skill_tol)[0]
    star = int(ok.min()) if len(ok) else int(np.nanargmax(skills))
    return radius_grid[star], skills, star


# ------------------------------- driver -------------------------------------- #
@dataclass
class ScalingPoint:
    n_mol: int
    n_env: int
    n_env_dedup: int
    rho_star: float
    nnz_row_mean_raw: float
    nnz_row_p95_raw: float
    nnz_row_mean_dedup: float
    nnz_row_p95_dedup: float
    skill_at_star: float
    exploded: bool


def run_scaling_test(
    make_population,
    n_mol_grid=(10_000, 30_000, 100_000),
    D=20,
    radius_grid=None,
    dedup_tol=1e-2,
    n_scored_mol=2000,
    workers=-1,
    fit_embedding=True,
    seed=0,
):
    """
    make_population(n_mol, seed) -> (X_raw_or_Z, mol_of, y_mol).
    If fit_embedding=True, X is raw features and one shared PCA embedding (D) is fit on the
    largest population. If fit_embedding=False (the OMol25Loader path), X is ALREADY the
    D-dim embedding (fit once inside the loader) and is used as-is.
    Returns the per-N table plus a verdict from a log-log fit of nnz/row(rho*, N).
    """
    from env_features_kernel import fit_env_embedding

    if radius_grid is None:
        radius_grid = np.linspace(0.5, 4.0, 8)

    emb = None
    if fit_embedding:
        print(
            f"[setup] fitting shared embedding (D={D}) on largest population "
            f"(n_mol={max(n_mol_grid)})...",
            flush=True,
        )
        X_big, _, _ = make_population(max(n_mol_grid), seed=seed)
        emb = fit_env_embedding(X_big, D=D)

    rows = []
    for n_mol in n_mol_grid:
        print(f"\n[N={n_mol}] loading + embedding...", flush=True)
        X, mol_of, y = make_population(n_mol, seed=seed)
        Z = emb.transform(X) if emb is not None else X  # loader already embedded
        A = _membership_matrix(mol_of, n_mol)

        rho, skills, star = skill_preserving_radius(
            Z, A, y, radius_grid, n_scored_mol=n_scored_mol, workers=workers, seed=seed
        )
        print(
            f"[N={n_mol}] rho*={rho:.2f}  measuring nnz/row (raw + dedup)...",
            flush=True,
        )
        s_raw = nnz_per_row_stats(Z, rho, seed=seed)

        Zd, mold, mult, n_ded = dedup_environments(Z, mol_of, tol=dedup_tol, seed=seed)
        s_ded = nnz_per_row_stats(Zd, rho, seed=seed)

        rows.append(
            ScalingPoint(
                n_mol=n_mol,
                n_env=len(Z),
                n_env_dedup=n_ded,
                rho_star=float(rho),
                nnz_row_mean_raw=s_raw["mean"],
                nnz_row_p95_raw=s_raw["p95"],
                nnz_row_mean_dedup=s_ded["mean"],
                nnz_row_p95_dedup=s_ded["p95"],
                skill_at_star=float(skills[star]),
                exploded=s_raw["exploded"] or s_ded["exploded"],
            )
        )
    return rows, verdict(rows)


def _membership_matrix(mol_of, n_mol):
    """Sparse sum-pooling A (M x n_env): A[m,i]=1 iff env i belongs to molecule m."""
    n_env = len(mol_of)
    data = np.ones(n_env)
    return sp.csr_matrix((data, (mol_of, np.arange(n_env))), shape=(n_mol, n_env))


def verdict(rows):
    """
    Fit log(nnz/row) ~ alpha * log(n_env) on the DEDUPED curve. alpha ~ 0 => bounded
    (linear storage; scale alive). alpha ~ 1 => grows linearly (quadratic storage;
    scale dead). Report both raw and dedup so leakage is visible.
    """
    ne = np.log(np.array([r.n_env_dedup for r in rows]))
    nz = np.log(np.array([max(r.nnz_row_mean_dedup, 1e-9) for r in rows]))
    if len(rows) >= 2 and np.ptp(ne) > 0:
        alpha_dedup = np.polyfit(ne, nz, 1)[0]
    else:
        alpha_dedup = np.nan
    ner = np.log(np.array([r.n_env for r in rows]))
    nzr = np.log(np.array([max(r.nnz_row_mean_raw, 1e-9) for r in rows]))
    alpha_raw = (
        np.polyfit(ner, nzr, 1)[0] if (len(rows) >= 2 and np.ptp(ner) > 0) else np.nan
    )

    if any(r.exploded for r in rows):
        call = "NO-GO (graph exploded at rho*: too dense to store)"
    elif np.isnan(alpha_dedup):
        call = "INCONCLUSIVE (need >=2 N points)"
    elif alpha_dedup < 0.15:
        call = "GO (nnz/row bounded on deduped envs -> linear storage plausible)"
    elif alpha_dedup < 0.6:
        call = (
            "MARGINAL (sub-linear growth; extrapolate carefully before trusting 100M)"
        )
    else:
        call = "NO-GO (nnz/row grows ~linearly -> quadratic storage -> 100M dead)"
    return {
        "alpha_dedup": float(alpha_dedup),
        "alpha_raw": float(alpha_raw),
        "call": call,
    }


if __name__ == "__main__":
    from env_features_kernel import synthetic_environments

    # Clean population (bounded env types) vs leaky population (duplicate conformers)
    def clean(n, seed=0):
        return synthetic_environments(n, dup_fraction=0.0, seed=seed)

    def leaky(n, seed=0):
        return synthetic_environments(n, dup_fraction=0.4, seed=seed)

    for name, pop in [("clean", clean), ("leaky", leaky)]:
        print(
            f"\n################  {name.upper()} population  ################",
            flush=True,
        )
        rows, v = run_scaling_test(pop, n_mol_grid=(1500, 4500, 13500), D=20)
        print(f"\n=== {name} population summary ===")
        for r in rows:
            print(
                f" n_mol={r.n_mol:6d} n_env={r.n_env:7d} dedup={r.n_env_dedup:7d} "
                f"rho*={r.rho_star:.2f} skill={r.skill_at_star:+.3f} "
                f"nnz/row raw={r.nnz_row_mean_raw:8.1f} dedup={r.nnz_row_mean_dedup:7.1f}"
            )
        print(
            f" VERDICT: alpha_dedup={v['alpha_dedup']:+.3f} alpha_raw={v['alpha_raw']:+.3f} -> {v['call']}"
        )
