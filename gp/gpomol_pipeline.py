#!/usr/bin/env python3
r"""
gpomol_pipeline.py
==================
ONE script that does, on a SINGLE shared molecule subsample:

  1. Option-4 representation + Wasserstein distance matrix W   (was option4_wasserstein.py)
  2. Per-element energy referencing -> residual target y       (was reference_energy.py)
  3. Exp-kernel-over-W validity + signal diagnostic            (was diagnose_exp_wasserstein.py)

Why combine them
----------------
The diagnostic only means anything if row i of W and entry i of y are the SAME
molecule. When W and y are produced by two separate scripts, that alignment has
to be reconstructed by hand from dataset indices -- and a silent mismatch makes
the diagnostic report "no signal" for the wrong reason. Here W and y come out of
ONE collection pass in ONE order, so alignment is structural: there is no index
map to get wrong. That is the entire point of merging the three scripts.

Caveat on provenance
---------------------
The Option-4 / Wasserstein block is reconstructed from the position paper
(Eqs. 13-14) and this project's conventions (alpha = atom-type weight, beta =
geometry weight, profiles resampled to a common quantile grid, exact EMD for the
outer transport). Sanity-check that it reproduces your standalone
option4_wasserstein.py numbers (e.g. the ~94% sparsity at the 75th-percentile
support radius) before trusting it as a drop-in.

Referencing leakage note
------------------------
For the actual GP MODEL, reference_energy.py fits the per-element constants on
the TRAIN split only (test leakage would inflate reported accuracy). This script
runs a SIGNAL PROBE with internal leave-one-out, so it fits the ~6-parameter
composition baseline on all N molecules being diagnosed; the per-molecule leakage
through a 6-DoF fit over N>>6 points is negligible for a yes/no signal verdict.
When you build the real model, keep the train-only discipline from the original.

Requirements
------------
    pip install ase numpy matplotlib fairchem-core pot
    (POT is used for exact EMD; a numpy Sinkhorn fallback runs if POT is absent.)

Usage
-----
    python gpomol_pipeline.py --src ../train_4M --n-molecules 400 \
        --elements organic --size 20 60 --add-charge \
        --alpha 1.0 --beta 1.0 --wasserstein-p 2 --outdir results/

    python gpomol_pipeline.py --demo                  # end-to-end self-test (signal)
    python gpomol_pipeline.py --demo --shuffle-energy # null: same geometry, no signal

Outputs (in --outdir, timestamped)
----------------------------------
    <ts>_pipeline.npz   : W, y, indices, eps, labels, elements, params
    <ts>_eigen.png / _distance_energy.png / _knn.png
    a formatted readout to stdout (paste-ready for Marcus)
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ot  # POT: exact EMD
    _HAVE_POT = True
except Exception:
    _HAVE_POT = False


# =========================================================================== #
#  PART 0 -- data collection (single pass; the alignment guarantee lives here)
# =========================================================================== #
def load_dataset(src):
    from fairchem.core.datasets import AseDBDataset
    ds = AseDBDataset({"src": src})
    print(f"Loaded dataset from {src!r}: {len(ds):,} structures.")
    return ds


def element_filter(mode):
    organic = {"H", "C", "N", "O", "S"}
    organic_ext = organic | {"P", "F", "Cl", "Br", "I"}
    return {
        "organic": lambda s: s <= organic,
        "organic_ext": lambda s: s <= organic_ext,
        "all": lambda s: True,
        "nonorganic": lambda s: not (s <= organic),
    }[mode]


def collect_subset(dataset, n_molecules, pool, size_range, mode, seed,
                   charge_filter=None, spin_filter=None):
    """One disk pass. Every per-molecule quantity is appended in the SAME order,
    so geometry (for W / WL) and energy (for y) can never desync downstream."""
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(dataset), size=min(pool, len(dataset)), replace=False)
    keep = element_filter(mode)
    lo, hi = size_range
    indices, symbols, positions, comps, energies, charges, spins = (
        [], [], [], [], [], [], [])
    for checked, idx in enumerate(pool_idx, 1):
        if checked % 1000 == 0:
            print(f"  scanned {checked}/{len(pool_idx)}, kept {len(indices)}", flush=True)
        atoms = dataset.get_atoms(int(idx))
        syms = atoms.get_chemical_symbols()
        if not (lo <= len(atoms) <= hi and keep(set(syms))):
            continue
        q = int(round(float(atoms.info.get("charge", 0))))
        sp = atoms.info.get("spin", None)
        if charge_filter is not None and q != charge_filter:
            continue
        if spin_filter is not None and sp is not None and int(sp) != spin_filter:
            continue
        try:
            e = atoms.get_potential_energy()
        except Exception:
            continue
        indices.append(int(idx))
        symbols.append(syms)
        positions.append(atoms.get_positions().astype(float))
        comps.append(Counter(syms))
        energies.append(float(e))
        charges.append(float(q))
        spins.append(int(sp) if sp is not None else 0)
        if len(indices) >= n_molecules:
            break
    print(f"kept {len(indices)} molecules ({mode}) in [{lo},{hi}] atoms"
          + (f", charge={charge_filter}" if charge_filter is not None else "")
          + (f", spin={spin_filter}" if spin_filter is not None else ""))
    return {
        "indices": np.array(indices),
        "symbols": symbols,
        "positions": positions,
        "comps": comps,
        "energies": np.array(energies, dtype=float),
        "charges": np.array(charges, dtype=float),
        "spins": np.array(spins, dtype=int),
    }


# =========================================================================== #
#  PART 1 -- Option-4 representation + Wasserstein distance matrix
# =========================================================================== #
def option4_profiles(positions, quantile_points):
    """Eq. 13-14. For each atom, the SORTED vector of distances to all other
    atoms is its profile (rotation/translation invariant). A sorted sample is an
    empirical quantile function, so we resample every profile onto a common grid
    of `quantile_points` -- this both handles unequal molecule sizes and makes
    the inner atom-to-atom comparison the closed-form 1-D OT (L2 of quantiles).

    Returns profiles array (n_atoms, Q). Atomic numbers are returned separately
    so the ground metric can include an atom-type term.
    """
    n = positions.shape[0]
    grid = np.linspace(0.0, 1.0, quantile_points)
    if n < 2:
        return np.zeros((max(n, 1), quantile_points))
    diff = positions[:, None, :] - positions[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    prof = np.empty((n, quantile_points))
    q_src = np.linspace(0.0, 1.0, n - 1)
    for i in range(n):
        d_sorted = np.sort(np.delete(D[i], i))
        prof[i] = np.interp(grid, q_src, d_sorted)
    return prof


def build_representation(positions_list, symbols_list, quantile_points):
    from ase.data import atomic_numbers
    reps = []
    for pos, syms in zip(positions_list, symbols_list):
        Z = np.array([atomic_numbers[s] for s in syms])
        reps.append((Z, option4_profiles(pos, quantile_points)))
    return reps


def _sinkhorn_cost(a, b, M, reg=0.05, iters=300):
    """numpy entropic-OT fallback (returns <gamma, M>; not a true metric)."""
    K = np.exp(-M / (reg * (M.max() + 1e-12)))
    u = np.ones_like(a)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    G = u[:, None] * K * v[None, :]
    return float((G * M).sum())


def molecule_distance(repA, repB, alpha, beta, p, use_pot):
    """Outer Wasserstein-p between two molecular atom-distributions (uniform
    marginals). Ground metric c = alpha*[Z_i != Z_j] + beta*||profile_i - profile_j||_2."""
    ZA, PA = repA
    ZB, PB = repB
    type_term = (ZA[:, None] != ZB[None, :]).astype(float)
    geom = np.sqrt(((PA[:, None, :] - PB[None, :, :]) ** 2).sum(-1))
    C = alpha * type_term + beta * geom            # ground metric c
    a = np.full(len(ZA), 1.0 / len(ZA))
    b = np.full(len(ZB), 1.0 / len(ZB))
    Cp = C ** p
    if use_pot and _HAVE_POT:
        wpp = ot.emd2(a, b, np.ascontiguousarray(Cp))   # exact: sum gamma * c^p
    else:
        wpp = _sinkhorn_cost(a, b, Cp)
    return float(wpp) ** (1.0 / p)


def _pair_block(reps, block, alpha, beta, p, use_pot):
    """Compute a CHUNK of pairs in one worker call. Chunking keeps the joblib
    task count in the dozens instead of in the millions, which is the difference
    between a real speedup and per-task overhead swamping the computation."""
    out = []
    for (i, j) in block:
        out.append((i, j, molecule_distance(reps[i], reps[j], alpha, beta, p, use_pot)))
    return out


def wasserstein_matrix(reps, alpha, beta, p, use_pot=True, n_jobs=1):
    n = len(reps)
    W = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    mode = "exact EMD" if (use_pot and _HAVE_POT) else "Sinkhorn fallback"
    print(f"computing W over {n} molecules ({len(pairs):,} pairs) [{mode}] ...", flush=True)

    if n_jobs == 1:
        results = []
        for k, (i, j) in enumerate(pairs, 1):
            if k % 20000 == 0:
                print(f"  {k:,}/{len(pairs):,} pairs", flush=True)
            results.append((i, j, molecule_distance(reps[i], reps[j], alpha, beta, p, use_pot)))
    else:
        from joblib import Parallel, delayed
        n_blocks = 4 * (os.cpu_count() or 8)
        block_size = max(1, -(-len(pairs) // n_blocks))  # ceil division
        blocks = [pairs[s:s + block_size] for s in range(0, len(pairs), block_size)]
        print(f"  parallel: {len(blocks)} blocks of <= {block_size:,} pairs "
              f"(n_jobs={n_jobs}); joblib prints progress below", flush=True)
        block_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_pair_block)(reps, b, alpha, beta, p, use_pot) for b in blocks)
        results = [r for blk in block_results for r in blk]

    for i, j, d in results:
        W[i, j] = W[j, i] = d
    return W


# =========================================================================== #
#  PART 2 -- per-element referencing (fit on the diagnosed set; see header note)
# =========================================================================== #
def build_composition_matrix(comps, charges, elements, add_charge, add_intercept):
    idx = {e: k for k, e in enumerate(elements)}
    C = np.zeros((len(comps), len(elements)))
    for m, comp in enumerate(comps):
        for e, n in comp.items():
            if e not in idx:
                raise KeyError(f"element {e!r} not in reference set {elements}")
            C[m, idx[e]] += n
    labels, cols = list(elements), [C]
    if add_charge:
        cols.append(charges.reshape(-1, 1)); labels.append("__charge__")
    if add_intercept:
        cols.append(np.ones((len(comps), 1))); labels.append("__intercept__")
    return np.hstack(cols), labels


def reference_energies(comps, charges, energies, add_charge, add_intercept):
    elements = sorted({e for c in comps for e in c})
    C, labels = build_composition_matrix(comps, charges, elements, add_charge, add_intercept)
    eps, *_ = np.linalg.lstsq(C, energies, rcond=None)
    residual = energies - C @ eps
    return residual, eps, labels, elements


# =========================================================================== #
#  PART 3 -- exp-kernel diagnostic (unchanged logic from diagnose_exp_wasserstein.py)
# =========================================================================== #
def lengthscale_grid(W, factors=(0.25, 0.5, 1.0, 2.0, 4.0)):
    iu = np.triu_indices_from(W, k=1)
    med = float(np.median(W[iu]))
    return med, [f * med for f in factors]


def pd_diagnostic(W, lengthscales):
    rows = []
    for ell in lengthscales:
        K = np.exp(-W / ell)
        ev = np.linalg.eigvalsh(K)
        lmin, lmax = float(ev[0]), float(ev[-1])
        neg = ev[ev < 0]
        rows.append({"ell": ell, "lambda_min": lmin, "lambda_max": lmax,
                     "ratio": lmin / lmax if lmax > 0 else np.nan,
                     "n_negative": int(neg.size), "neg_mass": float(-neg.sum())})
    return rows


def classify_pd(rows, tol=1e-8):
    worst = min(r["ratio"] for r in rows)
    if worst >= -tol:
        return "PD", worst
    if worst >= -1e-2:
        return "NEAR-PD", worst
    return "INDEFINITE", worst


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def distance_energy_readout(W, y, max_pairs=2_000_000, n_bins=12, seed=0):
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(W, k=1)
    wv, dE = W[iu], np.abs(y[iu[0]] - y[iu[1]])
    if wv.size > max_pairs:
        sel = rng.choice(wv.size, size=max_pairs, replace=False)
        wv_s, dE_s = wv[sel], dE[sel]
    else:
        wv_s, dE_s = wv, dE
    rho = spearman(wv_s, dE_s)
    edges = np.quantile(wv, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    idx = np.clip(np.digitize(wv, edges) - 1, 0, n_bins - 1)
    bin_mid = 0.5 * (edges[:-1] + edges[1:])
    bin_mean = np.array([dE[idx == b].mean() if np.any(idx == b) else np.nan
                         for b in range(n_bins)])
    valid = ~np.isnan(bin_mean)
    lift = (bin_mean[valid][-1] / bin_mean[valid][0]) if bin_mean[valid][0] > 0 else np.nan
    return {"spearman": rho, "bin_mid": bin_mid, "bin_mean": bin_mean, "lift": lift}


def knn_readout(W, y, ks=(1, 2, 3, 5, 10, 20)):
    n = W.shape[0]
    ks = tuple(k for k in ks if k < n) or (1,)
    order = np.argsort(W, axis=1)
    neighbors = order[:, 1:max(ks) + 1]
    loo_mean = (y.sum() - y) / (n - 1)
    mae_base = float(np.mean(np.abs(y - loo_mean)))
    per_k = []
    for k in ks:
        pred = y[neighbors[:, :k]].mean(axis=1)
        mae = float(np.mean(np.abs(y - pred)))
        per_k.append({"k": k, "mae": mae, "skill": 1.0 - mae / mae_base})
    return {"baseline_mae": mae_base, "per_k": per_k,
            "best": max(per_k, key=lambda r: r["skill"])}


def gp_loo_readout(W, y, ell, noise_frac=0.05, psd_tol=1e-8):
    yc = y - y.mean()
    sigf2 = float(np.var(yc))
    if sigf2 == 0:
        return {"ok": False, "flag": "zero variance"}
    K = sigf2 * np.exp(-W / ell)
    lmin = float(np.linalg.eigvalsh(K)[0])
    jitter, flag = 0.0, "clean (kernel PD at this lengthscale)"
    if lmin < -psd_tol:
        jitter = -lmin + 1e-6 * sigf2
        frac = jitter / sigf2
        flag = (f"jitter={jitter:.3g} to reach PSD (= {frac:.1%} of signal var; "
                f"{'tolerable' if frac < 0.2 else 'LARGE: patch likely drowns signal'})")
    Ky = K + (jitter + noise_frac ** 2 * sigf2) * np.eye(W.shape[0])
    try:
        Kinv = np.linalg.inv(Ky)
    except np.linalg.LinAlgError:
        return {"ok": False, "flag": "inversion failed"}
    alpha = Kinv @ yc
    dinv = np.diag(Kinv)
    resid = alpha / dinv
    sd = np.sqrt(np.clip(1.0 / dinv, 0, None))
    return {"ok": True, "ell": ell, "mae": float(np.mean(np.abs(resid))),
            "coverage": float(np.mean(np.abs(resid) <= 2 * sd)), "flag": flag}


# =========================================================================== #
#  Plots + readout
# =========================================================================== #
def make_plots(rows, de, knn, median_w, outdir, ts):
    p1 = f"{outdir}/{ts}_eigen.png"
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot([r["ell"] / median_w for r in rows], [r["ratio"] for r in rows], "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="lengthscale / median W", ylabel=r"$\lambda_{\min}/\lambda_{\max}$",
           title="PD of exp(-W/ell)  (negative => invalid kernel)", xscale="log")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)

    p2 = f"{outdir}/{ts}_distance_energy.png"
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(de["bin_mid"], de["bin_mean"], "o-")
    ax.set(xlabel="Wasserstein distance (bin midpoint)", ylabel="mean |energy gap| (eV)",
           title=f"Distance vs energy gap  (rho={de['spearman']:.3f}; rising => signal)")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)

    p3 = f"{outdir}/{ts}_knn.png"
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot([r["k"] for r in knn["per_k"]], [r["skill"] for r in knn["per_k"]], "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="k (neighbours in W)", ylabel="skill = 1 - MAE/MAE_base",
           title="k-NN-in-metric signal  (>0 => beats the mean)")
    fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig)
    return [p1, p2, p3]


def print_readout(n, median_w, eps, labels, rows, pd_v, worst,
                  de, knn, gp, expectation=None):
    out = []
    a = out.append
    a("=" * 72)
    a("  GPOMOL UNIFIED PIPELINE READOUT")
    a("=" * 72)
    a(f"  molecules diagnosed   : N = {n}  ({n*(n-1)//2:,} pairs)")
    a(f"  median pairwise W     : {median_w:.4g}")
    if expectation:
        a(f"  [self-test] expected  : {expectation}")
    a("")
    a("  per-element reference (eV), fit on these N molecules:")
    for lab, v in zip(labels, eps):
        a(f"      {lab:>14} : {v:12.3f}")
    a("")
    a("-" * 72)
    a("  (A) IS exp(-W/ell) PD ON THIS DATA?")
    a("-" * 72)
    a(f"  {'ell/med':>8} {'lambda_min':>12} {'lambda_max':>12} {'min/max':>11} "
      f"{'n_neg':>6} {'neg_mass':>10}")
    for r in rows:
        a(f"  {r['ell']/median_w:>8.2f} {r['lambda_min']:>12.4g} {r['lambda_max']:>12.4g} "
          f"{r['ratio']:>11.3e} {r['n_negative']:>6d} {r['neg_mass']:>10.4g}")
    a(f"\n  VERDICT (A): {pd_v}   (worst min/max = {worst:.2e})")
    a("")
    a("-" * 72)
    a("  (B) DOES W CARRY THE ENERGY SIGNAL?  [kernel-free]")
    a("-" * 72)
    a(f"  B1 distance vs |dE|   Spearman = {de['spearman']:.3f}  lift = {de['lift']:.2f}x")
    a(f"  B2 k-NN-in-metric (LOO), baseline MAE = {knn['baseline_mae']:.4g} eV")
    for r in knn["per_k"]:
        a(f"       k={r['k']:<3d} MAE={r['mae']:.4g} eV  skill={r['skill']:+.3f}")
    a(f"     best: k={knn['best']['k']} skill={knn['best']['skill']:+.3f}")
    if gp and gp.get("ok"):
        a(f"  B3 exp-kernel GP LOO  MAE={gp['mae']:.4g} eV  2sig-cov={gp['coverage']:.2f}"
          f"  ({gp['flag']})")
    a("")
    signal = de["spearman"] > 0.1 and knn["best"]["skill"] > 0.05
    a("-" * 72)
    a(f"  VERDICT (B): {'SIGNAL PRESENT' if signal else 'NO USABLE SIGNAL'}")
    a("-" * 72)
    a("\n  DECISION:")
    if pd_v in ("PD", "NEAR-PD") and signal:
        a("    -> exp kernel is a usable STOPGAP and the metric tracks energy.")
        a("       Use it for interim results; pursue sliced-Wasserstein for the")
        a("       scalable (compactly-supported) build -- exp(-W/ell) is dense.")
    elif pd_v == "INDEFINITE" and signal:
        a("    -> metric is good, kernel is the problem. exp kernel won't rescue this")
        a("       on our data. Move to a PD-by-construction metric (sliced-Wasserstein;")
        a("       your sorted Option-4 profiles are already the quantile functions).")
    elif (not signal) and pd_v in ("PD", "NEAR-PD"):
        a("    -> a VALID GP that predicts little: the descriptor/metric is the limit,")
        a("       not the kernel. Reconsider the representation before kernel work.")
    else:
        a("    -> Wasserstein over Option-4 looks like a dead end for energy here.")
        a("       Pivot the descriptor (e.g. WL graph distance) and re-run diagnostics.")
    a("\n  NOTE: exp(-W/ell) is dense -- this is a signal/validity probe, not the")
    a("  production gp2Scale kernel.")
    a("=" * 72)
    print("\n".join(out))


# =========================================================================== #
#  Reusable diagnostic + alpha/beta/p sweep
# =========================================================================== #
def run_diagnostic(W, y, no_gp):
    """All readouts for one W. Bundled so the single run and the sweep share
    identical logic."""
    median_w, grid = lengthscale_grid(W)
    rows = pd_diagnostic(W, grid)
    pd_v, worst = classify_pd(rows)
    de = distance_energy_readout(W, y)
    knn = knn_readout(W, y)
    gp = None if no_gp else gp_loo_readout(W, y, ell=median_w)
    return {"median_w": median_w, "pd_rows": rows, "pd_verdict": pd_v,
            "pd_worst": worst, "de": de, "knn": knn, "gp": gp}


def dedupe_combos(combos):
    """Every readout is invariant to an overall positive scaling of W, and
    (alpha, beta) -> (lambda*alpha, lambda*beta) scales W by lambda. So a metric
    is fully determined by the mixing fraction f = alpha/(alpha+beta) and p.
    Collapse the requested grid onto those distinct classes; pick the first
    (alpha, beta, p) in each class as its representative."""
    seen, unique = set(), []
    for (a, b, p) in combos:
        s = a + b
        if s <= 0:
            continue  # all-zero cost is degenerate
        key = (round(a / s, 6), p)
        if key in seen:
            continue
        seen.add(key)
        unique.append((a, b, p))
    return unique


def run_sweep(reps, y, agrid, bgrid, pgrid, use_pot, n_jobs, no_gp):
    requested = [(a, b, p) for p in pgrid for a in agrid for b in bgrid]
    unique = dedupe_combos(requested)
    print(f"sweep: {len(requested)} requested combo(s) -> {len(unique)} distinct metric(s)")
    print("       (readouts are invariant to overall W scale; only alpha:beta and p matter)\n")
    results, Ws = [], {}
    for k, (a, b, p) in enumerate(unique):
        print(f"[{k+1}/{len(unique)}] alpha={a} beta={b} p={p}")
        W = wasserstein_matrix(reps, a, b, p, use_pot=use_pot, n_jobs=n_jobs)
        d = run_diagnostic(W, y, no_gp)
        row = {
            "alpha": a, "beta": b, "p": p,
            "ratio": ("geom-only" if a == 0 else f"{a/b:.3g}" if b else "type-only"),
            "median_w": d["median_w"],
            "pd_verdict": d["pd_verdict"], "pd_worst": d["pd_worst"],
            "spearman": d["de"]["spearman"], "lift": d["de"]["lift"],
            "knn_k": d["knn"]["best"]["k"], "knn_skill": d["knn"]["best"]["skill"],
            "gp_mae": (d["gp"]["mae"] if d["gp"] and d["gp"].get("ok") else np.nan),
            "gp_cov": (d["gp"]["coverage"] if d["gp"] and d["gp"].get("ok") else np.nan),
        }
        results.append(row)
        Ws[f"W_{k:03d}"] = W
    return results, Ws


def print_sweep_table(results):
    rows = sorted(results, key=lambda r: r["knn_skill"], reverse=True)
    print("\n" + "=" * 88)
    print("  ALPHA / BETA / P SWEEP  (sorted by k-NN skill; the decision-relevant signal number)")
    print("=" * 88)
    hdr = (f"  {'alpha':>6} {'beta':>6} {'p':>3} {'a:b':>9} {'PD':>11} {'min/max':>10} "
           f"{'rho':>7} {'lift':>6} {'kNN':>10} {'GP MAE':>9} {'cov':>6}")
    print(hdr)
    print("  " + "-" * 84)
    for r in rows:
        knn = f"k{r['knn_k']}:{r['knn_skill']:+.3f}"
        gpm = "    n/a" if np.isnan(r["gp_mae"]) else f"{r['gp_mae']:.3g}"
        cov = "  n/a" if np.isnan(r["gp_cov"]) else f"{r['gp_cov']:.2f}"
        print(f"  {r['alpha']:>6.3g} {r['beta']:>6.3g} {r['p']:>3d} {r['ratio']:>9} "
              f"{r['pd_verdict']:>11} {r['pd_worst']:>10.2e} {r['spearman']:>7.3f} "
              f"{r['lift']:>6.2f} {knn:>10} {gpm:>9} {cov:>6}")
    print("  " + "-" * 84)
    best = rows[0]
    valid_signal = [r for r in rows
                    if r["pd_verdict"] in ("PD", "NEAR-PD")
                    and r["spearman"] > 0.1 and r["knn_skill"] > 0.05]
    print(f"  best k-NN skill : alpha={best['alpha']}, beta={best['beta']}, p={best['p']} "
          f"-> skill {best['knn_skill']:+.3f}")
    if valid_signal:
        print(f"  {len(valid_signal)} metric(s) clear the PD + signal bar; strongest is the row above.")
    else:
        print("  NO metric clears the PD + signal bar -- consistent with a global-vs-local")
        print("  mismatch in the representation rather than a kernel/lengthscale problem.")
    print("  (skill thresholds 0.1 / 0.05 are heuristics -- calibrate against the --shuffle null.)")
    print("=" * 88)


def make_sweep_plot(results, outdir, ts):
    rows = sorted(results, key=lambda r: (r["p"], r["alpha"], r["beta"]))
    labels = [f"a{r['alpha']:g}\nb{r['beta']:g}\np{r['p']}" for r in rows]
    skills = [r["knn_skill"] for r in rows]
    rhos = [r["spearman"] for r in rows]
    pd_ok = [r["pd_verdict"] in ("PD", "NEAR-PD") for r in rows]
    colors = ["#55a868" if ok else "#c44e52" for ok in pd_ok]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(max(7, 1.0 * len(rows)), 4), sharex=True)
    ax[0].bar(x, skills, color=colors)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(title="best k-NN skill per metric\n(green=PD, red=indefinite)",
              ylabel="skill = 1 - MAE/MAE_base")
    ax[1].bar(x, rhos, color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(title="distance-energy Spearman", ylabel="rho")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8)
    fig.tight_layout()
    path = f"{outdir}/{ts}_sweep.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def write_sweep_csv(results, path):
    cols = ["alpha", "beta", "p", "ratio", "median_w", "pd_verdict", "pd_worst",
            "spearman", "lift", "knn_k", "knn_skill", "gp_mae", "gp_cov"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


# =========================================================================== #
#  Self-test data (exercises the full chain without ASE/fairchem)
# =========================================================================== #
def make_demo(n=120, shuffle_energy=False, seed=0):
    """Synthetic molecules whose energy residual depends on molecular SCALE, a
    geometric property the Option-4 profiles capture -- so a correct pipeline
    should report PD-ish + signal. --shuffle-energy gives the matched null."""
    rng = np.random.default_rng(seed)
    elems = ["H", "C", "N", "O", "S"]
    true_eps = {"H": -13.6, "C": -1030.0, "N": -1480.0, "O": -2040.0, "S": -10800.0}
    symbols, positions, comps, energies, charges = [], [], [], [], []
    for _ in range(n):
        na = int(rng.integers(6, 15))
        syms = list(rng.choice(elems, size=na, p=[0.45, 0.30, 0.10, 0.10, 0.05]))
        scale = float(rng.uniform(0.8, 2.5))          # latent geometric scale
        pos = rng.normal(scale=scale, size=(na, 3))
        comp_e = sum(true_eps[s] for s in syms)
        e_geom = 6.0 * scale                          # signal living in geometry
        symbols.append(syms); positions.append(pos)
        comps.append(Counter(syms)); charges.append(0.0)
        energies.append(comp_e + e_geom + 0.2 * rng.normal())
    energies = np.array(energies)
    exp = "PD-ish + STRONG signal (geometry-linked energy)"
    if shuffle_energy:
        energies = rng.permutation(energies)
        exp = "PD-ish + NO signal (energies permuted)"
    return {"indices": np.arange(n), "symbols": symbols, "positions": positions,
            "comps": comps, "energies": energies,
            "charges": np.array(charges)}, exp

def run_wl_screen(data, y, args, ts):
    """Variogram + feature-kernel screen for the WL graph kernel, reusing the
    SAME subsample and referenced target y as the Wasserstein diagnostic. feats,
    y, and sizes share one molecule order, so the alignment guarantee carries
    over unchanged from the collection pass."""
    from ase.data import atomic_numbers

    from variogram_screen import (gp_loo_crps, gram_loo_crps, print_table,
                                  run_screen)
    from wl_kernel import make_wl_candidate

    # feats as (atomic_numbers, positions) tuples -- extract_atoms handles this form.
    feats = [(np.array([atomic_numbers[s] for s in syms]), pos)
             for syms, pos in zip(data["symbols"], data["positions"])]
    sizes = np.array([len(s) for s in data["symbols"]])

    cands = []
    for h in args.wl_depths:
        if args.wl_normalize in ("both", "raw"):
            cands.append(make_wl_candidate(f"wl_h{h}_raw", h=h,
                                           scale=args.wl_scale, normalize=False))
        if args.wl_normalize in ("both", "norm"):
            cands.append(make_wl_candidate(f"wl_h{h}_norm", h=h,
                                           scale=args.wl_scale, normalize=True))

    print("\n=== WL VARIOGRAM SCREEN (WL as a DISTANCE -> Wendland) ===")
    print(f"  feats={len(feats)}  referenced-y spread={np.ptp(y):.3g} eV  "
          f"sizes {sizes.min()}-{sizes.max()} atoms  bond-scale={args.wl_scale}\n")
    plot_dir = f"{args.outdir}/{ts}_wl_plots"
    results = run_screen(cands, feats, y, sizes=sizes, size_band=2,
                         plot_dir=plot_dir, use_cache=False)
    print_table(results)

    print("\n=== CONTRAST: the SAME WL kernel used as a FEATURE (dot-product) GP ===\n")
    print(f"  {'candidate':22s} {'as-distance (Wendland)':28s} {'as-feature (dot)'}")
    for cand in cands:
        K = cand.fn(feats)
        feat = gram_loo_crps(K, y)
        dist = gp_loo_crps(cand, feats, y, hps=(float(np.var(y)), None))
        print(f"  {cand.name:22s} RMSE={dist['rmse']:8.4g} CRPS={dist['crps']:8.4g}   "
              f"RMSE={feat['rmse']:8.4g} CRPS={feat['crps']:8.4g}")

    npz = f"{args.outdir}/{ts}_wl_screen.npz"
    np.savez(npz, y=y, indices=data["indices"], sizes=sizes,
             names=np.array([r.name for r in results]),
             struct_frac=np.array([r.structured_fraction for r in results]),
             struct_frac_sizectl=np.array([r.structured_fraction_sizectl for r in results]),
             vrange=np.array([r.vrange for r in results]),
             sparsity=np.array([r.sparsity_ratio for r in results]))
    print(f"\nsaved -> {npz}")
    print(f"saved -> {plot_dir}/  (variogram + distance-distribution per candidate)")
    print("\nfeats, y, sizes share one molecule order (reused from the collection pass).")

def _feats_from_data(data):
    from ase.data import atomic_numbers
    return [(np.array([atomic_numbers[s] for s in syms]), pos)
            for syms, pos in zip(data["symbols"], data["positions"])]


def sweep_one(feats, y, depth, scale, normalize, noise_frac):
    """Core support-radius sweep on ONE (feats, y): returns (rows, variogram_range).
    Shared by the single sweep and every comparison axis so they can't diverge."""
    from variogram_screen import empirical_variogram, wendland_loo_crps
    from wl_kernel import make_wl_candidate
    cand = make_wl_candidate(f"wl_h{depth}", h=depth, scale=scale, normalize=normalize)
    D, _ = cand.pairwise_distance(feats)
    dvec = D[np.triu_indices(len(feats), 1)]
    vg = empirical_variogram(D, y, n_bins=25)
    qs = np.array([0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35,
                   0.5, 0.65, 0.8, 0.95, 1.0])
    radii = np.unique(np.append(np.quantile(dvec, qs), vg.vrange))
    rows = [m for r in radii
            if (m := wendland_loo_crps(D, float(r), y, noise_frac=args_noise(noise_frac)))["ok"]]
    return rows, float(vg.vrange)


def args_noise(nf):           # tiny shim so sweep_one stays signature-stable
    return nf


def run_wl_support_sweep(data, y, args, ts):
    from variogram_screen import print_sweep_summary, summarize_support_sweep
    feats = _feats_from_data(data)
    normalize = (args.wl_sweep_normalize == "norm")
    tag = "norm" if normalize else "raw"
    print(f"\n=== WL SUPPORT-RADIUS SWEEP  (h={args.wl_sweep_depth}, {tag}, "
          f"bond-scale={args.wl_scale}, mean={args.mean_function}) ===")
    rows, vrange = sweep_one(feats, y, args.wl_sweep_depth, args.wl_scale,
                             normalize, args.noise_frac)
    print(f"  built WL distance for {len(feats)} molecules   variogram range={vrange:.3f}")
    print(f"\n  {'support r':>10} {'density':>9} {'sparsity':>9} {'RMSE':>8} "
          f"{'CRPS':>8} {'2s-cov':>7}")
    print("  " + "-" * 62)
    for m in rows:
        mark = "  <- variogram range" if abs(m["r"] - vrange) < 1e-9 else ""
        print(f"  {m['r']:>10.3f} {m['density']:>9.4f} {m['sparsity']:>9.4f} "
              f"{m['rmse']:>8.3f} {m['crps']:>8.3f} {m['coverage']:>7.2f}{mark}")
    print("  " + "-" * 62)
    s = summarize_support_sweep(rows)
    print_sweep_summary(s)
    plot = _plot_support_sweep(rows, vrange, s, args.outdir, ts, args.wl_sweep_depth, tag)
    npz = f"{args.outdir}/{ts}_wl_support_sweep.npz"
    np.savez(npz, radii=np.array([m["r"] for m in rows]),
             density=np.array([m["density"] for m in rows]),
             sparsity=np.array([m["sparsity"] for m in rows]),
             rmse=np.array([m["rmse"] for m in rows]),
             crps=np.array([m["crps"] for m in rows]),
             coverage=np.array([m["coverage"] for m in rows]),
             variogram_range=vrange, y=y, indices=data["indices"])
    print(f"\nsaved -> {npz}\nsaved -> {plot}")


def _plot_support_sweep(rows, vrange, s, outdir, ts, h, tag):
    dens = np.array([m["density"] for m in rows]); crps = np.array([m["crps"] for m in rows])
    cov = np.array([m["coverage"] for m in rows]); rr = np.array([m["r"] for m in rows])
    j = int(np.argmin(np.abs(rr - vrange)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(dens, crps, "o-")
    if s:                                                       # mark the true optimum (argmin)
        o = s["optimum"]
        ax[0].axhline(o["crps"], color="gray", ls=":", label="optimum CRPS")
        ax[0].plot(o["density"], o["crps"], "*", ms=16, color="goldenrod",
                   mec="k", label="optimum")
    ax[0].plot(dens[j], crps[j], "rs", ms=11, mfc="none", label="variogram range")
    ax[0].set(xscale="log", xlabel="matrix density (off-diag nonzero fraction)",
              ylabel="GP-LOO CRPS (eV)", title=f"signal vs sparsity: wl_h{h}_{tag}")
    ax[0].legend(fontsize=8)
    ax[1].plot(dens, cov, "o-", color="seagreen"); ax[1].axhline(0.95, color="k", ls=":")
    ax[1].plot(dens[j], cov[j], "rs", ms=11, mfc="none")
    ax[1].set(xscale="log", xlabel="matrix density", ylabel="2-sigma coverage",
              title="calibration vs sparsity")
    fig.tight_layout()
    path = f"{outdir}/{ts}_wl_support_sweep.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def _collect_for_compare(args, element_mode, n_molecules):
    ds = load_dataset(args.src)
    return collect_subset(ds, n_molecules, args.pool, tuple(args.size),
                          element_mode, args.seed,
                          charge_filter=args.charge, spin_filter=args.spin)


def run_wl_sweep_compare(args, ts):
    """Overlay support sweeps across one comparison axis. 'mean-function' reuses a
    single collection; 'elements' re-collects per element set; 'n' collects the
    largest N once and sweeps nested prefixes (a convergence study)."""
    from mean_function import compute_target
    from variogram_screen import print_sweep_summary, summarize_support_sweep
    if not args.src:
        raise SystemExit("--wl-sweep-compare needs --src (real data).")
    normalize = (args.wl_sweep_normalize == "norm")
    depth, scale, nf = args.wl_sweep_depth, args.wl_scale, args.noise_frac
    curves = []

    def target(data, basis):
        return compute_target(data, basis=basis, scale=scale,
                              add_charge=args.add_charge, add_intercept=args.add_intercept)["g"]

    if args.wl_sweep_compare == "mean-function":
        data = _collect_for_compare(args, args.elements, args.n_molecules)
        feats = _feats_from_data(data)
        for basis in args.compare_bases:
            rows, vr = sweep_one(feats, target(data, basis), depth, scale, normalize, nf)
            curves.append((f"mean={basis}", rows, vr, summarize_support_sweep(rows)))

    elif args.wl_sweep_compare == "elements":
        for em in args.compare_elements:
            data = _collect_for_compare(args, em, args.n_molecules)
            feats = _feats_from_data(data)
            rows, vr = sweep_one(feats, target(data, args.mean_function),
                                 depth, scale, normalize, nf)
            curves.append((f"{em} (N={len(feats)})", rows, vr, summarize_support_sweep(rows)))

    elif args.wl_sweep_compare == "n":
        nmax = max(args.compare_n)
        data = _collect_for_compare(args, args.elements, nmax)
        feats_all = _feats_from_data(data)
        y_all = target(data, args.mean_function)
        for n in sorted(args.compare_n):
            n = min(n, len(feats_all))
            rows, vr = sweep_one(feats_all[:n], y_all[:n], depth, scale, normalize, nf)
            curves.append((f"N={n}", rows, vr, summarize_support_sweep(rows)))

    print("\n" + "=" * 72)
    print(f"  SWEEP COMPARISON across: {args.wl_sweep_compare}")
    print("=" * 72)
    for label, rows, vr, s in curves:
        print(f"\n[{label}]  variogram range={vr:.3f}")
        print_sweep_summary(s)
    plot = _plot_sweep_overlay(curves, args.wl_sweep_compare, args.outdir, ts, depth)
    print(f"\nsaved -> {plot}")


def _plot_sweep_overlay(curves, axis, outdir, ts, depth):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(curves), 1)))
    for (label, rows, vr, s), col in zip(curves, colors):
        d = [m["density"] for m in rows]; cr = [m["crps"] for m in rows]
        cov = [m["coverage"] for m in rows]
        ax[0].plot(d, cr, "o-", color=col, label=label)
        ax[1].plot(d, cov, "o-", color=col, label=label)
        if s:
            o = s["optimum"]
            ax[0].plot(o["density"], o["crps"], "*", ms=15, color=col, mec="k")
    ax[0].set(xscale="log", xlabel="matrix density (off-diag nonzero fraction)",
              ylabel="GP-LOO CRPS (eV)",
              title=f"CRPS vs sparsity (h={depth});  * = optimum")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0.95, color="k", ls=":", label="0.95 target")
    ax[1].set(xscale="log", xlabel="matrix density", ylabel="2-sigma coverage",
              title="calibration vs sparsity")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = f"{outdir}/{ts}_wl_sweep_compare_{axis}.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path

def run_conformer_scan(data, y, args, ts):
    from ase.data import atomic_numbers

    import conformer_scan as cs
    feats = [(np.array([atomic_numbers[s] for s in syms]), pos)
             for syms, pos in zip(data["symbols"], data["positions"])]
    ref_rms = float(np.sqrt(np.mean(y ** 2)))            # residual RMS the kernel fights
    result = cs.scan(feats, data["energies"], depths=tuple(args.conformer_depths),
                     scale=args.wl_scale, reference_rms=ref_rms)
    cs.print_report(result)
    # surface atoms.info keys so you can find a true system/conformer id for the
    # direct (id-based) measurement, which doesn't depend on lucky co-sampling.
    try:
        info0 = load_dataset(args.src).get_atoms(int(data["indices"][0])).info
        print("\n  atoms.info keys on a sample structure (look for a system/molecule id):")
        print("   ", sorted(info0.keys()))
    except Exception:
        pass
    npz = f"{args.outdir}/{ts}_conformer_scan.npz"
    np.savez(npz, depths=np.array(args.conformer_depths),
             rms_within=np.array([r["rms_within"] for r in result["rows"]]),
             max_range=np.array([r["max_range"] for r in result["rows"]]),
             frac=np.array([r["frac"] for r in result["rows"]]),
             n_groups=np.array([r["n_groups"] for r in result["rows"]]),
             indices=data["indices"], reference_rms=ref_rms)
    print(f"\nsaved -> {npz}")

def run_sid_scan(args, ts):
    import sid_scan as ss
    ds = load_dataset(args.src)
    keep = element_filter(args.elements)
    lo, hi = args.size
    ids, energies, comps, charges, comp_sig = [], [], [], [], []
    kept = 0
    n = min(args.sid_pool, len(ds) - args.sid_start)
    for k in range(n):
        idx = args.sid_start + k
        if k % 5000 == 0:
            print(f"  scanned {k}/{n}, kept {kept}", flush=True)
        try:
            atoms = ds.get_atoms(int(idx))
        except Exception:
            continue
        syms = atoms.get_chemical_symbols()
        if not (lo <= len(atoms) <= hi and keep(set(syms))):
            continue
        q = int(round(float(atoms.info.get("charge", 0))))
        sp = atoms.info.get("spin", None)
        if args.charge is not None and q != args.charge:
            continue
        if args.spin is not None and sp is not None and int(sp) != args.spin:
            continue
        sid = atoms.info.get(args.sid_key, None)
        if sid is None:
            continue
        try:
            e = atoms.get_potential_energy()
        except Exception:
            continue
        c = Counter(syms)
        ids.append(sid); energies.append(float(e)); comps.append(c)
        charges.append(float(q))
        comp_sig.append(tuple(sorted(c.items())))
        kept += 1
    print(f"scanned {n} contiguous from {args.sid_start}, kept {kept} in-slice structures")

    # element-referencing residual RMS on the kept pool, for the "% of residual" context
    ref_rms = None
    if kept >= 10:
        elements = sorted({e for c in comps for e in c})
        idx_e = {e: i for i, e in enumerate(elements)}
        X = np.zeros((kept, len(elements) + 1))
        for m, c in enumerate(comps):
            for e, cnt in c.items():
                X[m, idx_e[e]] = cnt
        X[:, -1] = charges
        coef, *_ = np.linalg.lstsq(X, np.array(energies), rcond=None)
        g = np.array(energies) - X @ coef
        ref_rms = float(np.sqrt(np.mean(g ** 2)))

    result = ss.analyze_groups(ids, energies, comp_sig=comp_sig, reference_rms=ref_rms)
    ss.print_report(result, id_key=args.sid_key, scanned=n, kept=kept)
    npz = f"{args.outdir}/{ts}_sid_scan.npz"
    np.savez(npz, n_systems=result["n_systems"], n_structures=result["n_structures"],
             n_multi_systems=result["n_multi_systems"],
             frac_systems_multi=result["frac_systems_multi"],
             frac_struct_in_multi=result["frac_struct_in_multi"],
             rms_within=result["rms_within"], max_range=result["max_range"],
             median_group_std=result["median_group_std"], reference_rms=ref_rms or 0.0)
    print(f"\nsaved -> {npz}")
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--src", help="dir with the .aselmdb split (omit for --demo)")
    ap.add_argument("--n-molecules", type=int, default=400)
    ap.add_argument("--pool", type=int, default=80000)
    ap.add_argument("--elements", choices=["organic", "organic_ext", "all", "nonorganic"],
                    default="organic")
    ap.add_argument("--size", type=int, nargs=2, default=[20, 60], metavar=("LO", "HI"))
    ap.add_argument("--add-charge", action="store_true")
    ap.add_argument("--add-intercept", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # Option-4 / Wasserstein
    ap.add_argument("--alpha", type=float, default=1.0, help="atom-type mismatch weight")
    ap.add_argument("--beta", type=float, default=1.0, help="geometry (profile) weight")
    ap.add_argument("--wasserstein-p", type=int, default=2)
    ap.add_argument("--quantile-points", type=int, default=16, help="profile resample grid")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel pairs (needs joblib)")
    ap.add_argument("--sinkhorn", action="store_true", help="force Sinkhorn (skip exact EMD)")
    # sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep alpha/beta/p in ONE collection pass and tabulate")
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=[0.0, 1.0],
                    help="atom-type weights to try (0 = geometry-only control)")
    ap.add_argument("--beta-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="geometry weights to try (only the alpha:beta ratio matters)")
    ap.add_argument("--p-grid", type=int, nargs="+", default=[1, 2],
                    help="Wasserstein orders to try")
    # diagnostic / io
    ap.add_argument("--no-gp", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--shuffle-energy", action="store_true")
    # WL graph-kernel screen (reuses the same subsample + referencing)
    ap.add_argument("--wl-screen", action="store_true",
                    help="run the WL variogram screen instead of the Wasserstein diagnostic")
    ap.add_argument("--wl-depths", type=int, nargs="+", default=[0, 1, 2],
                    help="WL iteration depths h to screen")
    ap.add_argument("--wl-scale", type=float, default=1.2,
                    help="covalent-radius scale for bond perception (sweep this if unsure)")
    ap.add_argument("--wl-normalize", choices=["both", "raw", "norm"], default="both",
                    help="register raw, normalized, or both WL variants")
    # charge/spin filtering (WL graphs are blind to both -- filter for a clean target)
    ap.add_argument("--charge", type=int, default=None,
                    help="keep only molecules with this formal charge")
    ap.add_argument("--spin", type=int, default=None,
                    help="keep only molecules with this spin multiplicity (1 = singlet)")

    ap.add_argument("--wl-support-sweep", action="store_true",
                    help="sweep Wendland support radius on one WL kernel; trace CRPS vs density")
    ap.add_argument("--wl-sweep-depth", type=int, default=1)
    ap.add_argument("--wl-sweep-normalize", choices=["raw", "norm"], default="raw")
    ap.add_argument("--wl-sweep-compare", choices=["mean-function", "elements", "n"],
                    help="overlay support sweeps across a comparison axis (own data collection)")
    ap.add_argument("--wl-conformer-scan", action="store_true",
                    help="measure WL conformer blindness: DFT energy spread among WL-identical structures")
    ap.add_argument("--conformer-depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--compare-bases", nargs="+", default=["element", "element+bonds"])
    ap.add_argument("--compare-elements", nargs="+",
                    default=["organic", "organic_ext", "all"])
    ap.add_argument("--compare-n", type=int, nargs="+", default=[300, 600, 1200, 2400])
    ap.add_argument("--noise-frac", type=float, default=1e-2,
                    help="GP noise as a fraction of signal variance in the LOO sweep")
    ap.add_argument("--sid-scan", action="store_true",
                    help="conformer scan by system id: energy spread + prevalence")
    ap.add_argument("--sid-key", default="sid", help="atoms.info key to group on")
    ap.add_argument("--sid-pool", type=int, default=60000,
                    help="number of structures to scan (contiguous) for sid grouping")
    ap.add_argument("--sid-start", type=int, default=0, help="starting index of the scan window")

    ap.add_argument("--mean-function", choices=["element", "element+bonds"],
                    default="element",
                    help="extensive mean function for the WL residual target: "
                         "'element' = per-element referencing (your current target); "
                         "'element+bonds' adds a bond-inventory term to absorb cohesive energy")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.sid_scan:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_sid_scan(args, ts)
        return
    if args.wl_sweep_compare:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_wl_sweep_compare(args, ts)
        return

    expectation = None
    if args.demo:
        data, expectation = make_demo(shuffle_energy=args.shuffle_energy, seed=args.seed)
    else:
        if not args.src:
            ap.error("provide --src, or use --demo")
        data = collect_subset(load_dataset(args.src), args.n_molecules, args.pool,
                              tuple(args.size), args.elements, args.seed,
                              charge_filter=args.charge, spin_filter=args.spin)
        if len(data["indices"]) < 10:
            sys.exit("Too few molecules; widen --size/--pool/--elements.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- WL experiments: target is the mean-function residual g = E - m(M).
    #      --mean-function selects the extensive mean basis. compute_target prints
    #      the residual-vs-size diagnostic + verdict, which says whether g is
    #      intensive enough for a stationary compact kernel BEFORE you trust the
    #      support sweep. Because m(M) is deterministic, CRPS on g == CRPS on E,
    #      so these numbers are directly comparable to the total-energy runs. ----
    if args.wl_screen or args.wl_support_sweep or args.wl_conformer_scan:
        from mean_function import compute_target
        info = compute_target(data, basis=args.mean_function, scale=args.wl_scale,
                              add_charge=args.add_charge, add_intercept=args.add_intercept)
        y = info["g"]
        assert y.shape[0] == len(data["indices"])
        if args.wl_conformer_scan:
            run_conformer_scan(data, y, args, ts)
        elif args.wl_screen:
            run_wl_screen(data, y, args, ts)
        else:
            run_wl_support_sweep(data, y, args, ts)
        return

    # ---- Wasserstein / Option-4 path: per-element referencing as before ----
    y, eps, labels, elements = reference_energies(
        data["comps"], data["charges"], data["energies"],
        args.add_charge, args.add_intercept)
    print(f"\nreferencing: raw spread {np.ptp(data['energies']):,.1f} eV "
          f"-> residual spread {np.ptp(y):,.1f} eV  (RMS {np.sqrt(np.mean(y**2)):.3f} eV)")
    reps = build_representation(data["positions"], data["symbols"], args.quantile_points)
    assert len(reps) == y.shape[0]   # alignment is structural: row i of W is y[i]
    # ---- WL screen reuses ONLY the subsample + referenced y (no Option-4 reps) ----
    if args.wl_screen:
        run_wl_screen(data, y, args, ts)
        return
    if args.wl_support_sweep:
        run_wl_support_sweep(data, y, args, ts)
        return

    # ---- Option-4 representation (only needed for the Wasserstein diagnostic) ----
    reps = build_representation(data["positions"], data["symbols"], args.quantile_points)
    assert len(reps) == y.shape[0]   # alignment is structural: row i of W is y[i]

    # ======================================================================= #
    #  SWEEP MODE
    # ======================================================================= #
    if args.sweep:
        results, Ws = run_sweep(reps, y, args.alpha_grid, args.beta_grid, args.p_grid,
                                use_pot=not args.sinkhorn, n_jobs=args.n_jobs,
                                no_gp=args.no_gp)
        print_sweep_table(results)
        plot = make_sweep_plot(results, args.outdir, ts)
        csv = f"{args.outdir}/{ts}_sweep.csv"
        write_sweep_csv(results, csv)
        npz = f"{args.outdir}/{ts}_sweep.npz"
        np.savez(npz, y=y, indices=data["indices"], eps=eps,
                 labels=np.array(labels), elements=np.array(elements),
                 params=np.array([(r["alpha"], r["beta"], r["p"]) for r in results]),
                 quantile_points=args.quantile_points, **Ws)
        print(f"\nsaved -> {npz}\nsaved -> {csv}\nsaved -> {plot}")
        print("\nAll W_### in the .npz share the same molecule order as y (one collection pass).")
        return

    # ======================================================================= #
    #  SINGLE-METRIC MODE
    # ======================================================================= #
    W = wasserstein_matrix(reps, args.alpha, args.beta, args.wasserstein_p,
                           use_pot=not args.sinkhorn, n_jobs=args.n_jobs)
    assert W.shape[0] == y.shape[0]

    d = run_diagnostic(W, y, args.no_gp)
    paths = make_plots(d["pd_rows"], d["de"], d["knn"], d["median_w"], args.outdir, ts)
    print()
    print_readout(len(y), d["median_w"], eps, labels, d["pd_rows"], d["pd_verdict"],
                  d["pd_worst"], d["de"], d["knn"], d["gp"], expectation)

    npz = f"{args.outdir}/{ts}_pipeline.npz"
    np.savez(npz, W=W, y=y, indices=data["indices"], eps=eps,
             labels=np.array(labels), elements=np.array(elements),
             alpha=args.alpha, beta=args.beta, wasserstein_p=args.wasserstein_p,
             quantile_points=args.quantile_points)
    print(f"\nsaved -> {npz}")
    for p in paths:
        print(f"saved -> {p}")
    print("\nW and y in the .npz are aligned by construction (shared collection pass).")


if __name__ == "__main__":
    main()
