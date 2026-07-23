#!/usr/bin/env python
"""
same_wl_diagnostic.py  (descriptor_eval)
========================================
Does persistence-image distance carry signal ORTHOGONAL to the WL graph
descriptor, or does it just re-encode the same graph topology?

Method -- condition on the WL graph. Group molecules that WL maps to the
IDENTICAL descriptor (same per-depth WL label multiset), then ask whether PH
distance still tracks energy-residual differences WITHIN those groups. WL is held
constant inside a group, so any PH<->y structure there is signal WL provably
cannot represent (its best prediction for a whole group is the group mean).

Two numbers decide it:
  * WL blind-spot variance -- among molecules WL cannot tell apart, how much
    energy-residual spread is left? (If ~0, PH has nothing to add; if large, there
    is room.)
  * Within-group PH variogram / correlation -- among those same-WL pairs, does a
    smaller PH distance mean a smaller |dy|? A rising within-group semivariogram
    (or positive PH-dist vs semivariance correlation) = orthogonal signal; flat =
    PH is redundant with WL.

The WL partition uses the PRODUCTION label algorithm imported straight from
wl_gp2scale/wl_features.py (build_graph + wl_labels_per_depth are byte-identical
to the local features.py copy; importing the real one guarantees the grouping
matches the 200k model and is immune to any future drift). We import the module
file, not the package, to avoid pulling gpcam/fvgp/torch via __init__.

Run from inside descriptor_eval/ (env: omol).
    python same_wl_diagnostic.py --n 10000 --scaling pareto
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                  # descriptor_eval modules
sys.path.append(os.path.join(SCRIPT_DIR, "..", "wl_gp2scale"))  # production wl_features

import data as data_mod          # noqa: E402  (descriptor_eval)
import persistence as pers       # noqa: E402  (descriptor_eval)
import wl_features as wlf         # noqa: E402  (wl_gp2scale, production label code)


# ------------------------------ WL partition --------------------------------


def wl_signature(atoms, depth, depths, cutoff_mult=1.2):
    """Hashable canonical id of a molecule's WL descriptor: the per-depth
    (label -> count) multiset over the used depths. Two molecules with the same
    signature are IDENTICAL to the WL descriptor (identical count vector in every
    vocabulary, so identical after any min_count pruning and after PLS)."""
    adj, lab = wlf.build_graph(atoms, cutoff_mult)
    pdl = wlf.wl_labels_per_depth(adj, lab, depth)
    return tuple(tuple(sorted(Counter(pdl[d]).items())) for d in depths)


# ------------------------------ scaling -------------------------------------


def scale_columns(X, mode):
    """Population column pre-weighting (descriptive; no train/test split here).
    standard = z-score, pareto = /sqrt(std), center = mean-subtract only."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    w = {"standard": std, "pareto": np.sqrt(std), "center": np.ones_like(std)}[mode]
    return (X - mean) / w


# ------------------------------ variogram helper ----------------------------


def equal_count_variogram(dist, semivar, n_bins=15):
    """Equal-count binned semivariogram: (median lag, mean semivariance, count)."""
    edges = np.percentile(dist, np.linspace(0, 100, n_bins + 1))
    edges[-1] = np.inf
    lag, gamma, cnt = [], [], []
    for b in range(n_bins):
        m = (dist >= edges[b]) & (dist < edges[b + 1])
        if m.sum() == 0:
            continue
        lag.append(float(np.median(dist[m])))
        gamma.append(float(semivar[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(lag), np.array(gamma), np.array(cnt)


def pair_arrays(Z, y, ii, jj):
    """PH distance and semivariance 0.5*(yi-yj)^2 for the given index pairs."""
    d = np.linalg.norm(Z[ii] - Z[jj], axis=1)
    sv = 0.5 * (y[ii] - y[jj]) ** 2
    return d, sv


# ------------------------------ main ----------------------------------------


def main():
    ap = argparse.ArgumentParser(description="same-WL-graph orthogonality diagnostic")
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--include-depth0", action="store_true")
    ap.add_argument("--maxdim", type=int, default=1)
    ap.add_argument("--pixel-size", type=float, default=0.25)
    ap.add_argument("--ph-thresh", type=float, default=6.0)
    ap.add_argument("--scaling", default="pareto", choices=["pareto", "standard", "center"])
    ap.add_argument("--max-within-pairs", type=int, default=400_000,
                    help="cap on within-group pairs (subsample if exceeded)")
    ap.add_argument("--global-sample", type=int, default=4000,
                    help="molecules sampled for the global (unconditioned) variogram")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--plot", action="store_true", help="save a global-vs-within PNG")
    a = ap.parse_args()
    depths = tuple(range(0 if a.include_depth0 else 1, a.depth + 1))
    rng = np.random.default_rng(a.seed)

    # 1. shared frozen subset + descriptor-independent residual target
    atoms, y, _ = data_mod.get_data(a.src, n=a.n, seed=a.seed)
    y = np.asarray(y, float)
    N = len(atoms)
    total_ss = float(np.sum((y - y.mean()) ** 2))
    print(f"[data] {N} molecules, residual std={y.std():.3f} var={y.var():.3f}")

    # 2. WL partition via the PRODUCTION label algorithm
    print(f"[wl] computing WL signatures (depths={depths}) ...")
    groups = defaultdict(list)
    for i, at in enumerate(atoms):
        groups[wl_signature(at, a.depth, depths)].append(i)
        if (i + 1) % 2000 == 0:
            print(f"[wl]   ...{i + 1}/{N}")
    sizes = np.array([len(v) for v in groups.values()])
    multi = [v for v in groups.values() if len(v) >= 2]
    n_multi_mols = int(sum(len(v) for v in multi))

    # WL ceiling (predict the group mean) + blind-spot variance among collisions
    within_ss = sum(float(np.sum((y[v] - y[v].mean()) ** 2)) for v in groups.values())
    within_ss_multi = sum(float(np.sum((y[v] - y[v].mean()) ** 2)) for v in multi)
    blind_std = float(np.sqrt(within_ss_multi / max(n_multi_mols, 1)))
    print("\n================= WL PARTITION =================")
    print(f"  {len(groups)} distinct WL signatures over {N} molecules "
          f"(largest group={sizes.max()}, singletons={int((sizes==1).sum())})")
    print(f"  collision groups (size>=2): {len(multi)}  covering {n_multi_mols} "
          f"molecules ({100*n_multi_mols/N:.1f}%)")
    print(f"  WL lookup ceiling  R^2 <= {1 - within_ss/total_ss:.3f}  (in-sample, "
          f"optimistic -- singletons fit themselves)")
    frac_at_stake = within_ss_multi / total_ss
    print(f"  WL BLIND SPOT: among collision molecules, residual std left after WL "
          f"= {blind_std:.3f}  (vs global {y.std():.3f})")
    print(f"  VARIANCE AT STAKE: within-collision SS is {100*frac_at_stake:.2f}% of "
          f"total -- an UPPER BOUND on what PH could recover via this mechanism")
    if n_multi_mols < 50:
        print("  WARNING: very few exact WL collisions at this N -- the within-group "
              "variogram will be noisy; consider a larger --n.")

    # 3. PH embedding (the good config) + scaling
    print("\n[ph] featurizing ...")
    Xph = pers.PersistenceFeaturizer(
        maxdim=a.maxdim, pixel_size=a.pixel_size, thresh=a.ph_thresh
    ).fit_transform(atoms)
    Z = scale_columns(Xph, a.scaling)
    print(f"[ph] embedding {Z.shape} scaling={a.scaling}")

    # 4a. WITHIN-GROUP (WL-controlled) PH pairs
    ii, jj = [], []
    for v in multi:
        v = np.asarray(v)
        a_, b_ = np.triu_indices(len(v), k=1)
        ii.append(v[a_]); jj.append(v[b_])
    ii = np.concatenate(ii) if ii else np.array([], int)
    jj = np.concatenate(jj) if jj else np.array([], int)
    n_within = len(ii)
    if n_within > a.max_within_pairs:
        sel = rng.choice(n_within, a.max_within_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    d_w, sv_w = pair_arrays(Z, y, ii, jj) if len(ii) else (np.array([]), np.array([]))

    # 4b. GLOBAL (unconditioned) PH pairs, for the same PH embedding
    from scipy.spatial.distance import pdist
    gs = min(a.global_sample, N)
    gidx = rng.choice(N, gs, replace=False)
    d_g = pdist(Z[gidx], metric="euclidean")
    sv_g = 0.5 * pdist(y[gidx][:, None], metric="euclidean") ** 2

    # 5. report both variograms + the verdict
    def report(tag, d, sv):
        if len(d) < 100:
            print(f"\n[{tag}] only {len(d)} pairs -- too few to bin.")
            return
        lag, gamma, cnt = equal_count_variogram(d, sv, a.n_bins)
        q = np.quantile(d, [0.25, 0.75])
        near = sv[d <= q[0]].mean(); far = sv[d >= q[1]].mean()
        corr = float(np.corrcoef(d, sv)[0, 1])
        print(f"\n[{tag}] {len(d):,} pairs   corr(PHdist, semivar)={corr:+.3f}   "
              f"nearQ1_gamma={near:.2f}  farQ4_gamma={far:.2f}  ratio={near/far:.2f}")
        print(f"  {'lag':>8} {'gamma':>8} {'pairs':>10}")
        for L, G, C in zip(lag, gamma, cnt):
            print(f"  {L:>8.4f} {G:>8.2f} {C:>10,d}")
        return lag, gamma, corr, near / far

    print("\n================= PH VARIOGRAMS =================")
    g_glob = report("GLOBAL  (all pairs)", d_g, sv_g)
    g_with = report("WITHIN-WL (same-graph pairs)", d_w, sv_w)

    print("\n================= VERDICT =================")
    if g_with is None:
        print("  Not enough same-WL pairs to judge; increase --n.")
    else:
        _, _, corr_w, ratio_w = g_with
        se = 1.0 / np.sqrt(max(n_within - 3, 1))     # Fisher-z std error of corr
        z = corr_w / se
        print(f"  within-group corr={corr_w:+.3f}  n_pairs={n_within}  SE~{se:.3f}  "
              f"z={z:.1f}  (need |z|>~2.5 to trust)")
        if abs(z) < 2.5 or n_within < 500:
            print("  INCONCLUSIVE / UNDER-POWERED: too few same-WL pairs (or corr within "
                  "noise). This exact-match test cannot decide orthogonality at this N.")
        elif z >= 2.5 and ratio_w < 0.9:
            print("  PH distance SIGNIFICANTLY tracks the residual among WL-identical "
                  "molecules => ORTHOGONAL (geometric) signal WL cannot represent.")
            print(f"  BUT note only {100*frac_at_stake:.2f}% of total variance is at stake "
                  "via this mechanism -- see the additive kernel for the full-space test.")
        else:
            print("  PH ~FLAT among WL-identical molecules => largely REDUNDANT with WL.")

    if a.plot and g_glob is not None and g_with is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("variograms", exist_ok=True)
        ts = datetime.now().strftime("%m-%d-%H-%M-%S")
        path = os.path.join("variograms", f"same-wl-diagnostic-{a.scaling}-{ts}.png")
        with plt.style.context("fivethirtyeight"):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(g_glob[0], g_glob[1], "-o", label="global (all pairs)")
            ax.plot(g_with[0], g_with[1], "-s", label="within-WL (same graph)")
            ax.axhline(y.var(), ls="--", color="k", lw=1.5, label="sill = Var(y)")
            ax.set_xlabel("PH descriptor distance (lag)")
            ax.set_ylabel(r"semivariance  $\frac{1}{2}(y_i-y_j)^2$")
            ax.set_title("Does PH add signal orthogonal to WL?\n"
                         "within-WL rise = yes; flat = redundant")
            ax.legend(loc="lower right", fontsize=10)
            fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
