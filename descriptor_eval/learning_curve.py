#!/usr/bin/env python
"""
learning_curve.py
=================
R^2 vs training-set size, holding the TEST set fixed, to measure whether "more
data helps" (density-limited) before committing a large run.

Reads the curve by its LAST-SEGMENT slope, not endpoint-to-endpoint: a small
sample often starts with negative R^2 (undersampled GP) then rises to the
descriptor's true level and PLATEAUS. Endpoint-minus-startpoint would call that
"climbing" and mislead the scale-up decision; the last segment tells you whether
it's still climbing at the largest N you tested.

--pool N regenerates a fresh pool of N molecules (fit a new extensive mean +
residual on that pool, cached separately so the canonical 10k is untouched) so
you can extend the curve past the 10k cache -- e.g. test whether an 8k->16k
plateau holds on a 20k pool, which distinguishes a real ceiling from a 10k-pool
artifact.

Usage
-----
    python learning_curve.py --sizes 2000,4000,8000 --metric l2 --cutoff-pct 25
    python learning_curve.py --pool 20000 --sizes 4000,8000,12000,16000 --cutoff-pct 25
"""

import argparse
import math
import os
from datetime import datetime

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import gp_parity_l1 as gp
import matplotlib.pyplot as plt

# ------------------------------ larger-than-cache pool ----------------------


def build_pool(N, seed=0, charge_key="lowdin_charges"):
    """Draw a fresh N-molecule pool, fit a NEW extensive mean on it, return
    (atoms, residual). Cached to pool-specific files so the canonical 10k is
    never overwritten. R^2 across DIFFERENT pools shares neither the exact target
    nor the test set, so run a full curve WITHIN one pool for comparability."""
    import data

    idx_path = os.path.join(gp.CACHE, f"pool{N}_s{seed}_indices.npy")
    y_path = os.path.join(gp.CACHE, f"pool{N}_s{seed}_y.npy")
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": gp.SRC})

    if os.path.exists(idx_path) and os.path.exists(y_path):
        idx = np.load(idx_path)
        y = np.load(y_path)
        atoms = [ds.get_atoms(int(i)) for i in idx]
        print(f"[pool] reusing cached pool N={len(idx)} (seed {seed})")
        return atoms, y

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    idx, p = [], 0
    while len(idx) < N and p < len(ds):
        i = int(perm[p])
        p += 1
        q = ds.get_atoms(i).info.get(charge_key)
        if q is not None and not np.any(np.isnan(np.asarray(q, float))):
            idx.append(i)
    if len(idx) < N:
        raise RuntimeError(f"only {len(idx)} valid molecules (< {N})")
    idx = np.array(idx)
    atoms = [ds.get_atoms(int(i)) for i in idx]
    Z = [a.get_atomic_numbers().tolist() for a in atoms]
    E = [float(a.get_potential_energy()) for a in atoms]
    C = [float(a.info.get("charge", 0)) for a in atoms]
    S = [float(a.info.get("spin", 1)) for a in atoms]
    mm = data.ExtensiveMean().fit(Z, E, C, S)
    y = mm.residual(Z, E, C, S)
    np.save(idx_path, idx)
    np.save(y_path, y)
    print(f"[pool] built pool N={len(idx)} (scanned {p}); residual var {np.var(y):.4g}")
    return atoms, y


# ------------------------------ verdict (segment slope) ---------------------


def verdict(rows):
    """Judge by the LAST segment (R^2 gained per doubling of N), not endpoints."""
    Ns = [r[0] for r in rows]
    R2 = [r[1] for r in rows]
    if len(rows) < 2:
        return "[lc] need >= 2 sizes to judge slope"
    seg = [
        (R2[i + 1] - R2[i]) / math.log2(Ns[i + 1] / Ns[i]) for i in range(len(rows) - 1)
    ]
    last, total = seg[-1], R2[-1] - R2[0]
    proj = R2[-1] + last * math.log2(200_000 / Ns[-1])
    if last > 0.02:
        return (
            f"[lc] STILL CLIMBING: last-segment slope {last:+.3f} R^2/doubling "
            f"(total {total:+.3f}). Naive extrapolation to 200k ~ R^2 {proj:.3f}. "
            "A larger run is justified."
        )
    if total > 0.03:
        return (
            f"[lc] ROSE THEN PLATEAUED: overall {total:+.3f} but last segment flat "
            f"({last:+.3f}/doubling). Small-sample regime resolved; R^2 settled near "
            f"{R2[-1]:.3f}. More data unlikely to move it much on this descriptor."
        )
    return (
        f"[lc] FLAT (overall {total:+.3f}, last {last:+.3f}/doubling) -> representational; "
        "more data won't help."
    )


# ------------------------------ main ----------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", default="2000,4000,8000", help="TRAIN sizes (nested prefixes)"
    )
    ap.add_argument(
        "--pool",
        type=int,
        default=None,
        help="regenerate a fresh pool of this many molecules (default: cached 10k)",
    )
    ap.add_argument("--seed", type=int, default=0, help="pool draw seed")
    ap.add_argument("--wl-mode", default="explicit", choices=["explicit", "hashed"])
    ap.add_argument("--wl-depth", type=int, default=3)
    ap.add_argument("--include-depth0", action="store_true")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--pls-components", type=int, default=10)
    ap.add_argument("--no-pls", action="store_true")
    ap.add_argument("--jitter", type=float, default=None)
    a = ap.parse_args()
    sizes = sorted(int(s) for s in a.sizes.split(","))

    if a.pool is not None and a.pool != gp.SUBSET_N:
        atoms, y = build_pool(a.pool, a.seed)
    else:
        atoms, y = gp.build_atoms()

    a_tr, a_te, y_tr, y_te = train_test_split(
        atoms, y, test_size=gp.TEST_FRACTION, random_state=gp.RANDOM_STATE
    )
    print(f"[lc] fixed test={len(y_te)}, full train={len(y_tr)}; sizes={sizes}")

    rows = []
    for N in sizes:
        if N > len(a_tr):
            print(f"[lc] skip N={N} (> {len(a_tr)} available train)")
            continue
        res = gp.evaluate(
            a_tr[:N],
            y_tr[:N],
            a_te,
            y_te,
            wl_mode=a.wl_mode,
            wl_depth=a.wl_depth,
            include_depth0=a.include_depth0,
            min_count=a.min_count,
            use_pls=not a.no_pls,
            pls_components=a.pls_components,
            metric=a.metric,
            cutoff=a.cutoff,
            cutoff_pct=a.cutoff_pct,
            jitter=a.jitter,
            verbose=False,
        )
        print(
            f"[lc] N={N:>6}  R2={res['r2']:+.3f}  RMSE={res['rmse']:.3f}  "
            f"D={res['D']:>6}  OOV={res['oov']:.1%}  cutoff={res['cutoff']:.3g}"
        )
        rows.append((N, res["r2"], res["rmse"], res["oov"]))

    if not rows:
        print("[lc] no sizes ran")
        return

    Ns = [r[0] for r in rows]
    R2 = [r[1] for r in rows]
    os.makedirs(gp.GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    tag = f"pool{a.pool}" if a.pool else "cache10k"
    path = os.path.join(
        gp.GRAPHS, f"GP-learningcurve-WL-{a.wl_mode}-{a.metric}-{tag}-{ts}.png"
    )
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(Ns, R2, "-o", lw=2, color="#348ABD")
        if len(Ns) >= 2:
            ax.set_xscale("log")
        ax.set_xlabel("training set size N (log)")
        ax.set_ylabel("test $R^2$  (fixed test set)")
        ax.set_title(
            f"Learning curve — WL-{a.wl_mode} {a.metric.upper()} [{tag}]\n"
            "judge the LAST segment's slope, not the endpoints"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print(f"[saved] {path}")
    print(verdict(rows))


if __name__ == "__main__":
    main()
