"""
strain.py  (data_exploration)
=============================
Can a **positions-only** feature recover the off-equilibrium strain energy that the
census identified as ~40% of the regression target?

The census (REPORT.md §3) found that `max|F|` alone explains **R² = 0.397** of the
intensive residual `y`, while the best *input-side* column (n_atoms, charge, spin, R_g)
reaches R² < 0.001. Forces are DFT **labels**, not features -- so that 0.397 is a
diagnostic naming the missing physics, not a usable model. `train_4M` is emphatically
not relaxed minima (median max|F| = 3.4 eV/Å), so a large part of `y` is strain.

The existing geometry descriptor cannot express strain: `rdf`/`angle`/`torsion` are
population histograms divided by atom count, and strain is *localised* -- a few
over-compressed bonds in a 42-atom molecule barely move a normalised histogram. A
distribution cannot say "this bond is 0.1 Å too short".

This module computes strain directly, as molecular mechanics does (the harmonic bonded
terms of UFF / MMFF94 / AMBER):

    delta_ij   = r_ij - r0(Z_i, Z_j)          bond deviation
    delta_ijk  = theta_ijk - theta0(Z_j, deg) angle deviation
    E_strain  ~ sum delta^2                    (extensive, ~eV)

with the reference geometry **learned from the dataset** rather than tabulated: r0 is
the MODE of the observed bonded-distance distribution for that element pair (the mode,
not the mean/median, because the dataset's strained tail would drag those upward).
Reference learning is unsupervised -- it never sees `y`.

Every feature is emitted twice: **extensive** (a sum, ~eV, matching the fact that the
residual's std grows 3 -> 17 eV with system size) and **intensive** (divided by atom
count, matching the existing descriptor convention). Which one the model wants is an
empirical question, so both are reported.

Run the gate::

    python -m data_exploration.strain --n 20000

It scores every feature against `y` (compare to the max|F| benchmark) and against
`fmax` itself (does a positions-only feature recover the force signal?), overall and
per `data_id`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .census import CACHE

# Bond perception is shared with the production WL descriptor so the graph this module
# calls "bonded" is exactly the graph the descriptor calls "bonded" (PLAN.md §1:
# "reuse, don't reimplement -- the featurisers stay where they are").
from wl_gp2scale.wl_features import build_graph

# ----------------------------- per-structure scan ---------------------------


def _vdw_table():
    """van der Waals radii with a covalent fallback. ase's ``vdw_radii`` is both
    shorter than ``covalent_radii`` (104 vs 119 entries) and NaN for many elements,
    so pad to the covalent length before filling the gaps."""
    from ase.data import covalent_radii, vdw_radii

    cr = np.asarray(covalent_radii, dtype=float)
    v = np.full(len(cr), np.nan)
    src = np.asarray(vdw_radii, dtype=float)
    v[: len(src)] = src[: len(v)]
    bad = ~np.isfinite(v)
    v[bad] = 2.0 * cr[bad]
    v[~np.isfinite(v)] = 1.7
    return v


_VDW = _vdw_table()


def scan_one(atoms, cutoff_mult=1.2, clash_scale=1.0):
    """One structure -> the raw geometry strain needs, in ONE bond-perception pass.

    Returns a dict with:
      za, zb, r   : element pair and length of every covalent bond
      zc, deg, th : centre element, its coordination number, and each bond angle (deg)
      clash       : sum over NON-bonded, non-1-3 pairs closer than the vdW contact
                    distance of (r_contact - r)^2 -- steric repulsion, which no bonded
                    term can express
      n_atoms, n_bonds
    """
    P = np.asarray(atoms.get_positions(), dtype=float)
    Z = np.asarray(atoms.get_atomic_numbers())
    n = len(Z)
    adj, _ = build_graph(atoms, cutoff_mult)

    za, zb, da, db, rr = [], [], [], [], []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if j <= i:                       # each bond once
                continue
            d = float(np.linalg.norm(P[i] - P[j]))
            # Carry each end's COORDINATION NUMBER alongside its element. Coordination
            # proxies bond order (sp3 C has 4 neighbours, sp2 3, sp 2), and element
            # pair ALONE conflates single/aromatic/double/triple C-C (1.53/1.39/1.34/
            # 1.20 Å) into one mixture-mode reference -- deviations from which would
            # encode bond order, i.e. chemistry, rather than strain.
            ki, kj = (i, j) if Z[i] <= Z[j] else (j, i)
            za.append(int(Z[ki])); zb.append(int(Z[kj]))
            da.append(len(adj[ki])); db.append(len(adj[kj]))
            rr.append(d)

    zc, deg, th = [], [], []
    for i, nbrs in enumerate(adj):
        k = len(nbrs)
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                u = P[nbrs[a_]] - P[i]
                v = P[nbrs[b_]] - P[i]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 1e-9 and nv > 1e-9:
                    ang = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)))
                    zc.append(int(Z[i])); deg.append(k); th.append(float(ang))

    # steric clash: exclude bonded (1-2) and angle (1-3) pairs, which the bonded and
    # angular terms already own; what is left is genuine non-bonded contact.
    clash = 0.0
    if n >= 2:
        excl = set()
        for i, nbrs in enumerate(adj):
            for j in nbrs:
                excl.add((min(i, j), max(i, j)))
            for a_ in range(len(nbrs)):
                for b_ in range(a_ + 1, len(nbrs)):
                    p, q = nbrs[a_], nbrs[b_]
                    excl.add((min(p, q), max(p, q)))
        iu = np.triu_indices(n, k=1)
        diff = P[iu[0]] - P[iu[1]]
        r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        rc = clash_scale * (_VDW[Z[iu[0]]] + _VDW[Z[iu[1]]])
        hit = np.where(r < rc)[0]
        for h in hit:
            pair = (int(iu[0][h]), int(iu[1][h]))
            if pair not in excl:
                clash += float((rc[h] - r[h]) ** 2)

    return {
        "za": np.asarray(za, np.int16), "zb": np.asarray(zb, np.int16),
        "da": np.asarray(da, np.int8), "db": np.asarray(db, np.int8),
        "r": np.asarray(rr, np.float32),
        "zc": np.asarray(zc, np.int16), "deg": np.asarray(deg, np.int8),
        "th": np.asarray(th, np.float32),
        "clash": float(clash), "n_atoms": int(n), "n_bonds": len(rr),
    }


# ----------------------------- learned reference geometry -------------------


def _mode(x, lo, hi, width):
    """Histogram mode -- the MOST COMMON value, which for an off-equilibrium dataset
    is the right estimate of the relaxed value. The mean/median are pulled upward by
    the strained tail (median max|F| = 3.4 eV/Å across this dataset)."""
    if len(x) == 0:
        return float("nan")
    nb = max(int(round((hi - lo) / width)), 1)
    h, edges = np.histogram(x, bins=nb, range=(lo, hi))
    if h.sum() == 0:
        return float(np.median(x))
    k = int(np.argmax(h))
    return float(0.5 * (edges[k] + edges[k + 1]))


@dataclass
class Reference:
    """Learned equilibrium geometry: r0 per element pair, theta0 per (centre element,
    coordination number). Keying angles by coordination captures hybridisation
    directly -- 4-coordinate carbon sits near 109.5 deg, 3-coordinate near 120."""

    min_count: int = 30
    r0_: dict = field(default_factory=dict)          # (Za,Zb,dega,degb) -> r0
    r0_pair_: dict = field(default_factory=dict)     # (Za,Zb) -> r0   (fallback)
    th0_: dict = field(default_factory=dict)
    r0_fallback_: dict = field(default_factory=dict, repr=False)

    def fit(self, scans):
        from ase.data import covalent_radii

        bl, blp, ang = {}, {}, {}
        for s in scans:
            for a, b, da, db, r in zip(s["za"], s["zb"], s["da"], s["db"], s["r"]):
                bl.setdefault((int(a), int(b), int(da), int(db)), []).append(r)
                blp.setdefault((int(a), int(b)), []).append(r)
            for c, d, t in zip(s["zc"], s["deg"], s["th"]):
                ang.setdefault((int(c), int(d)), []).append(t)

        for k, v in bl.items():
            if len(v) >= self.min_count:
                self.r0_[k] = _mode(np.asarray(v), 0.4, 3.5, 0.005)
        for k, v in blp.items():
            if len(v) >= self.min_count:
                self.r0_pair_[k] = _mode(np.asarray(v), 0.4, 3.5, 0.005)
        for k, v in ang.items():
            if len(v) >= self.min_count:
                self.th0_[k] = _mode(np.asarray(v), 0.0, 180.0, 0.5)
        # Anything too rare to estimate falls back down the hierarchy and finally to
        # the covalent-radius sum, so a rare-element bond contributes ~0 strain rather
        # than a spurious spike.
        cr = np.asarray(covalent_radii, dtype=float)
        self.r0_fallback_ = {"cr": cr}
        print(f"[strain] reference: {len(self.r0_)} (pair, coordination) r0, "
              f"{len(self.r0_pair_)} pair-only r0, "
              f"{len(self.th0_)} (element, coordination) theta0 "
              f"(min_count={self.min_count})")
        return self

    def r0(self, a, b, da=None, db=None):
        """Coordination-resolved reference, falling back to element-pair, then to the
        covalent-radius sum."""
        if da is not None:
            v = self.r0_.get((int(a), int(b), int(da), int(db)))
            if v is not None:
                return v
        v = self.r0_pair_.get((int(a), int(b)))
        if v is not None:
            return v
        cr = self.r0_fallback_["cr"]
        return float(cr[int(a)] + cr[int(b)])

    def theta0(self, c, d):
        v = self.th0_.get((int(c), int(d)))
        if v is not None:
            return v
        return {1: 180.0, 2: 180.0, 3: 120.0, 4: 109.5}.get(int(d), 109.5)

    def report(self):
        """Sanity print: learned r0 against textbook covalent bond lengths. The
        coordination-resolved rows are the ones to read -- C-C at (4,4) should land on
        the sp3 single bond ~1.53, at (3,3) on aromatic ~1.39, at (2,2) on the triple
        bond ~1.20. A pair-only reference collapses all three into one mixture-mode."""
        from ase.data import chemical_symbols as S

        book_pair = {(6, 6): "1.20-1.53 by order", (1, 6): 1.09, (6, 8): 1.43,
                     (6, 7): 1.47, (1, 8): 0.97, (1, 7): 1.01, (6, 16): 1.82}
        print("[strain] learned r0 (Å), pair-only vs textbook:")
        for k in sorted(book_pair):
            if k in self.r0_pair_:
                print(f"          {S[k[0]]}-{S[k[1]]:<2} {self.r0_pair_[k]:.3f}"
                      f"  (textbook ~{book_pair[k]})")
        book_cc = {(4, 4): 1.53, (3, 3): 1.39, (2, 2): 1.20, (3, 4): 1.50}
        rows = [(k, v) for k, v in self.r0_.items() if k[0] == 6 and k[1] == 6]
        if rows:
            print("[strain] C-C by coordination (bond-order resolution):")
            for k, v in sorted(rows, key=lambda kv: -kv[1]):
                b = book_cc.get((k[2], k[3]))
                print(f"          deg({k[2]},{k[3]}) {v:.3f}"
                      + (f"  (textbook ~{b})" if b else ""))


# ----------------------------- features -------------------------------------

FEATURE_NAMES = [
    "bond_sq",        # sum delta^2                       (harmonic strain energy)
    "bond_abs",       # sum |delta|
    "bond_compress",  # sum delta^2 over delta < 0        (compression)
    "bond_extend",    # sum delta^2 over delta > 0        (extension)
    "bond_max",       # max |delta|                       (the LOCALISED signal)
    "bond_p95",       # 95th pctile |delta|
    "angle_sq",       # sum (theta - theta0)^2  [deg^2]
    "angle_max",      # max |theta - theta0|
    "clash",          # steric contact term
    "n_bonds",        # context
]


def features_one(scan, ref):
    """-> (extensive_vector, intensive_vector). Extensive = raw sums (~eV-like);
    intensive = divided by atom count (the existing descriptor convention)."""
    n = max(scan["n_atoms"], 1)
    r = scan["r"].astype(float)
    if len(r):
        r0 = np.array([ref.r0(a, b, da, db) for a, b, da, db
                       in zip(scan["za"], scan["zb"], scan["da"], scan["db"])])
        d = r - r0
        neg, pos = d[d < 0], d[d > 0]
        ad = np.abs(d)
        f_bond = [float((d ** 2).sum()), float(ad.sum()),
                  float((neg ** 2).sum()), float((pos ** 2).sum()),
                  float(ad.max()), float(np.percentile(ad, 95))]
    else:
        f_bond = [0.0] * 6

    t = scan["th"].astype(float)
    if len(t):
        t0 = np.array([ref.theta0(c, dg) for c, dg in zip(scan["zc"], scan["deg"])])
        dt = t - t0
        f_ang = [float((dt ** 2).sum()), float(np.abs(dt).max())]
    else:
        f_ang = [0.0, 0.0]

    ext = np.array(f_bond + f_ang + [scan["clash"], float(scan["n_bonds"])])
    # max/p95 are already localised (not sums), so dividing them by n would only
    # re-encode size; keep them identical in both variants.
    idx_local = {FEATURE_NAMES.index("bond_max"), FEATURE_NAMES.index("bond_p95"),
                 FEATURE_NAMES.index("angle_max")}
    inten = np.array([v if i in idx_local else v / n for i, v in enumerate(ext)])
    return ext, inten


# ----------------------------- the gate -------------------------------------


def _r2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100 or x[m].std() <= 0 or y[m].std() <= 0:
        return float("nan")
    r = np.corrcoef(x[m], y[m])[0, 1]
    return float(r * r)


def _joint_r2(X, y, holdout=True, seed=0):
    """Joint linear R². With ``holdout`` (the default) this is a HELD-OUT number:
    fit on a random half, score on the other. An in-sample multivariate fit with ~20
    features would be inflated by ~p/n, which is material for the per-subset slices
    (~600 rows each) -- so the reported headline must be the held-out one."""
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if m.sum() < 100:
        return float("nan")
    Xm, ym = X[m], y[m]
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-12)
    A = np.column_stack([np.ones(len(Xm)), Xm])
    if not holdout:
        beta, *_ = np.linalg.lstsq(A, ym, rcond=None)
        return float(1.0 - np.var(ym - A @ beta) / np.var(ym))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ym))
    h = len(ym) // 2
    tr, te = idx[:h], idx[h:]
    if len(te) < 50:
        return float("nan")
    beta, *_ = np.linalg.lstsq(A[tr], ym[tr], rcond=None)
    resid = ym[te] - A[te] @ beta
    return float(1.0 - np.var(resid) / np.var(ym[te]))


def run(n_sample, src, seed, cutoff_mult, min_count, workers, per_subset):
    from fairchem.core.datasets import AseDBDataset

    from .stats import load_all

    print(f"[strain] loading census columns + y ...")
    c, names, counts, subset, y, _ = load_all(
        columns=["gidx", "fmax", "frms", "n_atoms", "data_id"], need_y=True)
    if y is None:
        raise SystemExit("families.npz missing -- run `python -m data_exploration.families`")
    N = len(y)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(N, size=min(n_sample, N), replace=False))
    print(f"[strain] sampled {len(sel):,} of {N:,} structures (seed={seed})")

    ds = AseDBDataset({"src": src})
    t0 = time.perf_counter()
    scans = []
    for k, gi in enumerate(sel):
        scans.append(scan_one(ds.get_atoms(int(c["gidx"][gi])), cutoff_mult))
        if (k + 1) % 2000 == 0:
            print(f"[strain]   scanned {k+1}/{len(sel)}  "
                  f"({(time.perf_counter()-t0)/(k+1)*1e3:.1f} ms/mol)")
    print(f"[strain] scanned {len(scans):,} in {time.perf_counter()-t0:.0f}s")

    ref = Reference(min_count=min_count).fit(scans)
    ref.report()

    EXT = np.vstack([features_one(s, ref)[0] for s in scans])
    INT = np.vstack([features_one(s, ref)[1] for s in scans])
    ys, fmax, sub = y[sel], c["fmax"][sel].astype(float), subset[sel]

    # benchmark on THIS sample so the comparison is apples-to-apples
    bench = _r2(fmax, ys)
    print(f"\n[strain] BENCHMARK on this sample: max|F| -> y  R² = {bench:.4f} "
          f"(4M census value 0.397)")
    print(f"[strain] n_atoms -> y  R² = {_r2(c['n_atoms'][sel].astype(float), ys):.5f} "
          f"(input-side columns are all < 0.001 at 4M)")

    print(f"\n{'feature':>14} | {'ext->y':>8} {'int->y':>8} | {'ext->fmax':>10} {'int->fmax':>10}")
    print("-" * 60)
    for i, nm in enumerate(FEATURE_NAMES):
        print(f"{nm:>14} | {_r2(EXT[:, i], ys):>8.4f} {_r2(INT[:, i], ys):>8.4f} | "
              f"{_r2(EXT[:, i], fmax):>10.4f} {_r2(INT[:, i], fmax):>10.4f}")

    jx, ji = _joint_r2(EXT, ys), _joint_r2(INT, ys)
    BOTH = np.hstack([EXT, INT])
    both = _joint_r2(BOTH, ys)
    print("-" * 60)
    print(f"{'JOINT (held-out)':>14} | {jx:>8.4f} {ji:>8.4f} | "
          f"{_joint_r2(EXT, fmax):>10.4f} {_joint_r2(INT, fmax):>10.4f}")
    print(f"{'JOINT ext+int':>14} | {both:>8.4f} held-out  "
          f"({_joint_r2(BOTH, ys, holdout=False):.4f} in-sample)")
    print(f"{'':>14} | vs max|F| benchmark {bench:.4f} -> recovers "
          f"{100*both/max(bench,1e-9):.0f}% of the force-diagnostic signal")

    if per_subset:
        print(f"\n[strain] per data_id (joint ext+int -> y, HELD-OUT):")
        print(f"{'subset':>16}{'n':>8}{'strain_r2':>11}{'fmax_r2':>10}{'ratio':>8}")
        for s in sorted(set(sub.tolist())):
            m = sub == s
            if m.sum() < 400:
                continue
            sr = _joint_r2(np.hstack([EXT[m], INT[m]]), ys[m])
            fr = _r2(fmax[m], ys[m])
            print(f"{s:>16}{int(m.sum()):>8}{sr:>11.4f}{fr:>10.4f}"
                  f"{sr/fr if fr > 1e-6 else float('nan'):>8.2f}")

    out = os.path.join(CACHE, f"strain_{len(sel)}.npz")
    np.savez(out, sel=sel, ext=EXT, inten=INT, y=ys, fmax=fmax, subset=sub,
             names=np.array(FEATURE_NAMES))
    print(f"\n[strain] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="strain-from-positions gate")
    ap.add_argument("--n", type=int, default=20_000, help="molecules to sample")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff-mult", type=float, default=1.2)
    ap.add_argument("--min-count", type=int, default=30,
                    help="min observations before a pair/angle gets a learned reference")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-per-subset", action="store_true")
    a = ap.parse_args()
    run(a.n, a.src, a.seed, a.cutoff_mult, a.min_count, a.workers, not a.no_per_subset)


if __name__ == "__main__":
    main()
