"""
soap_screen.py  (data_exploration)
==================================
Is an off-the-shelf descriptor (SOAP) a better geometry channel than the hand-rolled
partial-RDF + angle channels?

Measured constraints that shape this screen (see the plan and the dscribe assessment):

* **MBTR is unusable here** -- dscribe 2.1.2's ``System.from_atoms`` passes 15 positional
  args to ``ase.Atoms.__init__``, whose signature changed in ase 3.28, so every MBTR
  call raises. Downgrading ase is a project-wide change. Also note our ``rdf`` channel
  at top_k=6/n_rdf=24 is 504 dims and MBTR k=2 at n=24 is *also* 504 -- they are the
  same object, so MBTR would buy nothing even if it ran.
* **ACE is not installed** (no pyace/Julia). It is the only candidate that natively
  reaches 4-body, i.e. the only one that could subsume our ``torsion`` channel.
* **ACSF** is per-atom with no pooling provided and O(n_species^2) angular terms.
* **SOAP is the one viable option** -- and only with ``compression="mu2"``, which makes
  the length independent of the species count (252 dims for ANY number of elements).
  Uncompressed SOAP at n_max=8/l_max=6 is 8,232 dims for 6 species and 1,545,460 for
  the dataset's 83 -- and 8,232 columns breaks ``reduce.py`` at 200k (it wraps X in a
  CSR and then does ``X.multiply(X)``; nnz 1.65e9 forces int64 indices, ~40 GB peak).

SOAP is 3-body and charge-blind, so it can only ever replace ``rdf`` + ``angle`` --
never ``torsion`` (4-body) or ``elec`` (Loewdin charges). This screen therefore asks the
narrow question it can actually answer: **does SOAP beat rdf+angle on the same
molecules?**

Scoring mirrors ``strain.py`` exactly (held-out joint linear R^2 against the cached 4M
target), and by default it runs on the SAME sampled molecules as the strain gate so
every number in this module is directly comparable.

    python -m data_exploration.soap_screen --n 20000
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .census import CACHE
from .strain import _joint_r2, _r2


def _pls_r2(X, y, n_comp=10, seed=0):
    """Held-out R² of **PLS(n_comp) with pareto scaling** -- the reduction the
    production pipeline actually applies (``reduce.SparsePLS``, 10 components,
    ``scaling="pareto"``).

    This, not a raw linear fit, is the fair comparison: descriptors here differ ~6x in
    width (252 vs 1602), and an unregularised least-squares fit on the wider one would
    be penalised purely for its dimensionality (p/n approaches 1), not for carrying less
    signal. Reducing both to the same 10 dims measures what the GP will actually see.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score

    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, yy = np.asarray(X, float)[m], np.asarray(y, float)[m]
    if len(yy) < 200:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yy))
    h = len(yy) // 2
    tr, te = idx[:h], idx[h:]
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / np.sqrt(sd)          # pareto, as in production
    k = min(n_comp, Xs.shape[1], len(tr) - 1)
    pls = PLSRegression(n_components=k, scale=False).fit(Xs[tr], yy[tr])
    return float(r2_score(yy[te], pls.predict(Xs[te]).ravel()))


def _soap(species, r_cut, n_max, l_max, compression):
    from dscribe.descriptors import SOAP

    kw = dict(species=species, r_cut=r_cut, n_max=n_max, l_max=l_max,
              periodic=False, average="inner", sparse=False)
    if compression != "off":
        kw["compression"] = {"mode": compression}
    return SOAP(**kw)


def run(n_sample, src, seed, r_cut, n_max, l_max, modes, per_subset, reuse, max_dims):
    from fairchem.core.datasets import AseDBDataset

    from .stats import load_all
    from wl_gp2scale.geometry_features import SparseGeometryFeaturizer

    c, names, counts, subset, y, _ = load_all(
        columns=["gidx", "fmax", "n_atoms", "data_id"], need_y=True)
    if y is None:
        raise SystemExit("families.npz missing -- run `python -m data_exploration.families`")

    cached = os.path.join(CACHE, f"strain_{n_sample}.npz")
    if reuse and os.path.exists(cached):
        sel = np.load(cached)["sel"]
        print(f"[soap] reusing the strain gate's sample ({len(sel):,}) -> numbers are "
              f"directly comparable to strain.py")
    else:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(y), size=min(n_sample, len(y)), replace=False))
        print(f"[soap] drew a fresh sample of {len(sel):,}")

    ds = AseDBDataset({"src": src})
    t0 = time.perf_counter()
    atoms = [ds.get_atoms(int(c["gidx"][g])) for g in sel]
    print(f"[soap] read {len(atoms):,} structures in {time.perf_counter()-t0:.0f}s")
    ys, sub = y[sel], subset[sel]

    # dscribe RAISES on an unseen species (unlike our featurizer, which folds off-vocab
    # atoms into a bucket), so the species list must cover every element present.
    species = sorted({int(z) for a in atoms for z in a.get_atomic_numbers()})
    print(f"[soap] {len(species)} distinct species in the sample")

    results = {}

    # --- baseline: the current hand-rolled rdf+angle channels -------------------
    t0 = time.perf_counter()
    gf = SparseGeometryFeaturizer(channels=("rdf", "angle")).fit(atoms)
    Xg = gf.transform(atoms)
    tg = time.perf_counter() - t0
    results["rdf+angle (ours)"] = (Xg, tg)

    # --- SOAP variants -----------------------------------------------------------
    for mode in modes:
        try:
            d = _soap(species, r_cut, n_max, l_max, mode)
            nf = d.get_number_of_features()
            if nf > max_dims:
                # crossover/mu1nu1/off grow with the species count, and this sample
                # carries dozens of elements. Beyond ~max_dims the descriptor both
                # exceeds reduce.py's envelope at 200k and stops being scoreable here
                # (p approaches n), so skip rather than report a meaningless number.
                print(f"[soap] mode={mode:>9} -> {nf} features for {len(species)} "
                      f"species: SKIPPED (> --max-dims {max_dims}; not viable at 200k "
                      f"and p/n too large to score at N={len(atoms)})")
                continue
            print(f"[soap] mode={mode:>9} -> {nf} features; featurising ...")
            t0 = time.perf_counter()
            X = np.vstack([d.create(a, n_jobs=1) for a in atoms])
            results[f"SOAP-{mode}"] = (X, time.perf_counter() - t0)
        except Exception as e:
            print(f"[soap] mode={mode} FAILED: {type(e).__name__}: {e}")

    # --- score --------------------------------------------------------------------
    print(f"\n{'descriptor':>20}{'dims':>8}{'ms/mol':>9}{'PLS10_R2':>10}{'linear_R2':>11}")
    print("-" * 58)
    print("  (PLS10 is the headline -- it is what the production pipeline reduces to,")
    print("   and it compares descriptors of different width fairly)")
    for k, (X, t) in results.items():
        print(f"{k:>20}{X.shape[1]:>8}{1e3*t/len(atoms):>9.2f}"
              f"{_pls_r2(X, ys):>10.4f}{_joint_r2(X, ys):>11.4f}")

    if per_subset:
        keys = list(results)
        print(f"\n[soap] per data_id (held-out PLS10 R² -> y):")
        hdr = "".join(f"{k[:13]:>15}" for k in keys)
        print(f"{'subset':>16}{'n':>7}{hdr}")
        for s in sorted(set(sub.tolist())):
            m = sub == s
            if m.sum() < 400:
                continue
            row = "".join(f"{_pls_r2(results[k][0][m], ys[m]):>15.4f}" for k in keys)
            print(f"{s:>16}{int(m.sum()):>7}{row}")

    out = os.path.join(CACHE, f"soap_screen_{len(sel)}.npz")
    np.savez(out, sel=sel, y=ys, subset=sub,
             **{f"X_{k.replace(' ', '_').replace('+', '_')}": v[0]
                for k, v in results.items()})
    print(f"\n[soap] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="SOAP vs hand-rolled geometry screen")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-cut", type=float, default=6.0)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--l-max", type=int, default=6)
    ap.add_argument("--modes", nargs="+", default=["mu2", "crossover"],
                    help="SOAP compression modes ('off' is 8232-dim at 6 species and "
                         "1.5M at 83 -- do not use at scale)")
    ap.add_argument("--no-reuse", action="store_true",
                    help="draw a fresh sample instead of reusing the strain gate's")
    ap.add_argument("--max-dims", type=int, default=5000,
                    help="skip SOAP modes wider than this (not viable at 200k, and p/n too large to score here)")
    ap.add_argument("--no-per-subset", action="store_true")
    a = ap.parse_args()
    run(a.n, a.src, a.seed, a.r_cut, a.n_max, a.l_max, a.modes,
        not a.no_per_subset, not a.no_reuse, a.max_dims)


if __name__ == "__main__":
    main()
