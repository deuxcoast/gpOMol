"""
orthogonality_organic.py  (data_exploration)
============================================
The strongest form of the question: **does a genuinely accurate chemical bond graph
carry signal that is orthogonal to the geometry channel?**

`orthogonality.py` compared ASE against `rdkit_ctd`, but connect-the-dots is itself
a distance heuristic (28.4% of same-molecule conformer families still split). The
object that actually deserves the name "chemical graph" is `DetermineBonds`
(xyz2mol): valence-satisfying, and **0.0%** conformer split. It is unusable across
`train_4M` -- 70% failure, 98% on metal complexes, 1.5 s/mol -- but that average is
entirely metal complexes grinding in the combinatorial search. On the organic
subsets it is **1.7 ms/mol at 91% success**, which makes the experiment affordable
on the third of the dataset where the question can be asked at all.

So: restrict to the organic `data_id`s, keep only rows all three perceivers can
handle (so every arm sees identical molecules), and compare

    solo R^2,  increment over geometry,  redundancy with geometry

for ASE 1.2, `rdkit_ctd`, and `rdkit_bonds`, under BOTH reducers -- supervised PLS
(production) and unsupervised SVD (the control that removes PLS's habit of fitting
spurious directions on weak features).

    python -m data_exploration.orthogonality_organic --n 20000 --seeds 42 7 123
"""

from __future__ import annotations

import argparse

import numpy as np

from .orthogonality import _canon_corr, _explained_by, _ols_r2, _residual, _ridge_r2

ORGANIC = ("reactivity", "ani2x", "trans1x", "rgd", "spice", "orbnet_denali")
PERCEIVERS = {"ase": "ase", "rdkit_ctd": "rdkit_ctd", "rdkit_bonds": "rdkit_bonds"}


def _usable(atoms_list):
    """Rows every perceiver can handle -- the arms must see identical molecules."""
    from wl_gp2scale.wl_features import build_graph

    ok = np.ones(len(atoms_list), dtype=bool)
    for k, a in enumerate(atoms_list):
        for p in PERCEIVERS.values():
            try:
                adj, _ = build_graph(a, 1.2, p)
                if sum(len(x) for x in adj) == 0:
                    ok[k] = False
            except Exception:
                ok[k] = False
                break
    return ok


def run(n=20000, seeds=(42, 7, 123), src="train_4M", dim=10, data_seed=0):
    from sklearn.model_selection import train_test_split

    from wl_gp2scale.data import get_data
    from wl_gp2scale.pipeline import GeometryPipeline, WLGPPipeline
    from wl_gp2scale.wl_features import SparseWLFeaturizer

    from .orthogonality import _svd_embed

    ds = get_data(src=src, n=n, seed=data_seed)
    sub = np.array([a.info.get("data_id", "?") for a in ds.atoms])
    org = np.flatnonzero(np.isin(sub, ORGANIC))
    print(f"[org] {len(org):,} organic-subset rows of {len(sub):,}")
    atoms_org = [ds.atoms[i] for i in org]
    ok = _usable(atoms_org)
    keep = org[ok]
    print(f"[org] {ok.sum():,} usable by ALL perceivers "
          f"({100 * ok.mean():.0f}% of the organic rows)")

    res = {r: {p: {"solo": [], "inc": [], "resid": [], "red": [], "cc": []}
               for p in PERCEIVERS} | {"geom": []} for r in ("pls", "svd")}

    for seed in seeds:
        tr, te = train_test_split(np.arange(len(keep)), test_size=0.2,
                                  random_state=seed)
        atr = [ds.atoms[keep[i]] for i in tr]
        ate = [ds.atoms[keep[i]] for i in te]
        ytr, yte = ds.y[keep[tr]], ds.y[keep[te]]
        ctr = ds.data_id[keep[tr]]

        pg = GeometryPipeline(top_k=6, channels=("rdf", "angle", "torsion", "elec"),
                              pls_components=dim, cutoff_percentile=None,
                              scaling="pareto")
        Gtr, Gte = pg.fit(atr, ytr, ctr), None
        Gte = pg.transform(ate)

        for red in ("pls", "svd"):
            g, _ = _ridge_r2(Gtr, ytr, Gte, yte)
            res[red]["geom"].append(g)
            rtr, rte = _residual(Gtr, ytr, Gte, yte)
            print(f"\n--- seed {seed}  reducer={red}  geom alone {g:.4f} ---")
            for name, perc in PERCEIVERS.items():
                if red == "pls":
                    pw = WLGPPipeline(depth=3, min_count=2, pls_components=dim,
                                      cutoff_percentile=None, scaling="pareto",
                                      vocab_sample=0, perceiver=perc)
                    Wtr, Wte = pw.fit(atr, ytr, ctr), pw.transform(ate)
                else:
                    f = SparseWLFeaturizer(depth=3, min_count=2, perceiver=perc)
                    f.fit(atr)
                    Wtr, Wte = _svd_embed(f.transform(atr), f.transform(ate), dim)
                solo, _ = _ridge_r2(Wtr, ytr, Wte, yte)
                both, _ = _ridge_r2(np.hstack([Gtr, Wtr]), ytr,
                                    np.hstack([Gte, Wte]), yte)
                rr, _ = _ridge_r2(Wtr, rtr, Wte, rte)
                redu, _ = _explained_by(None, Gtr, Wtr, Gte, Wte)
                cc = float(np.mean(_canon_corr(Wte, Gte)))
                for k, v in (("solo", solo), ("inc", both - g), ("resid", rr),
                             ("red", redu), ("cc", cc)):
                    res[red][name][k].append(v)
                print(f"  {name:<12} solo {solo:+.4f}  increment {both - g:+.4f}  "
                      f"resid {rr:+.4f}  redundancy {redu:5.1%}  CC {cc:.3f}")

    print("\n================ SUMMARY (organic subset, ridge, held-out) ================")
    for red in ("pls", "svd"):
        print(f"\nreducer = {red}   (geom alone {np.mean(res[red]['geom']):.4f})")
        print(f"{'perceiver':<14}{'solo':>9}{'increment':>11}{'resid':>9}"
              f"{'redundancy':>12}{'CC':>7}")
        for name in PERCEIVERS:
            d = res[red][name]
            print(f"{name:<14}{np.mean(d['solo']):>+9.4f}{np.mean(d['inc']):>+11.4f}"
                  f"{np.mean(d['resid']):>+9.4f}{np.mean(d['red']):>11.1%}"
                  f"{np.mean(d['cc']):>7.3f}")
        base = np.array(res[red]["ase"]["inc"])
        for name in PERCEIVERS:
            if name == "ase":
                continue
            dd = np.array(res[red][name]["inc"]) - base
            print(f"  paired increment({name}) - increment(ase) = {dd.mean():+.4f} "
                  f"+/- {dd.std(ddof=1) if len(dd) > 1 else 0:.4f}   "
                  + " ".join(f"{x:+.4f}" for x in dd))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument("--dim", type=int, default=10)
    a = p.parse_args()
    run(n=a.n, seeds=a.seeds, dim=a.dim)


if __name__ == "__main__":
    main()
