"""
analyze_residual.py
===================
Diagnose WHY the hybrid descriptor shows ~zero predictive skill on train_4M,
and whether there is any recoverable signal. Three questions:

  1. Where does the residual variance live? Between source subsets (systematic
     offsets the extensive mean is missing) or within them (a representation
     ceiling)? Plus: does molecule size / charge still explain residual (mean
     under-fit)?

  2. Was PCA hiding the signal? Compare unsupervised PCA+ridge vs SUPERVISED PLS
     at matched component counts, scored by held-out R^2 under a MOLECULE-DISJOINT
     split (group by composition, so conformers can't leak train->test and inflate
     the score). A full-feature ridge is the linear-signal reference. If PLS beats
     PCA and reaches positive R^2, signal exists and the reduction was the problem;
     if PLS is also ~0, that's evidence of a genuine ceiling.

  3. Are the ~44% multi-molecule records the problem? Re-score single-component
     vs multi-component vs all.

Model-based scoring (ridge/PLS) is used throughout, so nothing is O(n^2); this
runs comfortably at a few x 10^4.

Usage
-----
    python analyze_residual.py --src ../train_4M --n 40000
"""

import argparse
from collections import Counter

import numpy as np
from extensive_mean import ExtensiveEnergyModel
from features import HybridFeatureAssembler
from gp_fit import build_graph, charge_spin_features, n_connected_components
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Fresh load capturing the metadata the .npz doesn't have
# ----------------------------------------------------------------------------


def formula_key(Z) -> str:
    """Composition string from atomic numbers, e.g. '1x18_6x19_7x4_8x5_16x1'.
    Conformers of a molecule share it -> grouping by it removes conformer leakage
    (conservative: also groups isomers, which only makes the test harder)."""
    return "_".join(f"{z}x{c}" for z, c in sorted(Counter(int(x) for x in Z).items()))


def load_for_analysis(src, n, seed=0, charge_key="lowdin_charges", cutoff_mult=1.2):
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    N = len(ds)
    print(f"train_4M: {N:,}; sampling {min(n, N):,}")
    idxs = np.random.default_rng(seed).choice(N, size=min(n, N), replace=False)

    Z, G, P, Q, Y, NC, SP = [], [], [], [], [], [], []
    data_id, groups, natoms, single = [], [], [], []
    skipped = 0
    for i in idxs:
        atoms = ds.get_atoms(int(i))
        q = atoms.info.get(charge_key)
        if q is None or np.any(np.isnan(np.asarray(q, dtype=float))):
            skipped += 1
            continue
        Zi = atoms.get_atomic_numbers().tolist()
        adj, lab = build_graph(atoms, cutoff_mult)
        Z.append(Zi)
        P.append(atoms.get_positions())
        Q.append(np.asarray(q, float))
        Y.append(float(atoms.get_potential_energy()))
        NC.append(float(atoms.info.get("charge", 0)))
        SP.append(float(atoms.info.get("spin", 1)))
        G.append((adj, lab))
        data_id.append(atoms.info.get("data_id", "unknown"))
        groups.append(formula_key(Zi))
        natoms.append(len(Zi))
        single.append(n_connected_components(adj) == 1)
    print(f"  kept {len(Y):,} (skipped {skipped})")

    return dict(
        Z=Z,
        graphs=G,
        positions=P,
        charges=Q,
        y=np.array(Y),
        net_charges=np.array(NC),
        spins=np.array(SP),
        data_id=np.array(data_id),
        groups=np.array(groups),
        n_atoms=np.array(natoms),
        single=np.array(single, dtype=bool),
    )


# ----------------------------------------------------------------------------
# Shared scoring helper
# ----------------------------------------------------------------------------


def grouped_cv_r2(make_model, X, y, groups, n_splits=5) -> float:
    """Out-of-group R^2: every fold's test compositions are absent from its train,
    so conformers cannot leak. R^2 <= 0 means no better than predicting the mean."""
    ng = len(np.unique(groups))
    k = min(n_splits, ng)
    if k < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=k)
    yhat = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        yhat[te] = np.ravel(m.predict(X[te]))
    ok = ~np.isnan(yhat)
    return float(r2_score(y[ok], yhat[ok]))


# ----------------------------------------------------------------------------
# Diagnostic 1 — where does residual variance live?
# ----------------------------------------------------------------------------


def variance_decomposition(resid, data_id, n_atoms, net_charges, spins, groups):
    print("\n" + "=" * 64)
    print("1. RESIDUAL VARIANCE DECOMPOSITION")
    print("=" * 64)
    r = resid
    grand = r.mean()
    total_ss = np.sum((r - grand) ** 2)
    print(f"  total residual var = {np.var(r):.3g}  (std {np.std(r):.3g})   n={len(r)}")

    # between- vs within-subset (eta^2 = fraction of variance from subset identity)
    between_ss = 0.0
    print(f"\n  {'data_id':<22}{'n':>7}{'mean':>10}{'std':>9}")
    for did in sorted(set(data_id), key=lambda d: -np.sum(data_id == d)):
        m = data_id == did
        rg = r[m]
        between_ss += len(rg) * (rg.mean() - grand) ** 2
        print(f"  {str(did):<22}{len(rg):>7}{rg.mean():>10.3g}{rg.std():>9.3g}")
    eta2 = between_ss / total_ss
    print(
        f"\n  between-subset fraction (eta^2) = {eta2:.3f}  "
        f"-> {'MEAN under-fits: add per-subset/richer references' if eta2 > 0.3 else 'subset offsets are minor'}"
    )

    # residual still explained by size / charge / spin? (mean under-fit check)
    Z = np.column_stack([n_atoms, n_atoms**2, np.abs(net_charges), spins]).astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    r2_size = grouped_cv_r2(lambda: Ridge(alpha=1.0), Z, r, groups)
    print(
        f"  held-out R^2 of residual ~ [n_atoms, n_atoms^2, |charge|, spin] = {r2_size:.3f}"
        f"  -> {'size/charge leakage remains in the mean' if r2_size > 0.05 else 'mean has removed size/charge'}"
    )


# ----------------------------------------------------------------------------
# Diagnostic 2 — was PCA hiding the signal? PCA+ridge vs PLS, disjoint split
# ----------------------------------------------------------------------------


def pca_vs_pls(Xr, resid, groups, ks=(2, 5, 10, 25, 50)):
    print("\n" + "=" * 64)
    print("2. PCA+ridge  vs  PLS   (held-out R^2, composition-disjoint 5-fold)")
    print("=" * 64)
    print(f"  {'k':>4}{'PCA+ridge':>12}{'PLS':>10}")
    best_pls = -np.inf
    for k in ks:
        if k >= Xr.shape[1]:
            continue
        pca = grouped_cv_r2(
            lambda: Pipeline(
                [("pca", PCA(n_components=k)), ("ridge", Ridge(alpha=1.0))]
            ),
            Xr,
            resid,
            groups,
        )
        pls = grouped_cv_r2(lambda: PLSRegression(n_components=k), Xr, resid, groups)
        best_pls = max(best_pls, pls)
        print(f"  {k:>4}{pca:>12.3f}{pls:>10.3f}")
    full = grouped_cv_r2(lambda: Ridge(alpha=1.0), Xr, resid, groups)
    print(f"  full-feature ridge (reference upper bound on linear signal): {full:.3f}")
    verdict = (
        "signal EXISTS and PCA was hiding it -> switch to a supervised reduction"
        if best_pls > 0.05
        else "PLS also ~0 -> evidence of a genuine representation ceiling for this descriptor"
    )
    print(f"\n  => {verdict}")
    return best_pls


# ----------------------------------------------------------------------------
# Diagnostic 3 — single-molecule vs multi-molecule records
# ----------------------------------------------------------------------------


def single_vs_multi(Xr, resid, groups, single, k=25):
    print("\n" + "=" * 64)
    print("3. SINGLE- vs MULTI-MOLECULE RECORDS  (PLS held-out R^2)")
    print("=" * 64)
    print(f"  {'subset':<12}{'n':>8}{'resid_var':>12}{'PLS R^2':>10}")
    for name, mask in [
        ("single", single),
        ("multi", ~single),
        ("all", np.ones_like(single)),
    ]:
        if mask.sum() < 100:
            print(f"  {name:<12}{int(mask.sum()):>8}  (too few)")
            continue
        r2 = grouped_cv_r2(
            lambda: PLSRegression(n_components=min(k, Xr.shape[1] - 1)),
            Xr[mask],
            resid[mask],
            groups[mask],
        )
        print(
            f"  {name:<12}{int(mask.sum()):>8}{np.var(resid[mask]):>12.3g}{r2:>10.3f}"
        )


# ----------------------------------------------------------------------------


def analyze(data):
    # extensive mean (element counts + charge + spin) -> intensive residual
    ctx = list(zip(data["net_charges"], data["spins"]))
    mm = ExtensiveEnergyModel(extra_feature_fn=charge_spin_features).fit(
        data["Z"], data["y"], extra_context=ctx
    )
    resid = mm.residual(data["Z"], data["y"], extra_context=ctx)

    # standardized raw hybrid features (pre-PCA: PLS reduces from here)
    asm = HybridFeatureAssembler()
    Xr = asm.fit_transform(data["graphs"], data["positions"], data["charges"])
    print(
        f"[features] raw dim={Xr.shape[1]}  n={Xr.shape[0]}  "
        f"unique compositions (groups)={len(np.unique(data['groups']))}"
    )

    variance_decomposition(
        resid,
        data["data_id"],
        data["n_atoms"],
        data["net_charges"],
        data["spins"],
        data["groups"],
    )
    pca_vs_pls(Xr, resid, data["groups"])
    single_vs_multi(Xr, resid, data["groups"], data["single"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--charge_key", default="lowdin_charges")
    ap.add_argument("--cutoff_mult", type=float, default=1.2)
    a = ap.parse_args()
    data = load_for_analysis(a.src, a.n, a.seed, a.charge_key, a.cutoff_mult)
    analyze(data)


if __name__ == "__main__":
    main()
