"""
orthogonality.py  (data_exploration)
====================================
Is the WL channel's signal *redundant* with the geometry channel, and does a
chemically-accurate bond graph carry signal that the loose one does not?

`REPORT.md` §16b measured each perceiver's WL channel **standalone** and concluded
against RDKit. That is the wrong test for the production model, which is an
*additive* kernel: what matters is a channel's **marginal** contribution given the
geometry channel, not its solo R². The hypothesis this module tests:

    ASE-perceived WL scores well alone because it leaks geometry -- but geometry is
    already carried, explicitly and better, by the `geom` channel. A purer bond
    graph (rdkit_ctd) scores worse alone yet may be more ORTHOGONAL to geometry, and
    so contribute more on top of it.

Everything here is held-out and linear (OLS on the production PLS embeddings), which
makes it cheap enough to run every combination. The GP arms follow separately; if
the increments below are flat, the GP will not rescue them.

Reported per WL variant:

``solo``        R^2(y | wl)                      -- what §16b measured
``increment``   R^2(y | geom + wl) - R^2(y | geom) -- **the number that matters**
``on residual`` R^2 of the geom-residual explained by wl alone
``redundancy``  how much of the wl embedding geometry can already predict, and the
                canonical correlations between the two embeddings

    python -m data_exploration.orthogonality --n 20000 --seeds 42 7 123
"""

from __future__ import annotations

import argparse

import numpy as np


def _ols_r2(Xtr, ytr, Xte, yte):
    """Held-out R^2 of a least-squares fit with an intercept."""
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    B = np.hstack([np.ones((len(Xte), 1)), Xte])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    resid = yte - B @ beta
    return float(1.0 - np.var(resid) / np.var(yte)), beta


def _ridge_r2(Xtr, ytr, Xte, yte, alphas=(1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100),
              folds=4, seed=0):
    """Held-out R^2 of a ridge fit whose penalty is chosen by CV **on train only**.

    This exists to steelman the hypothesis. Appending ten poorly-conditioned columns
    to a good ten-column fit can lower held-out R^2 through pure variance inflation,
    even when those columns carry a little independent signal -- so an unregularised
    increment can be negative for reasons that have nothing to do with orthogonality.
    Ridge removes that failure mode: if a channel carries orthogonal signal, a
    CV-chosen penalty lets the fit keep it and shrink the rest.
    """
    rng = np.random.default_rng(seed)
    n, p = Xtr.shape
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    ym = ytr.mean()
    fold = rng.permutation(n) % folds
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        sc = []
        for f in range(folds):
            i_tr, i_va = fold != f, fold == f
            A = Ztr[i_tr]
            G = A.T @ A + a * np.eye(p)
            b = np.linalg.solve(G, A.T @ (ytr[i_tr] - ytr[i_tr].mean()))
            pred = Ztr[i_va] @ b + ytr[i_tr].mean()
            sc.append(1.0 - np.var(ytr[i_va] - pred) / np.var(ytr[i_va]))
        if np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    G = Ztr.T @ Ztr + best_a * np.eye(p)
    b = np.linalg.solve(G, Ztr.T @ (ytr - ym))
    resid = yte - (Zte @ b + ym)
    return float(1.0 - np.var(resid) / np.var(yte)), best_a


def _residual(Xtr, ytr, Xte, yte):
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    B = np.hstack([np.ones((len(Xte), 1)), Xte])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return ytr - A @ beta, yte - B @ beta


def _canon_corr(A, B):
    """Canonical correlations between two embeddings (both column-centred)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def _explained_by(target, source_tr, target_tr, source_te, target_te):
    """Mean held-out R^2 of predicting each column of `target` from `source`."""
    out = []
    for k in range(target_tr.shape[1]):
        r2, _ = _ols_r2(source_tr, target_tr[:, k], source_te, target_te[:, k])
        out.append(r2)
    return float(np.mean(out)), np.round(out, 3)


def _svd_embed(Xtr, Xte, dim, seed=0):
    """UNSUPERVISED reduction (truncated SVD), as a control for the reducer.

    `SparsePLS` is supervised: it picks directions maximising covariance with y on
    TRAIN. On a feature set that carries little real signal it will happily fit
    spurious directions that do not replicate -- which shows up as a NEGATIVE
    held-out R^2, exactly what rdkit_ctd produces under PLS. That is a property of
    the reducer, not proof that the features are empty, so the orthogonality question
    has to be re-asked with a reduction that never sees y.
    """
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=dim, random_state=seed)
    return svd.fit_transform(Xtr), svd.transform(Xte)


def run(n=20000, seeds=(42, 7, 123), src="train_4M", dim=10,
        geom_channels=("rdf", "angle", "torsion", "elec"), data_seed=0,
        reducer="pls"):
    from sklearn.model_selection import train_test_split

    from wl_gp2scale.data import get_data
    from wl_gp2scale.pipeline import GeometryPipeline, WLGPPipeline

    ds = get_data(src=src, n=n, seed=data_seed)
    variants = {"wl_ase": "ase", "wl_rdkit": "rdkit_ctd"}
    acc = {k: [] for k in
           ["geom", "geom_ridge", "wl_ase", "wl_rdkit", "geom+wl_ase",
            "geom+wl_rdkit", "geom+wl_ase_ridge", "geom+wl_rdkit_ridge",
            "resid_wl_ase", "resid_wl_rdkit", "resid_wl_ase_ridge",
            "resid_wl_rdkit_ridge", "red_wl_ase", "red_wl_rdkit",
            "cc_wl_ase", "cc_wl_rdkit"]}

    for seed in seeds:
        print(f"\n########## split seed {seed} ##########")
        idx = np.arange(len(ds))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        atr = [ds.atoms[i] for i in tr]
        ate = [ds.atoms[i] for i in te]
        ytr, yte = ds.y[tr], ds.y[te]
        ctr = ds.data_id[tr]

        pg = GeometryPipeline(top_k=6, channels=tuple(geom_channels),
                              pls_components=dim, cutoff_percentile=None,
                              scaling="pareto")
        Gtr = pg.fit(atr, ytr, ctr)
        Gte = pg.transform(ate)

        emb = {}
        for name, perc in variants.items():
            if reducer == "pls":
                pw = WLGPPipeline(depth=3, min_count=2, pls_components=dim,
                                  cutoff_percentile=None, scaling="pareto",
                                  vocab_sample=0, perceiver=perc)
                emb[name] = (pw.fit(atr, ytr, ctr), pw.transform(ate))
            else:
                from wl_gp2scale.wl_features import SparseWLFeaturizer

                f = SparseWLFeaturizer(depth=3, min_count=2, perceiver=perc)
                f.fit(atr)
                emb[name] = _svd_embed(f.transform(atr), f.transform(ate), dim)

        r2_g, _ = _ols_r2(Gtr, ytr, Gte, yte)
        acc["geom"].append(r2_g)
        rtr, rte = _residual(Gtr, ytr, Gte, yte)
        print(f"  geom alone                R2 = {r2_g:.4f}")

        r2_g_ridge, _ = _ridge_r2(Gtr, ytr, Gte, yte)
        acc["geom_ridge"].append(r2_g_ridge)
        print(f"  geom alone (ridge)        R2 = {r2_g_ridge:.4f}")

        for name in variants:
            Wtr, Wte = emb[name]
            solo, _ = _ols_r2(Wtr, ytr, Wte, yte)
            both, _ = _ols_r2(np.hstack([Gtr, Wtr]), ytr, np.hstack([Gte, Wte]), yte)
            both_r, alpha = _ridge_r2(np.hstack([Gtr, Wtr]), ytr,
                                      np.hstack([Gte, Wte]), yte)
            res, _ = _ols_r2(Wtr, rtr, Wte, rte)
            res_r, _ = _ridge_r2(Wtr, rtr, Wte, rte)
            red, per = _explained_by(None, Gtr, Wtr, Gte, Wte)
            cc = _canon_corr(Wte, Gte)
            acc[name].append(solo)
            acc[f"geom+{name}"].append(both)
            acc[f"geom+{name}_ridge"].append(both_r)
            acc[f"resid_{name}"].append(res)
            acc[f"resid_{name}_ridge"].append(res_r)
            acc[f"red_{name}"].append(red)
            acc[f"cc_{name}"].append(float(np.mean(cc)))
            print(f"  {name:<12} solo {solo:.4f} | geom+{name:<9} {both:.4f} "
                  f"(increment {both - r2_g:+.4f}) | RIDGE {both_r:.4f} "
                  f"(increment {both_r - r2_g_ridge:+.4f}, alpha={alpha:g}) | "
                  f"resid {res:+.4f}/ridge {res_r:+.4f} | "
                  f"geom explains {100 * red:.0f}% of its columns | CC {np.mean(cc):.3f}")

    print("\n================= SUMMARY over "
          f"{len(seeds)} seeds (held-out, linear on PLS-{dim}) =================")
    g = np.array(acc["geom"])
    print(f"{'geom alone':<26}{g.mean():>9.4f}")
    gr = np.array(acc["geom_ridge"])
    print(f"{'geom alone (ridge)':<26}{gr.mean():>9.4f}")
    print(f"\n{'variant':<22}{'solo':>9}{'incr OLS':>10}{'incr RIDGE':>12}"
          f"{'resid ridge':>13}{'redundancy':>12}{'mean CC':>9}")
    for name in variants:
        solo = np.array(acc[name])
        inc = np.array(acc[f"geom+{name}"]) - g
        incr = np.array(acc[f"geom+{name}_ridge"]) - gr
        print(f"{name:<22}{solo.mean():>9.4f}{inc.mean():>+10.4f}{incr.mean():>+12.4f}"
              f"{np.mean(acc['resid_' + name + '_ridge']):>+13.4f}"
              f"{np.mean(acc['red_' + name]):>11.1%}{np.mean(acc['cc_' + name]):>9.3f}")
    inc_a = np.array(acc["geom+wl_ase_ridge"]) - gr
    inc_r = np.array(acc["geom+wl_rdkit_ridge"]) - gr
    d = inc_r - inc_a
    print(f"\npaired per-seed: increment(rdkit) - increment(ase) = "
          f"{d.mean():+.4f} +/- {d.std(ddof=1) if len(d) > 1 else 0:.4f}  "
          f"per-seed " + " ".join(f"{x:+.4f}" for x in d) + "   [RIDGE increments]")
    print("\nreading: 'increment' is the marginal value of the WL channel GIVEN "
          "geometry.\n'redundancy' is how much of the WL embedding geometry alone "
          "can already predict.")
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument("--src", default="train_4M")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--reducer", default="pls", choices=["pls", "svd"],
                   help="svd = unsupervised control, removes the PLS confound")
    a = p.parse_args()
    run(n=a.n, seeds=a.seeds, src=a.src, dim=a.dim, reducer=a.reducer)


if __name__ == "__main__":
    main()
