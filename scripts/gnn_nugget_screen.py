"""Does a learned (GNN) representation beat our hand-crafted one on the NUGGET?

THE BASELINE TO BEAT, measured on the same seed, same split, same target:

    our descriptor, 10 dims/channel (production, 20 total)   nugget 0.5988   kNN R2 0.140
    our descriptor, 20 dims/channel (40 total)               nugget 0.5826   kNN R2 0.161
    our descriptor, 40 dims/channel (80 total)               nugget 0.5757   kNN R2 0.157

The floor is ~0.57 and it is representation-limited: quadrupling the PLS rank moved it
0.023, and L1/L0.5/Linf metrics all did worse on the axis that matters. If a GNN latent
space does not clear that, the descriptor direction is closed and we stop.

THIS IS AN UPPER BOUND. The model trained on all of OMol25, so it has seen the test
molecules -- and the PLS compression below is fit against the same target, which makes
it doubly optimistic. Fine for a go/no-go; not reportable. Read a NEGATIVE result as
decisive and a POSITIVE result as "worth doing properly with a clean split".

BOTH AXES, as always: a representation can lower the nugget by smearing molecules
together, which flatters the variogram and helps nothing. That failure mode has fired
once already in this project (L1 lowered the nugget while costing a third of the kNN
signal), so a nugget improvement only counts if kNN R^2 holds up beside it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nugget_dim_sweep import variogram_stats, knn_r2, gamma_by_rank  # noqa: E402

MEAN_CHAN = ("wl", "geom", "strain")
KERNEL_CHAN = ("wl", "geom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="cache/gnn_esen_sm_direct_all_omol_20000.npz")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb-dir", default="cache")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20, 40, 128],
                    help="PLS dimensions to compress the GNN embedding to; 128 = raw")
    a = ap.parse_args()

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import train_test_split
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean

    G = np.load(a.gnn, allow_pickle=True)
    P = np.load(os.path.join(a.emb_dir, f"emb_{a.n}_s{a.seed}.npz"), allow_pickle=True)
    ds = get_data(src="train_4M", n=a.n, seed=0)
    tr, te = train_test_split(np.arange(len(ds.atoms)), test_size=0.2,
                              random_state=a.seed)
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    y_tr, y_te = ds.y[tr], ds.y[te]

    # SAME FIXED TARGET as the dimension sweep: the M1 residual from the production
    # 10-dim embedding. Refitting the mean per representation would move the residual
    # underneath the measurement and confound descriptor quality with mean quality.
    Zm_tr = np.hstack([P[f"{n}_tr"] for n in MEAN_CHAN])
    Zm_te = np.hstack([P[f"{n}_te"] for n in MEAN_CHAN])
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=None)
    r_tr = y_tr - mean.predict(Zm_tr, size=None)
    r_te = y_te - mean.predict(Zm_te, size=None)

    print(f"[screen] target: M1 residual from the production embedding, "
          f"var={np.var(r_tr):.3f}")

    # OUR BASELINE IS RECOMPUTED HERE, not hardcoded. g(k) is NOT a scale-free property
    # of a descriptor: it depends on how densely the family was sampled, because the
    # k-th nearest of 2500 points is closer than the k-th nearest of 1500. Quoting a
    # baseline computed at a different sample size or a different family set silently
    # compares two different quantities -- which is exactly what an earlier hardcoded
    # header did, making the GNN look 0.12 better at g(1) than it is.
    print("  g(k) = semivariance at the k-th nearest same-family neighbour / sill.")
    print("  BOTH ARMS COMPUTED HERE with identical sampling, so the columns compare.")
    print("  g(1) is finest-scale resolution; g(200) is where the kernel operates.\n")
    print(f"{'descriptor':>12} {'dims':>6} {'g(1)':>8} {'g(10)':>8} {'g(200)':>8} "
          f"{'binned':>8} {'drift':>7} {'kNN R2':>8}")
    print("-" * 70)
    for d in (10, 20, 40):
        if 2 * d > sum(P[f"{c}_tr"].shape[1] for c in KERNEL_CHAN):
            continue
        Otr = np.hstack([P[f"{c}_tr"][:, :d] for c in KERNEL_CHAN])
        Ote = np.hstack([P[f"{c}_te"][:, :d] for c in KERNEL_CHAN])
        nug, dri = variogram_stats(Otr, r_tr, cat_tr)
        g = gamma_by_rank(Otr, r_tr, cat_tr)
        kr = knn_r2(Otr, Ote, r_tr, r_te, cat_tr, cat_te)
        print(f"{'ours':>12} {2 * d:>6} {g[1]:>8.4f} {g[10]:>8.4f} {g[200]:>8.4f} "
              f"{nug:>8.4f} {dri:>7.3f} {kr:>8.4f}")
    print()
    for pool in ("mean_pool", "sum_pool"):
        E = np.asarray(G[pool], dtype=float)
        Etr, Ete = E[tr], E[te]
        # standardise before PLS: raw GNN channels differ wildly in scale and PLS is
        # not scale-invariant
        mu, sd = Etr.mean(0), np.where(Etr.std(0) > 0, Etr.std(0), 1.0)
        Etr, Ete = (Etr - mu) / sd, (Ete - mu) / sd
        for k in a.dims:
            if k >= E.shape[1]:
                Ztr, Zte, lab = Etr, Ete, f"{E.shape[1]} raw"
            else:
                pls = PLSRegression(n_components=k, scale=False).fit(Etr, r_tr)
                Ztr, Zte, lab = pls.transform(Etr), pls.transform(Ete), str(k)
            nug, dri = variogram_stats(Ztr, r_tr, cat_tr)
            g = gamma_by_rank(Ztr, r_tr, cat_tr)
            kr = knn_r2(Ztr, Zte, r_tr, r_te, cat_tr, cat_te)
            print(f"{'gnn ' + pool[:4]:>12} {lab:>6} {g[1]:>8.4f} {g[10]:>8.4f} "
                  f"{g[200]:>8.4f} {nug:>8.4f} {dri:>7.3f} {kr:>8.4f}")
    print("\n  'ours' rows use the SAME sampling as the gnn rows, so g(k) is comparable")
    print("  across them. A lower g ONLY counts if kNN R2 holds up beside it -- L1")
    print("  lowered the binned nugget 0.027 while costing a third of the kNN signal.")
    print("\n  REMINDER: this model trained on OMol25 including our test molecules, and")
    print("  the PLS above is fit against the same target. Upper bound only.")


if __name__ == "__main__":
    main()
