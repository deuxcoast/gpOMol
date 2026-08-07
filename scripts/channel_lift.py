"""Do two channels pick the SAME neighbours? And does that change with N?

WHY THIS EXISTS. A product kernel's support is an INTERSECTION, so the only thing that
matters about a pair of channels is whether they agree about who is a neighbour. The
scalar for that is the LIFT:

    lift(A,B) = P(near in A and near in B) / [ P(near in A) * P(near in B) ]

lift = 1 means the channels pick neighbours independently -- a genuinely second view of
the data. The intersection is then much sparser than either channel alone, so at matched
density both radii can be widened a lot, and that widening is the mechanism by which the
product kernel improved sigma*. lift >> 1 means the second channel is re-describing the
first and the intersection buys little.

Marginals are held EQUAL across channels (each radius set so the same fraction of pairs
counts as near) so lift compares the SHAPE of the neighbourhoods rather than their size.
Only same-category pairs are counted, because the kernel zeroes everything else.

LIFT IS NECESSARY BUT NOT SUFFICIENT. A pure NOISE channel also scores lift 1.0 --
random neighbourhoods are independent of everything. So the second table reports each
channel's STANDALONE predictive power (k-NN regression on the M1 residual, same-category
neighbours only). A channel earns a kernel block by being independent AND predictive.

WHY IT IS PARAMETERISED BY N. lift was measured at 20k, and the wl x strain product
kernel that it motivated beat wl x geom at 20k (+0.0176 +- 0.0061 paired, 6 seeds) but
NOT at 80k (-0.0021, 1 seed). If strain's neighbourhoods become more redundant with
geometry as N grows, or if strain simply saturates as a 17-feature descriptor while geom
keeps resolving finer structure, that would explain the decline mechanically rather than
leaving it as single-seed noise. Run at every N whose embedding you have:

    python scripts/channel_lift.py --n 20000 40000 80000
"""
import argparse
import itertools
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEAN_CHAN = ("wl", "geom", "strain")
# marginals to evaluate lift at. 0.0125 is the production operating point: the kernel
# targets K=200 in-support neighbours, and 200 / n_train is ~1.25e-2 at 20k.
MARGINALS = (0.005, 0.0125, 0.02, 0.05)


def lift_table(Zc, cat, idx, marginal):
    """lift for every channel pair at one common marginal, same-category pairs only."""
    same = squareform(cat[idx][:, None] == cat[idx][None, :], checks=False)
    D = {n: pdist(Z[idx])[same] for n, Z in Zc.items()}
    near = {n: d < np.quantile(d, marginal) for n, d in D.items()}
    out = {}
    for a, b in itertools.combinations(sorted(Zc), 2):
        both = float((near[a] & near[b]).mean())
        indep = float(near[a].mean() * near[b].mean())
        out[(a, b)] = both / max(indep, 1e-300)
    return out, float(same.mean())


def knn_r2(Ztr, Zte, r_tr, r_te, cat_tr, cat_te, k=10, sample=1000, seed=0):
    """Held-out R^2 of a k-NN average over SAME-CATEGORY train neighbours.

    The cheap standalone-predictive-power probe. Not a GP: the point is to rank channels
    against each other on one axis, and to catch the case where a channel looks
    independent only because it is uninformative."""
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(Zte), size=min(sample, len(Zte)), replace=False)
    D = cdist(Zte[ti], Ztr)
    D[cat_te[ti][:, None] != cat_tr[None, :]] = np.inf
    kk = min(k, D.shape[1] - 1)
    nn = np.argpartition(D, kk, axis=1)[:, :kk]
    pred = r_tr[nn].mean(axis=1)
    ok = np.isfinite(pred)
    return float(r2_score(r_te[ti][ok], pred[ok])) if ok.sum() > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb-dir", default="cache")
    ap.add_argument("--sample", type=int, default=4000,
                    help="train rows sampled for the pairwise-distance tables. 4000 rows "
                         "is ~8e6 pairs per channel, tens of MB; it does NOT need to "
                         "grow with N because lift is a property of the pair geometry")
    a = ap.parse_args()

    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean

    lifts, knn = {}, {}
    for n in a.n:
        path = os.path.join(a.emb_dir, "emb_%d_s%d.npz" % (n, a.seed))
        if not os.path.exists(path):
            print("[lift] SKIP n=%d: no %s" % (n, path))
            continue
        d = np.load(path, allow_pickle=True)
        Ztr = {c: d[c + "_tr"] for c in MEAN_CHAN}
        Zte = {c: d[c + "_te"] for c in MEAN_CHAN}

        ds = get_data(src="train_4M", n=n, seed=0)
        tr, te = train_test_split(np.arange(len(ds.atoms)), test_size=0.2,
                                  random_state=a.seed)
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
        y_tr, y_te = ds.y[tr], ds.y[te]

        Zm_tr = np.hstack([Ztr[c] for c in MEAN_CHAN])
        Zm_te = np.hstack([Zte[c] for c in MEAN_CHAN])
        mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=None)
        r_tr = y_tr - mean.predict(Zm_tr, size=None)
        r_te = y_te - mean.predict(Zm_te, size=None)

        rng = np.random.default_rng(0)
        idx = rng.choice(len(cat_tr), size=min(a.sample, len(cat_tr)), replace=False)
        for m in MARGINALS:
            lifts[(n, m)], frac_same = lift_table(Ztr, cat_tr, idx, m)
        knn[n] = {c: knn_r2(Ztr[c], Zte[c], r_tr, r_te, cat_tr, cat_te)
                  for c in MEAN_CHAN}
        print("[lift] n=%d: %d train rows, %d sampled, %.1f%% of sampled pairs "
              "same-category" % (n, len(cat_tr), len(idx), 100 * frac_same))

    ns = sorted({k[0] for k in lifts})
    if not ns:
        raise SystemExit("no embeddings found -- nothing to report")
    pairs = sorted({p for v in lifts.values() for p in v})

    print("\n" + "=" * 78)
    print("LIFT  (1.0 = independent neighbourhoods; >>1 = one view described twice)")
    print("=" * 78)
    for m in MARGINALS:
        print("\n  marginal = %.4f%s" % (m, "   <- production operating point (K=200)"
                                         if m == 0.0125 else ""))
        print("    %-18s" % "pair" + "".join("%12s" % ("n=%dk" % (n // 1000))
                                             for n in ns))
        for p in pairs:
            row = "".join("%12.1f" % lifts[(n, m)][p] if (n, m) in lifts else "%12s" % "--"
                          for n in ns)
            print("    %-18s" % (p[0] + " x " + p[1]) + row)

    print("\n" + "=" * 78)
    print("STANDALONE PREDICTIVE POWER  (k-NN R^2 on the M1 residual, k=10)")
    print("=" * 78)
    print("  lift ~ 1 is necessary but NOT sufficient: a noise channel also scores 1.0.")
    print("    %-18s" % "channel" + "".join("%12s" % ("n=%dk" % (n // 1000))
                                            for n in ns))
    for c in MEAN_CHAN:
        print("    %-18s" % c + "".join("%12.4f" % knn[n][c] if n in knn
                                        else "%12s" % "--" for n in ns))
    print("\n  A channel earns a kernel block by being INDEPENDENT and PREDICTIVE.")
    print("  If a channel's R^2 stalls while another's climbs with N, the product that")
    print("  paired them will lose its advantage at scale even though the lift is flat.")


if __name__ == "__main__":
    main()
