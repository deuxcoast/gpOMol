"""Does more PLS rank reduce the NUGGET? The descriptor screen for sparsity.

WHY THE NUGGET IS THE QUANTITY. The marginal likelihood prefers a dense kernel for 9 of
10 chemical families, and the reason is visible in the variogram: gamma/sill is already
0.41-0.81 (mean 0.60) in the NEAREST distance bin. Between the two closest molecules in
descriptor space, ~60% of the energy variance is already there. A kernel whose nearest
neighbours are that dissimilar gains almost nothing from a short radius, so pooling
everything really is the better model and the likelihood says so. Sparsity cannot be
likelihood-preferred until that number comes down, and it is a DESCRIPTOR property, not
a kernel one.

Truncating the cached 10-component embedding showed the nugget still falling at 20 dims
with no floor in sight (0.79 at 2 dims -> 0.604 at 20). This goes past it.

TWO THINGS THIS CONTROLS FOR.

  * THE TARGET IS HELD FIXED at the M1 residual from the PRODUCTION 10-dim embedding.
    Refitting the prior mean at each rank would change the residual underneath the
    measurement, and a nugget ratio computed against a moving target measures the mean
    as much as the descriptor. We are asking one question: how well does the descriptor
    resolve short-range energy differences.

  * NESTEDNESS IS VERIFIED, NOT ASSUMED. PLS components are built sequentially with
    deflation, so the first k columns of a rank-R fit should equal a rank-k fit, which
    is what makes one expensive featurisation serve every dimension. If that fails the
    whole sweep is invalid, so it is checked against the cached production embedding
    before any sweeping happens.

Screened on BOTH axes, for the same reason the channel screen is: a descriptor can lower
the nugget by smearing everything together, so a falling nugget only counts if k-NN
predictive power holds up or improves alongside it.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from percat_radius import FEAT, MEAN_CHAN                       # noqa: E402

KERNEL_CHAN = ("wl", "geom")


def variogram_stats(Z, r, cat, n_bins=12, sample=2500, seed=0):
    """Mean over families of (nugget, drift), within-family pairs only.

    nugget = gamma(nearest bin)/sill, drift = gamma(farthest bin)/sill. Equal-count
    bins, so neither end is a sparse-tail artifact."""
    rng = np.random.default_rng(seed)
    nug, dri = [], []
    for c in np.unique(cat):
        idx = np.where(cat == c)[0]
        if len(idx) < 150:
            continue
        idx = rng.choice(idx, size=min(sample, len(idx)), replace=False)
        D = pdist(Z[idx])
        sv = 0.5 * pdist(r[idx][:, None]) ** 2
        B = np.array_split(np.argsort(D), n_bins)
        g = np.array([sv[b].mean() for b in B])
        sill = float(np.var(r[idx]))
        nug.append(g[0] / sill); dri.append(g[-1] / sill)
    return float(np.mean(nug)), float(np.mean(dri))


def knn_r2(Ztr, Zte, r_tr, r_te, cat_tr, cat_te, k=10, sample=1500, seed=0):
    """Held-out k-NN R^2 over same-category neighbours -- the second screening axis."""
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(Zte), size=min(sample, len(Zte)), replace=False)
    D = cdist(Zte[ti], Ztr)
    D[cat_te[ti][:, None] != cat_tr[None, :]] = np.inf
    kk = max(min(k, D.shape[1] - 1), 1)
    nn = np.argpartition(D, kk, axis=1)[:, :kk]
    pred = r_tr[nn].mean(axis=1)
    ok = np.isfinite(pred)
    return float(r2_score(r_te[ti][ok], pred[ok])) if ok.sum() > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=40,
                    help="PLS components to fit per channel; the sweep truncates this")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[1, 2, 3, 4, 6, 8, 10, 14, 20, 28, 40],
                    help="per-channel dimensions to report (must be <= --rank)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None)
    ap.add_argument("--emb-dir", default="cache")
    ap.add_argument("--out", default="cache/nugget_sweep.npz")
    a = ap.parse_args()

    from wl_gp2scale.data import get_data
    from wl_gp2scale.pipeline import build_channels
    from wl_gp2scale.reduce import LinearEmbeddingMean

    prod = os.path.join(a.emb_dir, f"emb_{a.n}_s{a.seed}.npz")
    if not os.path.exists(prod):
        raise SystemExit(f"need the production embedding {prod} to hold the target "
                         f"fixed and to verify nestedness; build it first")
    P = np.load(prod, allow_pickle=True)

    ds = get_data(src="train_4M", n=a.n, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=0.2, random_state=a.seed)
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    y_tr, y_te = ds.y[tr], ds.y[te]

    # THE FIXED TARGET: the M1 residual from the production 10-dim embedding.
    Zm_tr = np.hstack([P[f"{n}_tr"] for n in MEAN_CHAN])
    Zm_te = np.hstack([P[f"{n}_te"] for n in MEAN_CHAN])
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=None)
    r_tr = y_tr - mean.predict(Zm_tr, size=None)
    r_te = y_te - mean.predict(Zm_te, size=None)
    print(f"[sweep] target held fixed: M1 residual from the 10-dim production "
          f"embedding, var={np.var(r_tr):.3f}")

    hi = os.path.join(a.emb_dir, f"embR{a.rank}_{a.n}_s{a.seed}.npz")
    if os.path.exists(hi):
        E = np.load(hi, allow_pickle=True)
        print(f"[sweep] reused {hi}")
    else:
        feat = dict(FEAT); feat["pls"] = a.rank
        t0 = time.time()
        if a.scheduler_file:
            from wl_gp2scale.pipeline import connect_dask
            cl = connect_dask(a.scheduler_file, n_workers=a.workers)
        else:
            from distributed import Client
            cl = Client(n_workers=a.workers, threads_per_worker=1, processes=False,
                        dashboard_address=None, silence_logs=50)
        try:
            emb = build_channels([ds.atoms[i] for i in tr], y_tr, cat_tr,
                                 [ds.atoms[i] for i in te], cat_te,
                                 set(KERNEL_CHAN), client=cl, target_neighbors=0,
                                 cutoff_pct=25.0, **feat)
        finally:
            cl.close()
        np.savez(hi, **{f"{n}_{s}": emb[n][f"Z{s}"] for n in emb for s in ("tr", "te")})
        E = np.load(hi, allow_pickle=True)
        print(f"[sweep] featurised at rank {a.rank} in {time.time() - t0:.0f}s -> {hi}")

    # NESTEDNESS GATE -- the whole sweep rests on it
    print("\n[sweep] nestedness check: first 10 components of the rank-%d fit vs the "
          "production rank-10 fit" % a.rank)
    ok = True
    for n in KERNEL_CHAN:
        A, B = E[f"{n}_tr"][:, :10], P[f"{n}_tr"]
        # sign of a PLS component is arbitrary; compare per-column correlation magnitude
        cc = [abs(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(B.shape[1])]
        worst = float(np.min(cc))
        print(f"    {n:>5}: min |corr| over the 10 components = {worst:.4f}")
        ok &= worst > 0.99
    if not ok:
        raise SystemExit(
            "NESTEDNESS FAILED: the rank-%d fit's leading components differ from the "
            "rank-10 fit, so truncation is NOT equivalent to refitting and this sweep "
            "would compare different embeddings at every dimension. Fit each dimension "
            "separately instead." % a.rank)
    print("    -> nested; truncation is equivalent to refitting, sweep is valid\n")

    dims = [d for d in a.dims if d <= a.rank]
    print(f"{'dims/chan':>10} {'total':>7} {'NUGGET':>9} {'drift':>8} {'kNN R2':>9}")
    print("-" * 48)
    rows = []
    for k in dims:
        Ztr = np.hstack([E[f"{n}_tr"][:, :k] for n in KERNEL_CHAN])
        Zte = np.hstack([E[f"{n}_te"][:, :k] for n in KERNEL_CHAN])
        nug, dri = variogram_stats(Ztr, r_tr, cat_tr)
        kr = knn_r2(Ztr, Zte, r_tr, r_te, cat_tr, cat_te)
        rows.append((k, nug, dri, kr))
        print(f"{k:>10} {k * len(KERNEL_CHAN):>7} {nug:>9.4f} {dri:>8.3f} {kr:>9.4f}")
    R = np.array(rows)
    np.savez(a.out, dims=R[:, 0], nugget=R[:, 1], drift=R[:, 2], knn_r2=R[:, 3],
             n=a.n, seed=a.seed, rank=a.rank)
    best = int(np.argmin(R[:, 1]))
    print(f"\n  lowest nugget {R[best, 1]:.4f} at {int(R[best, 0])} dims/channel "
          f"(production is 10, nugget {R[dims.index(10), 1]:.4f})"
          if 10 in dims else "")
    print("  A falling nugget only counts if kNN R^2 holds up: a descriptor can lower it")
    print("  by smearing molecules together, which helps the variogram and nothing else.")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
