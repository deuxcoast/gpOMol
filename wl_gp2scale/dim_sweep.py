"""
dim_sweep.py  (wl_gp2scale)
===========================
Confirm on the ACTUAL gp2Scale Wendland kernel what the truncated-R^2 diagnostic
(reduce.truncated_r2_curve) predicted on OLS: that a low-dim slice of the natural-
scaled PLS embedding is MORE predictive than the full 10-D, because the tail
components are noise. Sweeps embedding dim x split-seed and reports predictive R^2.

For each split seed:
  * fit vocab (ALL train -> 0% train OOV) + natural-scaled PLS(pls_components) on
    train, transform test -> a single 10-D embedding shared by every dim slice, so
    the dims-1..d comparison is apples-to-apples (SIMPLS components are sequential:
    the first d columns are identical whatever the total).
  * for each dim d: slice Z[:, :d], RECALIBRATE the cutoff at that dim (distances
    change with d; same percentile keeps the in-support fraction comparable), build
    the frozen-hyperparameter gp2Scale GP, predict test mean, score R^2 vs truth.
    Also prints the dense OLS R^2 on the same slice as a cross-check against the
    reduce.truncated_r2_curve numbers.

Frozen hyperparameters only (signal_var = var(y_tr), fixed cutoff) -- no training,
so each build is CG solves only. The cutoff is held at one percentile across dims;
that is the honest default but it does NOT hold neighbour count / conditioning fixed
across dims, so the per-dim median-neighbour line is printed to keep that visible.
"""

from __future__ import annotations

import argparse

import numpy as np

from .cutoff import recalibrate, sparsity_report
from .pipeline import (build_gp, predict, release_gp, sort_by_category,
                       with_category_tag, WLGPPipeline)
from .reduce import regression_r2


def _one_gp_r2(Z_tr, y_tr, cat_tr, Z_te, y_te, cat_te, cutoff, dim, client, args):
    """Build a frozen-hp gp2Scale GP on the dim-sliced embedding, predict test mean,
    return predictive R^2. Category-tagged + sorted so cross-category blocks skip.

    TEST must carry its REAL categories: the kernel zeroes cross-category covariance,
    so a test molecule tagged with the wrong category draws only on training molecules
    of that wrong category -> its posterior mean collapses to the prior. (Tagging all
    test rows with a single dummy category is only valid when TRAIN uses that same
    single category, as in validate.sparse_vs_dense_parity.)"""
    from sklearn.metrics import r2_score

    Xtr = with_category_tag(Z_tr[:, :dim], cat_tr)
    Xtr, y_tr_s, _ = sort_by_category(Xtr, y_tr)
    Xte = with_category_tag(Z_te[:, :dim], cat_te)
    sv = float(np.var(y_tr))

    gp, _ = build_gp(
        Xtr, y_tr_s, cutoff, dim, client,
        signal_var=sv, jitter=args.jitter, batch_size=args.batch_size,
        compute_device="cpu", device=args.device, linalg_mode=args.linalg,
    )
    m, _ = predict(gp, Xte, batch=args.pred_batch, variance=False)
    r2 = float(r2_score(y_te, m))
    del gp
    release_gp(client)
    return r2


def run(args, client):
    from sklearn.model_selection import train_test_split

    from .data import get_data

    ds = get_data(src=args.src, n=args.n, seed=args.data_seed)
    dims = sorted({int(d) for d in args.dims})
    rows = []

    for seed in args.seeds:
        print(f"\n########## split seed {seed} ##########")
        idx = np.arange(len(ds))
        tr, te = train_test_split(idx, test_size=args.test_size, random_state=seed)
        atoms_tr = [ds.atoms[i] for i in tr]
        atoms_te = [ds.atoms[i] for i in te]
        y_tr, y_te = ds.y[tr], ds.y[te]
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]

        # one embedding per seed; dims slice it. vocab_sample=0 -> vocab on ALL train.
        pipe = WLGPPipeline(
            depth=args.depth, min_count=args.min_count,
            pls_components=args.pls, cutoff_percentile=args.cutoff_pct,
            scaling=args.scaling, vocab_sample=0,
        )
        Z_tr = pipe.fit(atoms_tr, y_tr, cat_tr, client=client)
        Z_te = pipe.transform(atoms_te, client=client)

        for d in dims:
            if args.cutoff is not None:
                cutoff = float(args.cutoff)          # absolute radius, same for all dims
            else:
                cutoff, _ = recalibrate(Z_tr[:, :d], percentile=args.cutoff_pct, dim=d)
            rep = sparsity_report(Z_tr[:, :d], cutoff, dim=d, data_id=cat_tr)
            ols = regression_r2(Z_tr[:, :d], y_tr, Z_te[:, :d], y_te)
            gp_r2 = _one_gp_r2(Z_tr, y_tr, cat_tr, Z_te, y_te, cat_te, cutoff, d,
                               client, args)
            print(f"[sweep] seed={seed} dim={d:>2}  cutoff={cutoff:.4f}  "
                  f"median_nbr={rep['median_neighbors']:.0f}  "
                  f"density={rep['density']:.2e}  OLS_R2={ols:.4f}  GP_R2={gp_r2:.4f}")
            rows.append((seed, d, cutoff, rep['median_neighbors'], ols, gp_r2))

    print("\n================= SUMMARY =================")
    print(f"{'seed':>5} {'dim':>4} {'cutoff':>8} {'med_nbr':>8} {'OLS_R2':>8} {'GP_R2':>8}")
    for s, d, c, nb, ols, gp in rows:
        print(f"{s:>5} {d:>4} {c:>8.4f} {nb:>8.0f} {ols:>8.4f} {gp:>8.4f}")
    # per-dim mean/std across seeds
    print("\n--- GP_R2 across seeds, by dim ---")
    for d in dims:
        vals = np.array([gp for (s, dd, c, nb, ols, gp) in rows if dd == d])
        print(f"  dim={d:>2}: GP_R2 mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"n={len(vals)}  values={np.round(vals,4)}")
    if args.out:
        np.savez(args.out, rows=np.array(rows, dtype=float),
                 dims=np.array(dims), seeds=np.array(args.seeds))
        print(f"\n[sweep] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="wl_gp2scale embedding-dim x seed sweep")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--data-seed", type=int, default=0, help="subset draw seed (frozen in cache)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                    help="TRAIN/TEST split seeds (the stability axis)")
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--pls", type=int, default=10, help="PLS components fit (>= max dim)")
    ap.add_argument("--scaling", default="pareto",
                    choices=["pareto", "standard", "center"],
                    help="SparsePLS column pre-weighting (default pareto)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="absolute compact-support radius; overrides --cutoff-pct "
                         "(applied to every --dims slice)")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--pred-batch", type=int, default=2000)
    ap.add_argument("--linalg", default="sparseCG")
    ap.add_argument("--device", default="cuda", help="OUR kernel's torch device")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None)
    ap.add_argument("--out", default=None, help="optional .npz of the summary rows")
    args = ap.parse_args()
    if max(args.dims) > args.pls:
        ap.error(f"--dims max {max(args.dims)} exceeds --pls {args.pls}")

    if args.scheduler_file:
        from .pipeline import connect_dask
        client = connect_dask(args.scheduler_file, n_workers=args.workers)
    else:
        from distributed import Client
        client = Client(n_workers=args.workers, threads_per_worker=1)
        client.wait_for_workers(args.workers)
        print(f"[sweep] local dask: {args.workers} workers")

    try:
        run(args, client)
    finally:
        client.close()
    print("[sweep] done.")


if __name__ == "__main__":
    main()
