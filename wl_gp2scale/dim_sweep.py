"""
dim_sweep.py  (wl_gp2scale)
===========================
Confirm on the ACTUAL gp2Scale Wendland kernel what the truncated-R^2 diagnostic
(reduce.truncated_r2_curve) predicted on OLS. Sweeps embedding dim x cutoff x
split-seed and reports predictive R^2 + realised density.

Cutoff sweep: pass ``--cutoffs 0.16 0.22 0.28 ...`` to test several ABSOLUTE radii
while REUSING one embedding per seed (featurise + PLS are cutoff-independent; only
the kernel changes). This is the cheap way to pick the cutoff -- look for the largest
GP_R2 whose median in-support neighbour count is still well-conditioned (tens, not
hundreds). ``--cutoff`` (single) and ``--cutoff-pct`` (per-dim percentile) still work.

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
import time

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
    single category, as in validate.sparse_vs_dense_parity.)

    ``--prior-mean linear`` fits an OLS mean on the embedding, has the GP model the
    residual y - m(z), and adds m(z*) back to the prediction (gp2Scale Eq. 2 with a
    linear m). This makes uncovered test points revert to the OLS prediction instead
    of 0, fixing the compact-support mean reversion."""
    from sklearn.metrics import r2_score

    from .reduce import LinearEmbeddingMean

    Ztr_d, Zte_d = Z_tr[:, :dim], Z_te[:, :dim]
    if getattr(args, "prior_mean", "none") == "linear":
        mean = LinearEmbeddingMean().fit(Ztr_d, y_tr)
        y_fit = y_tr - mean.predict(Ztr_d)          # GP models the residual
    else:
        mean, y_fit = None, y_tr

    Xtr = with_category_tag(Ztr_d, cat_tr)
    Xtr, y_fit_s, _ = sort_by_category(Xtr, y_fit)  # sort residual consistently
    Xte = with_category_tag(Zte_d, cat_te)
    sv = float(np.var(y_fit))                       # signal var of what the GP models

    t0 = time.time()
    gp, _ = build_gp(
        Xtr, y_fit_s, cutoff, dim, client,
        signal_var=sv, jitter=args.jitter, batch_size=args.batch_size,
        compute_device="cpu", device=args.device, linalg_mode=args.linalg,
        cg_maxiter=args.cg_maxiter, cg_tol=args.cg_tol,
    )
    t_build = time.time() - t0                       # kernel build + logdet + KVinvY solve
    t1 = time.time()
    m, _ = predict(gp, Xte, batch=args.pred_batch, variance=False)
    t_pred = time.time() - t1
    if mean is not None:
        m = m + mean.predict(Zte_d)                 # add the linear mean back (Eq. 2)
    r2 = float(r2_score(y_te, m))
    print(f"[sweep]   timing: build(+solve+logdet)={t_build:.1f}s  predict={t_pred:.1f}s")
    del gp
    release_gp(client)
    return r2


def _cutoffs_for_dim(Z_tr, d, args):
    """Cutoffs to test at embedding dim ``d``. Precedence: --cutoffs (an absolute
    sweep, reusing the embedding) > --cutoff (single absolute) > --cutoff-pct (one
    per-dim percentile of the pairwise distances, the legacy default)."""
    if args.cutoffs:
        return [float(c) for c in args.cutoffs]
    if args.cutoff is not None:
        return [float(args.cutoff)]
    c, _ = recalibrate(Z_tr[:, :d], percentile=args.cutoff_pct, dim=d)
    return [float(c)]


def run(args, client):
    from sklearn.model_selection import train_test_split

    from .data import get_data

    ds = get_data(src=args.src, n=args.n, seed=args.data_seed)
    dims = sorted({int(d) for d in args.dims})
    rows = []
    print(f"[sweep] config: scaling={args.scaling} prior_mean={args.prior_mean} "
          f"jitter={args.jitter:g} linalg={args.linalg} dims={dims}")

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
            # OLS R^2 is cutoff-INDEPENDENT (linear probe on Z[:, :d]) -> compute once
            # per dim and reuse across every cutoff. The GP kernel is the only thing that
            # changes with the cutoff, so the whole embedding (featurise + PLS) is reused.
            ols = regression_r2(Z_tr[:, :d], y_tr, Z_te[:, :d], y_te)
            for cutoff in _cutoffs_for_dim(Z_tr, d, args):
                rep = sparsity_report(Z_tr[:, :d], cutoff, dim=d, data_id=cat_tr)
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
    # mean/std across seeds, by (dim, cutoff). For an absolute cutoff sweep the cutoff
    # is identical across seeds so each group has one value per seed; for the percentile
    # default the recalibrated cutoff varies slightly per seed (groups of ~1).
    print("\n--- GP_R2 across seeds, by (dim, cutoff) ---")
    for (d, c) in sorted({(dd, round(cc, 4)) for (s, dd, cc, nb, ols, gp) in rows}):
        vals = np.array([gp for (s, dd, cc, nb, ols, gp) in rows
                         if dd == d and round(cc, 4) == c])
        print(f"  dim={d:>2} cutoff={c:>7.4f}: GP_R2 mean={vals.mean():.4f}  "
              f"std={vals.std():.4f}  n={len(vals)}  values={np.round(vals,4)}")
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
    ap.add_argument("--prior-mean", default="none", choices=["none", "linear"],
                    help="GP prior mean: 'linear' fits OLS on the embedding, GPs the "
                         "residual, adds it back (gp2Scale Eq. 2; fixes mean reversion)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=None,
                    help="sweep these ABSOLUTE cutoffs, reusing the embedding (built "
                         "once per seed) across all of them; overrides --cutoff / "
                         "--cutoff-pct. e.g. --cutoffs 0.16 0.22 0.28 0.35 0.44")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="absolute compact-support radius; overrides --cutoff-pct "
                         "(applied to every --dims slice)")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--pred-batch", type=int, default=2000)
    ap.add_argument("--linalg", default="sparseCG")
    ap.add_argument("--cg-maxiter", type=int, default=None,
                    help="cap CG iterations so an ill-conditioned split fails fast "
                         "(fvgp warns 'CG not successful') instead of grinding")
    ap.add_argument("--cg-tol", type=float, default=None,
                    help="CG relative tolerance (fvgp sparse_cg_tol; default 1e-5)")
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
