"""
dim_sweep.py  (wl_gp2scale)
===========================
Channel comparison on the ACTUAL gp2Scale Wendland kernel: for each requested
channel mode -- ``wl`` (graph), ``geom`` (3D geometry+charge), ``additive``
(k = k_WL + k_geom) -- report held-out predictive R^2 at the current N. This is the
ladder driver for the 40k/80k/200k comparison established in descriptor_eval, where
geometry dominates at small N but WL scales faster, so the two are expected to
converge near ~200k (exactly where the additive kernel should pay off).

For each split seed the two embeddings are built ONCE (WLGPPipeline for the graph
channel, GeometryPipeline for the geometry channel), then each requested mode reuses
them -- no re-featurisation. Each channel gets its OWN compact-support cutoff tuned to
``--target-neighbors`` (the validated sparsity lever; percentile mis-tracks the median
neighbour count on the clustered embedding, see cutoff.cutoff_for_neighbors). The
additive kernel sums the two blocks; ``--prior-mean linear`` detrends on the
CONCATENATED embedding (gp2Scale Eq. 2). Per-channel signal variances are frozen at
var(y)/n_channels by default (diagonal = var(y)); ``--train`` optimises them by
marginal likelihood (the principled auto-weighting -- avoids the equal-split dilution
seen at 10k).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .cutoff import cutoff_for_neighbors, recalibrate, sparsity_report
from .pipeline import (GeometryPipeline, WLGPPipeline, build_gp, predict,
                       release_gp, sort_by_category, train_hyperparameters,
                       with_category_tag)
from .reduce import LinearEmbeddingMean, regression_r2

CHANNEL_CODE = {"wl": 0, "geom": 1, "additive": 2}   # numeric tag for the --out rows


def _channel_cutoff(Z_tr, Z_te, cat_tr, cat_te, args):
    """Per-channel compact-support radius: tuned to ~--target-neighbors (default), else
    the --cutoff-pct percentile of this channel's own pairwise distances."""
    if args.target_neighbors:
        return cutoff_for_neighbors(
            Z_tr, Z_te, args.target_neighbors, dim=Z_tr.shape[1],
            data_id_tr=cat_tr, data_id_te=cat_te,
        )
    c, _ = recalibrate(Z_tr, percentile=args.cutoff_pct, dim=Z_tr.shape[1])
    return c


def _gp_r2_channels(chan, y_tr, cat_tr, y_te, cat_te, client, args):
    """Build a frozen (or --train'd) additive gp2Scale GP over the given channels and
    return held-out R^2. ``chan`` is a list of {"Ztr","Zte","cutoff"} dicts; the kernel
    sums one Wendland block per channel on its own column range/cutoff. Category-tagged +
    sorted; ``--prior-mean linear`` detrends on the concatenated embedding and adds the
    OLS mean back (fixes compact-support mean reversion)."""
    from sklearn.metrics import r2_score

    Ztr = np.hstack([c["Ztr"] for c in chan])
    Zte = np.hstack([c["Zte"] for c in chan])
    specs, off = [], 0                                # (start, stop, cutoff) per channel
    for c in chan:
        d = c["Ztr"].shape[1]
        specs.append((off, off + d, float(c["cutoff"])))
        off += d

    if getattr(args, "prior_mean", "none") == "linear":
        mean = LinearEmbeddingMean().fit(Ztr, y_tr)
        y_fit = y_tr - mean.predict(Ztr)              # GP models the residual
    else:
        mean, y_fit = None, y_tr

    Xtr = with_category_tag(Ztr, cat_tr)
    Xtr, y_fit_s, _ = sort_by_category(Xtr, y_fit)
    Xte = with_category_tag(Zte, cat_te)
    C = len(specs)
    var = float(np.var(y_fit))
    signal_var = [var / C] * C                        # equal split -> diagonal = var(y)

    t0 = time.time()
    gp, _ = build_gp(
        Xtr, y_fit_s, None, None, client,
        channels=specs, signal_var=signal_var,
        jitter=args.jitter, batch_size=args.batch_size,
        compute_device="cpu", device=args.device, linalg_mode=args.linalg,
        cg_maxiter=args.cg_maxiter, cg_tol=args.cg_tol,
    )
    if getattr(args, "train", False):
        bounds = [[1e-6, 10.0 * var]] * C             # per-channel signal-var bounds
        hps = train_hyperparameters(gp, bounds, max_iter=args.train_iter)
        print(f"[sweep]   trained signal vars -> {np.round(np.asarray(hps), 5)}")
    t_build = time.time() - t0
    m, _ = predict(gp, Xte, batch=args.pred_batch, variance=False)
    if mean is not None:
        m = m + mean.predict(Zte)                     # add the linear mean back
    r2 = float(r2_score(y_te, m))
    print(f"[sweep]   build(+solve){'+train' if args.train else ''}={t_build:.1f}s")
    del gp
    release_gp(client)
    return r2


def _ols_r2(chan, y_tr, y_te):
    """Cross-check: held-out OLS R^2 on the concatenated embedding (cutoff-free)."""
    Ztr = np.hstack([c["Ztr"] for c in chan])
    Zte = np.hstack([c["Zte"] for c in chan])
    return regression_r2(Ztr, y_tr, Zte, y_te)


def run(args, client):
    from sklearn.model_selection import train_test_split

    from .data import get_data

    ds = get_data(src=args.src, n=args.n, seed=args.data_seed)
    modes = args.channels
    need_wl = any(m in ("wl", "additive") for m in modes)
    need_geom = any(m in ("geom", "additive") for m in modes)
    rows = []
    print(f"[sweep] config: channels={modes} scaling={args.scaling} "
          f"prior_mean={args.prior_mean} target_nbrs={args.target_neighbors} "
          f"train={args.train} pls={args.pls} jitter={args.jitter:g} linalg={args.linalg}")

    for seed in args.seeds:
        print(f"\n########## split seed {seed} ##########")
        idx = np.arange(len(ds))
        tr, te = train_test_split(idx, test_size=args.test_size, random_state=seed)
        atoms_tr = [ds.atoms[i] for i in tr]
        atoms_te = [ds.atoms[i] for i in te]
        y_tr, y_te = ds.y[tr], ds.y[te]
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]

        emb = {}   # channel name -> {"Ztr","Zte","cutoff"}
        if need_wl:
            pw = WLGPPipeline(depth=args.depth, min_count=args.min_count,
                              pls_components=args.pls, cutoff_percentile=None,
                              scaling=args.scaling, vocab_sample=0)
            Zw_tr = pw.fit(atoms_tr, y_tr, cat_tr, client=client)
            Zw_te = pw.transform(atoms_te, client=client)
            emb["wl"] = {"Ztr": Zw_tr, "Zte": Zw_te,
                         "cutoff": _channel_cutoff(Zw_tr, Zw_te, cat_tr, cat_te, args)}
        if need_geom:
            pg = GeometryPipeline(top_k=args.geom_top_k, channels=tuple(args.geom_channels),
                                  r_max=args.geom_r_max, pls_components=args.pls,
                                  cutoff_percentile=None, scaling=args.scaling,
                                  charge_key=args.charge_key)
            Zg_tr = pg.fit(atoms_tr, y_tr, cat_tr, client=client)
            Zg_te = pg.transform(atoms_te, client=client)
            emb["geom"] = {"Ztr": Zg_tr, "Zte": Zg_te,
                           "cutoff": _channel_cutoff(Zg_tr, Zg_te, cat_tr, cat_te, args)}

        for mode in modes:
            chan = ([emb["wl"]] if mode == "wl"
                    else [emb["geom"]] if mode == "geom"
                    else [emb["wl"], emb["geom"]])
            for c in chan:
                sparsity_report(c["Ztr"], c["cutoff"], dim=c["Ztr"].shape[1], data_id=cat_tr)
            ols = _ols_r2(chan, y_tr, y_te)
            gp_r2 = _gp_r2_channels(chan, y_tr, cat_tr, y_te, cat_te, client, args)
            cuts = "+".join(f"{c['cutoff']:.3f}" for c in chan)
            print(f"[sweep] seed={seed} mode={mode:>8}  cutoffs={cuts}  "
                  f"OLS_R2={ols:.4f}  GP_R2={gp_r2:.4f}")
            rows.append((seed, CHANNEL_CODE[mode], ols, gp_r2))

    print("\n================= SUMMARY =================")
    print(f"{'seed':>5} {'mode':>9} {'OLS_R2':>8} {'GP_R2':>8}")
    inv = {v: k for k, v in CHANNEL_CODE.items()}
    for s, code, ols, gp in rows:
        print(f"{s:>5} {inv[int(code)]:>9} {ols:>8.4f} {gp:>8.4f}")
    print("\n--- GP_R2 across seeds, by channel ---")
    for mode in modes:
        vals = np.array([gp for (s, code, ols, gp) in rows if int(code) == CHANNEL_CODE[mode]])
        if len(vals):
            print(f"  {mode:>8}: GP_R2 mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"n={len(vals)}  values={np.round(vals, 4)}")
    if args.out:
        np.savez(args.out, rows=np.array(rows, dtype=float),
                 channels=np.array(modes), seeds=np.array(args.seeds),
                 channel_code=np.array([CHANNEL_CODE[m] for m in modes]))
        print(f"\n[sweep] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="wl_gp2scale channel comparison (wl | geom | additive)")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--data-seed", type=int, default=0, help="subset draw seed (frozen in cache)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                    help="TRAIN/TEST split seeds (the stability axis)")
    ap.add_argument("--channels", nargs="+", default=["wl", "geom", "additive"],
                    choices=["wl", "geom", "additive"],
                    help="which channel modes to evaluate (embeddings built once, reused)")
    ap.add_argument("--pls", type=int, default=10, help="PLS components per channel")
    ap.add_argument("--scaling", default="pareto",
                    choices=["pareto", "standard", "center"],
                    help="SparsePLS column pre-weighting (default pareto)")
    ap.add_argument("--prior-mean", default="linear", choices=["none", "linear"],
                    help="'linear' fits OLS on the (concatenated) embedding, GPs the "
                         "residual, adds it back (gp2Scale Eq. 2; fixes mean reversion)")
    ap.add_argument("--target-neighbors", type=int, default=60,
                    help="per-channel cutoff tuned to this median in-support neighbour "
                         "count (0 -> fall back to --cutoff-pct)")
    ap.add_argument("--cutoff-pct", type=float, default=25.0,
                    help="fallback percentile per channel when --target-neighbors 0")
    ap.add_argument("--train", action="store_true",
                    help="optimise per-channel signal variances by marginal likelihood "
                         "(default frozen at var(y)/n_channels)")
    ap.add_argument("--train-iter", type=int, default=50)
    # geometry channel
    ap.add_argument("--geom-channels", default="rdf,angle,torsion,elec",
                    help="comma-separated subset of rdf,angle,torsion,elec")
    ap.add_argument("--geom-top-k", type=int, default=6)
    ap.add_argument("--geom-r-max", type=float, default=6.0)
    ap.add_argument("--charge-key", default="lowdin_charges")
    # WL channel
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    # solver / cluster
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--pred-batch", type=int, default=2000)
    ap.add_argument("--linalg", default="sparseCG")
    ap.add_argument("--cg-maxiter", type=int, default=None,
                    help="cap CG iterations so an ill-conditioned split fails fast")
    ap.add_argument("--cg-tol", type=float, default=None, help="CG relative tolerance")
    ap.add_argument("--device", default="cuda", help="OUR kernel's torch device")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None)
    ap.add_argument("--out", default=None, help="optional .npz of the summary rows")
    args = ap.parse_args()
    args.geom_channels = [c.strip() for c in args.geom_channels.split(",") if c.strip()]

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
