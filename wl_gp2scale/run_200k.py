"""
run_200k.py  (wl_gp2scale)
==========================
Full-scale driver: 200k molecules, 16 GPUs (4 Perlmutter nodes), gp2Scale +
sparseCG. Connects to the Dask scheduler file written by launch-dask-conda.sh.

Usage (inside the allocation, after launching Dask)::

    python -m wl_gp2scale.run_200k --n 200000 --workers 16 --min-count 2 --no-variance

Default flow is PREDICT-ONLY: nothing is trained, signal_var = var(y_train) and the
cutoff are both set analytically, and --train (marginal-likelihood optimisation) is
off.

NOTE: predict-only does NOT avoid the log-determinant. fvgp computes it inside the
GP CONSTRUCTOR -- GPkv.__init__ -> _refresh -> `self.logdet_KV = self.logdet()`
(gp_kv.py:62,216) -- so every gp2Scale GP pays for one imate stochastic-Lanczos
logdet no matter what, and at 200k that is a real cost, not a rounding error. It is
also why `imate` is needed just to instantiate the GP. --train only adds MORE of
them (one per likelihood evaluation).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def plot_parity(y_true, y_pred, std, out_path, r2, rmse, n_train):
    """Parity plot (pred vs true residual), points coloured by GP posterior std."""
    import matplotlib
    matplotlib.use("Agg")  # headless (Perlmutter compute node)
    import matplotlib.pyplot as plt

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    have_std = std is not None and np.isfinite(std).any()
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(7, 7))
        if have_std:
            sc = ax.scatter(y_true, y_pred, c=std, s=8, cmap="viridis", alpha=0.6)
            cb = fig.colorbar(sc, ax=ax); cb.set_label("posterior std")
        else:  # --no-variance: nothing to colour by
            ax.scatter(y_true, y_pred, s=8, color="#348ABD", alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
        ax.set_xlabel("true residual  y = E − m(x)")
        ax.set_ylabel("predicted residual")
        ax.set_title(f"wl_gp2scale parity — N_train={n_train:,}\n"
                     f"R²={r2:.4f}  RMSE={rmse:.4f}")
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def build_argparser():
    ap = argparse.ArgumentParser(description="wl_gp2scale 200k distributed GP run")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.02)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--pls", type=int, default=10)
    # channel selection (wl = graph; geom = 3D geometry+charge; additive = k_WL + k_geom)
    ap.add_argument("--channel", default="wl", choices=["wl", "geom", "additive"],
                    help="which kernel channel(s) to run (additive = k_WL + k_geom)")
    ap.add_argument("--target-neighbors", type=int, default=0,
                    help="per-channel cutoff tuned to this median neighbour count; "
                         "0 = keep the WL --cutoff/--cutoff-pct path (recommended 60 for "
                         "the geom/additive comparison)")
    ap.add_argument("--geom-channels", default="rdf,angle,torsion,elec",
                    help="geometry: comma-separated subset of rdf,angle,torsion,elec")
    ap.add_argument("--geom-top-k", type=int, default=6)
    ap.add_argument("--geom-r-max", type=float, default=6.0)
    ap.add_argument("--charge-key", default="lowdin_charges")
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="absolute compact-support radius (embedding units); OVERRIDES "
                         "--cutoff-pct. The embedding scale is N-invariant, so a radius "
                         "picked once (variogram/R_inf) transfers across N.")
    ap.add_argument("--prior-mean", default="none", choices=["none", "linear"],
                    help="GP prior mean: 'linear' fits OLS on the embedding, GPs the "
                         "residual, adds it back (gp2Scale Eq. 2; fixes mean reversion)")
    ap.add_argument("--vocab-sample", type=int, default=0,
                    help="0 = fit WL vocab on ALL train molecules (recommended: no "
                         "train OOV, no dropped signal). >0 caps it to a stratified "
                         "sample of that many molecules if memory/time bites.")
    ap.add_argument("--chunk", type=int, default=500, help="molecules per WL task")
    ap.add_argument("--batch-size", type=int, default=10_000, help="gp2Scale block")
    ap.add_argument("--backend", default="wendland32",
                    choices=["wendland32", "wendland_d0"])
    ap.add_argument("--linalg", default="sparseCG")
    ap.add_argument("--cg-maxiter", type=int, default=None,
                    help="cap CG iterations (fail fast on an ill-conditioned split "
                         "instead of grinding; fvgp warns 'CG not successful')")
    ap.add_argument("--cg-tol", type=float, default=None,
                    help="CG relative tolerance (fvgp sparse_cg_tol; default 1e-5)")
    ap.add_argument("--logdet-verbose", action="store_true",
                    help="print imate stochastic-Lanczos log-det progress")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--signal-var", type=float, default=None,
                    help="frozen signal variance; default var(y_train)")
    ap.add_argument("--scheduler-file", default=None,
                    help="default $SCRATCH/scheduler_file_gpOmol.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda",
                    help="OUR kernel's torch device (this is what uses the GPU)")
    ap.add_argument("--compute-device", default="cpu", choices=["cpu", "gpu"],
                    help="fvgp's device. Keep 'cpu': 'gpu' routes imate's logdet to a "
                         "CUDA backend that a pip-installed imate does not have, and "
                         "fvgp gates it on TORCH having CUDA, not imate. Costs nothing "
                         "-- the kernel still runs on --device.")
    ap.add_argument("--predict-batch", type=int, default=500,
                    help="test points per prediction batch; bounds the DENSE "
                         "cross-covariance k (n_train x batch)")
    ap.add_argument("--no-variance", action="store_true",
                    help="mean only. posterior_covariance costs ONE SOLVE PER TEST "
                         "POINT on the full NxN system -- at 200k that is the "
                         "dominant cost. Use with a large test set; keep the test "
                         "set in the hundreds if you want variance.")
    ap.add_argument("--logdet-rtol", type=float, default=0.5,
                    help="imate SLQ error_rtol for the log-determinant. fvgp computes "
                         "the logdet in the GP CONSTRUCTOR no matter what, but "
                         "predict-only never READS it -- so pay the floor (loose rtol "
                         "-> min_num_samples=10). Forced to 0.01 under --train, which "
                         "actually uses it.")
    ap.add_argument("--train", action="store_true",
                    help="marginal-likelihood training (needs imate)")
    ap.add_argument("--out", default="cache/preds_200k.npz")
    return ap


def main():
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from .cutoff import cutoff_for_neighbors, sparsity_report
    from .data import get_data
    from .pipeline import (
        GeometryPipeline, WLGPPipeline, build_gp, connect_dask, predict,
        sort_by_category, train_hyperparameters, with_category_tag,
    )

    args = build_argparser().parse_args()
    t0 = time.time()

    ds = get_data(src=args.src, n=args.n, seed=args.seed)
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=42)
    atoms_tr = [ds.atoms[i] for i in tr]
    atoms_te = [ds.atoms[i] for i in te]
    y_tr, y_te = ds.y[tr], ds.y[te]
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    print(f"[run] train={len(tr):,} test={len(te):,}")
    if len(te) < 1000:
        print(f"[run] WARNING: only {len(te)} test points -> R² will be noisy "
              f"(~+/-0.03-0.05). --test-size defaults to {args.test_size:g}, tuned for "
              f"the 200k run; for a small-N baseline pass e.g. --test-size 0.2 to get "
              f"~{int(len(idx)*0.2):,} test points.")

    client = connect_dask(args.scheduler_file, n_workers=args.workers)

    # -- build the requested channel embedding(s) --------------------------------
    # WL and geometry each get their own natural-scaled PLS embedding + compact-support
    # cutoff; the additive kernel (build_gp channels=) sums one Wendland block per
    # channel. A single channel routes through the additive kernel too -- it is
    # byte-identical to the WLBlockKernel (validated), so --channel wl is unchanged.
    geom_channels = [c.strip() for c in args.geom_channels.split(",") if c.strip()]
    chan = []  # list of {"Ztr","Zte","cutoff"}
    if args.channel in ("wl", "additive"):
        pw = WLGPPipeline(
            depth=args.depth, min_count=args.min_count, pls_components=args.pls,
            cutoff_percentile=(None if args.target_neighbors else args.cutoff_pct),
            cutoff_abs=args.cutoff, vocab_sample=args.vocab_sample,
        )
        Zw_tr = pw.fit(atoms_tr, y_tr, cat_tr, client=client, chunk=args.chunk)
        Zw_te = pw.transform(atoms_te, client=client, chunk=args.chunk)
        cw = (cutoff_for_neighbors(Zw_tr, Zw_te, args.target_neighbors,
                                   dim=Zw_tr.shape[1], data_id_tr=cat_tr, data_id_te=cat_te)
              if args.target_neighbors else pw.cutoff_)
        chan.append({"Ztr": Zw_tr, "Zte": Zw_te, "cutoff": cw})
    if args.channel in ("geom", "additive"):
        pg = GeometryPipeline(
            top_k=args.geom_top_k, channels=tuple(geom_channels), r_max=args.geom_r_max,
            pls_components=args.pls, cutoff_percentile=None, charge_key=args.charge_key,
        )
        Zg_tr = pg.fit(atoms_tr, y_tr, cat_tr, client=client, chunk=args.chunk)
        Zg_te = pg.transform(atoms_te, client=client, chunk=args.chunk)
        # geometry has no historical cutoff-pct convention -> always neighbour-tuned
        # (default 60, matching descriptor_eval).
        cg = cutoff_for_neighbors(Zg_tr, Zg_te, args.target_neighbors or 60,
                                  dim=Zg_tr.shape[1], data_id_tr=cat_tr, data_id_te=cat_te)
        chan.append({"Ztr": Zg_tr, "Zte": Zg_te, "cutoff": cg})

    Z_tr = np.hstack([c["Ztr"] for c in chan])
    Z_te = np.hstack([c["Zte"] for c in chan])
    specs, off = [], 0                                # (start, stop, cutoff) per channel
    for c in chan:
        d = c["Ztr"].shape[1]
        specs.append((off, off + d, float(c["cutoff"])))
        off += d
    dim = Z_tr.shape[1]

    # optional linear prior mean (gp2Scale Eq. 2 with linear m): the GP models the
    # residual y - m(z) and predictions add m(z*) back, so uncovered test points revert
    # to the OLS prediction instead of 0 (fixes compact-support mean reversion).
    if args.prior_mean == "linear":
        from .reduce import LinearEmbeddingMean
        emean = LinearEmbeddingMean().fit(Z_tr, y_tr)
        y_fit = y_tr - emean.predict(Z_tr)
        print(f"[run] linear prior mean: GP models residual (var {np.var(y_fit):.4g} "
              f"vs y var {np.var(y_tr):.4g})")
    else:
        emean, y_fit = None, y_tr

    # tag with category, sort train into contiguous category blocks
    X_tr = with_category_tag(Z_tr, cat_tr)
    X_te = with_category_tag(Z_te, cat_te)
    X_tr, y_fit, order = sort_by_category(X_tr, y_fit)

    # Know the memory bill BEFORE paying it: fvgp gathers every COO component to the
    # DRIVER and builds one scipy CSR there (gp_prior.py:294-306), so this number --
    # not worker RAM -- is what can kill the run. Report per channel.
    for c in chan:
        sparsity_report(c["Ztr"], c["cutoff"], dim=c["Ztr"].shape[1], data_id=cat_tr)
    n_blocks = max(1, len(X_tr) // args.batch_size)
    print(f"[run] channel={args.channel}  specs={specs}")
    print(f"[run] gp2Scale: {n_blocks} batches -> ~{n_blocks*(n_blocks+1)//2} blocks "
          f"over {args.workers} workers")

    # per-channel signal variances: frozen equal split (diagonal = var(y_fit)), or the
    # single --signal-var if given; --train optimises them by marginal likelihood.
    C = len(specs)
    if args.signal_var is not None:
        signal_var = [float(args.signal_var) / C] * C
    else:
        signal_var = None                             # build_gp -> var(y_fit)/C each

    print("[run] building gp2Scale GP (this includes the unavoidable imate logdet) ...")
    t_gp = time.time()
    gp, kern = build_gp(
        X_tr, y_fit, None, dim, client,
        channels=specs, signal_var=signal_var,
        jitter=args.jitter, batch_size=args.batch_size,
        backend=args.backend, linalg_mode=args.linalg,
        compute_device=args.compute_device,
        device=args.device,
        logdet_rtol=(0.01 if args.train else args.logdet_rtol),
        cg_maxiter=args.cg_maxiter, cg_tol=args.cg_tol,
        logdet_verbose=args.logdet_verbose,
    )
    print(f"[run] GP constructed in {time.time()-t_gp:.0f}s "
          f"(kernel assembly + logdet + KVinvY solve)")

    if args.train:
        sv0 = float(args.signal_var or np.var(y_fit))
        bounds = np.array([[1e-3, max(10 * sv0, 1e-2)]] * C)
        print("[run] marginal-likelihood training (requires imate) ...")
        hps = train_hyperparameters(gp, bounds, max_iter=50)
        print(f"[run] trained hyperparameters: {hps}")

    want_var = not args.no_variance
    print(f"[run] predicting on {len(X_te):,} test points "
          f"(batch={args.predict_batch}, variance={want_var}) ...")
    if want_var and len(X_te) > 1000:
        print(f"[run] WARNING: variance costs ~1 solve per test point on the "
              f"{len(y_tr):,}-point system; {len(X_te):,} test points may take "
              f"hours. Consider --no-variance or a smaller --test-size.")
    m, v = predict(gp, X_te, batch=args.predict_batch, variance=want_var, verbose=True)
    if emean is not None:
        m = m + emean.predict(Z_te)              # add the linear mean back (Eq. 2)
    E_pred_resid = m
    rmse = float(np.sqrt(np.mean((m - y_te) ** 2)))
    r2 = float(r2_score(y_te, m))
    print(f"[run] TEST residual RMSE={rmse:.4f}  R²={r2:.4f}  "
          f"(baseline std {np.std(y_te):.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out, y_true=y_te, y_pred=E_pred_resid, var=v,
        channel=args.channel, cutoffs=np.array([c["cutoff"] for c in chan]),
        channel_specs=np.array(specs, dtype=float),
        r2=r2, rmse=rmse, signal_var=float(args.signal_var or np.var(y_fit)),
        prior_mean=args.prior_mean,
        dim=dim, min_count=args.min_count, depth=args.depth, pls=args.pls,
        cutoff_pct=args.cutoff_pct, category_order=order,
    )
    plot_path = os.path.splitext(args.out)[0] + "_parity.png"
    plot_parity(y_te, m, np.sqrt(v), plot_path, r2, rmse, len(y_tr))
    print(f"[run] saved predictions -> {args.out}")
    print(f"[run] saved parity plot -> {plot_path}  (elapsed {time.time()-t0:.0f}s)")
    client.close()


if __name__ == "__main__":
    main()
