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
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--vocab-sample", type=int, default=0,
                    help="0 = fit WL vocab on ALL train molecules (recommended: no "
                         "train OOV, no dropped signal). >0 caps it to a stratified "
                         "sample of that many molecules if memory/time bites.")
    ap.add_argument("--chunk", type=int, default=500, help="molecules per WL task")
    ap.add_argument("--batch-size", type=int, default=10_000, help="gp2Scale block")
    ap.add_argument("--backend", default="wendland32",
                    choices=["wendland32", "wendland_d0"])
    ap.add_argument("--linalg", default="sparseCG")
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

    from .cutoff import sparsity_report
    from .data import get_data
    from .pipeline import (
        WLGPPipeline, build_gp, connect_dask, predict, sort_by_category,
        train_hyperparameters, with_category_tag,
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

    client = connect_dask(args.scheduler_file, n_workers=args.workers)

    pipe = WLGPPipeline(
        depth=args.depth, min_count=args.min_count, pls_components=args.pls,
        cutoff_percentile=args.cutoff_pct, vocab_sample=args.vocab_sample,
    )
    Z_tr = pipe.fit(atoms_tr, y_tr, cat_tr, client=client, chunk=args.chunk)
    Z_te = pipe.transform(atoms_te, client=client, chunk=args.chunk)
    cutoff = pipe.cutoff_
    dim = pipe.dim_

    # tag with category, sort train into contiguous category blocks
    X_tr = with_category_tag(Z_tr, cat_tr)
    X_te = with_category_tag(Z_te, cat_te)
    X_tr, y_tr, order = sort_by_category(X_tr, y_tr)

    # Know the memory bill BEFORE paying it: fvgp gathers every COO component to the
    # DRIVER and builds one scipy CSR there (gp_prior.py:294-306), so this number --
    # not worker RAM -- is what can kill the run.
    rep = sparsity_report(Z_tr, cutoff, dim=dim, data_id=cat_tr)
    n_blocks = max(1, len(X_tr) // args.batch_size)
    print(f"[run] gp2Scale: {n_blocks} batches -> ~{n_blocks*(n_blocks+1)//2} blocks "
          f"over {args.workers} workers; driver-side CSR ~{rep['est_gb']:.1f} GB "
          f"(assembly peak roughly 3x that)")

    print("[run] building gp2Scale GP (this includes the unavoidable imate logdet) ...")
    t_gp = time.time()
    gp, kern = build_gp(
        X_tr, y_tr, cutoff, dim, client,
        signal_var=args.signal_var, jitter=args.jitter, batch_size=args.batch_size,
        backend=args.backend, linalg_mode=args.linalg,
        compute_device=args.compute_device,
        device=args.device,
        logdet_rtol=(0.01 if args.train else args.logdet_rtol),
    )
    print(f"[run] GP constructed in {time.time()-t_gp:.0f}s "
          f"(kernel assembly + logdet + KVinvY solve)")

    if args.train:
        sv0 = float(args.signal_var or np.var(y_tr))
        bounds = np.array([[1e-3, max(10 * sv0, 1e-2)]])
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
    E_pred_resid = m
    rmse = float(np.sqrt(np.mean((m - y_te) ** 2)))
    r2 = float(r2_score(y_te, m))
    print(f"[run] TEST residual RMSE={rmse:.4f}  R²={r2:.4f}  "
          f"(baseline std {np.std(y_te):.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out, y_true=y_te, y_pred=E_pred_resid, var=v, cutoff=cutoff,
        r2=r2, rmse=rmse, signal_var=float(args.signal_var or np.var(y_tr)),
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
