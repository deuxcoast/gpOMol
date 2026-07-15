"""
run_200k.py  (wl_gp2scale)
==========================
Full-scale driver: 200k molecules, 16 GPUs (4 Perlmutter nodes), gp2Scale +
sparseCG. Connects to the Dask scheduler file written by launch-dask-moduleGPU.sh.

Usage (inside the allocation, after launching Dask)::

    python -m wl_gp2scale.run_200k --n 200000 --workers 16

Default flow is PREDICT-ONLY with hyperparameters frozen from the validation-scale
fit (posterior mean/variance need only CG solves, no log-determinant). Pass
--train only if `imate` is installed and validated (see pipeline.train_hyperparameters).
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
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(y_true, y_pred, c=std, s=8, cmap="viridis", alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
        cb = fig.colorbar(sc, ax=ax); cb.set_label("posterior std")
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
    ap.add_argument("--vocab-sample", type=int, default=20_000)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", action="store_true",
                    help="marginal-likelihood training (needs imate)")
    ap.add_argument("--out", default="cache/preds_200k.npz")
    return ap


def main():
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

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

    gp, kern = build_gp(
        X_tr, y_tr, cutoff, dim, client,
        signal_var=args.signal_var, jitter=args.jitter, batch_size=args.batch_size,
        backend=args.backend, linalg_mode=args.linalg,
        compute_device=("gpu" if args.device == "cuda" else "cpu"),
        device=args.device,
    )

    if args.train:
        sv0 = float(args.signal_var or np.var(y_tr))
        bounds = np.array([[1e-3, max(10 * sv0, 1e-2)]])
        print("[run] marginal-likelihood training (requires imate) ...")
        hps = train_hyperparameters(gp, bounds, max_iter=50)
        print(f"[run] trained hyperparameters: {hps}")

    print("[run] predicting on test set ...")
    m, v = predict(gp, X_te)
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
