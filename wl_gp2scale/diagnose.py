"""
diagnose.py  (wl_gp2scale)
==========================
Bisect a gp2Scale predictive-accuracy failure.

Motivation. On the real OMol25 20k slice, `validate` reported:

    parity: max|Δ|=3.07e+01 (rel 1.1e+01), corr=0.750  [sparseCG] -> FAIL
    predictive R² vs truth: sparse=-0.1810  dense=0.1188

i.e. the DENSE path reproduces descriptor_eval/gp_parity.py (~0.09-0.12) but the
gp2Scale/sparseCG path does not, even though both use the same embedding and the
same Wendland math. Two hypotheses were already falsified by measurement:

  * "CG can't handle the conditioning": FALSE. On the real embedding scipy's cg
    converges (exit 0) and matches Cholesky to 4 decimals even at cond ~9.4e9.
  * "CG runs out of iterations": FALSE. fvgp passes maxiter=None -> scipy's 10*n.

So the fault is somewhere in the gp2Scale path itself. This script isolates it by
holding the embedding fixed and varying ONE thing at a time:

  (a) dense scipy cdist kernel + Cholesky      <- the gp_parity.py reference
  (b) our GPU block kernel  + Cholesky         <- kernel correct? (no gp2Scale)
  (c) gp2Scale, 1 block,   sparseLU            <- assembly correct? (direct solve)
  (d) gp2Scale, 1 block,   sparseCG            <- solver?
  (e) gp2Scale, k blocks,  sparseLU            <- multi-block assembly?
  (f) gp2Scale, k blocks,  sparseCG

Read the table by comparing R² / max|Δ| against (a):
  * (b) differs      -> the kernel itself is wrong.
  * (c) differs      -> gp2Scale ASSEMBLY is wrong (block masking / mirroring).
  * (d) differs, (c) fine -> the sparseCG SOLVER path is wrong.
  * only multi-block differs -> block-count / range handling.

Note `batch_size` interacts with N: fvgp computes num_batches = N // batch_size
and falls back to a single range when that is 0 (gp_prior.py::_ranges), so
batch_size >= N means "one big block".
"""

from __future__ import annotations

import argparse

import numpy as np


def build_argparser():
    ap = argparse.ArgumentParser(description="bisect a gp2Scale accuracy failure")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=6000, help="molecules to load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2,
                    help="2 matches descriptor_eval/gp_parity.py")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--pls", type=int, default=10)
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--ntr", type=int, default=3000, help="train rows for the bisect")
    ap.add_argument("--nte", type=int, default=1000, help="test rows for the bisect")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scheduler-file", default=None)
    return ap


def main():
    from distributed import Client
    from gpcam import GPOptimizer
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from .data import get_data
    from .kernel import dense_wendland_reference, make_wl_block_kernel
    from .pipeline import (WLGPPipeline, _first, build_gp, connect_dask, predict,
                           release_gp, with_category_tag)

    args = build_argparser().parse_args()

    ds = get_data(src=args.src, n=args.n, seed=args.seed)
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=42)

    pipe = WLGPPipeline(depth=args.depth, min_count=args.min_count,
                        pls_components=args.pls, cutoff_percentile=args.cutoff_pct)
    Z_tr = pipe.fit([ds.atoms[i] for i in tr], ds.y[tr], ds.data_id[tr], client=None)
    Z_te = pipe.transform([ds.atoms[i] for i in te], client=None)
    cut, dim = pipe.cutoff_, pipe.dim_

    ntr = min(args.ntr, len(Z_tr))
    nte = min(args.nte, len(Z_te))
    Ztr, ytr = Z_tr[:ntr], ds.y[tr][:ntr]
    Zte, yte = Z_te[:nte], ds.y[te][:nte]
    sv = float(np.var(ytr))
    # single dummy category so the category mask is inert and we compare the
    # sparse machinery against the dense reference in isolation
    Xtr = with_category_tag(Ztr, np.zeros(ntr))
    Xte = with_category_tag(Zte, np.zeros(nte))
    jit = args.jitter

    print(f"\n[diag] ntr={ntr} nte={nte} dim={dim} cutoff={cut:.5f} "
          f"signal_var={sv:.3f} jitter={jit:g}\n")
    hdr = f"{'case':<44}{'R2':>9}{'max|d| vs (a)':>16}{'corr':>8}"
    print(hdr); print("-" * len(hdr))

    def row(label, m, m_ref=None):
        r2 = r2_score(yte, m)
        if m_ref is None:
            print(f"{label:<44}{r2:>+9.4f}{'--':>16}{'--':>8}")
        else:
            d = float(np.max(np.abs(m - m_ref)))
            c = float(np.corrcoef(m, m_ref)[0, 1])
            print(f"{label:<44}{r2:>+9.4f}{d:>16.3e}{c:>8.4f}")
        return r2

    # (a) dense scipy cdist kernel + Cholesky (the gp_parity.py reference)
    def dref(x1, x2, hps):
        return dense_wendland_reference(x1, x2, hps, cut, dim=None)

    g = GPOptimizer(x_data=Ztr, y_data=ytr, init_hyperparameters=np.array([sv]),
                    kernel_function=dref, noise_variances=jit * np.ones(ntr))
    m_ref = _first(g.posterior_mean(Zte), ["f(x)", "m(x)"])
    row("(a) dense scipy cdist + Cholesky [REF]", m_ref)

    # (b) our block kernel + Cholesky -> is the kernel itself right?
    kern = make_wl_block_kernel(cut, dim=dim, use_category_tag=True,
                                device=args.device, dtype="float64")
    g2 = GPOptimizer(x_data=Xtr, y_data=ytr, init_hyperparameters=np.array([sv]),
                     kernel_function=kern, noise_variances=jit * np.ones(ntr))
    row("(b) our kernel + Cholesky (no gp2Scale)",
        _first(g2.posterior_mean(Xte), ["f(x)", "m(x)"]), m_ref)

    if args.scheduler_file:
        client = connect_dask(args.scheduler_file, n_workers=args.workers)
    else:
        client = Client(n_workers=args.workers, threads_per_worker=1)
        client.wait_for_workers(args.workers)

    # (c)-(f) gp2Scale: vary block count and solver independently
    one_block = ntr * 2          # >= ntr  -> num_batches = 0 -> single range
    multi = max(1, ntr // 3)     # -> ~3 blocks
    cases = [
        ("(c) gp2Scale 1 block   + sparseLU", one_block, "sparseLU"),
        ("(d) gp2Scale 1 block   + sparseCG", one_block, "sparseCG"),
        ("(e) gp2Scale ~3 blocks + sparseLU", multi, "sparseLU"),
        ("(f) gp2Scale ~3 blocks + sparseCG", multi, "sparseCG"),
    ]
    for label, bs, mode in cases:
        try:
            gp, _ = build_gp(Xtr, ytr, cut, dim, client, signal_var=sv, jitter=jit,
                             batch_size=bs, backend="wendland32",
                             compute_device=("gpu" if args.device == "cuda" else "cpu"),
                             device=args.device, linalg_mode=mode)
            m, _v = predict(gp, Xte)
            row(f"{label} (nb={ntr // bs})", m, m_ref)
            del gp
        except Exception as e:
            print(f"{label:<44}{'ERROR':>9}  {type(e).__name__}: {str(e)[:60]}")
        release_gp(client)

    client.close()
    print("\n[diag] Interpretation:")
    print("  (b) != (a)            -> the block kernel is wrong")
    print("  (c) != (a)            -> gp2Scale ASSEMBLY is wrong (masking/mirroring)")
    print("  (d) != (c)            -> the sparseCG solver path is wrong")
    print("  (e)/(f) != (c)/(d)    -> block-count / range handling is wrong")


if __name__ == "__main__":
    main()
