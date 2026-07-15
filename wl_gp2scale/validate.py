"""
validate.py  (wl_gp2scale)
==========================
Pre-run checklist. Run on a 20k-50k slice with 1-2 GPUs (or locally on CPU)
BEFORE the full 200k run:

  1. Sparse-vs-dense parity   -- the sparse GPU kernel reproduces the dense CPU
                                 kernel's GP predictions.
  2. Sparsity / memory        -- realised sparsity + total nnz confirm memory fits.
  3. CG convergence           -- sparseCG converges at jitter 1e-6.
  4. Streaming-PLS parity     -- streaming SIMPLS R^2 == batch PLSRegression R^2.
  5. PD guard                 -- min eigenvalue of a subsample Gram > -tol.

Checks 1 and 3 use a SINGLE dummy category so the category mask is a no-op and the
sparse/GPU machinery is compared against the dense reference in isolation.
"""

from __future__ import annotations

import numpy as np

from . import cutoff as cutoff_mod
from .kernel import check_kernel_psd, dense_wendland_reference, make_wl_block_kernel
from .pipeline import _first, build_gp, predict, with_category_tag
from .reduce import SparsePLS, batch_pls_r2, regression_r2


# ----------------------------- 1 + 3: parity & CG --------------------------


def sparse_vs_dense_parity(
    Z_tr, y_tr, Z_te, cutoff, client, y_te=None, jitter=1e-6, tol=1e-3, device="cpu",
    linalg_mode="sparseCG",
):
    """Compare gp2Scale/sparseCG posterior mean vs the dense scipy.cdist Wendland
    GP on the SAME embedding. Returns a dict with max abs diff and correlation.
    If ``y_te`` is given, also reports each GP's R^2 against truth -- this is the
    number to compare against descriptor_eval/gp_parity.py (~0.09-0.12). All rows
    share one dummy category so masking is inert."""
    from gpcam import GPOptimizer
    from sklearn.metrics import r2_score

    dim = Z_tr.shape[1]
    sv = float(np.var(y_tr))

    # dense reference (descriptor_eval path): dense kernel, no gp2Scale
    def dense_ref(x1, x2, hps):
        return dense_wendland_reference(x1, x2, hps, cutoff, dim=None)

    gp_dense = GPOptimizer(
        x_data=np.asarray(Z_tr, float),
        y_data=np.asarray(y_tr, float),
        init_hyperparameters=np.array([sv]),
        kernel_function=dense_ref,
        noise_variances=jitter * np.ones(len(y_tr)),
    )
    m_dense = _first(gp_dense.posterior_mean(np.asarray(Z_te, float)), ["f(x)", "m(x)"])

    # sparse GPU path with one dummy category (all zeros)
    Xtr = with_category_tag(Z_tr, np.zeros(len(Z_tr)))
    Xte = with_category_tag(Z_te, np.zeros(len(Z_te)))
    gp_sparse, _ = build_gp(
        Xtr, y_tr, cutoff, dim, client, signal_var=sv, jitter=jitter,
        batch_size=10_000, compute_device=device, device=device,
        linalg_mode=linalg_mode,
    )
    m_sparse, _ = predict(gp_sparse, Xte)

    diff = float(np.max(np.abs(m_sparse - m_dense)))
    scale = float(np.std(m_dense)) or 1.0
    corr = float(np.corrcoef(m_sparse, m_dense)[0, 1])
    ok = diff / scale < tol
    print(
        f"[val] parity: max|Δ|={diff:.3e} (rel {diff/scale:.1e}), corr={corr:.6f} "
        f"[{linalg_mode}] -> {'PASS' if ok else 'FAIL'}"
    )
    out = {"max_abs_diff": diff, "rel_diff": diff / scale, "corr": corr, "pass": ok}
    if y_te is not None:
        r2_sp = float(r2_score(y_te, m_sparse))
        r2_de = float(r2_score(y_te, m_dense))
        print(
            f"[val] predictive R² vs truth: sparse={r2_sp:.4f}  dense={r2_de:.4f}  "
            f"(compare to descriptor_eval/gp_parity.py ~0.09-0.12)"
        )
        out.update({"r2_sparse": r2_sp, "r2_dense": r2_de})
    return out


# ----------------------------- 2: sparsity / memory ------------------------


def sparsity(Z, cutoff, dim, data_id=None):
    return cutoff_mod.sparsity_report(Z, cutoff, dim=dim, data_id=data_id)


# ----------------------------- 4: streaming-PLS parity ---------------------


def streaming_pls_parity(X_tr, y_tr, X_te, y_te, n_components=10, tol=0.02):
    """Assert streaming SIMPLS embedding R^2 matches batch PLSRegression on a small
    slice (dense reference allowed here). Gate for the 200k reduction."""
    spls = SparsePLS(n_components=n_components).fit(X_tr, y_tr)
    Z_tr, Z_te = spls.transform(X_tr), spls.transform(X_te)
    r2_stream = regression_r2(Z_tr, y_tr, Z_te, y_te)
    r2_batch_emb, r2_batch_pls = batch_pls_r2(X_tr, y_tr, X_te, y_te, n_components)
    ok = abs(r2_stream - r2_batch_emb) < tol
    print(
        f"[val] PLS parity: streaming R²={r2_stream:.4f} vs batch R²={r2_batch_emb:.4f} "
        f"(sklearn .score={r2_batch_pls:.4f}) -> {'PASS' if ok else 'FAIL'}"
    )
    return {
        "r2_streaming": r2_stream,
        "r2_batch_embed": r2_batch_emb,
        "r2_batch_pls_score": r2_batch_pls,
        "pass": ok,
    }


# ----------------------------- 5: PD guard ---------------------------------


def psd_guard(Z_sample, cutoff, backend="wendland32", tol=1e-8):
    dim = Z_sample.shape[1]
    kern = make_wl_block_kernel(cutoff, dim=dim, use_category_tag=False,
                                backend=backend, device="cpu")
    res = check_kernel_psd(kern, Z_sample, np.array([1.0]), tol=tol)
    print(
        f"[val] PSD [{backend}]: min_eig={res['min_eigenvalue']:.3e} "
        f"is_psd={res['is_psd']} gram_density={res['gram_density']:.3e}"
    )
    return res


# ----------------------------- CLI driver ----------------------------------


def main():
    import argparse

    from distributed import Client
    from sklearn.model_selection import train_test_split

    from .data import get_data
    from .pipeline import WLGPPipeline

    ap = argparse.ArgumentParser(description="wl_gp2scale pre-run validation")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--pls", type=int, default=10)
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--parity-n", type=int, default=3000, help="train rows for parity")
    ap.add_argument("--device", default="cpu", help="cpu (local) or cuda")
    ap.add_argument("--workers", type=int, default=2, help="dask workers")
    ap.add_argument("--scheduler-file", default=None,
                    help="connect to srun-launched GPU workers (proper per-task GPU "
                         "binding) instead of a local cluster; e.g. "
                         "$SCRATCH/scheduler_file_gpOmol.json")
    args = ap.parse_args()

    ds = get_data(src=args.src, n=args.n, seed=args.seed)
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=42)

    atoms_tr = [ds.atoms[i] for i in tr]
    atoms_te = [ds.atoms[i] for i in te]
    y_tr, y_te = ds.y[tr], ds.y[te]
    cat_tr = ds.data_id[tr]

    if args.scheduler_file:
        from .pipeline import connect_dask
        client = connect_dask(args.scheduler_file, n_workers=args.workers)
    else:
        client = Client(n_workers=args.workers, threads_per_worker=1)
        client.wait_for_workers(args.workers)
        print(f"[val] local dask: {args.workers} workers")

    pipe = WLGPPipeline(
        depth=args.depth, min_count=args.min_count, pls_components=args.pls,
        cutoff_percentile=args.cutoff_pct,
    )
    Z_tr = pipe.fit(atoms_tr, y_tr, cat_tr, client=client)
    Z_te = pipe.transform(atoms_te, client=client)
    cutoff = pipe.cutoff_

    # 4: PLS parity needs the sparse feature matrices (recompute on the slice)
    X_tr = pipe.featurizer.transform(atoms_tr, client=client)
    X_te = pipe.featurizer.transform(atoms_te, client=client)
    streaming_pls_parity(X_tr, y_tr, X_te, y_te, n_components=args.pls)

    # 2: sparsity / memory (with category block-sparsity)
    sparsity(Z_tr, cutoff, dim=Z_tr.shape[1], data_id=cat_tr)

    # 5: PD guard (both backends)
    psd_guard(Z_tr[: min(1500, len(Z_tr))], cutoff, backend="wendland32")
    psd_guard(Z_tr[: min(1500, len(Z_tr))], cutoff, backend="wendland_d0")

    # 1 + 3: parity + CG on a parity-sized slice, single dummy category
    k = min(args.parity_n, len(Z_tr))
    kt = min(args.parity_n // 3 or 1, len(Z_te))
    sparse_vs_dense_parity(
        Z_tr[:k], y_tr[:k], Z_te[:kt], cutoff, client, y_te=y_te[:kt],
        device=args.device, linalg_mode="sparseCG",
    )

    client.close()
    print("[val] done.")


if __name__ == "__main__":
    main()
