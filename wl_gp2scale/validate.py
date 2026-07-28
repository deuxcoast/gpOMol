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
  6. Additive-kernel parity   -- 1-channel additive == WLBlockKernel; 2-channel ==
                                 sum of dense references (protects the numerics when
                                 combining WL + geometry).

Checks 1 and 3 use a SINGLE dummy category so the category mask is a no-op and the
sparse/GPU machinery is compared against the dense reference in isolation.
"""

from __future__ import annotations

import numpy as np

from . import cutoff as cutoff_mod
from .kernel import (check_kernel_diagonal, check_kernel_psd,
                     dense_wendland_reference, make_wl_block_kernel)
from .pipeline import _first, build_gp, predict, with_category_tag
from .reduce import SparsePLS, batch_pls_r2, regression_r2


# ----------------------------- 1 + 3: parity & CG --------------------------


def sparse_vs_dense_parity(
    Z_tr, y_tr, Z_te, cutoff, client, y_te=None, jitter=1e-6, tol=1e-3, device="cpu",
    linalg_mode="sparseCG", solve_maxiter=None, solve_tol=None,
):
    """Compare gp2Scale/sparseCG posterior mean vs the dense scipy.cdist Wendland
    GP on the SAME embedding. Returns a dict with max abs diff and correlation.
    If ``y_te`` is given, also reports each GP's R^2 against truth -- this is the
    number to compare against descriptor_eval/gp_parity.py (~0.09-0.12). All rows
    share one dummy category so masking is inert."""
    from gpcam import GPOptimizer
    from scipy.spatial.distance import cdist
    from sklearn.metrics import r2_score

    dim = Z_tr.shape[1]
    sv = float(np.var(y_tr))

    # Report the neighbour count THIS GP actually sees. It trains on len(Z_tr) rows
    # (the parity slice), NOT on the full train split, so the regime here is
    # density*len(Z_tr) -- much smaller than density*N_train. Comparing R² across
    # runs is only meaningful when THIS number matches; a cutoff percentile chosen
    # to preserve the neighbour count at the full N will silently starve the slice.
    _nb = (cdist(Z_tr, Z_tr) < cutoff).sum(axis=1) - 1
    print(
        f"[val] parity GP trains on {len(Z_tr):,} rows and sees "
        f"median={np.median(_nb):.0f} in-support neighbours/point "
        f"(mean={_nb.mean():.0f}). Compare R² across runs ONLY at a matching value."
    )

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
        batch_size=10_000, compute_device="cpu", device=device,
        linalg_mode=linalg_mode, solve_maxiter=solve_maxiter, solve_tol=solve_tol,
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


def streaming_pls_parity(X_tr, y_tr, X_te, y_te, n_components=10, tol=0.02,
                         scaling="pareto"):
    """Assert streaming SIMPLS embedding R^2 matches batch PLSRegression on a small
    slice (dense reference allowed here). Gate for the 200k reduction. ``scaling``
    must match the production SparsePLS scaling on BOTH sides or the comparison is
    between two different embeddings."""
    spls = SparsePLS(n_components=n_components, scaling=scaling).fit(X_tr, y_tr)
    Z_tr, Z_te = spls.transform(X_tr), spls.transform(X_te)
    r2_stream = regression_r2(Z_tr, y_tr, Z_te, y_te)
    r2_batch_emb, r2_batch_pls = batch_pls_r2(
        X_tr, y_tr, X_te, y_te, n_components, scaling=scaling
    )
    ok = abs(r2_stream - r2_batch_emb) < tol
    print(
        f"[val] PLS parity [{scaling}]: streaming R²={r2_stream:.4f} vs batch "
        f"R²={r2_batch_emb:.4f} (sklearn .score={r2_batch_pls:.4f}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return {
        "r2_streaming": r2_stream,
        "r2_batch_embed": r2_batch_emb,
        "r2_batch_pls_score": r2_batch_pls,
        "pass": ok,
    }


# ----------------------------- 5: PD guard ---------------------------------


def diagonal_guard(Z_sample, cutoff, signal_var, backend="wendland32"):
    """Regression guard: K[i,i] must be EXACTLY signal_var. Catches a distance
    backend returning nonzero self-distances (torch.cdist's mm expansion), which
    silently corrupts the solve on this near-singular Gram instead of erroring."""
    dim = Z_sample.shape[1]
    kern = make_wl_block_kernel(cutoff, dim=dim, use_category_tag=False,
                                backend=backend, device="cpu")
    res = check_kernel_diagonal(kern, Z_sample, signal_var)
    print(
        f"[val] diagonal [{backend}]: max err={res['max_diag_err']:.3e} "
        f"(rel {res['rel_diag_err']:.1e}) diag=[{res['diag_min']:.6f},"
        f"{res['diag_max']:.6f}] expected={signal_var:.6f} -> "
        f"{'PASS' if res['pass'] else 'FAIL'}"
    )
    return res


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


# ----------------------------- 6: additive-kernel parity -------------------


def additive_kernel_guard(Z_sample, cutoff, cat_sample=None, tol=1e-10):
    """Guard the AdditiveWendlandKernel against the proven single-block kernel.
    Pure-numpy (CPU), no gp2Scale needed:
      (a) a ONE-channel additive kernel is byte-identical to WLBlockKernel -- protects
          the conditioning-critical numerics (donot_use_mm cdist, float64) on the
          near-singular Gram;
      (b) a TWO-channel additive kernel equals the sum of two dense Wendland references
          (different per-channel cutoffs), masked by category;
      (c) the diagonal is exactly sum_c sv_c;
      (d) cross-category pairs are zeroed;
      (e) with ``cutoffs_are_hp=True``, ``hps[C+c]`` and ``ChannelSpec.cutoff`` are the
          SAME knob -- byte-exactly, across scale factors -- and the realised nnz is
          strictly monotone in the radius."""
    from .kernel import (AdditiveWendlandKernel, ChannelSpec, WLBlockKernel,
                         dense_wendland_reference)

    Z = np.asarray(Z_sample, float)
    n, D = Z.shape
    cat = np.zeros(n) if cat_sample is None else np.asarray(cat_sample, float)[:n]
    x = np.hstack([Z, cat[:, None]])

    # (a) single channel == WLBlockKernel
    wl = WLBlockKernel(cutoff=cutoff, dim=D, use_category_tag=True, device="cpu")
    add1 = AdditiveWendlandKernel(channels=[ChannelSpec(0, D, cutoff)], device="cpu")
    Kw = wl(x, x, np.array([2.0]))
    Ka = add1(x, x, np.array([2.0]))
    a_diff = float(np.max(np.abs(Kw - Ka)))
    a_ok = bool(np.array_equal(Kw, Ka))

    # (b) two channels == sum of two masked dense refs (distinct cutoffs)
    d0 = D // 2 or 1
    c1 = 1.3 * cutoff
    add2 = AdditiveWendlandKernel(
        channels=[ChannelSpec(0, d0, cutoff), ChannelSpec(d0, D, c1)], device="cpu")
    sv0, sv1 = 1.3, 0.7
    K2 = add2(x, x, np.array([sv0, sv1]))
    mask = (cat[:, None] == cat[None, :]).astype(float)
    ref = (sv0 * dense_wendland_reference(Z[:, :d0], Z[:, :d0], [1.0], cutoff)
           + sv1 * dense_wendland_reference(Z[:, d0:], Z[:, d0:], [1.0], c1)) * mask
    b_diff = float(np.max(np.abs(K2 - ref)))
    b_ok = b_diff <= tol

    # (c) diagonal == sv0 + sv1
    c_ok = bool(np.allclose(np.diag(K2), sv0 + sv1, atol=1e-9))

    # (d) cross-category zeroed
    i, j = np.where(cat[:, None] != cat[None, :])
    d_ok = (i.size == 0) or bool(np.allclose(K2[i, j], 0.0))

    # (e) the trainable radius IS the frozen radius.
    # This is the core correctness check for `cutoffs_are_hp`: a kernel built with the
    # flag and handed hps[C+c] = r must produce EXACTLY what a kernel built with
    # ChannelSpec.cutoff = r produces. Byte equality (not a tolerance) is the right bar
    # -- both paths run identical arithmetic and differ only in where the number came
    # from, so any discrepancy is a wiring bug rather than numerical drift. The monotone
    # nnz assertion is the half that actually answers "does the hyperparameter move the
    # SPARSITY", which is what makes a trainable radius meaningful at all.
    add2_hp = AdditiveWendlandKernel(
        channels=[ChannelSpec(0, d0, cutoff), ChannelSpec(d0, D, c1)],
        cutoffs_are_hp=True, device="cpu")
    e_exact, e_nnz = True, []
    for s in (0.5, 1.0, 2.0):
        ref_s = AdditiveWendlandKernel(
            channels=[ChannelSpec(0, d0, cutoff * s), ChannelSpec(d0, D, c1 * s)],
            device="cpu")
        Kt = add2_hp(x, x, np.array([sv0, sv1, cutoff * s, c1 * s]))
        e_exact &= bool(np.array_equal(np.asarray(Kt), np.asarray(ref_s(x, x, np.array([sv0, sv1])))))
        e_nnz.append(int(np.count_nonzero(np.asarray(Kt))))
    e_mono = e_nnz[0] < e_nnz[1] < e_nnz[2]
    # and the flag must GATE it: frozen kernels ignore any trailing hps
    e_gated = bool(np.array_equal(
        np.asarray(K2), np.asarray(add2(x, x, np.array([sv0, sv1, 4 * cutoff, 4 * c1])))))
    e_ok = e_exact and e_mono and e_gated

    ok = a_ok and b_ok and c_ok and d_ok and e_ok
    print(
        f"[val] additive kernel: (a) ==WLBlock diff={a_diff:.1e} {'ok' if a_ok else 'FAIL'} | "
        f"(b) ==sum-dense diff={b_diff:.1e} {'ok' if b_ok else 'FAIL'} | "
        f"(c) diag {'ok' if c_ok else 'FAIL'} | (d) cat-mask {'ok' if d_ok else 'FAIL'} | "
        f"(e) hps-radius exact={e_exact} nnz={e_nnz} mono={e_mono} gated={e_gated} "
        f"{'ok' if e_ok else 'FAIL'} -> {'PASS' if ok else 'FAIL'}"
    )
    return {"a_wlblock": a_ok, "b_sum_dense": b_ok, "c_diag": c_ok,
            "d_catmask": d_ok, "e_hps_radius": e_ok, "e_nnz": e_nnz, "pass": ok}


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
    ap.add_argument("--cutoff", type=float, default=None,
                    help="absolute compact-support radius; overrides --cutoff-pct")
    ap.add_argument("--parity-n", type=int, default=3000, help="train rows for parity")
    ap.add_argument("--device", default="cpu", help="cpu (local) or cuda")
    ap.add_argument("--linalg", default="sparseCG",
                    help="fvgp solver for the parity check: sparseCG, sparseLU, or "
                         "sparseCGpre (preconditioned; append _amg/_ilu to pick the "
                         "type). Use sparseCGpre to confirm it converges to the same "
                         "answer as the dense reference before the 200k run.")
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
        cutoff_percentile=args.cutoff_pct, cutoff_abs=args.cutoff,
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

    # 5: PD guard (both backends) + exact-diagonal regression guard
    ns = min(1500, len(Z_tr))
    psd_guard(Z_tr[:ns], cutoff, backend="wendland32")
    psd_guard(Z_tr[:ns], cutoff, backend="wendland_d0")
    diagonal_guard(Z_tr[:ns], cutoff, float(np.var(y_tr)))

    # 6: additive-kernel parity (one channel == WLBlockKernel; two == sum of refs)
    additive_kernel_guard(Z_tr[:ns], cutoff, cat_sample=cat_tr[:ns])

    # 1 + 3: parity + CG on a parity-sized slice, single dummy category
    k = min(args.parity_n, len(Z_tr))
    kt = min(args.parity_n // 3 or 1, len(Z_te))
    sparse_vs_dense_parity(
        Z_tr[:k], y_tr[:k], Z_te[:kt], cutoff, client, y_te=y_te[:kt],
        device=args.device, linalg_mode=args.linalg,
    )

    client.close()
    print("[val] done.")


if __name__ == "__main__":
    main()
