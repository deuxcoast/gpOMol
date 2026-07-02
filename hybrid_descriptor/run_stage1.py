"""
run_stage1.py
=============
Stage 1: the decisive gate on a subsample of train_4M, BEFORE any cluster fit.

  load 10^5 -> fit extensive mean + hybrid features + PCA
            -> run falsification gate + sparsity/accuracy sweep on a few-thousand
               sub-sample (pairwise diagnostics are O(n^2) in memory)
            -> save the fitted preprocessor for reuse in Stages 2-3.

The number that decides everything is s* at r_full in the sweep: whether an exact
GP over the full target_N fits the storage budget, or how much correlation you'd
trade to make it fit.

Usage
-----
    # quick smoke first (confirms keys/accessor/graph build work):
    python run_stage1.py --n_fit 2000 --n_diag 1500 --out smoke

    # the real Stage 1:
    python run_stage1.py --src ../train_4M --n_fit 100000 --n_diag 4000 --out stage1_4M
"""

import argparse
import pickle

import diagnostics as dg
import numpy as np
from embedding_kernel import (
    check_kernel_psd,
    default_hp_bounds,
    make_wendland_mahalanobis,
)
from gp_fit import HybridPreprocessor, MolBatch, load_omol25_subset  # noqa: F401


def stage1(
    batch: "MolBatch",
    n_components: int = 15,
    wendland_k: int = 2,
    n_diag: int = 4000,
    target_N: int = 4_000_000,
    seed: int = 0,
    out: str = None,
):
    # ---- fit preprocessor on the FULL subsample (cheap: OLS + one SVD) --------
    pre = HybridPreprocessor(n_components=n_components)
    X, resid = pre.fit(batch)
    print(
        f"[preprocess] N={len(X):,}  D={X.shape[1]}  "
        f"retained_var={pre.reducer.retained_variance():.3f}  "
        f"residual_var={np.var(resid):.4g}  mean_extra_cols={pre.mean_model.n_extra_}"
    )

    # ---- pairwise diagnostics on a sub-sample (O(n^2) memory guard) ----------
    rng = np.random.default_rng(seed)
    m = min(n_diag, len(X))
    sel = rng.choice(len(X), size=m, replace=False)
    Xd, rd = X[sel], resid[sel]
    Xwl = pre.wl_only_embedding(batch)[sel]
    print(f"[gate] running diagnostics on {m:,} of {len(X):,} points\n")

    report = dg.run_falsification(Xd, Xwl, rd, target_N=target_N)
    print(report.summary())

    # ---- PD guard on the REAL embedding at the chosen D ----------------------
    kernel = make_wendland_mahalanobis(
        dim=n_components, k=wendland_k, backend="explicit"
    )
    bounds = default_hp_bounds(Xd, rd)
    init = np.concatenate([[np.var(rd)], 0.5 * (bounds[1:, 0] + bounds[1:, 1])])
    psd = check_kernel_psd(kernel, Xd[:800], init)
    print(
        f"\n[PD guard] min_eig={psd['min_eigenvalue']:.3e}  is_psd={psd['is_psd']}  "
        f"gram_density={psd['gram_density']:.3f}"
    )

    # ---- the decisive sweep --------------------------------------------------
    sweep = dg.sparsity_accuracy_sweep(Xd, rd, target_N=target_N)
    print()
    print(dg.format_sweep(sweep))

    # ---- persist the fitted preprocessor (reuse for Stage 2/3) ---------------
    if out:
        np.savez(
            f"{out}.npz",
            X=X,
            resid=resid,
            net_charges=batch.net_charges,
            spins=batch.spins,
        )
        with open(f"{out}_preprocessor.pkl", "wb") as f:
            pickle.dump(pre, f)
        print(
            f"\n[saved] {out}.npz  +  {out}_preprocessor.pkl "
            f"(apply this same fitted preprocessor to the full 4M in Stage 3)"
        )
    return pre, report, sweep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n_fit", type=int, default=100_000, help="subsample to fit on")
    ap.add_argument("--n_diag", type=int, default=4000, help="pairwise-diagnostic size")
    ap.add_argument(
        "--n_components", type=int, default=15, help="PCA dim (<= Wendland d0)"
    )
    ap.add_argument("--wendland_k", type=int, default=2, help="Wendland smoothness")
    ap.add_argument("--target_N", type=int, default=4_000_000)
    ap.add_argument("--charge_key", default="lowdin_charges")
    ap.add_argument("--cutoff_mult", type=float, default=1.2)
    ap.add_argument(
        "--size_cap", type=int, default=None, help="skip records > this many atoms"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stage1_4M")
    a = ap.parse_args()

    batch = load_omol25_subset(
        a.src,
        n=a.n_fit,
        seed=a.seed,
        charge_key=a.charge_key,
        cutoff_mult=a.cutoff_mult,
        size_cap=a.size_cap,
    )
    stage1(
        batch,
        n_components=a.n_components,
        wendland_k=a.wendland_k,
        n_diag=a.n_diag,
        target_N=a.target_N,
        seed=a.seed,
        out=a.out,
    )


if __name__ == "__main__":
    main()
