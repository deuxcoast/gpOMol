#!/usr/bin/env python
"""
reduction_compare.py
====================
Does "PLS beats PCA" still hold for the NEW explicit-vocabulary WL descriptor, and
which reduction should feed the 200k GPU run?

At 200k the WL feature matrix is huge and sparse, and sklearn PLSRegression
densifies it. The candidates:
  1. TruncatedSVD          -- unsupervised, natively sparse, scales; but y-blind.
  2. Incremental/mini PLS  -- supervised, streams sparse minibatches; more code.
  3. SVD -> PLS (two-stage)-- TruncatedSVD to ~100-200 dims (sparse) then dense
                              PLS(10) on that; sparse-scalable AND y-aware.

This measures the QUALITY axis directly on the current 10k/20k data: held-out GP
R^2 through the identical Wendland pipeline, varying only the reduction. It
compares supervised PLS(10) against unsupervised TruncatedSVD at several dims
(does unsupervised need more components to catch up, as the old hybrid finding
showed?) and against the two-stage SVD->PLS(10). Cutoff is auto-calibrated per
method (--cutoff-pct) so no method is advantaged by scale.

Scalability note (not measured here, but decisive at 200k): centering/standardizing
densifies a sparse matrix, and PLSRegression densifies regardless. TruncatedSVD and
the SVD stage of the two-stage method run on the raw sparse counts. So if the
two-stage method matches PLS quality here, it is the recommended 200k path
(option 3): keep sparsity through the SVD stage, recover the supervised benefit in
a tiny dense PLS.

Usage
-----
    python reduction_compare.py --metric l2 --cutoff-pct 25 --min-count 2
    python reduction_compare.py --pool 20000 --subsample 8000 --cutoff-pct 25
"""

import argparse
import os
from datetime import datetime

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import gp_parity_l1 as gp
import learning_curve as lc
import matplotlib.pyplot as plt

# (reduction, final_k, svd_predims) configs to compare
CONFIGS = [
    ("pls", 10, None),  # supervised baseline (current)
    ("pca", 10, None),  # unsupervised, same dim  -> the direct PLS-vs-PCA test
    ("pca", 50, None),  # does unsupervised need more dims to catch up?
    ("pca", 100, None),
    ("svd_then_pls", 10, 200),  # two-stage (option 3): sparse-scalable + supervised
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="cap train size (for speed); test held fixed",
    )
    ap.add_argument("--wl-depth", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--cutoff-pct", type=float, default=25.0)
    ap.add_argument("--jitter", type=float, default=None)
    a = ap.parse_args()

    if a.pool is not None and a.pool != gp.SUBSET_N:
        atoms, y = lc.build_pool(a.pool, a.seed)
    else:
        atoms, y = gp.build_atoms()
    a_tr, a_te, y_tr, y_te = train_test_split(
        atoms, y, test_size=gp.TEST_FRACTION, random_state=gp.RANDOM_STATE
    )
    if a.subsample is not None and a.subsample < len(a_tr):
        a_tr, y_tr = a_tr[: a.subsample], y_tr[: a.subsample]
    print(
        f"[rc] train={len(y_tr)} test={len(y_te)}; comparing reductions "
        f"(metric={a.metric}, cutoff-pct={a.cutoff_pct})"
    )

    rows = []
    for reduction, k, pre in CONFIGS:
        res = gp.evaluate(
            a_tr,
            y_tr,
            a_te,
            y_te,
            wl_mode="explicit",
            wl_depth=a.wl_depth,
            min_count=a.min_count,
            reduction=reduction,
            pls_components=k,
            svd_predims=(pre if pre else 200),
            metric=a.metric,
            cutoff_pct=a.cutoff_pct,
            jitter=a.jitter,
            verbose=False,
        )
        label = res["embed"]
        print(
            f"[rc] {label:>14}  R2={res['r2']:+.3f}  RMSE={res['rmse']:.3f}  "
            f"cutoff={res['cutoff']:.3g}"
        )
        rows.append((label, reduction, k, res["r2"], res["rmse"]))

    # --- verdict ---
    def r2_of(reduction, k):
        for lab, red, kk, r2, rm in rows:
            if red == reduction and kk == k:
                return r2
        return None

    pls10 = r2_of("pls", 10)
    pca10 = r2_of("pca", 10)
    pca_best = max(r2_of("pca", 10), r2_of("pca", 50), r2_of("pca", 100))
    two_stage = r2_of("svd_then_pls", 10)

    print("\n[verdict]")
    print(
        f"  PLS(10)={pls10:+.3f}  vs  TruncatedSVD(10)={pca10:+.3f}  "
        f"(supervised edge at same dim: {pls10 - pca10:+.3f})"
    )
    print(f"  best unsupervised TruncatedSVD (<=100 dims) = {pca_best:+.3f}")
    print(f"  two-stage SVD200->PLS(10) = {two_stage:+.3f}")
    if pls10 - pca10 > 0.02:
        print("  => supervised STILL helps: plain TruncatedSVD(10) loses signal.")
        if two_stage >= pls10 - 0.01:
            print(
                "  => RECOMMEND option 3 (SVD->PLS): two-stage recovers PLS-level R^2 "
                "AND stays sparse-scalable. Best of both."
            )
        elif pca_best >= pls10 - 0.01:
            print(
                "  => TruncatedSVD catches up only with more dims; option 1 viable at "
                "higher k, but option 3 is cheaper for the same quality."
            )
        else:
            print(
                "  => neither unsupervised nor two-stage matches PLS; option 2 "
                "(Incremental/mini-batch PLS on sparse) is the faithful choice."
            )
    else:
        print(
            "  => supervised no longer helps on WL-explicit: PLS-vs-PCA gap has closed. "
            "RECOMMEND option 1 (TruncatedSVD) -- simplest and natively sparse."
        )

    # --- plot ---
    os.makedirs(gp.GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(gp.GRAPHS, f"GP-reduction-compare-{a.metric}-{ts}.png")
    labels = [r[0] for r in rows]
    r2s = [r[3] for r in rows]
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = [
            "#348ABD" if rows[i][1] in ("pls", "svd_then_pls") else "#E24A33"
            for i in range(len(rows))
        ]
        ax.bar(range(len(labels)), r2s, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("held-out test $R^2$")
        ax.set_title(
            f"Reduction comparison on WL-explicit (train={len(y_tr)})\n"
            "blue = supervised (y-aware), red = unsupervised"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
