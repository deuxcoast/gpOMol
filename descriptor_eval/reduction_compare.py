#!/usr/bin/env python
"""
reduction_compare.py
====================
Which reduction to a 10-D embedding should feed the 200k GPU run, for the new
explicit-vocabulary WL descriptor?

Candidates: supervised PLS (current, densifies), unsupervised TruncatedSVD (sparse
but y-blind), two-stage SVD->PLS, and a supervised univariate PRESCREEN->PLS
(keep the screen_k columns most correlated with y -- one sparse matvec -- then
dense PLS).

CRITICAL FIX vs the first version: each reduction produces a different distance
scale, so a single --cutoff-pct lands each method at a different point on its own
neighbor curve -- confounding reduction quality with cutoff. This version
featurizes ONCE, reduces once per method, then SWEEPS the cutoff percentile per
method and reports each method's BEST R^2 (apples-to-apples at each one's optimum).

Usage
-----
    python reduction_compare.py --pool 20000 --subsample 8000 --metric l2
    python reduction_compare.py --sweep 5,10,25,50,75 --subsample 8000
"""

import argparse
import os
from datetime import datetime

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import gp_parity as gp
import learning_curve as lc
import matplotlib.pyplot as plt

# (reduction, final_k, extra_kwargs)
CONFIGS = [
    ("pls", 10, {}),
    ("pca", 10, {}),
    ("svd_then_pls", 10, {"svd_predims": 200}),
    ("screen_then_pls", 10, {"screen_k": 300}),
    ("screen_then_pls", 10, {"screen_k": 1000}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--wl-depth", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument(
        "--sweep",
        default="10,25,50,75",
        help="cutoff percentiles to sweep per method; best R^2 is reported",
    )
    ap.add_argument("--jitter", type=float, default=None)
    a = ap.parse_args()
    sweep = [float(s) for s in a.sweep.split(",")]

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
        f"[rc] train={len(y_tr)} test={len(y_te)}; metric={a.metric}; "
        f"cutoff-pct sweep={sweep}"
    )

    # featurize ONCE (shared by every reduction)
    Xs_tr, Xs_te, feat = gp.featurize_wl(
        a_tr,
        a_te,
        wl_mode="explicit",
        wl_depth=a.wl_depth,
        min_count=a.min_count,
        verbose=True,
    )

    rows = []
    for reduction, k, extra in CONFIGS:
        Z_tr, Z_te, embed = gp._reduce(
            Xs_tr,
            y_tr,
            Xs_te,
            reduction,
            k,
            extra.get("svd_predims", 200),
            extra.get("screen_k", 300),
        )
        best, detail = None, []
        for pct in sweep:
            try:
                res = gp.gp_eval_embedding(
                    Z_tr,
                    y_tr,
                    Z_te,
                    y_te,
                    metric=a.metric,
                    cutoff_pct=pct,
                    jitter=a.jitter,
                )
                detail.append((pct, res["r2"]))
                if best is None or res["r2"] > best[1]:
                    best = (pct, res["r2"], res["cutoff"])
            except RuntimeError:
                detail.append((pct, None))  # non-PD at this cutoff
        sweepstr = " ".join(
            f"{p:g}:{('%+.3f' % r) if r is not None else 'PD!'}" for p, r in detail
        )
        if best is None:
            print(f"[rc] {embed:>16}  non-PD at ALL cutoffs   [{sweepstr}]")
            rows.append((embed, reduction, k, None, None))
        else:
            pct, r2, cut = best
            print(
                f"[rc] {embed:>16}  BEST R2={r2:+.3f} @pct{pct:g} (cutoff {cut:.3g})"
                f"   [sweep {sweepstr}]"
            )
            rows.append((embed, reduction, k, r2, pct))

    # --- verdict (best-over-sweep per method; None = non-PD everywhere) ---
    def best_of(reduction):
        vals = [
            r2 for lab, red, kk, r2, pct in rows if red == reduction and r2 is not None
        ]
        return max(vals) if vals else None

    pls10 = best_of("pls")
    pca10 = best_of("pca")
    two_stage = best_of("svd_then_pls")
    screen = best_of("screen_then_pls")

    def fmt(v):
        return f"{v:+.3f}" if v is not None else "FAILED"

    print("\n[verdict] (each at its OWN best cutoff)")
    print(
        f"  PLS={fmt(pls10)}   TruncatedSVD={fmt(pca10)}   "
        f"SVD->PLS={fmt(two_stage)}   SCREEN->PLS={fmt(screen)}"
    )
    if pls10 is None:
        print("  => PLS failed at all cutoffs; widen --sweep.")
    else:
        if pca10 is not None and pls10 - pca10 > 0.02:
            print(
                f"  => supervised required (PLS beats TruncatedSVD by {pls10 - pca10:+.3f})."
            )
        if screen is not None and screen >= pls10 - 0.01:
            print(
                f"  => RECOMMEND supervised PRESCREEN->PLS ({screen:+.3f} vs PLS "
                f"{pls10:+.3f}): sparse-scalable, no streaming PLS needed. Best 200k option."
            )
        elif two_stage is not None and two_stage >= pls10 - 0.01:
            print(
                f"  => SVD->PLS matches PLS at its best cutoff ({two_stage:+.3f}); viable."
            )
        else:
            alt = max([v for v in (screen, two_stage) if v is not None], default=None)
            print(
                f"  => even at best cutoff, no sparse-scalable reduction matches PLS "
                f"(best alt {fmt(alt)} vs {pls10:+.3f}). Signal is DIFFUSE across columns "
                "-> univariate screening loses it. Option 2 (Incremental PLS) confirmed."
            )

    # --- plot ---
    ok = [(lab, red, r2) for lab, red, kk, r2, pct in rows if r2 is not None]
    if ok:
        os.makedirs(gp.GRAPHS, exist_ok=True)
        ts = datetime.now().strftime("%m-%d-%H-%M-%S")
        path = os.path.join(gp.GRAPHS, f"GP-reduction-sweep-{a.metric}-{ts}.png")
        labels = [o[0] for o in ok]
        r2s = [o[2] for o in ok]
        colors = [
            (
                "#348ABD"
                if o[1] in ("pls", "svd_then_pls", "screen_then_pls")
                else "#E24A33"
            )
            for o in ok
        ]
        with plt.style.context("fivethirtyeight"):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.bar(range(len(labels)), r2s, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("best held-out $R^2$ (over cutoff sweep)")
            ax.set_title(
                f"Reduction comparison (cutoff-swept) — WL-explicit, train={len(y_tr)}\n"
                "blue = supervised, red = unsupervised"
            )
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
