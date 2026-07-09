"""
run_eval.py  (descriptor_eval)
=============================
Orchestrator for Candidate A. Produces TWO variogram clouds against the shared,
descriptor-independent residual target:

  1. WL-feature-raw  : Euclidean distance on the standardized 323-dim descriptor
                       (validates the descriptor itself).
  2. WL-feature-pls  : Euclidean distance on the PLS-reduced embedding the kernel
                       would actually see. PLS is supervised (fit on y) IN-SAMPLE
                       here -- the cloud is descriptive, not a generalization
                       claim -- and is labeled as such on the plot.

Usage
-----
    python run_eval.py --src ../train_4M --n 10000 --pls_components 10
"""

import argparse

import data as data_mod
import features as feat
import numpy as np
from variogram import build_variogram, plot_cloud, plot_empirical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pls_components", type=int, default=10)
    ap.add_argument(
        "--n_lags", type=int, default=20, help="empirical bins (equal-count)"
    )
    ap.add_argument(
        "--cloud",
        action="store_true",
        help="render the hexbin cloud instead of the empirical curve (default)",
    )
    a = ap.parse_args()
    mode = "cloud" if a.cloud else "empirical"

    # 1. shared subset + descriptor-independent residual target
    atoms_list, y, _ = data_mod.get_data(a.src, n=a.n, seed=a.seed)

    # 2. raw standardized descriptor
    X_raw = feat.feature_matrix(atoms_list)
    X_std, _, _ = feat.standardize(X_raw)
    print(f"[features] standardized matrix {X_std.shape}")

    # 3. PLS-reduced embedding (supervised, in-sample -- descriptive only)
    from sklearn.cross_decomposition import PLSRegression

    Z = (
        PLSRegression(n_components=a.pls_components, scale=False)
        .fit(X_std, y)
        .transform(X_std)
    )
    print(f"[features] PLS embedding {Z.shape}")

    # 4. build each Variogram ONCE, render the chosen mode
    for label, coords, sub in [
        ("WL-feature-raw", X_std, "Euclidean on standardized 323-dim descriptor"),
        ("WL-feature-pls", Z, f"Euclidean on {a.pls_components}-comp PLS (in-sample)"),
    ]:
        V = build_variogram(coords, y, n_lags=a.n_lags)
        render = plot_cloud if a.cloud else plot_empirical
        sub_full = sub + (
            f"; n_lags={a.n_lags}, uniform bins, Matheron" if not a.cloud else ""
        )
        path = render(V, label, subtitle=sub_full)
        print(f"[saved:{mode}] {path}")


if __name__ == "__main__":
    main()
