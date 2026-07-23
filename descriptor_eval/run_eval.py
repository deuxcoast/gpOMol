"""
run_eval.py  (descriptor_eval)
=============================
Orchestrator. For the chosen `--descriptor`, produces TWO variogram clouds
against the shared, descriptor-independent residual target:

  1. <tag>-raw  : Euclidean distance on the standardized raw descriptor
                  (validates the descriptor itself).
  2. <tag>-pls  : Euclidean distance on the PLS-reduced embedding the kernel
                  would actually see. PLS is supervised (fit on y) IN-SAMPLE
                  here -- the cloud is descriptive, not a generalization
                  claim -- and is labeled as such on the plot.

Descriptors:
  wl (default)  : the 323-dim hybrid WL+geometry+charge vector (features.py).
  persistence   : Rips persistent-homology images (persistence.py) -- tests
                  whether distances between persistence diagrams track the
                  property (Marcus's curiosity). Needs ripser + persim installed.

Usage
-----
    python run_eval.py --src ../train_4M --n 10000 --pls_components 10
    python run_eval.py --descriptor persistence --n 10000 --maxdim 1
"""

import argparse

import data as data_mod
import features as feat
import numpy as np
from variogram import build_variogram, plot_cloud, plot_empirical

# friendly aliases -> scipy pdist metric names. Anything not listed is passed
# straight through to scipy (e.g. 'cosine', 'chebyshev', 'braycurtis', 'jaccard').
METRIC_ALIASES = {
    "l1": "cityblock",
    "manhattan": "cityblock",
    "taxicab": "cityblock",
    "l2": "euclidean",
    "euclid": "euclidean",
    "linf": "chebyshev",
    "l_inf": "chebyshev",
    "chebyshev": "chebyshev",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pls_components", type=int, default=10)
    ap.add_argument(
        "--descriptor",
        choices=["wl", "persistence", "element-ph"],
        default="wl",
        help="which candidate descriptor to featurize (default wl). "
        "element-ph = element-specific persistent homology",
    )
    ap.add_argument(
        "--esph-top-k",
        type=int,
        default=6,
        help="element-ph: number of most-common elements to use as channels",
    )
    ap.add_argument(
        "--esph-pairs",
        default="none",
        choices=["none", "all"],
        help="element-ph: also add every element pair as an interactive channel",
    )
    ap.add_argument(
        "--maxdim",
        type=int,
        default=1,
        help="persistence: max homology dim (0=H0, 1=+rings, 2=+voids)",
    )
    ap.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="persistence: PersistenceImager pixel size (smaller = higher res)",
    )
    ap.add_argument(
        "--ph-thresh",
        type=float,
        default=None,
        help="persistence: cap the Rips filtration radius in Angstrom (speed/range guard)",
    )
    ap.add_argument(
        "--n_lags", type=int, default=20, help="empirical bins (equal-count)"
    )
    ap.add_argument(
        "--metric",
        default="euclidean",
        help="pairwise distance metric: euclidean (default), l1/cityblock, "
        "cosine, chebyshev, or any scipy pdist metric name",
    )
    ap.add_argument(
        "--cloud",
        action="store_true",
        help="render the hexbin cloud instead of the empirical curve (default)",
    )
    a = ap.parse_args()
    mode = "cloud" if a.cloud else "empirical"
    metric_label = a.metric.lower()  # goes in the filename/title
    metric_func = METRIC_ALIASES.get(metric_label, metric_label)  # goes to scipy
    print(f"[metric] {metric_label}  (scipy: {metric_func})")

    # 1. shared subset + descriptor-independent residual target
    atoms_list, y, _ = data_mod.get_data(a.src, n=a.n, seed=a.seed)

    # 2. raw descriptor matrix (candidate-specific), then standardize
    if a.descriptor == "element-ph":
        import persistence as pers

        X_raw = pers.ElementPHFeaturizer(
            maxdim=a.maxdim, pixel_size=a.pixel_size, thresh=a.ph_thresh,
            top_k=a.esph_top_k, pairs=a.esph_pairs,
        ).fit_transform(atoms_list)
        tag = f"ESPH-k{a.esph_top_k}-{a.esph_pairs}"
        raw_sub = f"Euclidean on standardized element-specific PH images ({X_raw.shape[1]}-dim)"
    elif a.descriptor == "persistence":
        import persistence as pers

        X_raw = pers.PersistenceFeaturizer(
            maxdim=a.maxdim, pixel_size=a.pixel_size, thresh=a.ph_thresh
        ).fit_transform(atoms_list)
        tag = f"PH-maxdim{a.maxdim}"
        raw_sub = f"Euclidean on standardized persistence images ({X_raw.shape[1]}-dim)"
    else:
        X_raw = feat.feature_matrix(atoms_list)
        tag = "WL-feature"
        raw_sub = f"Euclidean on standardized {X_raw.shape[1]}-dim descriptor"
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

    # 4. build each Variogram ONCE with the chosen metric, render the chosen mode
    for label, coords, sub in [
        (f"{tag}-raw", X_std, raw_sub),
        (f"{tag}-pls", Z, f"{a.pls_components}-comp PLS (in-sample)"),
    ]:
        V = build_variogram(coords, y, dist_func=metric_func, n_lags=a.n_lags)
        render = plot_cloud if a.cloud else plot_empirical
        sub_full = sub + (
            f"; n_lags={a.n_lags}, uniform bins, Matheron" if not a.cloud else ""
        )
        path = render(V, label, metric=metric_label, subtitle=sub_full)
        print(f"[saved:{mode}] {path}")


if __name__ == "__main__":
    main()
