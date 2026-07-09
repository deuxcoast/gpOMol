"""
variogram.py  (descriptor_eval)
==============================
Two views of the SAME scikit-gstat Variogram (built once -- the ~50M-pair distance
computation is the bottleneck, so we pay it a single time and choose how to render):

  * empirical (DEFAULT): the experimental semivariogram curve -- bin centers vs
    Matheron mean semivariance -- with marker size + colour scaled by per-bin pair
    count so under-populated lags are visible. fivethirtyeight style.
  * cloud (--cloud): the full pairwise hexbin density (previous behaviour).

Binning uses skgstat's own experimental values (V.bins / V.experimental /
V.bin_count) so the curve can never desync from the cloud. Defaults: n_lags=20,
bin_func='uniform' (equal-count -- robust when descriptor distances cluster),
estimator='matheron' (the literal 1/2 (y_i - y_j)^2 we plot in the cloud).
dist_func is passed through for later candidates (e.g. 'jaccard' for Morgan).
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skgstat as skg

# ----------------------------- build (once) --------------------------------


def build_variogram(
    coordinates,
    values,
    dist_func: str = "euclidean",
    n_lags: int = 20,
    bin_func: str = "uniform",
):
    """Build the skgstat Variogram once; both renderers read from it."""
    return skg.Variogram(
        np.asarray(coordinates, dtype=float),
        np.asarray(values, dtype=float),
        dist_func=dist_func,
        n_lags=n_lags,
        bin_func=bin_func,
        estimator="matheron",
        normalize=False,
    )


def cloud_arrays(V):
    """(distance, semivariance) over all pairs, scipy-pdist order."""
    return np.asarray(V.distance), 0.5 * np.asarray(V._diff) ** 2


def empirical_arrays(V):
    """(lag, semivariance, count) per bin, NaN bins dropped."""
    bins = np.asarray(V.bins, dtype=float)
    gamma = np.asarray(V.experimental, dtype=float)
    try:
        counts = np.asarray(V.bin_count, dtype=float)
    except Exception:
        lg = V.lag_groups()
        counts = np.bincount(lg[lg >= 0], minlength=len(bins)).astype(float)
    counts = counts[: len(bins)]
    ok = ~np.isnan(gamma)
    return bins[ok], gamma[ok], counts[ok]


# ----------------------------- filename ------------------------------------


def _path(descriptor: str, mode: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    return os.path.join(out_dir, f"{descriptor}-{mode}-{ts}.png")


# ----------------------------- renderers -----------------------------------


def plot_empirical(
    V, descriptor: str, out_dir: str = "variograms", subtitle: str = ""
) -> str:
    """Experimental semivariogram curve; marker size & colour = per-bin pair count."""
    lag, gamma, counts = empirical_arrays(V)
    path = _path(descriptor, "empirical", out_dir)
    cmax = counts.max() if counts.max() > 0 else 1.0
    sizes = 40.0 + 400.0 * (counts / cmax)

    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(lag, gamma, "-", lw=2, color="#777777", zorder=1)
        sc = ax.scatter(
            lag,
            gamma,
            s=sizes,
            c=counts,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        fig.colorbar(sc, ax=ax, label="pairs per bin")
        ax.set_xlabel("descriptor distance (lag)")
        ax.set_ylabel(r"semivariance  $\frac{1}{2}(y_i-y_j)^2$")
        ax.set_title(
            f"Empirical semivariogram — {descriptor}"
            + (f"\n{subtitle}" if subtitle else "")
        )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


def plot_cloud(
    V,
    descriptor: str,
    out_dir: str = "variograms",
    subtitle: str = "",
    gridsize: int = 120,
) -> str:
    """Full pairwise hexbin density (log colour scale)."""
    distance, semivariance = cloud_arrays(V)
    path = _path(descriptor, "cloud", out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        distance, semivariance, gridsize=gridsize, bins="log", mincnt=1, cmap="viridis"
    )
    fig.colorbar(hb, ax=ax, label="pair count (log)")
    ax.set_xlabel("descriptor distance")
    ax.set_ylabel(r"semivariance  $\frac{1}{2}(y_i-y_j)^2$   (residual energy)")
    ax.set_title(
        f"Variogram cloud — {descriptor}   (n_pairs={len(distance):,})"
        + (f"\n{subtitle}" if subtitle else "")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
