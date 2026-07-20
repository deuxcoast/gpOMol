"""
plots.py  (wl_gp2scale)
=======================
Headless PNG plots for the radius diagnostics (matplotlib Agg -- safe on a compute
node with no display). Two figures:

  * plot_semivariogram: gamma(h) vs embedding distance, with the sill and the fitted
    range marked -- the a-priori, GP-free radius picker.
  * plot_rmse_vs_nn:    test RMSE vs distance-to-nearest-training-neighbour, with the
    predict-the-mean baseline and the informative radius R_inf -- the a-posteriori
    cross-check (consumes the dict from radius.rmse_vs_nn_distance).

Both take an explicit output path and return it; the caller builds the path (see
``_path``) so filenames stay consistent across the two figures of one run.
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _path(name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    return os.path.join(out_dir, f"{name}-{ts}.png")


def plot_semivariogram(
    lag, gamma, sill, range_, out_dir="diagnostics", subtitle="", name="semivariogram"
):
    """Empirical gamma(h) with sill (dashed) and effective range (vertical) marked."""
    lag = np.asarray(lag, float)
    gamma = np.asarray(gamma, float)
    path = _path(name, out_dir)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lag, gamma, "o-", color="#2b6cb0", lw=1.5, ms=5, label=r"$\gamma(h)$")
    if np.isfinite(sill) and sill > 0:
        ax.axhline(
            sill, ls="--", color="#718096", lw=1.2, label=f"sill = Var(y) = {sill:.3g}"
        )
    if range_ is not None:
        ax.axvline(
            range_, ls=":", color="#c53030", lw=1.5, label=f"range = {range_:.4f}"
        )
    ax.set_xlabel("embedding distance  h")
    ax.set_ylabel(r"semivariance  $\gamma(h)=\frac{1}{2}\langle(y_i-y_j)^2\rangle$")
    title = "Semivariogram of the target over embedding distance"
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=8)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_rmse_vs_nn(curve, out_dir="diagnostics", subtitle="", name="rmse_vs_nn"):
    """RMSE vs nearest-train-neighbour distance (per-bin + cumulative), baseline and
    R_inf marked. ``curve`` is the dict returned by radius.rmse_vs_nn_distance."""
    med = np.asarray(curve["bin_median_nn"], float)
    rms = np.asarray(curve["bin_rmse"], float)
    base = float(curve["baseline"])
    R_inf = curve.get("radius")
    path = _path(name, out_dir)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(med, rms, "o-", color="#2b6cb0", lw=1.5, ms=5, label="per-bin RMSE")
    if len(curve.get("cum_nn", [])):
        ax.plot(
            curve["cum_nn"],
            curve["cum_rmse"],
            "s--",
            color="#805ad5",
            lw=1.2,
            ms=4,
            label="cumulative RMSE",
        )
    ax.axhline(
        base,
        ls="--",
        color="#718096",
        lw=1.2,
        label=f"baseline (predict mean) = {base:.3g}",
    )
    if R_inf is not None:
        ax.axvline(
            R_inf, ls=":", color="#c53030", lw=1.5, label=f"$R_{{inf}}$ = {R_inf:.4f}"
        )
    ax.set_xlabel("distance to nearest training neighbour")
    ax.set_ylabel("test RMSE")
    title = "Prediction error vs nearest-neighbour distance"
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=8)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
