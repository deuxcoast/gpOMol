#!/usr/bin/env python
"""
gp_parity.py  (L2 / Euclidean variant)
======================================
Validate the 323-dim WL feature vector with a COMPACT-SUPPORT WENDLAND kernel via
a gpCAM Gaussian process, and output a test-set parity plot. The distance NORM is
parameterized (default L2/euclidean; set 'cityblock' for L1).

PD status -- read carefully, it is NOT as simple as "L2 fixes it":
The Wendland used here, psi(r) = (1-r)^4 (4r+1), is the C2 Wendland psi_{3,1},
which is positive-definite as a radial function on Euclidean R^d ONLY for d <= 3.
Our feature space is 323-dimensional, so even with the L2 norm the Gram matrix is
NOT guaranteed PD -- the classic "dimension trap." What the L2 switch buys is
removing the *metric* violation (L1 is not a Euclidean distance at all); what it
does NOT buy is a dimension guarantee at d=323. Two things then decide whether the
solve succeeds at low jitter:
  (1) the smaller cutoff (10 vs 110) makes the Gram much sparser and more
      diagonally dominant, which empirically pushes it toward PD; but
  (2) if the cutoff is small relative to typical L2 distances in 323-standardized
      space (~sqrt(2*323) ~ 25), many test points will have FEW OR NO in-support
      training neighbours -> their kernel row is ~zero -> they revert to the prior
      mean anyway (a sparsity failure, distinct from the L1 jitter failure).
The final jitter level AND the parity spread together tell you which regime you're
in. A genuinely PD-guaranteed L2 kernel at d=323 needs psi_{323,k} (the
dimension-correct Wendland), not psi_{3,1} -- available on request.

Run from inside descriptor_eval/ (like run_eval.py); paths are anchored to this file.
"""

import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ EASILY MODIFIABLE KNOBS ========================
NORM = "euclidean"  # scipy metric: "euclidean" (L2) | "cityblock" (L1)
NORM_LABEL = "L2"  # goes in the filename / title
CUTOFF = 10.0  # compact-support radius in NORM space
SUBSET_N = 10_000  # number of frozen indices to use (all of them)
TEST_FRACTION = 0.20  # 80/20 split
RANDOM_STATE = 42  # deterministic split -- identical across runs
SIGMA_MULT = 2.0  # error bars = +/- 2 sigma (~95% interval)
JITTER_EXPONENTS = range(-6, 0)  # try 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so `import features` resolves regardless of cwd
import features  # your existing descriptor_eval/features.py

SRC = os.path.join(SCRIPT_DIR, "..", "train_4M")
CACHE = os.path.join(SCRIPT_DIR, "cache")
GRAPHS = os.path.join(SCRIPT_DIR, "graphs")


# ------------------------------ 1. data & preprocessing ---------------------


def build_dataset():
    """Load frozen indices + residual target, regenerate the standardized 323-dim
    WL feature matrix on the fly, in the SAME row order as y (X[k] <-> y[k])."""
    idx = np.load(os.path.join(CACHE, "subset_indices.npy"))
    y = np.load(os.path.join(CACHE, "y_residual.npy"))
    assert len(idx) == len(y), f"indices ({len(idx)}) and y ({len(y)}) length mismatch"
    if SUBSET_N < len(idx):
        idx, y = idx[:SUBSET_N], y[:SUBSET_N]  # frozen order => deterministic
    print(f"[data] {len(idx)} molecules; regenerating 323-dim features on the fly")

    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": SRC})
    X_raw = np.vstack([features.featurize(ds.get_atoms(int(i))) for i in idx])

    # Standardize over the FULL matrix (unsupervised): the cutoff was chosen in the
    # fully-standardized distance space, so the kernel geometry must match.
    X_std, _, _ = features.standardize(X_raw)
    print(f"[data] X {X_std.shape}, y {y.shape}  (residual var {np.var(y):.4g})")
    return X_std, y


# ------------------------------ 2. custom Wendland kernel -------------------


def wendland_kernel(x1, x2, hps):
    """
    gpCAM kernel callable: kernel(x1, x2, hps) -> (len(x1), len(x2)).

    Compact-support C2 Wendland psi(r) = (1-r)^4 (4r+1) over the NORM distance
    (default L2), scaled by signal variance hps[0]. psi = 0 for distance >= CUTOFF
    -> compact support. See module docstring re: PD at d=323 (not guaranteed).
    """
    signal_var = hps[0]
    D = cdist(x1, x2, metric=NORM)  # (N1, N2) distances
    r = np.clip(D / CUTOFF, 0.0, 1.0)
    psi = (1.0 - r) ** 4 * (4.0 * r + 1.0)  # = 0 exactly at r == 1
    return signal_var * psi


# ------------------------------ posterior extraction helpers ----------------


def _first(result_dict, keys):
    for k in keys:
        if k in result_dict:
            return np.asarray(result_dict[k]).ravel()
    raise KeyError(f"none of {keys} found in gpCAM result keys {list(result_dict)}")


def fit_and_predict(X_tr, y_tr, X_te, jitter, signal_var):
    """Build the GP with `jitter` as diagonal noise, factorize, predict the test
    set. Non-PD failures raise numpy.linalg.LinAlgError here (caught upstream)."""
    from gpcam import GPOptimizer

    gp = GPOptimizer(
        x_data=X_tr,
        y_data=y_tr,
        init_hyperparameters=np.array([signal_var]),  # hps[0] = signal variance
        kernel_function=wendland_kernel,
        noise_variances=jitter * np.ones(len(y_tr)),  # jitter = diagonal nudge
    )
    mean = _first(gp.posterior_mean(X_te), ["f(x)", "m(x)"])
    var = _first(
        gp.posterior_covariance(X_te, variance_only=True), ["v(x)", "S(x)", "variance"]
    )
    return mean, np.maximum(var, 0.0)  # clip tiny negatives from non-PD numerics


# ------------------------------ 3. jitter retry loop ------------------------


def fit_with_jitter(X_tr, y_tr, X_te, signal_var):
    """Escalating-jitter PD fallback: 1e-6 -> x10 -> ... -> 1e-1, then abort."""
    for e in JITTER_EXPONENTS:
        jitter = 10.0**e
        try:
            print(f"[jitter] trying diagonal noise = {jitter:.0e} ...")
            mean, var = fit_and_predict(X_tr, y_tr, X_te, jitter, signal_var)
            print(f"[jitter] SUCCESS at {jitter:.0e}")
            return mean, var, jitter
        except np.linalg.LinAlgError:
            print(f"[jitter] Cholesky failed at {jitter:.0e}; escalating x10")
    raise RuntimeError(
        f"{NORM_LABEL} Wendland covariance could not be regularized to PD up to "
        "jitter=1e-1; aborting."
    )


# ------------------------------ 4. parity plot ------------------------------


def parity_plot(y_obs, y_pred, y_std, rmse, r2, jitter):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-parity-{NORM_LABEL}-{ts}.png")

    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.errorbar(
            y_obs,
            y_pred,
            yerr=SIGMA_MULT * y_std,
            fmt="o",
            ms=4,
            alpha=0.45,
            elinewidth=0.7,
            capsize=0,
            color="#348ABD",
            ecolor="#B0B0B0",
            label=rf"test predictions ($\pm{int(SIGMA_MULT)}\sigma$)",
        )
        lo = float(min(y_obs.min(), y_pred.min()))
        hi = float(max(y_obs.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="perfect (y = x)")
        ax.set_xlabel("Observed Residual Energy")
        ax.set_ylabel("Predicted Residual Energy")
        ax.set_title(
            f"GP parity — {NORM_LABEL} Wendland (cutoff={CUTOFF:g})\n"
            f"RMSE = {rmse:.4g}    $R^2$ = {r2:.3f}    jitter = {jitter:.0e}"
        )
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


# ------------------------------ main ----------------------------------------


def main():
    X, y = build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    print(
        f"[split] train {X_tr.shape[0]}  test {X_te.shape[0]}  (random_state={RANDOM_STATE})"
    )

    signal_var = float(np.var(y_tr))  # fixed: we validate the kernel SHAPE, not HPs
    mean, var, jitter = fit_with_jitter(X_tr, y_tr, X_te, signal_var)

    rmse = float(np.sqrt(np.mean((mean - y_te) ** 2)))
    r2 = float(r2_score(y_te, mean))
    print(f"[result] RMSE = {rmse:.4g}   R^2 = {r2:.3f}   (final jitter {jitter:.0e})")

    path = parity_plot(y_te, mean, np.sqrt(var), rmse, r2, jitter)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
