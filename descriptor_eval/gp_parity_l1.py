#!/usr/bin/env python
"""
gp_parity_l1.py
===============
Validate the 323-dim WL feature vector with a COMPACT-SUPPORT WENDLAND kernel
built on the L1 (Manhattan) norm, via a gpCAM Gaussian process. Output: a
test-set parity plot.

PREMISE / WARNING (this is the whole point of the experiment):
The Wendland polynomial psi(r) = (1-r)^4 (4r+1) is positive-definite as a RADIAL
function on EUCLIDEAN (L2) space, up to a design dimension. Feeding it an L1
distance breaks that guarantee -- psi(||.||_1) is not a known PD kernel -- so the
covariance matrix can carry negative eigenvalues and Cholesky can fail. The jitter
retry loop (Section 3) is the mitigation, and the FINAL JITTER LEVEL is itself a
result: the larger the diagonal we had to add to force PD-ness, the more the L1
kernel is being bent into shape, and the more skeptically the predictions and
uncertainties should be read.

Run from inside descriptor_eval/ (like run_eval.py); paths are anchored to this
file, so cwd doesn't matter.
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
CUTOFF = 110.0  # compact-support radius in L1 space (fixed, per spec)
SUBSET_N = 10_000  # number of frozen indices to use (all of them)
TEST_FRACTION = 0.20  # 80/20 split
RANDOM_STATE = 42  # deterministic split -- identical across runs
SIGMA_MULT = 2.0  # error bars = +/- 2 sigma (~95% interval)
JITTER_EXPONENTS = range(-1, 2)  # try 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so `import features` resolves regardless of cwd
import features  # your existing descriptor_eval/features.py

SRC = os.path.join(SCRIPT_DIR, "..", "train_4M")
CACHE = os.path.join(SCRIPT_DIR, "cache")
GRAPHS = os.path.join(SCRIPT_DIR, "graphs")


# ------------------------------ 1. data & preprocessing ---------------------


def build_dataset():
    """Load the frozen indices + residual target, regenerate the standardized
    323-dim WL feature matrix on the fly from the LMDB, in the SAME row order as
    y (so X[k] <-> y[k])."""
    idx = np.load(os.path.join(CACHE, "subset_indices.npy"))
    y = np.load(os.path.join(CACHE, "y_residual.npy"))
    assert len(idx) == len(y), f"indices ({len(idx)}) and y ({len(y)}) length mismatch"
    if SUBSET_N < len(idx):
        idx, y = idx[:SUBSET_N], y[:SUBSET_N]  # frozen order => deterministic
    print(f"[data] {len(idx)} molecules; regenerating 323-dim features on the fly")

    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": SRC})
    X_raw = np.vstack([features.featurize(ds.get_atoms(int(i))) for i in idx])

    # Standardize over the FULL matrix (not train-only) on purpose: the CUTOFF of
    # 110 was derived from the L1 semivariogram of the fully-standardized features,
    # so the kernel geometry only matches that analysis if we standardize the same
    # way here. Standardization is unsupervised (uses X, not y).
    X_std, _, _ = features.standardize(X_raw)
    print(f"[data] X {X_std.shape}, y {y.shape}  (residual var {np.var(y):.4g})")
    return X_std, y


# ------------------------------ 2. custom L1 Wendland kernel -----------------


def l1_wendland_kernel(x1, x2, hps):
    """
    gpCAM kernel callable: kernel(x1, x2, hps) -> (len(x1), len(x2)).

    Compact-support C2 Wendland psi(r) = (1-r)^4 (4r+1) applied to the L1
    (Manhattan) distance, scaled by signal variance hps[0]. psi = 0 for r >= 1,
    i.e. for L1 distance >= CUTOFF -> compact support (and the source of sparsity
    the full-scale kernel would exploit).

    NOTE: L1 breaks the Wendland PD guarantee (see module docstring). The Gram
    matrix may not be PD; that is handled upstream by the jitter loop.
    """
    signal_var = hps[0]
    D = cdist(x1, x2, metric="cityblock")  # (N1, N2) L1 distances
    r = np.clip(D / CUTOFF, 0.0, 1.0)
    psi = (1.0 - r) ** 4 * (4.0 * r + 1.0)  # = 0 exactly at r == 1
    return signal_var * psi


# ------------------------------ posterior extraction helpers ----------------


def _first(result_dict, keys):
    """Pull the first matching key from a gpCAM posterior dict (key names vary
    slightly across versions)."""
    for k in keys:
        if k in result_dict:
            return np.asarray(result_dict[k]).ravel()
    raise KeyError(f"none of {keys} found in gpCAM result keys {list(result_dict)}")


def fit_and_predict(X_tr, y_tr, X_te, jitter, signal_var):
    """Build the GP with `jitter` as diagonal noise, factorize, and predict the
    test set. Any non-PD failure raises numpy.linalg.LinAlgError here, which the
    jitter loop catches."""
    from gpcam import GPOptimizer

    gp = GPOptimizer(
        x_data=X_tr,
        y_data=y_tr,
        init_hyperparameters=np.array([signal_var]),  # hps[0] = signal variance
        kernel_function=l1_wendland_kernel,
        noise_variances=jitter * np.ones(len(y_tr)),  # jitter = diagonal nudge
    )
    # These calls force the Cholesky / solve (where non-PD blows up).
    mean = _first(gp.posterior_mean(X_te), ["f(x)", "m(x)"])
    var = _first(
        gp.posterior_covariance(X_te, variance_only=True), ["v(x)", "S(x)", "variance"]
    )
    return mean, np.maximum(var, 0.0)  # clip tiny negatives from non-PD numerics


# ------------------------------ 3. jitter retry loop ------------------------


def fit_with_jitter(X_tr, y_tr, X_te, signal_var):
    """Escalating-jitter PD fallback: 1e-6 -> x10 -> ... -> 10, then abort."""
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
        "L1 Wendland covariance could not be regularized to PD up to jitter=1e-1. "
        "The kernel is too far from positive-definite at this cutoff/data; aborting."
    )


# ------------------------------ 4. parity plot ------------------------------


def parity_plot(y_obs, y_pred, y_std, rmse, r2, jitter):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-parity-L1-{ts}.png")

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
            f"GP parity — L1 Wendland (cutoff={CUTOFF:g})\n"
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

    # Fixed signal variance: the kernel SHAPE (L1 Wendland at this cutoff) is what
    # we are validating, so we don't tune hyperparameters -- keeps it deterministic.
    signal_var = float(np.var(y_tr))
    mean, var, jitter = fit_with_jitter(X_tr, y_tr, X_te, signal_var)

    rmse = float(np.sqrt(np.mean((mean - y_te) ** 2)))
    r2 = float(r2_score(y_te, mean))
    print(f"[result] RMSE = {rmse:.4g}   R^2 = {r2:.3f}   (final jitter {jitter:.0e})")

    path = parity_plot(y_te, mean, np.sqrt(var), rmse, r2, jitter)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
