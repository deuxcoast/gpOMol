#!/usr/bin/env python
"""
gp_parity.py  (PLS + L2 Wendland variant)
=========================================
Validate the WL descriptor with a compact-support Wendland GP on the SUPERVISED
PLS embedding (not the raw 323-dim vector), and output a test-set parity plot.

Why PLS here: on the raw 323-dim descriptor the L2 Wendland was PSD at jitter=1e-6
but gave R^2 ~ 0.02 -- a SPARSITY failure, because a cutoff of ~10 is far below the
~25 median pairwise L2 distance in 323-standardized space, so test points had no
in-support neighbours. PLS compresses the signal into ~10 dimensions where typical
distances are much smaller, so a cutoff near the PLS variogram range should give a
well-populated (non-degenerate) kernel.

Leakage control: PLS is supervised, so the standardizer AND the PLS model are fit
on the TRAIN split only and then applied to test. Fitting PLS on all data would
leak test labels into the embedding and inflate R^2.

PD note: the Wendland psi_{3,1} is PD on Euclidean R^d only for d <= 3, so even at
d=PLS_COMPONENTS it is not guaranteed PD; the jitter loop handles the rest. If
jitter climbs, switch to the dimension-correct psi_{d,k} (available on request).

A scale diagnostic prints BEFORE the GP so you can calibrate the cutoff from the
actual PLS-space distances rather than guessing.

Run from inside descriptor_eval/ (like run_eval.py); paths are anchored to this file.
"""

import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ EASILY MODIFIABLE KNOBS ========================
NORM = "euclidean"  # scipy metric: "euclidean" (L2) | "cityblock" (L1)
NORM_LABEL = "L2"
PLS_COMPONENTS = 10  # supervised reduction dimension
CUTOFF = 4.0  # compact-support radius in PLS-embedding NORM space
SUBSET_N = 10_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42  # deterministic split
SIGMA_MULT = 2.0  # error bars = +/- 2 sigma
JITTER_EXPONENTS = range(-6, 0)  # 1e-6 ... 1e-1
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import features

SRC = os.path.join(SCRIPT_DIR, "..", "train_4M")
CACHE = os.path.join(SCRIPT_DIR, "cache")
GRAPHS = os.path.join(SCRIPT_DIR, "graphs")
RUN_LABEL = f"PLS{PLS_COMPONENTS}-{NORM_LABEL}"


# ------------------------------ 1. data (raw features) ----------------------


def build_raw():
    """Load frozen indices + residual target; regenerate the RAW 323-dim feature
    matrix (standardization + PLS happen AFTER the split, train-only)."""
    idx = np.load(os.path.join(CACHE, "subset_indices.npy"))
    y = np.load(os.path.join(CACHE, "y_residual.npy"))
    assert len(idx) == len(y), f"indices ({len(idx)}) vs y ({len(y)}) mismatch"
    if SUBSET_N < len(idx):
        idx, y = idx[:SUBSET_N], y[:SUBSET_N]
    print(f"[data] {len(idx)} molecules; regenerating raw 323-dim features")

    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": SRC})
    X_raw = np.vstack([features.featurize(ds.get_atoms(int(i))) for i in idx])
    print(f"[data] X_raw {X_raw.shape}, y {y.shape} (residual var {np.var(y):.4g})")
    return X_raw, y


# ------------------------------ 2. Wendland kernel --------------------------


def wendland_kernel(x1, x2, hps):
    """Compact-support C2 Wendland psi(r)=(1-r)^4 (4r+1) over the NORM distance,
    scaled by hps[0]. psi = 0 for distance >= CUTOFF."""
    signal_var = hps[0]
    D = cdist(x1, x2, metric=NORM)
    r = np.clip(D / CUTOFF, 0.0, 1.0)
    return signal_var * (1.0 - r) ** 4 * (4.0 * r + 1.0)


# ------------------------------ scale diagnostic ----------------------------


def report_scale(Z_tr, Z_te):
    """Print PLS-space distance percentiles and in-support neighbour counts so the
    cutoff can be calibrated (this is the check that would have caught the raw-L2
    sparsity failure before it happened)."""
    dtr = pdist(Z_tr, metric=NORM)
    pct = np.percentile(dtr, [5, 25, 50, 75, 95])
    print(
        f"[scale] train pairwise {NORM} dist pctiles "
        f"[5,25,50,75,95] = {np.round(pct, 3)}"
    )
    print(
        f"[scale] CUTOFF={CUTOFF:g} sits at the "
        f"{(dtr < CUTOFF).mean() * 100:.1f}th percentile of train pair distances"
    )
    nbr = (cdist(Z_te, Z_tr, metric=NORM) < CUTOFF).sum(axis=1)
    print(
        f"[scale] in-support train neighbours per TEST point: "
        f"median={np.median(nbr):.0f} min={int(nbr.min())} "
        f"frac_with_zero={np.mean(nbr == 0):.1%}"
    )
    frac_in = (dtr < CUTOFF).mean()
    if np.mean(nbr == 0) > 0.05:
        print(
            "[scale] WARNING: many test points have zero in-support neighbours "
            "-> expect mean-reversion; RAISE CUTOFF."
        )
    elif frac_in > 0.98:
        print(
            "[scale] WARNING: cutoff exceeds ~all pair distances -> kernel is "
            "near-dense/near-rank-1 and may be ill-conditioned (jitter may climb "
            "or predictions smooth to the mean); consider LOWERING CUTOFF toward "
            "the variogram range."
        )


# ------------------------------ posterior helpers ---------------------------


def _first(result_dict, keys):
    for k in keys:
        if k in result_dict:
            return np.asarray(result_dict[k]).ravel()
    raise KeyError(f"none of {keys} in gpCAM keys {list(result_dict)}")


def fit_and_predict(Z_tr, y_tr, Z_te, jitter, signal_var):
    from gpcam import GPOptimizer

    gp = GPOptimizer(
        x_data=Z_tr,
        y_data=y_tr,
        init_hyperparameters=np.array([signal_var]),
        kernel_function=wendland_kernel,
        noise_variances=jitter * np.ones(len(y_tr)),
    )

    # --- UNFREEZE SIGNAL VARIANCE ---
    # Provide search boundaries for hps[0]:
    # e.g., from 1% of the empirical variance up to 1000%
    bounds = np.array([[signal_var * 0.01, signal_var * 10.0]])

    # This triggers the marginal log-likelihood optimization
    gp.train(hyperparameter_bounds=bounds)
    # --------------------------------

    mean = _first(gp.posterior_mean(Z_te), ["f(x)", "m(x)"])
    var = _first(
        gp.posterior_covariance(Z_te, variance_only=True), ["v(x)", "S(x)", "variance"]
    )
    return mean, np.maximum(var, 0.0)


# ------------------------------ 3. jitter retry loop ------------------------


def fit_with_jitter(Z_tr, y_tr, Z_te, signal_var):
    for e in JITTER_EXPONENTS:
        jitter = 10.0**e
        try:
            print(f"[jitter] trying diagonal noise = {jitter:.0e} ...")
            mean, var = fit_and_predict(Z_tr, y_tr, Z_te, jitter, signal_var)
            print(f"[jitter] SUCCESS at {jitter:.0e}")
            return mean, var, jitter
        except np.linalg.LinAlgError:
            print(f"[jitter] Cholesky failed at {jitter:.0e}; escalating x10")
    raise RuntimeError(
        f"{RUN_LABEL} covariance not PD-regularizable up to 1e-1; aborting."
    )


# ------------------------------ 4. parity plot ------------------------------


def parity_plot(y_obs, y_pred, y_std, rmse, r2, jitter):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-parity-{RUN_LABEL}-{ts}.png")
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
            f"GP parity — PLS({PLS_COMPONENTS}) + {NORM_LABEL} Wendland "
            f"(cutoff={CUTOFF:g})\n"
            f"RMSE = {rmse:.4g}    $R^2$ = {r2:.3f}    jitter = {jitter:.0e}"
        )
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


# ------------------------------ main ----------------------------------------


def main():
    X_raw, y = build_raw()

    # split FIRST, then fit standardizer + PLS on train only (no label leakage)
    Xr_tr, Xr_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    print(f"[split] train {len(y_tr)}  test {len(y_te)}  (random_state={RANDOM_STATE})")

    Xs_tr, mean_, std_ = features.standardize(Xr_tr)  # train stats
    Xs_te = (Xr_te - mean_) / std_  # apply to test

    pls = PLSRegression(n_components=PLS_COMPONENTS, scale=False).fit(Xs_tr, y_tr)
    Z_tr, Z_te = pls.transform(Xs_tr), pls.transform(Xs_te)
    print(f"[pls] embedding train {Z_tr.shape} test {Z_te.shape} (fit on train only)")

    report_scale(Z_tr, Z_te)  # calibrate the cutoff before spending the solve

    signal_var = float(np.var(y_tr))
    mean, var, jitter = fit_with_jitter(Z_tr, y_tr, Z_te, signal_var)

    rmse = float(np.sqrt(np.mean((mean - y_te) ** 2)))
    r2 = float(r2_score(y_te, mean))
    print(f"[result] RMSE = {rmse:.4g}   R^2 = {r2:.3f}   (final jitter {jitter:.0e})")
    print(f"[saved] {parity_plot(y_te, mean, np.sqrt(var), rmse, r2, jitter)}")


if __name__ == "__main__":
    main()
