#!/usr/bin/env python
"""
gp_parity.py  (WL-only, explicit-vocabulary PLS + Wendland parity test)
======================================================================
WL-ONLY descriptor (geometry + charge channels dropped -- reintroduce later via
an additive kernel). The WL vector is built by an EXPLICIT per-depth vocabulary
fitted on the training split (no hashing collisions), then standardized -> PLS
-> compact-support Wendland GP. Output: test parity plot + nearest-neighbour
coverage diagnostic.

Key flags:
  --wl-mode {explicit,hashed}   explicit = exact vocab (default); hashed = legacy
                                256-bucket, for an A/B on the SAME split.
  --wl-depth N                  WL refinement depth (default 3).
  --include-depth0              keep depth-0 (bare element counts); dropped by
                                default since the extensive mean removes composition.
  --metric {l2,l1,...}          distance metric (default l2).
  --pls-components / --no-pls   supervised reduction (fit on train only).
  --cutoff / --cutoff-pct       compact-support radius (explicit, or auto by pctile).
  --jitter X                    single fixed-jitter fit (skip the escalating loop).

Leakage control: WL vocabulary, standardizer, and PLS are all fit on TRAIN only.
Run from inside descriptor_eval/.
"""

import argparse
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

# ============================ DEFAULT KNOBS (CLI overrides) ==================
NORM = "euclidean"
NORM_LABEL = "L2"
WL_MODE = "explicit"  # explicit | hashed
WL_DEPTH = 3
INCLUDE_DEPTH0 = False
USE_PLS = True
PLS_COMPONENTS = 10
CUTOFF = 10.0
SUBSET_N = 10_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42
SIGMA_MULT = 2.0
JITTER_EXPONENTS = range(-6, 0)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import features

SRC = os.path.join(SCRIPT_DIR, "..", "train_4M")
CACHE = os.path.join(SCRIPT_DIR, "cache")
GRAPHS = os.path.join(SCRIPT_DIR, "graphs")
RUN_LABEL = "WL-explicit-PLS10-L2"

METRIC_ALIASES = {
    "l1": "cityblock",
    "manhattan": "cityblock",
    "l2": "euclidean",
    "euclid": "euclidean",
    "linf": "chebyshev",
}
METRIC_LABELS = {"cityblock": "L1", "euclidean": "L2", "chebyshev": "Linf"}


# ------------------------------ 1. data (atoms + target) --------------------


def build_atoms():
    """Load frozen indices + residual target and the ase.Atoms (featurization is
    fit-on-train, so it happens AFTER the split, not here)."""
    idx = np.load(os.path.join(CACHE, "subset_indices.npy"))
    y = np.load(os.path.join(CACHE, "y_residual.npy"))
    assert len(idx) == len(y), f"indices ({len(idx)}) vs y ({len(y)}) mismatch"
    if SUBSET_N < len(idx):
        idx, y = idx[:SUBSET_N], y[:SUBSET_N]
    print(f"[data] loading {len(idx)} ase.Atoms")
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": SRC})
    atoms = [ds.get_atoms(int(i)) for i in idx]
    print(f"[data] {len(atoms)} molecules, y (residual var {np.var(y):.4g})")
    return atoms, y


# ------------------------------ 2. Wendland kernel --------------------------


def wendland_kernel(x1, x2, hps):
    signal_var = hps[0]
    D = cdist(x1, x2, metric=NORM)
    r = np.clip(D / CUTOFF, 0.0, 1.0)
    return signal_var * (1.0 - r) ** 4 * (4.0 * r + 1.0)


# ------------------------------ scale diagnostic ----------------------------


def report_scale(Z_tr, Z_te):
    dtr = pdist(Z_tr, metric=NORM)
    pct = np.percentile(dtr, [5, 25, 50, 75, 95])
    print(
        f"[scale] train pairwise {NORM} dist pctiles [5,25,50,75,95] = {np.round(pct,3)}"
    )
    frac_in = (dtr < CUTOFF).mean()
    print(
        f"[scale] CUTOFF={CUTOFF:g} sits at the {frac_in*100:.1f}th percentile of train pairs"
    )
    nbr = (cdist(Z_te, Z_tr, metric=NORM) < CUTOFF).sum(axis=1)
    print(
        f"[scale] in-support train neighbours per TEST point: median={np.median(nbr):.0f} "
        f"min={int(nbr.min())} frac_with_zero={np.mean(nbr==0):.1%}"
    )
    if np.mean(nbr == 0) > 0.05:
        print(
            "[scale] WARNING: many zero-neighbour test points -> mean-reversion; RAISE cutoff."
        )
    elif frac_in > 0.98:
        print("[scale] WARNING: near-dense / near-rank-1 kernel; LOWER cutoff.")


# ------------------------------ nearest-neighbour diagnostic ----------------


def nn_distances(Z_te, Z_tr):
    return cdist(Z_te, Z_tr, metric=NORM).min(axis=1)


def error_vs_distance(nn_dist, y_te, y_pred, n_bins=10):
    baseline = float(np.std(y_te))
    edges = np.percentile(nn_dist, np.linspace(0, 100, n_bins + 1))
    print("\n[nn-error] test error vs distance-to-nearest-train-neighbour")
    print(
        f"  baseline RMSE (predict the mean) = {baseline:.3f}   "
        f"(overall RMSE = {np.sqrt(np.mean((y_pred - y_te)**2)):.3f})"
    )
    print(
        f"  {'bin':>3}{'n':>6}{'med_nn_dist':>13}{'RMSE':>9}{'RMSE/base':>11}  informative?"
    )
    med, rms, informative_radius, left_zone = [], [], None, False
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (
            (nn_dist >= lo) & (nn_dist <= hi)
            if b == n_bins - 1
            else (nn_dist >= lo) & (nn_dist < hi)
        )
        if m.sum() == 0:
            continue
        rmse_b = float(np.sqrt(np.mean((y_pred[m] - y_te[m]) ** 2)))
        md = float(np.median(nn_dist[m]))
        ratio = rmse_b / baseline if baseline else np.nan
        flag = "yes" if ratio < 0.9 else ("~" if ratio < 1.0 else "no (>=mean)")
        if not left_zone:
            if ratio < 0.9:
                informative_radius = md
            elif ratio >= 1.0:
                left_zone = True
        print(
            f"  {b:>3}{int(m.sum()):>6}{md:>13.3f}{rmse_b:>9.3f}{ratio:>11.2f}  {flag}"
        )
        med.append(md)
        rms.append(rmse_b)
    if informative_radius is not None:
        print(
            f"  => informative out to nn-dist ~ {informative_radius:.3f}. More data helps "
            "IF it puts test points inside this."
        )
    else:
        print(
            "  => NO bin beats the mean baseline -> error flat vs neighbour distance; "
            "looks REPRESENTATIONAL, not a density problem."
        )
    return np.array(med), np.array(rms), baseline


def plot_error_vs_distance(med, rms, baseline):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-nnerror-{RUN_LABEL}-{ts}.png")
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(med, rms, "-o", lw=2, color="#348ABD", label="RMSE per distance bin")
        ax.axhline(baseline, ls="--", color="k", lw=2, label="baseline (predict mean)")
        ax.set_xlabel("distance to nearest train neighbour (bin median)")
        ax.set_ylabel("test RMSE")
        ax.set_title(
            f"Informative radius — {RUN_LABEL}\n"
            "RMSE below the dashed line = better than guessing the mean"
        )
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


# ------------------------------ posterior helpers ---------------------------


def _first(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k]).ravel()
    raise KeyError(f"none of {keys} in gpCAM keys {list(d)}")


def fit_and_predict(Z_tr, y_tr, Z_te, jitter, signal_var):
    from gpcam import GPOptimizer

    gp = GPOptimizer(
        x_data=Z_tr,
        y_data=y_tr,
        init_hyperparameters=np.array([signal_var]),
        kernel_function=wendland_kernel,
        noise_variances=jitter * np.ones(len(y_tr)),
    )
    mean = _first(gp.posterior_mean(Z_te), ["f(x)", "m(x)"])
    var = _first(
        gp.posterior_covariance(Z_te, variance_only=True), ["v(x)", "S(x)", "variance"]
    )
    return mean, np.maximum(var, 0.0)


# ------------------------------ 3. jitter handling --------------------------


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


def fit_single(Z_tr, y_tr, Z_te, jitter, signal_var):
    frac = jitter / np.var(y_tr) * 100.0
    print(
        f"[jitter] single-shot fixed jitter = {jitter:g} eV^2 "
        f"({frac:.1f}% of residual variance); no escalation"
    )
    try:
        mean, var = fit_and_predict(Z_tr, y_tr, Z_te, jitter, signal_var)
        print(f"[jitter] Cholesky SUCCEEDED at {jitter:g}")
        return mean, var, jitter
    except np.linalg.LinAlgError:
        raise RuntimeError(
            f"Cholesky FAILED at fixed jitter={jitter:g}: not PD. "
            "Raise --jitter or change --cutoff / --metric."
        )


# ------------------------------ 4. parity plot ------------------------------


def parity_plot(y_obs, y_pred, y_std, rmse, r2, jitter, nn_dist=None):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-parity-{RUN_LABEL}-{ts}.png")
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8.6, 8))
        ax.errorbar(
            y_obs,
            y_pred,
            yerr=SIGMA_MULT * y_std,
            fmt="none",
            ecolor="#CFCFCF",
            elinewidth=0.6,
            zorder=1,
        )
        if nn_dist is not None:
            sc = ax.scatter(
                y_obs,
                y_pred,
                c=nn_dist,
                cmap="viridis",
                s=20,
                alpha=0.85,
                zorder=2,
                label=rf"test predictions ($\pm{int(SIGMA_MULT)}\sigma$)",
            )
            fig.colorbar(sc, ax=ax, label="distance to nearest train neighbour")
        else:
            ax.scatter(
                y_obs,
                y_pred,
                s=20,
                color="#348ABD",
                alpha=0.6,
                zorder=2,
                label=rf"test predictions ($\pm{int(SIGMA_MULT)}\sigma$)",
            )
        lo = float(min(y_obs.min(), y_pred.min()))
        hi = float(max(y_obs.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="perfect (y = x)")
        ax.set_xlabel("Observed Residual Energy")
        ax.set_ylabel("Predicted Residual Energy")
        ax.set_title(
            f"GP parity — {RUN_LABEL} (cutoff={CUTOFF:g})\n"
            f"RMSE = {rmse:.4g}    $R^2$ = {r2:.3f}    jitter = {jitter:g}"
        )
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


# ------------------------------ main ----------------------------------------


def main():
    global WL_MODE, WL_DEPTH, INCLUDE_DEPTH0, USE_PLS, PLS_COMPONENTS, CUTOFF
    global RUN_LABEL, NORM, NORM_LABEL
    ap = argparse.ArgumentParser(
        description="WL-only explicit-vocab PLS + Wendland parity test"
    )
    ap.add_argument("--wl-mode", default=WL_MODE, choices=["explicit", "hashed"])
    ap.add_argument("--wl-depth", type=int, default=WL_DEPTH)
    ap.add_argument("--include-depth0", action="store_true")
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument(
        "--cutoff-pct",
        type=float,
        default=None,
        help="cutoff = this percentile of embedding train distances (overrides --cutoff)",
    )
    ap.add_argument("--pls-components", type=int, default=PLS_COMPONENTS)
    ap.add_argument("--no-pls", action="store_true")
    ap.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="single fixed diagonal noise (eV^2), skip the escalating loop",
    )
    a = ap.parse_args()
    WL_MODE, WL_DEPTH, INCLUDE_DEPTH0 = a.wl_mode, a.wl_depth, a.include_depth0
    USE_PLS, PLS_COMPONENTS, CUTOFF = not a.no_pls, a.pls_components, a.cutoff
    NORM = METRIC_ALIASES.get(a.metric.lower(), a.metric.lower())
    NORM_LABEL = METRIC_LABELS.get(NORM, a.metric.upper())
    print(f"[metric] {a.metric} -> scipy '{NORM}' (label {NORM_LABEL})")

    atoms, y = build_atoms()
    a_tr, a_te, y_tr, y_te = train_test_split(
        atoms, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    print(f"[split] train {len(y_tr)}  test {len(y_te)}  (random_state={RANDOM_STATE})")

    # WL featurizer: fit vocabulary on TRAIN only, transform both
    feat = features.WLFeaturizer(
        depth=WL_DEPTH, include_depth0=INCLUDE_DEPTH0, mode=WL_MODE
    )
    Xr_tr = feat.fit_transform(a_tr)
    Xr_te = feat.transform(a_te)
    print(
        f"[wl] mode={WL_MODE} depths={feat.depths_} D={feat.n_features_}  "
        f"test OOV rate={feat.last_oov_rate_:.1%}"
    )

    Xs_tr, mean_, std_ = features.standardize(Xr_tr)
    Xs_te = (Xr_te - mean_) / std_

    if USE_PLS:
        ncomp = min(PLS_COMPONENTS, Xr_tr.shape[1])
        if ncomp < PLS_COMPONENTS:
            print(
                f"[pls] clamping components {PLS_COMPONENTS} -> {ncomp} (feature width)"
            )
        pls = PLSRegression(n_components=ncomp, scale=False).fit(Xs_tr, y_tr)
        Z_tr, Z_te = pls.transform(Xs_tr), pls.transform(Xs_te)
        embed = f"PLS{ncomp}"
        print(f"[pls] embedding train {Z_tr.shape} (fit on train only)")
    else:
        Z_tr, Z_te = Xs_tr, Xs_te
        embed = "raw"
        print(f"[embed] standardized WL features directly {Z_tr.shape}")

    if a.cutoff_pct is not None:
        CUTOFF = float(np.percentile(pdist(Z_tr, metric=NORM), a.cutoff_pct))
        print(
            f"[cutoff] set to {a.cutoff_pct:g}th pct of embedding distances = {CUTOFF:.4g}"
        )

    RUN_LABEL = f"WL-{WL_MODE}-{embed}-{NORM_LABEL}"
    report_scale(Z_tr, Z_te)

    signal_var = float(np.var(y_tr))
    if a.jitter is not None:
        mean, var, jitter = fit_single(Z_tr, y_tr, Z_te, a.jitter, signal_var)
    else:
        mean, var, jitter = fit_with_jitter(Z_tr, y_tr, Z_te, signal_var)

    rmse = float(np.sqrt(np.mean((mean - y_te) ** 2)))
    r2 = float(r2_score(y_te, mean))
    print(
        f"[result] {RUN_LABEL}  RMSE = {rmse:.4g}   R^2 = {r2:.3f}   (jitter {jitter:g})"
    )

    nn = nn_distances(Z_te, Z_tr)
    med, rms, base = error_vs_distance(nn, y_te, mean)
    print(
        f"[saved] {parity_plot(y_te, mean, np.sqrt(var), rmse, r2, jitter, nn_dist=nn)}"
    )
    if len(med) >= 2:
        print(f"[saved] {plot_error_vs_distance(med, rms, base)}")


if __name__ == "__main__":
    main()
