#!/usr/bin/env python
"""
gp_parity.py  (per-channel PLS + Wendland parity test)
======================================================
Validate a chosen descriptor CHANNEL with a compact-support Wendland GP on its
(train-only) PLS embedding, and output a test-set parity plot.

Channels (--channel): 'wl' (topology), 'geometry' (distance histogram),
'charge' (Loewdin scalars), or 'all' (the full 323-dim hybrid).

Motivation: on the full descriptor this pipeline gave R^2 ~ 0.08 at jitter 1e-6 --
a representational ceiling, not a tuning problem (static vs trained signal variance
were identical). The per-channel variograms said WL is the LOCAL + informative
channel and geometry is the GLOBAL one, so the key test is WL ALONE through a
compact-support kernel (the tool WL's locality actually suits).

Leakage control: standardizer and PLS are fit on TRAIN only, applied to test.
Cutoff calibration: distance scale changes per channel, so either set --cutoff
explicitly or use --cutoff-pct P to put the cutoff at the P-th percentile of the
embedding's train pairwise distances (the [scale] diagnostic reports where it lands).

PD note: psi_{3,1} is PD on Euclidean R^d only for d<=3; the jitter loop covers
the rest. Run from inside descriptor_eval/.
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

# ============================ DEFAULT KNOBS (CLI overrides below) ============
NORM = "euclidean"  # "euclidean" (L2) | "cityblock" (L1)
NORM_LABEL = "L2"
CHANNEL = "all"  # wl | geometry | charge | all
USE_PLS = True
PLS_COMPONENTS = 10
CUTOFF = 10.0  # compact-support radius (in embedding NORM space)
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

# channel column layout of the 323-dim vector (WL=256, hist=64, charge=3)
_WL = features.__dict__.get("N_WL", 256)
_HIST = len(features.default_distance_bins()) - 1
_CHG = 3
CHANNEL_SLICES = {
    "wl": slice(0, _WL),
    "geometry": slice(_WL, _WL + _HIST),
    "charge": slice(_WL + _HIST, _WL + _HIST + _CHG),
    "all": slice(0, _WL + _HIST + _CHG),
}
RUN_LABEL = "all-PLS10-L2"  # recomputed in main()

# --metric aliases -> scipy pdist name, and pretty labels for filename/title
METRIC_ALIASES = {
    "l1": "cityblock",
    "manhattan": "cityblock",
    "l2": "euclidean",
    "euclid": "euclidean",
    "linf": "chebyshev",
}
METRIC_LABELS = {"cityblock": "L1", "euclidean": "L2", "chebyshev": "Linf"}


# ------------------------------ 1. data (raw features) ----------------------


def build_raw():
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
        f"[scale] CUTOFF={CUTOFF:g} sits at the {frac_in*100:.1f}th percentile "
        "of train pair distances"
    )
    nbr = (cdist(Z_te, Z_tr, metric=NORM) < CUTOFF).sum(axis=1)
    print(
        f"[scale] in-support train neighbours per TEST point: median={np.median(nbr):.0f} "
        f"min={int(nbr.min())} frac_with_zero={np.mean(nbr==0):.1%}"
    )
    if np.mean(nbr == 0) > 0.05:
        print(
            "[scale] WARNING: many test points have zero in-support neighbours "
            "-> mean-reversion; RAISE cutoff."
        )
    elif frac_in > 0.98:
        print(
            "[scale] WARNING: cutoff exceeds ~all pair distances -> near-dense / "
            "near-rank-1 kernel, may be ill-conditioned; LOWER cutoff."
        )


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


def fit_single(Z_tr, y_tr, Z_te, jitter, signal_var):
    """Single fixed-jitter fit -- NO escalation. Reports whether the Cholesky
    actually succeeded at this jitter (so a degenerate fit can't pass silently)."""
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
            f"Cholesky FAILED at fixed jitter={jitter:g}: the kernel is not PD at "
            "this level. Raise --jitter, or change --cutoff / --metric."
        )


# ------------------------------ nearest-neighbour diagnostic ----------------


def nn_distances(Z_te, Z_tr):
    """Distance from each TEST point to its NEAREST TRAIN point, in the SAME
    embedding + metric the kernel uses. This is the quantity Marcus flagged: how
    'covered' each test point is by training data."""
    return cdist(Z_te, Z_tr, metric=NORM).min(axis=1)


def error_vs_distance(nn_dist, y_te, y_pred, n_bins=10):
    """Bin test points by nearest-train-neighbour distance (equal count) and print
    RMSE per bin vs the mean-prediction baseline. The distance at which RMSE
    climbs to ~baseline is the 'informative radius': inside it the GP is better
    than guessing the mean; beyond it, it isn't. Returns (median_dist, rmse) per
    bin for plotting."""
    baseline = float(np.std(y_te))  # RMSE of predicting the test mean
    edges = np.percentile(nn_dist, np.linspace(0, 100, n_bins + 1))
    print("\n[nn-error] test error vs distance-to-nearest-train-neighbour")
    print(
        f"  baseline RMSE (predict the mean) = {baseline:.3f}   "
        f"(overall RMSE = {np.sqrt(np.mean((y_pred - y_te) ** 2)):.3f})"
    )
    print(
        f"  {'bin':>3}{'n':>6}{'med_nn_dist':>13}{'RMSE':>9}{'RMSE/base':>11}  informative?"
    )
    med, rms = [], []
    informative_radius = None
    left_zone = False  # once error reaches baseline, stop
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
        # contiguous informative zone: extend while clearly below baseline; stop
        # updating once a bin reaches baseline (later dips are small-bin noise)
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
            f"  => informative out to nn-dist ~ {informative_radius:.3f} "
            "(bins below baseline). More data helps IF it puts test points inside this."
        )
    else:
        print(
            "  => NO bin beats the mean baseline -> error is flat vs neighbour "
            "distance; this looks REPRESENTATIONAL, not a density problem."
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


# ------------------------------ 4. parity plot ------------------------------


def parity_plot(y_obs, y_pred, y_std, rmse, r2, jitter, nn_dist=None):
    os.makedirs(GRAPHS, exist_ok=True)
    ts = datetime.now().strftime("%m-%d-%H-%M-%S")
    path = os.path.join(GRAPHS, f"GP-parity-{RUN_LABEL}-{ts}.png")
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8.6, 8))
        # error bars drawn neutral/underneath so the point colour reads clearly
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
    global CHANNEL, USE_PLS, PLS_COMPONENTS, CUTOFF, RUN_LABEL, NORM, NORM_LABEL
    ap = argparse.ArgumentParser(
        description="per-channel PLS + Wendland GP parity test"
    )
    ap.add_argument("--channel", default=CHANNEL, choices=list(CHANNEL_SLICES))
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument(
        "--cutoff-pct",
        type=float,
        default=None,
        help="if set, cutoff = this percentile of embedding train distances "
        "(auto-calibrate; e.g. 80). Overrides --cutoff.",
    )
    ap.add_argument("--pls-components", type=int, default=PLS_COMPONENTS)
    ap.add_argument(
        "--no-pls",
        action="store_true",
        help="use standardized channel " "features directly, skipping PLS",
    )
    ap.add_argument(
        "--metric",
        default="euclidean",
        help="distance metric: l2/euclidean (default), l1/cityblock, "
        "linf/chebyshev, or any scipy pdist metric name",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="if set, fit ONCE at this exact diagonal noise (eV^2), "
        "skipping the escalating retry loop",
    )
    a = ap.parse_args()
    CHANNEL, USE_PLS, PLS_COMPONENTS, CUTOFF = (
        a.channel,
        not a.no_pls,
        a.pls_components,
        a.cutoff,
    )
    NORM = METRIC_ALIASES.get(a.metric.lower(), a.metric.lower())
    NORM_LABEL = METRIC_LABELS.get(NORM, a.metric.upper())
    print(f"[metric] {a.metric} -> scipy '{NORM}' (label {NORM_LABEL})")

    X_raw, y = build_raw()
    Xc = X_raw[:, CHANNEL_SLICES[CHANNEL]]
    print(f"[channel] '{CHANNEL}' -> {Xc.shape[1]} raw dims")

    Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        Xc, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    print(f"[split] train {len(y_tr)}  test {len(y_te)}  (random_state={RANDOM_STATE})")

    Xs_tr, mean_, std_ = features.standardize(Xc_tr)
    Xs_te = (Xc_te - mean_) / std_

    if USE_PLS:
        ncomp = min(PLS_COMPONENTS, Xc.shape[1])  # clamp for narrow channels
        if ncomp < PLS_COMPONENTS:
            print(
                f"[pls] clamping components {PLS_COMPONENTS} -> {ncomp} (channel width)"
            )
        pls = PLSRegression(n_components=ncomp, scale=False).fit(Xs_tr, y_tr)
        Z_tr, Z_te = pls.transform(Xs_tr), pls.transform(Xs_te)
        embed = f"PLS{ncomp}"
        print(f"[pls] embedding train {Z_tr.shape} (fit on train only)")
    else:
        Z_tr, Z_te = Xs_tr, Xs_te
        embed = "raw"
        print(f"[embed] using standardized channel features directly {Z_tr.shape}")

    if a.cutoff_pct is not None:  # auto-calibrate the cutoff
        CUTOFF = float(np.percentile(pdist(Z_tr, metric=NORM), a.cutoff_pct))
        print(
            f"[cutoff] set to {a.cutoff_pct:g}th pct of embedding distances = {CUTOFF:.4g}"
        )

    RUN_LABEL = f"{CHANNEL}-{embed}-{NORM_LABEL}"
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

    # nearest-neighbour coverage diagnostic (Marcus): is error a density effect?
    nn = nn_distances(Z_te, Z_tr)
    med, rms, base = error_vs_distance(nn, y_te, mean)
    print(
        f"[saved] {parity_plot(y_te, mean, np.sqrt(var), rmse, r2, jitter, nn_dist=nn)}"
    )
    if len(med) >= 2:
        print(f"[saved] {plot_error_vs_distance(med, rms, base)}")


if __name__ == "__main__":
    main()
