"""
radius.py  (wl_gp2scale)
========================
The informative radius, and what it predicts about scaling to 200k.

Motivation. descriptor_eval's RMSE-vs-nearest-neighbour-distance plot shows a
cliff: test molecules whose nearest TRAIN neighbour is close are predicted well,
and beyond a threshold the prediction is WORSE than guessing the mean. Overall R^2
is just the blend of those two populations. That makes the radius the single most
decision-relevant number we have:

  1. It sets the cutoff. Beyond the informative radius a neighbour carries no
     usable signal, so extending compact support past it buys nothing and costs
     memory. (Choosing the cutoff as "the loosest that fits memory" is a
     memory-driven argument, not a signal-driven one.)
  2. It explains the ONLY mechanism by which more data helps: extra training
     molecules pull test points across the radius, from the bad population into
     the good one. So "does 200k help?" becomes the much cheaper question "does
     P(nn-dist < R_inf) rise enough?" -- which needs no GP at 200k at all.

UNITS WARNING. Our embedding is NOT in descriptor_eval's units, so the ~1.8 seen
in that plot cannot be read across. SIMPLS normalises every score column to unit
norm, so our embedding scales as ~1/sqrt(N) (median pairwise distance measured at
0.095 for n=2000 vs 0.048 for n=8000 -- exactly sqrt(4)), while sklearn's PLS
scores have an N-independent scale (~4.3 at both). Ours are ~100x smaller at 16k
and ~350x at 196k. Everything here is therefore ALSO reported as a percentile of
the pairwise-distance distribution, which is invariant to that rescaling and is
what transfers across N.
"""

from __future__ import annotations

import numpy as np


def radius_from_bins(bin_median_nn, bin_rmse, baseline):
    """The informative radius, given a per-bin RMSE profile.

    MARGINAL question -- "does a neighbour at this distance still carry signal?" --
    because that is what decides the cutoff. (The cumulative curve answers a
    different question and stays under the baseline well past the point where
    individual predictions are already bad, since the good near points dominate the
    aggregate: measured 0.01012 cumulative vs a ~0.006 per-bin cliff.)

    Scan from the FAR end, walk in over the contiguous run of bins that fail the
    baseline, and return the median of the last bin inside it. On a clean monotonic
    curve this agrees with gp_parity.py; unlike its scan-outward-and-stop rule it is
    not aborted by an isolated noisy near bin. Returns None only if every bin fails.
    """
    med = np.asarray(bin_median_nn, float)
    rms = np.asarray(bin_rmse, float)
    if med.size == 0 or not baseline:
        return None
    bad = rms >= baseline
    if bad.all():
        return None                     # nothing beats the mean, at any distance
    k = len(bad)
    while k > 0 and bad[k - 1]:         # walk in over the failing tail
        k -= 1
    return float(med[k - 1]) if k > 0 else None


def rmse_vs_nn_distance(nn_dist, y_true, y_pred, n_bins=10, tol=0.9):
    """RMSE per nearest-neighbour-distance decile, plus the informative radius.

    descriptor_eval/gp_parity.py::error_vs_distance walks the bins outward and stops
    at the first one reaching the baseline. That assumes RMSE rises MONOTONICALLY
    with distance. It does on a well-resolved run, but per-bin RMSE is noisy at
    small n (each decile is only nte/10 points), and a single bad near bin then
    aborts the scan and reports no radius at all -- even when most bins beat the
    baseline.

    So the radius is taken from the CUMULATIVE curve instead: RMSE over ALL test
    points with nn-dist <= d, which is both what "how far out can I trust this"
    actually means and far less noisy (it aggregates rather than partitions). The
    radius is the largest bin edge whose cumulative RMSE is still < tol * baseline.

    `radius` is None only when even the nearest points fail to beat tol*baseline --
    which really would mean the error is flat in distance, i.e. representational.
    """
    nn_dist = np.asarray(nn_dist, float)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    baseline = float(np.std(y_true))
    se = (y_pred - y_true) ** 2
    edges = np.percentile(nn_dist, np.linspace(0, 100, n_bins + 1))

    med, rms, frac = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = ((nn_dist >= lo) & (nn_dist <= hi)) if b == n_bins - 1 else \
            ((nn_dist >= lo) & (nn_dist < hi))
        if m.sum() == 0:
            continue
        med.append(float(np.median(nn_dist[m])))
        rms.append(float(np.sqrt(se[m].mean())))
        frac.append(float(m.mean()))

    cum_d, cum_rmse = [], []
    for d in edges[1:]:
        m = nn_dist <= d
        if m.sum() < 20:            # too few points to mean anything yet
            continue
        cum_d.append(float(d))
        cum_rmse.append(float(np.sqrt(se[m].mean())))

    radius = radius_from_bins(med, rms, baseline)

    return {
        "bin_median_nn": np.array(med),
        "bin_rmse": np.array(rms),
        "bin_frac": np.array(frac),
        "cum_nn": np.array(cum_d),
        "cum_rmse": np.array(cum_rmse),
        "baseline": baseline,
        "radius": radius,
    }


def nn_scaling_exponent(Z_te, Z_tr, sizes, seed=0):
    """How the nearest-neighbour distance shrinks as the training set grows.

    For points on an effective manifold of dimension d_eff, nn-dist ~ n^(-1/d_eff),
    so a log-log fit of the median nn-dist against n gives slope = -1/d_eff. The
    embedding is held FIXED and only the train set is subsampled, so this isolates
    the density effect from any change in the representation.

    Returns (slope, d_eff, sizes, median_nn_per_size, nn_at_largest)."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    sizes = [int(s) for s in sizes if s <= len(Z_tr)]
    meds, nn_last = [], None
    for n in sizes:
        idx = rng.choice(len(Z_tr), size=n, replace=False)
        nn = cdist(Z_te, Z_tr[idx]).min(axis=1)
        meds.append(float(np.median(nn)))
        nn_last = nn
    slope = float(np.polyfit(np.log(sizes), np.log(meds), 1)[0])
    d_eff = (-1.0 / slope) if slope < 0 else float("inf")
    return slope, d_eff, np.array(sizes), np.array(meds), nn_last


def predict_r2_at_n(curve, nn_now, n_now, n_target, slope, y_te):
    """Predict R^2 at n_target from the density shift alone.

    Assumes RMSE is a function of nearest-neighbour distance -- i.e. a test point
    with nn-dist d is predicted about as well regardless of how many other training
    points exist. That is the assumption descriptor_eval's diagnostic already makes
    ("more data helps IF it puts test points inside this radius").

    Each test point's nn-dist is scaled by (n_target/n_now)**slope, then mapped
    through the measured RMSE curve.

    This DELIBERATELY holds the embedding fixed, so it captures only the density
    effect. The representation also improves with N (the PLS linear probe went
    -0.266 at 16k to +0.1212 at 40k), so treat this as a conservative floor.
    """
    scaled = np.asarray(nn_now, float) * (float(n_target) / float(n_now)) ** slope
    # np.interp clamps outside the measured range, which is what we want: below the
    # closest bin we assume the best measured RMSE, above the farthest the worst.
    rmse_hat = np.interp(scaled, curve["bin_median_nn"], curve["bin_rmse"])
    mse = float(np.mean(rmse_hat ** 2))
    var = float(np.var(y_te))
    return {
        "pred_mse": mse,
        "pred_rmse": float(np.sqrt(mse)),
        "pred_r2": 1.0 - mse / var if var else float("nan"),
        "median_nn_scaled": float(np.median(scaled)),
    }


def frac_within(nn_dist, radius):
    """Fraction of test molecules with a training neighbour inside the radius."""
    if radius is None:
        return float("nan")
    return float(np.mean(np.asarray(nn_dist, float) <= radius))
