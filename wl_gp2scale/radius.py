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

Scale is now N-INVARIANT (historical note). This module was written when SparsePLS
normalised every score column to unit norm, so the embedding scaled as ~1/sqrt(N)
(median pairwise distance 0.095 at n=2000 vs 0.048 at n=8000 -- exactly sqrt(4)) and
an absolute radius could NOT be read across N. We therefore reported everything as a
percentile of the pairwise-distance distribution. That normalisation has since been
removed (natural + pareto scaling in reduce.py): var(t_a)=w_a^T Cov(Xtilde) w_a is a
population quantity, so the embedding scale -- and hence an absolute radius -- now
TRANSFERS across N. Absolute distances are the primary form here; the percentile is
retained only as a density / conditioning guard (it still tracks neighbour count,
which scales with N regardless of the absolute scale).

The semivariogram below picks that absolute radius directly from data (gamma(h)
reaching its sill = the correlation length beyond which neighbours carry no signal),
independently of and as a cross-check on the RMSE-vs-nn-distance informative radius.
"""

from __future__ import annotations

import numpy as np


def _rolling_median(a, w=3):
    """Odd-window rolling median (edges shrink the window). De-noises a per-bin RMSE
    profile so one fluctuating bin cannot move the radius."""
    a = np.asarray(a, float)
    n = len(a)
    if n < w:
        return a.copy()
    h = w // 2
    return np.array([np.median(a[max(0, i - h): min(n, i + h + 1)]) for i in range(n)])


def radius_from_bins(bin_median_nn, bin_rmse, baseline, tol=0.9, smooth=True):
    """The informative radius from a per-bin RMSE profile.

    MARGINAL question -- "does a neighbour at this distance still carry USEFUL signal?"
    A bin is uninformative once its RMSE reaches ``tol * baseline`` (default 0.9):
    predicting only marginally better than the mean is not useful, so the threshold is
    ``tol*baseline``, NOT the bare baseline. (Using the bare baseline was a bug -- a
    noisy tail that tops out at ~0.98*baseline then never "fails", so the radius ran all
    the way out to the last bin.)

    Robustness: per-bin RMSE is noisy (~nte/n_bins points per bin), so the tail can
    alternate fail/good/fail. We median-smooth the profile (window 3), then scan
    OUTWARD for the first SUSTAINED crossing: the first bin that is itself
    uninformative AND whose outward tail is predominantly uninformative
    (``median(r[i:]) >= tol*baseline``). The tail-median guard means an isolated near
    spike does NOT trigger (its tail is still mostly good) and an isolated good bin
    does NOT rescue a bad tail. The radius is the last informative bin before that
    cliff. Returns None if the very first bin already triggers (error flat in distance
    -> representational, not density); returns the last bin if no cliff is found.
    """
    med = np.asarray(bin_median_nn, float)
    rms = np.asarray(bin_rmse, float)
    if med.size == 0 or not baseline:
        return None
    r = _rolling_median(rms) if (smooth and len(rms) >= 3) else rms
    thresh = tol * baseline
    for i in range(len(r)):
        if r[i] >= thresh and float(np.median(r[i:])) >= thresh:
            return float(med[i - 1]) if i > 0 else None
    return float(med[-1])               # tail never predominantly bad -> informative to end


def rmse_vs_nn_distance(nn_dist, y_true, y_pred, n_bins=10, tol=0.9):
    """RMSE per nearest-neighbour-distance decile, plus the informative radius.

    descriptor_eval/gp_parity.py::error_vs_distance walks the bins outward and stops
    at the first one reaching the baseline. That assumes RMSE rises MONOTONICALLY
    with distance. It does on a well-resolved run, but per-bin RMSE is noisy at
    small n (each decile is only nte/10 points), and a single bad near bin then
    aborts the scan and reports no radius at all -- even when most bins beat the
    baseline.

    The radius (via radius_from_bins) is the per-bin MARGINAL cliff: the last bin
    before RMSE reaches ``tol*baseline``, computed on a MEDIAN-SMOOTHED profile so a
    single noisy bin cannot move it. The CUMULATIVE curve is also returned for context
    but is NOT used for the radius -- the good near points dominate the aggregate, so it
    stays under the baseline well past the point where individual (marginal) predictions
    are already useless.

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

    radius = radius_from_bins(med, rms, baseline, tol=tol)

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


# ----------------------------- semivariogram -------------------------------


def semivariogram(Z, y, sample=5000, n_bins=20, cat=None, seed=0):
    """Empirical semivariogram of the target y over embedding distance.

    gamma(h) = 0.5 * < (y_i - y_j)^2 >  binned by pairwise embedding distance h.
    Since gamma(inf) = Var(y) for independent pairs, the distance where gamma reaches
    the sill (= Var(y)) is the correlation length -- beyond it neighbours carry no
    usable signal, so it is the natural compact-support radius. GP-free and cheap:
    one pdist over a ``sample``-row subset (5000 rows -> ~1.25e7 pairs).

    ``cat``: if given, only SAME-category pairs are kept -- faithful to the kernel,
    which zeroes cross-category covariance, so the relevant sill is the within-category
    variance the kernel actually sees (<= total Var(y)).

    Returns dict: ``lag`` (per-bin median distance), ``gamma`` (per-bin semivariance),
    ``count`` (pairs per bin), ``sill`` (Var(y), or within-category variance if masked),
    ``n_pairs``.
    """
    from scipy.spatial.distance import pdist

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    m = min(sample, len(Z))
    idx = rng.choice(len(Z), size=m, replace=False)
    Zs, ys = Z[idx], y[idx]

    d = pdist(Zs)                                  # (m*(m-1)/2,) pairwise distances
    dy = pdist(ys[:, None])                        # |y_i - y_j| for the same pairs
    sv = 0.5 * dy**2                               # per-pair semivariance

    if cat is not None:
        cats = np.asarray(cat)[idx]
        same = pdist(cats[:, None]) == 0           # 0 iff i,j share a category
        d, sv = d[same], sv[same]
        # gamma plateau for within-category (independent) pairs is the within-category
        # variance, which the mean semivariance over same-category pairs estimates
        # (0.5*mean(dy^2) = Var_within when pairs decorrelate). <= total Var(y).
        sill = float(np.mean(sv)) if len(sv) else float("nan")
    else:
        sill = float(np.var(y))                     # exact gamma(inf) over all pairs

    if len(d) == 0:
        return {"lag": np.array([]), "gamma": np.array([]), "count": np.array([]),
                "sill": sill, "n_pairs": 0}

    edges = np.percentile(d, np.linspace(0, 100, n_bins + 1))
    lag, gamma, count = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = ((d >= lo) & (d <= hi)) if b == n_bins - 1 else ((d >= lo) & (d < hi))
        if sel.sum() == 0:
            continue
        lag.append(float(np.median(d[sel])))
        gamma.append(float(np.mean(sv[sel])))
        count.append(int(sel.sum()))
    return {
        "lag": np.array(lag), "gamma": np.array(gamma), "count": np.array(count),
        "sill": sill, "n_pairs": int(len(d)),
    }


def range_from_variogram(lag, gamma, sill, sill_frac=0.95):
    """Effective range: the smallest lag at which gamma reaches ``sill_frac`` * sill.

    Model-free (no covariance-model fit). Returns None if gamma never reaches the
    threshold within the sampled distances (no decorrelation observed -> the radius is
    larger than the data resolves, or the signal is representational not spatial).
    A scipy.optimize.curve_fit spherical/exponential-model range is a drop-in
    refinement if a smoother estimate is wanted.
    """
    lag = np.asarray(lag, float)
    gamma = np.asarray(gamma, float)
    if lag.size == 0 or not np.isfinite(sill) or sill <= 0:
        return None
    hit = np.where(gamma >= sill_frac * sill)[0]
    return float(lag[hit[0]]) if hit.size else None
