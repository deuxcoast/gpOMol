"""Proper scoring rules for the GP's predictive distribution: CRPS, NLPD, coverage.

WHAT THIS IS FOR, and how it differs from score_percat.py. That script answers a
MODEL-SELECTION question -- does sigma* rank its own errors better than a
nearest-neighbour distance -- using Spearman correlations. This one answers the
question the paper reports: are the predictive distributions themselves any good, on
proper scoring rules, at each N.

THE PREDICTIVE VARIANCE IS NOT v_gp. fvgp's `posterior_covariance` defaults to
add_noise=False, so the stored `v_gp` is the LATENT FUNCTION variance. The predictive
variance of an OBSERVATION is

    sigma_pred^2 = v_gp + noise            (noise = the jitter passed to build_gp)

Measured on the 20k product arm: std(z) = 1.178 with the latent variance alone and
0.975 once the noise is added, i.e. 18% overconfident vs essentially calibrated. Rank
correlations cannot see this -- adding a constant to every variance does not change any
ranking -- which is exactly why it survived every UQ comparison run so far. It dominates
CRPS, NLPD and coverage. Both are reported so the size of the effect stays visible.

Still NOT included, and it makes these numbers slightly conservative: the prior mean's
own parameter variance (`LinearEmbeddingMean.predict_var`). We detrend manually, so the
GP's variance describes the residual only. Under M1 that is 31 fitted coefficients'
worth of uncertainty being dropped.

IS THE UNCERTAINTY MEANINGFUL? The CRPS_flat column answers that directly. It scores a
HOMOSCEDASTIC predictive distribution -- same mu, but one constant variance for every
point, set to the realised MSE. If the GP's per-point sigma carries real information,
CRPS must beat it. A model can be perfectly calibrated on average and still useless if
its sigma is the same everywhere.
"""
import argparse
import glob
import os

import numpy as np
from scipy.special import erf
from sklearn.metrics import r2_score


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _Phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def crps_gaussian(y, mu, sd):
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting & Raftery).

        CRPS = sd * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],   z = (y - mu)/sd

    Lower is better, in the units of y. Unlike NLPD it is finite and well behaved when a
    point lands far into the tail, which is why it is the headline UQ number here: one
    badly overconfident outlier can dominate a mean NLPD entirely."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sd
    return sd * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / np.sqrt(np.pi))


def nlpd_gaussian(y, mu, sd):
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sd
    return 0.5 * np.log(2.0 * np.pi * sd ** 2) + 0.5 * z ** 2


def coverage(err, sd, level):
    """Fraction of points inside the central `level` interval. Should equal `level`."""
    zc = np.sqrt(2.0) * _erfinv(level)
    return float(np.mean(np.abs(err) <= zc * sd))


def _erfinv(x):
    from scipy.special import erfinv
    return erfinv(x)


def split_scale(err, sd, seed=0):
    """Fit ONE scalar multiplier on sigma using half the points; return it plus the
    other half's index, so scoring is out-of-sample.

    WHY THIS IS NOT OPTIONAL. CRPS punishes a badly SCALED sigma and a badly SHAPED one
    together, and the two have different fixes. Measured at 200k: sv is set to the MEAN
    model's residual variance with the jitter added on top, so the model asserts ~58%
    more variance than the errors have, std(z) falls to 0.795, and CRPS skill goes
    NEGATIVE -- while the per-family sv lever underneath it is still worth +2.1 points.
    Reporting only the unscaled number would have called a working lever a failure.
    The original calibration work did exactly this and score_calibration dropped it."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(err))
    h1, h2 = p[:len(p) // 2], p[len(p) // 2:]
    # the CRPS-optimal scalar for a Gaussian is the one matching realised RMS z
    a = float(np.sqrt(np.mean((err[h1] / np.maximum(sd[h1], 1e-12)) ** 2)))
    return a, h2


def report(d, noise, label):
    y, mu, v = d["y"], d["mu"], np.maximum(d["v_gp"], 0.0)
    err = y - mu
    rows = {}
    for name, sd in (("latent only", np.sqrt(v)), ("+ noise", np.sqrt(v + noise))):
        rows[name] = dict(
            crps=float(np.mean(crps_gaussian(y, mu, sd))),
            nlpd=float(np.mean(nlpd_gaussian(y, mu, sd))),
            stdz=float(np.std(err / np.maximum(sd, 1e-12))),
            c50=coverage(err, sd, 0.50), c90=coverage(err, sd, 0.90),
            c95=coverage(err, sd, 0.95),
            sharp=float(np.mean(sd)),
        )
    mse = float(np.mean(err ** 2))
    flat = float(np.mean(crps_gaussian(y, mu, np.full(len(y), np.sqrt(mse)))))
    # SCALE-FREE skill: every candidate gets its own best global scale on one half and
    # is scored on the other, so this measures the SHAPE of sigma alone.
    sd = np.sqrt(v + noise)
    a, h2 = split_scale(err, sd)
    af, _ = split_scale(err, np.full(len(err), np.sqrt(mse)))
    c_gp = float(np.mean(crps_gaussian(y[h2], mu[h2], a * sd[h2])))
    c_fl = float(np.mean(crps_gaussian(y[h2], mu[h2],
                                       np.full(len(h2), af * np.sqrt(mse)))))
    return dict(rows=rows, mse=mse, rmse=np.sqrt(mse), mae=float(np.mean(np.abs(err))),
                r2=float(r2_score(y, mu)), crps_flat=flat, n=len(y), label=label,
                skill_scaled=100.0 * (c_fl - c_gp) / c_fl, alpha=a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories (cache/product_200k_uq ...)")
    ap.add_argument("--noise", type=float, default=4.30,
                    help="observation-noise variance the GP was built with (the jitter). "
                         "Older result files do not record it; 4.30 is the variogram "
                         "nugget used throughout")
    a = ap.parse_args()

    res = []
    for dpath in a.dirs:
        for f in sorted(glob.glob(os.path.join(dpath, "*_s*.npz"))):
            d = np.load(f, allow_pickle=True)
            if "has_variance" in d.files and not int(d["has_variance"]):
                continue                       # accuracy-only arm: no predictive dist
            if not np.isfinite(d["v_gp"]).any():
                continue
            n = int(d["n"]) if "n" in d.files else -1
            noise = float(d["jitter"]) if "jitter" in d.files else a.noise
            lab = "%s n=%s s%d" % (str(d["arm"]), ("%dk" % (n // 1000)) if n > 0 else "?",
                                   int(d["seed"]))
            res.append(report(d, noise, lab))

    if not res:
        raise SystemExit("no result files with a posterior variance found")

    print("=" * 100)
    print("ACCURACY")
    print("=" * 100)
    print("%-26s %7s %9s %9s %9s %9s" % ("arm", "n_test", "R2", "MSE", "RMSE", "MAE"))
    for r in res:
        print("%-26s %7d %9.4f %9.4f %9.4f %9.4f"
              % (r["label"], r["n"], r["r2"], r["mse"], r["rmse"], r["mae"]))

    print()
    print("=" * 100)
    print("PREDICTIVE DISTRIBUTION  (CRPS and NLPD lower is better; std(z) and coverage")
    print("should hit 1.00 / 0.50 / 0.90 / 0.95 exactly if calibrated)")
    print("=" * 100)
    print("%-26s %-12s %9s %9s %7s %6s %6s %6s %8s"
          % ("arm", "variance", "CRPS", "NLPD", "std(z)", "cov50", "cov90", "cov95",
             "mean sd"))
    for r in res:
        for name in ("latent only", "+ noise"):
            v = r["rows"][name]
            print("%-26s %-12s %9.4f %9.4f %7.3f %6.3f %6.3f %6.3f %8.3f"
                  % (r["label"] if name == "latent only" else "", name, v["crps"],
                     v["nlpd"], v["stdz"], v["c50"], v["c90"], v["c95"], v["sharp"]))

    print()
    print("=" * 100)
    print("ARE THE UNCERTAINTIES MEANINGFUL?  CRPS against a constant-variance model")
    print("=" * 100)
    print("  CRPS_flat uses the SAME mu with one variance for every point (= MSE), so it")
    print("  isolates what the per-point sigma is worth. Negative skill = the GP's")
    print("  varying sigma is worse than no sigma model at all.")
    print("  'skill' is as-is; 'scale-free' gives EVERY candidate its own best global")
    print("  sigma scale on half the points and scores on the other half, so it isolates")
    print("  the SHAPE of sigma. alpha is the fitted multiplier: alpha<1 means the model")
    print("  asserted too much variance, >1 too little. They diverge when the variance")
    print("  is mis-SCALED rather than mis-shaped, which is a different bug with a")
    print("  different fix.")
    print("%-26s %9s %9s %8s %11s %7s"
          % ("arm", "CRPS", "CRPS_flat", "skill", "scale-free", "alpha"))
    for r in res:
        c = r["rows"]["+ noise"]["crps"]
        sk = 100 * (r["crps_flat"] - c) / r["crps_flat"]
        print("%-26s %9.4f %9.4f %7.1f%% %10.2f%% %7.3f"
              % (r["label"], c, r["crps_flat"], sk, r["skill_scaled"], r["alpha"]))


if __name__ == "__main__":
    main()
