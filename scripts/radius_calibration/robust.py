"""Addendum: dist-to-NN's proper-scoring-rule numbers are not robust, and Step 3's were
measured on the one seed where the pathology is absent.

A sigma candidate proportional to the distance to the nearest training point asserts
sigma -> 0 for a molecule that has a near-duplicate in the training set. The Gaussian MLE
scale c = sqrt(mean((err/s)^2)) is then dominated by that single smallest s, so one
molecule sets the scale for all 800. Seed 7 has such a point (dist 3.7e-5); seeds 42 and
123 do not. CALIBRATION.md's Step 3 ran seed 42 only.

Scale-FREE metrics (Spearman, top-5% recall) are untouched by this -- a global scale
cannot reorder anything -- so the headline finding stands either way. It is only the
NLPD/CRPS/miscalibration column that has to be reported differently.

Two repairs, both reported, neither flattering to the raw candidate:
  * FLOORED: s = sqrt(dnn^2 + d0^2), d0 = the 1st percentile of dnn. The minimum defensible
    regularisation of a sigma model that can otherwise claim infinite confidence.
  * MEDIAN-based scale instead of the MLE: c = median(|err|/s) / 0.6745, which no single
    point can move.
"""
import os, glob
import numpy as np
from scipy.stats import norm

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")


def crps_gauss(y, mu, sig):
    z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def miscal_area(z):
    qs = np.linspace(0.05, 0.95, 19)
    emp = np.array([np.mean(np.abs(z) <= norm.ppf(0.5 + q / 2)) for q in qs])
    return float(np.mean(np.abs(emp - qs)))


def score(s, err, y, mu, how="mle", seed=1):
    n = len(s)
    h = np.random.default_rng(seed).permutation(n)
    folds = (h[:n // 2], h[n // 2:])
    nl, cr, z_all, cs = [], [], np.empty(n), []
    for fit, ev in (folds, folds[::-1]):
        if how == "mle":
            c = float(np.sqrt(np.mean((err[fit] / s[fit]) ** 2)))
        else:                                   # median-absolute-deviation scale
            c = float(np.median(np.abs(err[fit]) / s[fit]) / 0.6745)
        cs.append(c)
        sig = c * s[ev]
        z = err[ev] / sig
        z_all[ev] = z
        nl.append(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2))
        cr.append(np.mean(crps_gauss(y[ev], mu[ev], sig)))
    return np.mean(nl), np.mean(cr), miscal_area(z_all), z_all.std(), max(cs) / min(cs)


def main():
    base = {int(np.load(f)["seed"]): np.load(f)
            for f in sorted(glob.glob(os.path.join(ARMS, "base_s*.npz")))}
    seeds = sorted(base)
    for K in (60, 200, 500):
        print("=" * 100)
        print(f"K = {K}   (mean over seeds {seeds}; 'c hi/lo' = ratio of the two "
              f"cross-fitted scales -- instability)")
        print("=" * 100)
        print(f"{'candidate':>26} {'scale fit':>10} {'NLPD':>9} {'CRPS':>9} "
              f"{'miscal':>8} {'std(z)':>8} {'c hi/lo':>9}")
        rows = []
        for s in seeds:
            a = np.load(os.path.join(ARMS, f"arm_s{s}_k{K}.npz"))
            b = base[s]
            dnn = b["dnn"]
            d0 = np.percentile(dnn, 1.0)
            cand = {
                ("size+category", "mle"):       b["sd_cheap"],
                ("dist-to-NN raw", "mle"):      dnn,
                ("dist-to-NN raw", "median"):   dnn,
                ("dist-to-NN floored", "mle"):  np.sqrt(dnn ** 2 + d0 ** 2),
                ("GP sigma* (latent)", "mle"):  np.sqrt(a["v_gp"]),
                ("GP sigma* + noise", "mle"):   np.sqrt(a["v_gp"] + a["v_mean"]
                                                        + float(a["jitter"])),
            }
            for (nm, how), sg in cand.items():
                rows.append((nm, how) + score(sg, a["err"], a["y"], a["mu"], how=how))
        seen = []
        for nm, how in [k for k in
                        [("size+category", "mle"), ("dist-to-NN raw", "mle"),
                         ("dist-to-NN raw", "median"), ("dist-to-NN floored", "mle"),
                         ("GP sigma* (latent)", "mle"), ("GP sigma* + noise", "mle")]]:
            v = np.array([r[2:] for r in rows if r[0] == nm and r[1] == how], float)
            print(f"{nm:>26} {how:>10} {v[:,0].mean():>9.3f} {v[:,1].mean():>9.4f} "
                  f"{v[:,2].mean():>8.4f} {v[:,3].mean():>8.4f} {v[:,4].mean():>9.1f}")
        print()

    print("Per-seed NLPD for dist-to-NN raw @ K=200 -- the instability, not an average:")
    for s in seeds:
        a = np.load(os.path.join(ARMS, f"arm_s{s}_k200.npz"))
        r = score(base[s]["dnn"], a["err"], a["y"], a["mu"])
        print(f"  seed {s:>4}: NLPD={r[0]:>10.3f}  CRPS={r[1]:>8.3f}  "
              f"min(dnn)={base[s]['dnn'].min():.2e}  c hi/lo={r[4]:.1f}")
    print("\nStep 3 reported dist-to-NN NLPD 2.379 (best of five). That was seed 42, the")
    print("one seed with no near-duplicate test molecule. The metric is not reproducible.")


if __name__ == "__main__":
    main()
