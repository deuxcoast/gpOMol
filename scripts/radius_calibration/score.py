"""Score the radius grid on UNCERTAINTY rather than R^2.

Reads the per-arm .npz files radius_cal.py wrote and answers one question: does the GP's
sigma* overtake the GP-free dist-to-NN baseline at a tighter support radius?

TWO DELIBERATE DEPARTURES from CALIBRATION.md Step 3's scoring, both of which buy
precision the earlier run flagged as its main weakness ("the evaluation half is 400
points, so a 0.05 difference in top-5% recall is one molecule"):

 1. Spearman and top-5% recall are SCALE-FREE -- a global scale factor cannot change a
    ranking. Step 3 nonetheless computed them on the held-out half only, because the
    scale-fitting split was applied to every metric uniformly. Here they are computed on
    all NTE points, halving their standard error for free.
 2. NLPD/CRPS/miscalibration DO need a fitted scale, so they keep the split -- but
    2-fold cross-fitted (fit on A score B, fit on B score A, average) instead of using
    half the data and discarding the rest.

The headline statistic is the PAIRED per-seed difference Spearman(sigma*) -
Spearman(dist-to-NN). dist-to-NN does not depend on K, and within a seed the embedding,
the prior mean and the test subset are identical across K, so the only thing moving
between arms is the radius. Comparing arm means across seeds would instead be dominated
by seed-to-seed spread, which at 20k is large (WL std 0.076, [[channel-placement-criterion]]).
"""
import os, glob
import numpy as np
from scipy.stats import spearmanr, norm

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")


def crps_gauss(y, mu, sig):
    z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def miscal_area(z):
    qs = np.linspace(0.05, 0.95, 19)
    emp = np.array([np.mean(np.abs(z) <= norm.ppf(0.5 + q / 2)) for q in qs])
    return float(np.mean(np.abs(emp - qs)))


def rank_metrics(s, aerr, frac=0.05):
    """Spearman and top-frac recall on ALL points -- both are invariant to the scale of
    ``s``, so no scale needs fitting and no data needs holding out."""
    rho = float(spearmanr(s, aerr).statistic)
    k = max(int(frac * len(s)), 1)
    hit = len(set(np.argsort(aerr)[-k:]) & set(np.argsort(s)[-k:])) / k
    return rho, hit, k


def scale_metrics(s, err, y, mu, seed=1):
    """NLPD / CRPS / miscalibration at each candidate's OWN best global scale, 2-fold
    cross-fitted so every point is scored out-of-fold and none is thrown away."""
    n = len(s)
    h = np.random.default_rng(seed).permutation(n)
    folds = (h[:n // 2], h[n // 2:])
    nl, cr, z_all = [], [], np.empty(n)
    for fit, ev in (folds, folds[::-1]):
        c = float(np.sqrt(np.mean((err[fit] / s[fit]) ** 2)))   # Gaussian MLE scale
        sig = c * s[ev]
        z = err[ev] / sig
        z_all[ev] = z
        nl.append(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2))
        cr.append(np.mean(crps_gauss(y[ev], mu[ev], sig)))
    return float(np.mean(nl)), float(np.mean(cr)), miscal_area(z_all), float(z_all.std())


def load():
    arms = {}
    for f in sorted(glob.glob(os.path.join(ARMS, "arm_s*_k*.npz"))):
        if "_smoke" in f:
            continue
        d = np.load(f, allow_pickle=True)
        arms[(int(d["seed"]), int(d["K"]))] = d
    base = {int(np.load(f)["seed"]): np.load(f)
            for f in sorted(glob.glob(os.path.join(ARMS, "base_s*.npz")))}
    return arms, base


def candidates(a, b):
    """The sigma candidates for one arm. dist-to-NN and size+category come from the
    per-seed baseline file and are byte-identical across K, which is what makes the
    K comparison paired."""
    v_gp, v_mean = a["v_gp"], a["v_mean"]
    return {
        "constant":           np.ones(len(v_gp)),
        "size+category":      b["sd_cheap"],
        "dist-to-NN":         b["dnn"],
        "GP sigma* (latent)": np.sqrt(v_gp),
        "GP sigma* + noise":  np.sqrt(v_gp + v_mean + float(a["jitter"])),
    }


def main():
    from sklearn.metrics import r2_score
    arms, base = load()
    seeds = sorted({s for s, _ in arms})
    Ks = sorted({k for _, k in arms})
    print(f"loaded {len(arms)} arms: seeds={seeds} K={Ks}\n")

    # ------------------------------------------------------- what the radius does to fit
    print("=" * 96)
    print("THE TRADEOFF: what the radius buys and what it costs")
    print("=" * 96)
    print(f"{'K':>5} {'cut_wl':>7} {'cut_gm':>7} {'nbrs/test pt':>14} {'frac 0-nbr':>11} "
          f"{'R2(GP)':>8} {'sd(sigma*)/mean':>16} {'warn':>5}")
    for K in Ks:
        r2s, med, fz, cv, wn, cw, cg = [], [], [], [], [], [], []
        for s in seeds:
            a = arms.get((s, K))
            if a is None:
                continue
            r2s.append(r2_score(a["y"], a["mu"]))
            nb = a["nbr"].sum(axis=0)              # union over the additive channels
            med.append(np.median(nb)); fz.append(np.mean(nb == 0))
            sg = np.sqrt(a["v_gp"]); cv.append(sg.std() / sg.mean())
            wn.append(int(a["n_warn"])); cw.append(a["cutoffs"][0]); cg.append(a["cutoffs"][1])
        print(f"{K:>5} {np.mean(cw):>7.3f} {np.mean(cg):>7.3f} {np.mean(med):>14.0f} "
              f"{np.mean(fz):>11.1%} {np.mean(r2s):>8.4f} {np.mean(cv):>16.4f} "
              f"{sum(wn):>5}")
    print("  sd(sigma*)/mean is the COEFFICIENT OF VARIATION of the posterior sigma. A")
    print("  sigma that does not vary cannot rank anything, whatever its average level.")

    # ------------------------------------------- is sigma* just rediscovering distance?
    print()
    print("=" * 96)
    print("MECHANISM: how much of sigma* IS the baseline?")
    print("=" * 96)
    print(f"{'K':>5} {'rho(sigma*, dist-NN)':>22} {'rho(sigma*, #nbrs)':>20} "
          f"{'sigma*/sqrt(prior)':>20}")
    for K in Ks:
        rd, rn, lv = [], [], []
        for s in seeds:
            a = arms.get((s, K))
            if a is None:
                continue
            sg = np.sqrt(a["v_gp"])
            rd.append(spearmanr(sg, base[s]["dnn"]).statistic)
            rn.append(spearmanr(sg, a["nbr"].sum(axis=0)).statistic)
            lv.append(np.median(sg) / np.sqrt(float(a["prior_var"])))
        print(f"{K:>5} {np.mean(rd):>+22.4f} {np.mean(rn):>+20.4f} {np.mean(lv):>20.4f}")
    print("  If sigma* converges on dist-to-NN as the radius tightens, the GP is not")
    print("  adding information -- it is recomputing the baseline at the cost of a solve")
    print("  per test point. sigma*/sqrt(prior) -> 1 means reversion to the prior.")

    # ------------------------------------------------------------- the ranking question
    print()
    print("=" * 96)
    print("RANKING: Spearman(sigma, |err|) and top-5% recall, all points, mean over seeds")
    print("=" * 96)
    names = ["size+category", "dist-to-NN", "GP sigma* (latent)", "GP sigma* + noise"]
    print(f"{'K':>5} | " + " | ".join(f"{n[:17]:>17}" for n in names))
    print(f"{'':>5} | " + " | ".join(f"{'rho':>8}{'rec':>9}" for _ in names))
    print("-" * 96)
    rho_tab, rec_tab = {}, {}
    for K in Ks:
        cells = []
        for nm in names:
            rr, hh = [], []
            for s in seeds:
                a = arms.get((s, K))
                if a is None:
                    continue
                sg = candidates(a, base[s])[nm]
                rho, hit, _ = rank_metrics(sg, np.abs(a["err"]))
                rr.append(rho); hh.append(hit)
            rho_tab[(K, nm)] = np.array(rr); rec_tab[(K, nm)] = np.array(hh)
            cells.append(f"{np.mean(rr):>+8.4f}{np.mean(hh):>9.3f}")
        print(f"{K:>5} | " + " | ".join(cells))

    # ------------------------------- the paired statistic that actually answers the ask
    print()
    print("=" * 96)
    print("THE ANSWER: sigma* MINUS dist-to-NN, paired within seed")
    print("=" * 96)
    print("  positive = the GP's posterior variance beats a distance with no GP in it")
    for tab, lab in ((rho_tab, "Spearman"), (rec_tab, "top-5% recall")):
        print(f"\n  {lab}:")
        print(f"{'K':>7} {'per-seed differences':>34} {'mean':>9} {'SE':>8} {'verdict':>12}")
        for K in Ks:
            d = tab[(K, "GP sigma* (latent)")] - tab[(K, "dist-to-NN")]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            v = "GP wins" if d.mean() > 0 and abs(d.mean()) > 2 * se else (
                "baseline" if d.mean() < 0 and abs(d.mean()) > 2 * se else "tie")
            print(f"{K:>7} {np.array2string(np.round(d, 4), precision=4):>34} "
                  f"{d.mean():>+9.4f} {se:>8.4f} {v:>12}")

    # ----------------------------------------------------- scale-dependent metrics too
    print()
    print("=" * 96)
    print("PROPER SCORING RULES at each candidate's own best scale (2-fold cross-fitted)")
    print("=" * 96)
    print(f"{'K':>5} {'candidate':>20} {'NLPD':>9} {'CRPS':>9} {'miscal':>9} {'std(z)':>9}")
    for K in Ks:
        for nm in ["constant"] + names:
            n_, c_, m_, z_ = [], [], [], []
            for s in seeds:
                a = arms.get((s, K))
                if a is None:
                    continue
                sg = candidates(a, base[s])[nm]
                r = scale_metrics(sg, a["err"], a["y"], a["mu"])
                n_.append(r[0]); c_.append(r[1]); m_.append(r[2]); z_.append(r[3])
            print(f"{K:>5} {nm:>20} {np.mean(n_):>9.3f} {np.mean(c_):>9.4f} "
                  f"{np.mean(m_):>9.4f} {np.mean(z_):>9.4f}")
        print()

    # ------------------------------------------------------- the small-category caveat
    print("=" * 96)
    print("CAVEAT: categories too small to supply K same-category neighbours")
    print("=" * 96)
    print("  cutoff_for_neighbors masks cross-category distances to inf, then DROPS the")
    print("  non-finite k-th distances -- so these categories do not help set the radius,")
    print("  and the arms are not measuring the same thing across the dataset.")
    a0 = arms[(seeds[0], Ks[0])]
    cn = a0["cat_names"]
    for K in Ks:
        a = arms[(seeds[0], K)]
        sc = a["short_cats"]
        nm = ", ".join(str(cn[c]) for c in sc) if len(sc) else "(none)"
        print(f"  K={K:>4}: {len(sc)} categories excluded from the median "
              f"({float(a['short_frac_test']):.1%} of test points) -- {nm}")

    # per-category ranking at the tightest and widest arm, pooled over seeds
    print(f"\n  per-category Spearman(sigma*, |err|), pooled over seeds:")
    print(f"{'category':>22} {'n_test':>7} " +
          " ".join(f"{'K='+str(K):>9}" for K in Ks) + f" {'dist-NN':>9}")
    cats = sorted({int(c) for s in seeds for c in arms[(s, Ks[0])]["cat"]})
    for c in cats:
        row, ntot = [], 0
        for K in Ks:
            rr = []
            for s in seeds:
                a = arms[(s, K)]
                m = a["cat"] == c
                if m.sum() >= 15:
                    rr.append(spearmanr(np.sqrt(a["v_gp"][m]), np.abs(a["err"][m])).statistic)
            row.append(np.mean(rr) if rr else np.nan)
        for s in seeds:
            ntot += int((arms[(s, Ks[0])]["cat"] == c).sum())
        dd = []
        for s in seeds:
            a, b = arms[(s, Ks[0])], base[s]
            m = a["cat"] == c
            if m.sum() >= 15:
                dd.append(spearmanr(b["dnn"][m], np.abs(a["err"][m])).statistic)
        f = lambda x: f"{x:>+9.3f}" if np.isfinite(x) else f"{'--':>9}"
        print(f"{str(cn[c]):>22} {ntot:>7} " + " ".join(f(x) for x in row) +
              f" {f(np.mean(dd) if dd else np.nan)}")
    print("  (categories with <15 test points in a seed are skipped for that seed)")


if __name__ == "__main__":
    main()
