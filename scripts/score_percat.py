"""Score the per-category radius arms against the global-radius baseline.

The question is the position paper's, not an accuracy question: does allocating the
support radius per chemical family make the GP's posterior variance a better ranker of
its own errors, and specifically does it close the gap to `dist-to-NN`, which has beaten
sigma* at every radius, every mean strength and every kernel tried so far.

EVERYTHING IS PAIRED WITHIN SEED. Same split, same 800 test points, same M1 mean, same
jitter, same additive wl+geom kernel. Only the radius allocation differs. The baselines
are recomputed per arm because |err| changes when the kernel changes, so a cross-arm
comparison of raw rho would be confounded by the errors moving underneath it.

READ THE DENSITY COLUMN FIRST. RADIUS_CALIBRATION.md established that R^2 and sigma*'s
ranking skill BOTH rise monotonically with matrix density. If the per-category arm is
denser than the global arm, any improvement may be the density rather than the
allocation, which is what the `matched` arm exists to rule out.
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "cache/percat"
ARMS = ("global", "percat", "matched", "product")


def rank(s, aerr, frac=0.05):
    rho = float(spearmanr(s, aerr).statistic)
    k = max(int(frac * len(s)), 1)
    hit = len(set(np.argsort(aerr)[-k:]) & set(np.argsort(s)[-k:])) / k
    return rho, hit


def main():
    A = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*_s*.npz"))):
        d = np.load(f, allow_pickle=True)
        A[(str(d["arm"]), int(d["seed"]))] = d
    seeds = sorted({s for _, s in A})
    arms = [a for a in ARMS if any((a, s) in A for s in seeds)]
    print(f"loaded {len(A)} arms: {arms} x seeds {seeds}\n")

    print("=" * 104)
    print("ACCURACY AND SPARSITY")
    print("=" * 104)
    print(f"{'arm':>9} {'density':>11} {'OLS R2':>9} {'GP R2':>9} {'kernel adds':>12} "
          f"{'warn':>5}")
    for arm in arms:
        dn, o, g, w = [], [], [], []
        for s in seeds:
            d = A.get((arm, s))
            if d is None:
                continue
            dn.append(float(d["dens_percat"])); o.append(float(d["ols_r2"]))
            g.append(r2_score(d["y"], d["mu"])); w.append(int(d["n_warn"]))
        print(f"{arm:>9} {np.mean(dn):>11.4e} {np.mean(o):>9.4f} {np.mean(g):>9.4f} "
              f"{np.mean(g) - np.mean(o):>+12.4f} {sum(w):>5}")

    print()
    print("=" * 104)
    print("UNCERTAINTY: does a per-category radius make sigma* a better ranker?")
    print("=" * 104)
    print(f"{'arm':>9} | {'rho(sigma*)':>12} {'top5%':>7} | {'rho(dist-NN)':>13} "
          f"{'rho(size+cat)':>14} | {'sigma* - dNN':>13} {'SE':>7} {'verdict':>10}")
    print("-" * 104)
    tab = {}
    for arm in arms:
        rs, rec, rd, rc, dif = [], [], [], [], []
        for s in seeds:
            d = A.get((arm, s))
            if d is None:
                continue
            ae = np.abs(d["err"])
            x1, h1 = rank(np.sqrt(np.maximum(d["v_gp"], 0)), ae)
            x2, _ = rank(d["dnn"], ae)
            x3, _ = rank(d["sd_cheap"], ae)
            rs.append(x1); rec.append(h1); rd.append(x2); rc.append(x3)
            dif.append(x1 - x2)
        dif = np.array(dif)
        se = dif.std(ddof=1) / np.sqrt(len(dif)) if len(dif) > 1 else np.nan
        v = ("GP WINS" if dif.mean() > 2 * se else
             "baseline" if -dif.mean() > 2 * se else "tie")
        tab[arm] = dif
        print(f"{arm:>9} | {np.mean(rs):>+12.4f} {np.mean(rec):>7.3f} | "
              f"{np.mean(rd):>+13.4f} {np.mean(rc):>+14.4f} | "
              f"{dif.mean():>+13.4f} {se:>7.4f} {v:>10}")

    if "global" in tab:
        print()
        print("=" * 104)
        print("PAIRED CHANGE vs the GLOBAL-RADIUS baseline, per seed")
        print("=" * 104)
        print(f"{'arm':>9} {'d GP_R2':>10} {'d rho(sigma*)':>15} "
              f"{'d [sigma*-dNN]':>16} {'SE':>8}")
        for arm in arms:
            if arm == "global":
                continue
            dr2, drho, dgap = [], [], []
            for s in seeds:
                x, b = A.get((arm, s)), A.get(("global", s))
                if x is None or b is None:
                    continue
                dr2.append(r2_score(x["y"], x["mu"]) - r2_score(b["y"], b["mu"]))
                rx = rank(np.sqrt(np.maximum(x["v_gp"], 0)), np.abs(x["err"]))[0]
                rb = rank(np.sqrt(np.maximum(b["v_gp"], 0)), np.abs(b["err"]))[0]
                drho.append(rx - rb)
                dgap.append(tab[arm][seeds.index(s)] - tab["global"][seeds.index(s)])
            dgap = np.array(dgap)
            se = dgap.std(ddof=1) / np.sqrt(len(dgap)) if len(dgap) > 1 else np.nan
            print(f"{arm:>9} {np.mean(dr2):>+10.4f} {np.mean(drho):>+15.4f} "
                  f"{dgap.mean():>+16.4f} {se:>8.4f}")

    # where the allocation actually differs, and whether those categories improved
    print()
    print("=" * 104)
    print("PER-CATEGORY: radius ratio vs global, and rho(sigma*,|err|) by arm")
    print("=" * 104)
    d0 = A.get(("percat", seeds[0]))
    if d0 is not None:
        cn = d0["cat_names"]
        ratio = d0["cuts_cat"] / d0["cuts_glob"][:, None]
        print(f"{'category':>18} {'rho_wl/glob':>12} {'rho_gm/glob':>12} "
              + " ".join(f"{a[:8]:>9}" for a in arms))
        for c in range(len(cn)):
            row = []
            for arm in arms:
                rr = []
                for s in seeds:
                    d = A.get((arm, s))
                    if d is None:
                        continue
                    m = d["cat"] == c
                    if m.sum() >= 15:
                        rr.append(spearmanr(np.sqrt(np.maximum(d["v_gp"][m], 0)),
                                            np.abs(d["err"][m])).statistic)
                row.append(np.mean(rr) if rr else np.nan)
            f = lambda x: f"{x:>+9.3f}" if np.isfinite(x) else f"{'--':>9}"
            print(f"{str(cn[c]):>18} {ratio[0, c]:>12.3f} {ratio[1, c]:>12.3f} "
                  + " ".join(f(x) for x in row))
        print("  (categories with <15 test points in a seed are skipped for that seed)")


if __name__ == "__main__":
    main()
