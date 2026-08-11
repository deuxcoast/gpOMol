"""Leave-one-family-out: does the uncertainty survive chemistry it has never seen?

THE BLIND SPOT THIS EXISTS TO CLOSE. Every calibration number in this project comes from
a RANDOM 80/20 split -- cov95 0.955, std(z) ~ 1.0, CRPS skill ~3%. Random test molecules
are drawn from the training distribution, so a descriptor that is confidently wrong OFF
distribution still looks perfectly calibrated. For active learning the entire point is
querying unusual molecules, so the number that matters is UQ quality on chemistry the
model has NOT seen, and it has never been measured.

THE SPECIFIC SUSPICION. A GNN trained on energies maps molecules with similar ENERGIES
close together. That is right for interpolation and wrong for novelty detection: two
structurally unrelated molecules with similar energies land near each other, the GP sees
coverage, and reports low uncertainty on a query unlike anything it has seen. Our
hand-crafted descriptor measures what a molecule IS rather than what its energy is --
worse for accuracy, better geometry for novelty. Prediction, stated in advance: the two
look similar on a random split and the GNN degrades MORE on held-out chemistry.

THE CATEGORY MASK HAD TO GO, AND THAT IS ITSELF A FINDING. Production zeroes
cross-category covariance, so with a whole family held out EVERY test point has zero
covariance with EVERY training point: sigma* is exactly the prior for all of them and the
mean reverts entirely. The GP contributes literally nothing on unseen chemistry, and
sigma* is CONSTANT there, hence useless for ranking -- which is precisely the acquisition
case. Honest (unseen chemistry really is maximally uncertain) but uninformative. So both
arms here run WITHOUT the mask; otherwise there is nothing to compare.

FAIRNESS. Both arms get the identical pipeline -- same mean (fit on training families
only, from the production channels), same kernel shape, same neighbour-count target, same
test points. ONLY the descriptor the kernel measures distance in changes. The mean is
deliberately NOT switched to the GNN embedding: that would confound descriptor-for-UQ
with descriptor-for-accuracy, and the question here is only about the uncertainty.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_calibration import crps_gaussian, coverage          # noqa: E402

MEAN_CHAN = ("wl", "geom", "strain")
KERNEL_CHAN = ("wl", "geom")
JITTER, K_NBR = 4.30, 200


def wendland(t):
    s = np.clip(1.0 - t, 0.0, None)
    return np.where(t < 1.0, s ** 4 * (4.0 * t + 1.0), 0.0)


def radius_for_nbrs(Ztr, target, sample=1500, seed=0):
    """One global radius giving ~target in-support neighbours. No category mask."""
    rng = np.random.default_rng(seed)
    i = rng.choice(len(Ztr), size=min(sample, len(Ztr)), replace=False)
    D = cdist(Ztr[i], Ztr)
    k = min(target, D.shape[1] - 1)
    return float(np.median(np.partition(D, k, axis=1)[:, k]))


def gp_predict(Ztr, r_tr, Zte, rho, sv, jitter=JITTER):
    """Dense exact GP on one descriptor. Small enough here to solve directly, which
    removes the sparse solver as a source of difference between the arms."""
    D = cdist(Ztr, Ztr)
    Kt = sv * wendland(D / rho)
    Kt[np.diag_indices_from(Kt)] += jitter
    Ks = sv * wendland(cdist(Zte, Ztr) / rho)
    L = np.linalg.cholesky(Kt + 1e-8 * np.eye(len(Kt)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r_tr))
    mu = Ks @ alpha
    V = np.linalg.solve(L, Ks.T)
    var = sv - np.einsum("ij,ij->j", V, V)
    return mu, np.maximum(var, 0.0)


def uq_metrics(y, mu, var, jitter=JITTER):
    err = y - mu
    sd = np.sqrt(var + jitter)
    mse = float(np.mean(err ** 2))
    c_gp = float(np.mean(crps_gaussian(y, mu, sd)))
    c_fl = float(np.mean(crps_gaussian(y, mu, np.full(len(y), np.sqrt(mse)))))
    return dict(
        r2=float(r2_score(y, mu)), rmse=float(np.sqrt(mse)),
        rho=float(spearmanr(sd, np.abs(err)).statistic),
        skill=100.0 * (c_fl - c_gp) / c_fl,
        cov95=coverage(err, sd, 0.95), stdz=float(np.std(err / sd)),
        rng=float(np.percentile(sd, 90) / np.percentile(sd, 10)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gnn", default="cache/gnn_esen_sm_direct_all_omol_20000.npz")
    ap.add_argument("--gnn-dims", type=int, default=40)
    ap.add_argument("--our-dims", type=int, default=10, help="per channel")
    ap.add_argument("--ntr", type=int, default=6000, help="train subsample (dense solve)")
    ap.add_argument("--nte", type=int, default=600)
    ap.add_argument("--families", nargs="+", default=None,
                    help="families to hold out; default = the 4 largest")
    a = ap.parse_args()

    from sklearn.cross_decomposition import PLSRegression
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean

    P = np.load(os.path.join("cache", f"emb_{a.n}_s{a.seed}.npz"), allow_pickle=True)
    G = np.load(a.gnn, allow_pickle=True)
    ds = get_data(src="train_4M", n=a.n, seed=0)
    idx = np.arange(len(ds.atoms))
    tr0, te0 = train_test_split(idx, test_size=0.2, random_state=a.seed)
    names = list(ds.category_names)
    cat_all = ds.data_id
    y_all = ds.y

    # rebuild full-length arrays in the original order so families can be re-split
    ours = np.zeros((len(idx), sum(P[f"{c}_tr"].shape[1] for c in KERNEL_CHAN)))
    meanZ = np.zeros((len(idx), sum(P[f"{c}_tr"].shape[1] for c in MEAN_CHAN)))
    ours[tr0] = np.hstack([P[f"{c}_tr"] for c in KERNEL_CHAN])
    ours[te0] = np.hstack([P[f"{c}_te"] for c in KERNEL_CHAN])
    meanZ[tr0] = np.hstack([P[f"{c}_tr"] for c in MEAN_CHAN])
    meanZ[te0] = np.hstack([P[f"{c}_te"] for c in MEAN_CHAN])
    gnn_raw = np.asarray(G["sum_pool"], float)          # sum-pool won the screen

    sizes = sorted(((int((cat_all == c).sum()), c) for c in np.unique(cat_all)),
                   reverse=True)
    fams = ([names.index(f) for f in a.families] if a.families
            else [c for _, c in sizes[:4]])
    print(f"[lofo] holding out: {[names[c] for c in fams]}")
    print(f"[lofo] NO category mask (production would give sigma* = prior for every "
          f"held-out point, i.e. nothing to compare)\n")

    rng = np.random.default_rng(0)
    rows = []
    for split in ("random", "held-out family"):
        for c in fams:
            if split == "random":
                # matched control: same sizes, same family, but drawn IN-distribution
                pool = np.where(cat_all == c)[0]
                te = rng.choice(pool, min(a.nte, len(pool) // 2), replace=False)
                rest = np.setdiff1d(idx, te)
            else:
                te = np.where(cat_all == c)[0]
                te = rng.choice(te, min(a.nte, len(te)), replace=False)
                rest = np.where(cat_all != c)[0]        # the family is fully absent
            tr = rng.choice(rest, min(a.ntr, len(rest)), replace=False)

            mean = LinearEmbeddingMean().fit(meanZ[tr], y_all[tr], size=None)
            r_tr = y_all[tr] - mean.predict(meanZ[tr], size=None)
            r_te = y_all[te] - mean.predict(meanZ[te], size=None)
            sv = float(np.var(r_tr))

            arms = {}
            d = a.our_dims
            arms["ours"] = (np.hstack([ours[tr][:, :d], ours[tr][:, 10:10 + d]]),
                            np.hstack([ours[te][:, :d], ours[te][:, 10:10 + d]]))
            mu_, sd_ = gnn_raw[tr].mean(0), np.where(gnn_raw[tr].std(0) > 0,
                                                     gnn_raw[tr].std(0), 1.0)
            Etr, Ete = (gnn_raw[tr] - mu_) / sd_, (gnn_raw[te] - mu_) / sd_
            pls = PLSRegression(n_components=a.gnn_dims, scale=False).fit(Etr, r_tr)
            arms["gnn"] = (pls.transform(Etr), pls.transform(Ete))

            for arm, (Ztr, Zte) in arms.items():
                rho = radius_for_nbrs(Ztr, K_NBR)
                m, v = gp_predict(Ztr, r_tr, Zte, rho, sv)
                mt = uq_metrics(r_te, m, v)
                rows.append((split, names[c], arm, mt))
                print(f"  {split:>16} {names[c]:>16} {arm:>5}  "
                      f"R2 {mt['r2']:>+6.3f}  rho {mt['rho']:>+6.3f}  "
                      f"skill {mt['skill']:>+6.2f}%  cov95 {mt['cov95']:.3f}  "
                      f"sd-range {mt['rng']:.2f}", flush=True)

    print("\n" + "=" * 78)
    print("DEGRADATION  (held-out family MINUS random, per arm; negative = worse "
          "on unseen chemistry)")
    print("=" * 78)
    print(f"{'arm':>6} {'d R2':>9} {'d rho':>9} {'d skill':>10} {'d cov95':>9} "
          f"{'d sd-range':>11}")
    for arm in ("ours", "gnn"):
        g = lambda s, k: np.mean([m[k] for sp, _, ar, m in rows
                                  if ar == arm and sp == s])
        print(f"{arm:>6} {g('held-out family','r2') - g('random','r2'):>+9.3f} "
              f"{g('held-out family','rho') - g('random','rho'):>+9.3f} "
              f"{g('held-out family','skill') - g('random','skill'):>+10.2f} "
              f"{g('held-out family','cov95') - g('random','cov95'):>+9.3f} "
              f"{g('held-out family','rng') - g('random','rng'):>+11.2f}")
    print("\n  The PREDICTION, made before running: both look similar on the random")
    print("  split and the GNN degrades MORE on held-out chemistry, because an")
    print("  energy-trained embedding places similar-ENERGY molecules together and so")
    print("  reports coverage for structurally novel queries. If 'd rho' and 'd skill'")
    print("  fall further for gnn than for ours, that is the effect. If they do not,")
    print("  the objection is wrong and a learned descriptor is simply better.")


if __name__ == "__main__":
    main()
