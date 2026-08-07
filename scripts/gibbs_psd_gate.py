"""PHASE 1 GATE: is a Gibbs construction with a WENDLAND shape actually PSD?

WHY THIS IS IN DOUBT. The Gibbs / Paciorek-Schervish non-stationary kernel

    k(x,x') = sigma(x) sigma(x') * [2 l(x) l(x') / (l(x)^2+l(x')^2)]^(d/2)
                                 * R( |x-x'| / sqrt(l(x)^2+l(x')^2) )

is proven positive definite only when the SHAPE R is a valid correlation function in
R^d for EVERY d (Paciorek & Schervish 2004, Thm 1; equivalently R is completely
monotone in the squared argument, by Schoenberg). The Gaussian and the Matern satisfy
that. WENDLAND FUNCTIONS DO NOT -- each is positive definite only up to a fixed maximum
dimension, and ours, psi(r) = (1-r)^4 (4r+1), is the d <= 3 member. We already apply it
to a 10-dimensional PLS embedding, so the question predates the Gibbs proposal; the
Gibbs construction just makes it load-bearing, because the prefactor is the only thing
standing between a varying length scale and an invalid covariance.

So: measure it. Nothing downstream is worth building until this is settled.

DESIGN. Every run reports a POSITIVE CONTROL and a TREATMENT on the same points, the
same length scales and the same target density:

  * control   = Gibbs with a GAUSSIAN shape. This IS proven PSD. If it comes out
                indefinite, the bug is in this file, not in the mathematics -- which is
                the only way to tell a real result from a transcription error.
  * treatment = Gibbs with the Wendland shape.

l(x) is the real thing we intend to use: c * (distance to the k-th nearest
same-category neighbour), on real embedding data, with c bisected to a production-like
neighbour count -- because a very sparse Gram is nearly diagonal and trivially PSD, so
testing at the wrong density would give a falsely reassuring answer.

`--alpha` controls how much l varies: l_i propto sigma_k(i)^alpha, so alpha=0 is exactly
constant (must be PSD, a second control) and larger alpha exaggerates the variation.

READING THE RESULT. Exact PSD is not the practical bar -- we add a jitter of 4.30 to the
diagonal in production regardless. The number that matters is `min_eig / jitter`: if the
most negative eigenvalue is far smaller than the jitter we already add, the construction
is usable even if not provably PSD, and the paper says so honestly.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEAN_CHAN = ("wl", "geom", "strain")


def wendland(t):
    """psi_{3,1}(r) = (1-r)^4 (4r+1), zero beyond r=1. PD in R^d for d <= 3."""
    s = np.clip(1.0 - t, 0.0, None)
    return np.where(t < 1.0, s ** 4 * (4.0 * t + 1.0), 0.0)


def gaussian(t):
    """exp(-t^2). PD in R^d for EVERY d -- the positive control."""
    return np.exp(-t * t)


SHAPES = {"wendland": wendland, "gaussian": gaussian}


def local_scale(D, k):
    """sigma_i = distance to the k-th nearest neighbour of i (self excluded)."""
    Dk = D.copy()
    np.fill_diagonal(Dk, np.inf)
    return np.sort(Dk, axis=1)[:, k - 1]


def gibbs_gram(D, ell, shape, d_pref, sig=None):
    """The Gibbs/Paciorek-Schervish Gram with an arbitrary shape function.

    denom_ij = l_i^2 + l_j^2 ; the effective support between i and j is sqrt(denom_ij),
    so compact support SURVIVES the construction -- the prefactor is a positive scalar.
    With l constant the prefactor is identically 1 and this reduces to the stationary
    kernel at radius l*sqrt(2), which is the alpha=0 control."""
    li = ell[:, None]
    lj = ell[None, :]
    denom = li ** 2 + lj ** 2
    pref = (2.0 * li * lj / denom) ** (0.5 * d_pref)
    K = pref * shape(D / np.sqrt(denom))
    if sig is not None:
        K = K * sig[:, None] * sig[None, :]
    return K


def median_neighbors(D, ell):
    """Median in-support neighbour count under the Gibbs support sqrt(l_i^2+l_j^2)."""
    nz = D < np.sqrt(ell[:, None] ** 2 + ell[None, :] ** 2)
    return float(np.median(nz.sum(axis=1) - 1))


def fit_scale(D, base, target):
    """Bisect one global multiplier on l until the median neighbour count hits target.

    Density is the confound here: a nearly diagonal Gram is trivially PSD, so testing at
    a sparser support than production would report a reassuring non-result."""
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if median_neighbors(D, base * mid) > target:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="cache/emb_20000_s42.npz")
    ap.add_argument("--n", type=int, default=20000, help="N the embedding was built at")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chan", default="wl", choices=list(MEAN_CHAN))
    ap.add_argument("--k", type=int, default=10, help="k for the local scale sigma_k")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                    help="l_i propto sigma_k(i)^alpha. 0 = constant l (a control that "
                         "MUST be PSD); larger exaggerates the variation")
    ap.add_argument("--target-nbrs", type=int, default=200,
                    help="median in-support neighbours, matched to production")
    ap.add_argument("--max-rows", type=int, default=2500,
                    help="rows per category block (eigvalsh is O(n^3))")
    ap.add_argument("--jitter", type=float, default=4.30,
                    help="the production jitter, for the practical-usability column")
    a = ap.parse_args()

    from wl_gp2scale.data import get_data

    d = np.load(a.emb, allow_pickle=True)
    Z = d[a.chan + "_tr"]
    ds = get_data(src="train_4M", n=a.n, seed=0)
    tr, _ = train_test_split(np.arange(len(ds.atoms)), test_size=0.2,
                             random_state=a.seed)
    cat = ds.data_id[tr]
    names = list(ds.category_names)

    dim = Z.shape[1]
    print("channel %s, embedding dim %d, Wendland psi_{3,1} is PD in R^d for d<=3"
          % (a.chan, dim))
    print("prefactor exponent d/2 uses the AMBIENT dimension d=%d\n" % dim)

    # the largest few categories, which are the blocks that actually matter
    sizes = [(int((cat == c).sum()), c) for c in np.unique(cat)]
    sizes.sort(reverse=True)
    picks = [c for _, c in sizes[:3]]

    print("%-16s %6s %7s %8s %11s %12s %12s %9s"
          % ("category", "rows", "alpha", "shape", "l spread", "min eig", "min/max",
             "min/jitter"))
    print("-" * 96)
    verdict = []
    for c in picks:
        idx = np.where(cat == c)[0]
        if len(idx) > a.max_rows:
            idx = np.random.default_rng(0).choice(idx, a.max_rows, replace=False)
        Zc = Z[idx]
        D = cdist(Zc, Zc)
        sk = local_scale(D, a.k)
        sk = np.maximum(sk, 1e-9)
        for alpha in a.alpha:
            base = (sk / np.exp(np.mean(np.log(sk)))) ** alpha
            scale = fit_scale(D, base, a.target_nbrs)
            ell = base * scale
            spread = float(np.percentile(ell, 90) / np.percentile(ell, 10))
            for sname, shape in SHAPES.items():
                K = gibbs_gram(D, ell, shape, d_pref=dim)
                ev = np.linalg.eigvalsh(K)
                mn, mx = float(ev[0]), float(ev[-1])
                print("%-16s %6d %7.2f %8s %11.2f %12.3e %12.2e %9.1e"
                      % (names[c] if sname == "wendland" else "", len(idx), alpha,
                         sname, spread, mn, mn / mx, mn / a.jitter))
                verdict.append((sname, alpha, mn, mx))
        print()

    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    for sname in SHAPES:
        v = [x for x in verdict if x[0] == sname]
        worst = min(v, key=lambda x: x[2])
        rel = worst[2] / worst[3]
        print("  %-9s worst min-eig %11.3e  (rel %9.2e, %.1e x the production jitter)"
              % (sname, worst[2], rel, worst[2] / a.jitter))
    # ---- POWER CHECK -------------------------------------------------------------
    # A gate that cannot fail is not a gate. Re-run the worst case with a DELIBERATELY
    # WRONG prefactor exponent: d_pref=0 removes the prefactor entirely, and d_pref=3 is
    # the mistake worth naming, since psi_{3,1} is "the d<=3 Wendland" and matching the
    # exponent to the SHAPE's design dimension instead of the AMBIENT dimension is the
    # natural wrong guess.
    print()
    print("=" * 96)
    print("POWER CHECK: is this test able to detect an invalid kernel?")
    print("=" * 96)
    c = picks[0]
    idx = np.where(cat == c)[0]
    if len(idx) > a.max_rows:
        idx = np.random.default_rng(0).choice(idx, a.max_rows, replace=False)
    D = cdist(Z[idx], Z[idx])
    sk = np.maximum(local_scale(D, a.k), 1e-9)
    alpha_hi = max(max(a.alpha), 4.0)
    base = (sk / np.exp(np.mean(np.log(sk)))) ** alpha_hi
    ell = base * fit_scale(D, base, a.target_nbrs)
    print("  %s, alpha=%.1f, l spread %.1fx"
          % (names[c], alpha_hi, np.percentile(ell, 90) / np.percentile(ell, 10)))
    print("  %-28s %13s %12s" % ("prefactor exponent", "min eig", "min/max"))
    for dp, note in ((0, "none (no prefactor)"), (1, "d=1"), (3, "d=3 (shape's design d)"),
                     (dim, "d=%d (AMBIENT -- correct)" % dim)):
        ev = np.linalg.eigvalsh(gibbs_gram(D, ell, wendland, d_pref=dp))
        print("  %-28s %13.3e %12.2e" % (note, ev[0], ev[0] / ev[-1]))
    print("\n  If the wrong exponents come out strongly negative and the ambient one does")
    print("  not, the test is sensitive AND the prefactor exponent must be the EMBEDDING")
    print("  dimension, not the Wendland's design dimension. That is a real trap: psi is")
    print("  'the d<=3 Wendland', so d=3 is the natural guess and it is badly wrong.")

    ctrl = min([x[2] for x in verdict if x[0] == "gaussian"])
    if ctrl < -1e-8 * max(x[3] for x in verdict if x[0] == "gaussian"):
        print("\n  CONTROL FAILED: the Gaussian Gibbs kernel is PROVEN PSD, so a")
        print("  materially negative eigenvalue here means this script is wrong.")
        print("  Fix the implementation before reading the Wendland row at all.")
    else:
        print("\n  Control OK (Gaussian Gibbs is PSD as the theorem says), so the")
        print("  Wendland row is a statement about the mathematics, not about this code.")
    print("\n  alpha=0 is constant l: it MUST be PSD for both shapes, since the")
    print("  prefactor is then identically 1 and the kernel is ordinary stationary.")
    print("  Practical bar is the last column: a negative eigenvalue far smaller than")
    print("  the jitter we already add to the diagonal is usable in production.")


if __name__ == "__main__":
    main()
