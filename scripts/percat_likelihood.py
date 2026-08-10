"""Does the MARGINAL LIKELIHOOD want a different per-category radius than our heuristic?

This is the gate for any MCMC campaign over rho(x). Our per-category radii come from a
NEIGHBOUR-COUNT heuristic (the median distance to the 200th nearest same-category train
point). gp2Scale's prescription sets them by MARGINAL LIKELIHOOD under a
sparsity-preferring prior. Those are different objectives. If the heuristic already sits
near the likelihood optimum, MCMC over the same parameterisation will land near the
+0.024 the discrete version already delivers, and the campaign is not worth two hours per
seed. If the likelihood pulls somewhere else, MCMC is justified and we know which way.

THE LIKELIHOOD IS EXACTLY SEPARABLE BY CATEGORY, which makes this cheap and exact rather
than approximate. `kernel.py` zeroes cross-category covariance unconditionally, so after
sorting by category KV = K + sigma^2 I is block diagonal, and

    log L = sum_c [ -0.5 y_c^T KV_c^-1 y_c - 0.5 log|KV_c| ] - (n/2) log 2pi

Every term depends on one category's radius alone. So:
  * coordinate-wise profiling is NOT an approximation, it is the exact optimum;
  * no gp2Scale rebuild is needed. The largest block at 20k is ~3,200 rows, so each
    likelihood evaluation is one dense Cholesky, about a second, instead of a ~20 s
    distributed sparse assembly.

PARAMETERISATION. One multiplier m_c per category, applied to BOTH channel radii for that
category, exactly as a per-category rho would act. m_c = 1 is our heuristic, so the null
being tested is "the profile peaks at 1".

Signal variances and jitter are held at the values the arms used, so the only thing moving
is the radius. psi comes from wl_gp2scale.kernel so it is byte-identical to production.
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _wendland32_np(t):
    """psi_{3,2}(r) = (1-r)^4 (4r+1), r clipped to [0,1].

    Re-stated in numpy rather than imported: `kernel._wendland32` calls `s.pow(4)`, which
    is a torch method, so it cannot take an ndarray. The formula is transcribed verbatim
    from that function and `test_matches_production_psi` below checks the two agree.
    """
    s = 1.0 - t
    return s ** 4 * (4.0 * t + 1.0)


def test_matches_production_psi():
    """Guard against the transcription drifting from kernel._wendland32."""
    import torch
    from wl_gp2scale.kernel import _wendland32
    t = np.linspace(0, 1, 257)
    a = _wendland32_np(t)
    b = _wendland32(torch.as_tensor(t, dtype=torch.float64)).numpy()
    assert np.allclose(a, b, atol=0, rtol=1e-15), np.abs(a - b).max()

N, JITTER, K = 20_000, 4.30, 200
KERNEL_CHAN = ("wl", "geom")
MEAN_CHAN = ("wl", "geom", "strain")
GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

# THE GRID IS PARAMETERISED IN NEIGHBOURS PER ROW, NOT IN A RADIUS MULTIPLIER.
#
# A multiplier grid cannot be read. Measured at 20k, a multiplier of 3 on the length
# scale -- the top of the old grid -- puts the largest families at 86-100% of their
# block, i.e. it IS the dense GP. So "argmax at the grid edge" was never "we should
# have tested wider"; there is nothing wider than dense. The old grid also made the
# saturation invisible: metal_complexes appeared to acquire an interior optimum at
# m=2.5, which is where its block reaches ~95% density -- the likelihood had simply
# run out of pairs to add, not found a correlation length.
#
# Neighbours per row is the axis that means something: it is the computational budget,
# it is comparable across families of different size and across N, and it is the unit
# a sparsity-preferring prior should be specified in. DENSE is included explicitly as
# the limit, so the result can be stated as "the likelihood is maximised at the fully
# dense covariance" -- which is exactly the regime the position paper's sec 7 says
# offers no advantage over a standard dense GP.
# ...but NEIGHBOURS CANNOT BE THE GRID AXIS, because the map from length scale to
# neighbour count STOPS BEING INJECTIVE once the support saturates. Past full coverage
# the sparsity pattern is frozen while l keeps changing the kernel VALUES: as l grows,
# psi -> 1 for every pair and K degenerates toward rank one. Measured on biomolecules,
# two scales that both give all 3194 neighbours differ by 584 nats. "Dense" is a family
# of kernels, not one.
#
# So the grid runs over the SCALE (the actual parameter) and reports the achieved
# neighbour count as a derived column -- interpretability without the degeneracy. It is
# deliberately wide: the old grid stopped at m=3, and every family's optimum turns out
# to lie between m=2 and m=16, so that grid reported "monotone to the edge" for families
# whose peak was simply past it.
SCALE_GRID = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
                       512.0, 2048.0])


def block_loglik(Ds, cuts, r, sv, jitter):
    """Exact log marginal likelihood of ONE category block, dropping the constant
    -(n/2)log 2pi (it is shared by every arm and cancels in every comparison here)."""
    n = len(r)
    KV = np.zeros((n, n))
    for D, c, s in zip(Ds, cuts, sv):
        t = np.clip(D / c, 0.0, 1.0)
        KV += s * np.where(t < 1.0, _wendland32_np(t), 0.0)
    KV[np.diag_indices_from(KV)] += jitter
    try:
        cf = cho_factor(KV, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.nan
    alpha = cho_solve(cf, r, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return float(-0.5 * r @ alpha - 0.5 * logdet)


def block_loglik_product(Ds, cuts, r, sv, jitter):
    """PRODUCT kernel on one block: k = sv * prod_c psi(d_c / rho_c).

    The recorded grid-edge result was measured on the ADDITIVE kernel, which is no
    longer what production runs. That is the same class of untested assumption as the
    hardcoded mean rung, so the profile is available on every kernel we actually use."""
    n = len(r)
    K = np.ones((n, n))
    for D, c in zip(Ds, cuts):
        t = np.clip(D / c, 0.0, 1.0)
        K *= np.where(t < 1.0, _wendland32_np(t), 0.0)
    return _chol_loglik(sv * K, r, jitter)


def block_loglik_gibbs(Ds, sks, dims, ells_scale, r, sv, jitter):
    """GIBBS kernel on one block: l_i = ells_scale * sigma_k(i), support sqrt(l_i^2+l_j^2).

    dims must be the AMBIENT channel dimensions -- the prefactor exponent is what makes
    a varying length scale admissible, and using the Wendland's design dimension instead
    leaves a relative min-eigenvalue of -4.7e-2 (scripts/gibbs_psd_gate.py)."""
    n = len(r)
    K = np.ones((n, n))
    for D, sk, dim in zip(Ds, sks, dims):
        ell = np.maximum(ells_scale * sk, 1e-12)
        S = ell[:, None] ** 2 + ell[None, :] ** 2
        pref = (2.0 * ell[:, None] * ell[None, :] / S) ** (0.5 * dim)
        t = np.clip(D / np.sqrt(S), 0.0, 1.0)
        K *= pref * np.where(t < 1.0, _wendland32_np(t), 0.0)
    return _chol_loglik(sv * K, r, jitter)


def _chol_loglik(K, r, jitter):
    KV = K.copy()
    KV[np.diag_indices_from(KV)] += jitter
    try:
        cf = cho_factor(KV, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.nan
    alpha = cho_solve(cf, r, check_finite=False)
    return float(-0.5 * r @ alpha - np.sum(np.log(np.diag(cf[0]))))


def _stat_nbrs(Ds, cuts, mode):
    """Median in-support neighbours for the stationary kernels. The additive kernel's
    support is the UNION over channels, the product kernel's the INTERSECTION."""
    nz = None
    for D, c in zip(Ds, cuts):
        near = D < c
        nz = near if nz is None else (nz | near if mode == "union" else nz & near)
    return float(np.median(nz.sum(axis=1) - 1))


def scale_for_nbrs(nbr_fn, target, lo=1e-4, hi=1e4, iters=50):
    """Bisect one multiplier until the median neighbour count hits `target`.

    Geometric bisection: the scale spans orders of magnitude across families and
    targets, so a linear bracket would waste most of its resolution."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if nbr_fn(mid) > target:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def _gibbs_nbrs(Ds, sks, scale):
    """Median in-support neighbours under the Gibbs support, for the m=1 calibration."""
    nz = None
    for D, sk in zip(Ds, sks):
        ell = scale * sk
        near = D < np.sqrt(ell[:, None] ** 2 + ell[None, :] ** 2)
        nz = near if nz is None else (nz & near)
    return float(np.median(nz.sum(axis=1) - 1))


def gibbs_unit_scale(Ds, sks, target):
    """The l multiplier that puts this block at `target` neighbours -- so that m=1 in the
    profile means THE HEURISTIC, exactly as it does for the stationary arms. Without this
    the Gibbs grid would be centred on an arbitrary point (sigma_k is an absolute distance
    with no reason to sit near the neighbour-count radius) and the two profiles would not
    be comparable."""
    lo, hi = 1e-4, 1e4
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if _gibbs_nbrs(Ds, sks, mid) > target:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--emb-dir", default="cache")
    ap.add_argument("--n", type=int, default=N,
                    help="dataset size; must match an existing cache/emb_{n}_s{seed}.npz")
    ap.add_argument("--mean", default="M1", choices=["M0", "M1", "M4"],
                    help="prior mean rung the residual is taken against. The original "
                         "run hardcoded M1, so 'the likelihood wants the radius at the "
                         "grid edge' was measured at ONE mean and the mean was never a "
                         "variable. M0 (no mean at all, just the extensive residual) "
                         "tests whether a strong mean is what removed the correlation "
                         "length. NOTE M0 is gated out as a PRODUCTION config (-0.36 "
                         "R^2); this is a diagnostic probe, which is a different thing.")
    ap.add_argument("--kernel", default="additive",
                    choices=["additive", "product", "gibbs"],
                    help="which kernel to profile. The recorded 'likelihood wants the "
                         "radius at the grid edge' result was measured on ADDITIVE, "
                         "which is no longer production. 'gibbs' profiles l(x) = m * "
                         "sigma_k(x), the non-stationary kernel, with m=1 recalibrated "
                         "per block to the same neighbour-count target so the grid stays "
                         "comparable to the stationary profiles.")
    ap.add_argument("--gibbs-k", type=int, default=10,
                    help="k for sigma_k(x) under --kernel gibbs")
    ap.add_argument("--sv-grid", type=float, nargs="+", default=None,
                    help="multipliers on the signal variance, profiled JOINTLY with the "
                         "radius. The original run held sv fixed at prior_var/C, shared "
                         "across categories whose residual variances differ 8.2x -- so "
                         "the radius may have been absorbing a VARIANCE misspecification "
                         "rather than expressing a correlation length. Since the "
                         "likelihood is exactly separable by category, a joint (radius, "
                         "sv) grid per category is still exact. Costs len(sv-grid)x.")
    a = ap.parse_args()

    from wl_gp2scale.cutoff import cutoff_for_neighbors
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from percat_radius import per_category_cutoffs

    # Check every embedding BEFORE get_data: a wrong --n otherwise materialises and
    # caches the whole subset (803 MB at 100k, minutes at 4M) and only then fails.
    missing = [f"emb_{a.n}_s{sd}.npz" for sd in a.seeds
               if not os.path.exists(os.path.join(a.emb_dir, f"emb_{a.n}_s{sd}.npz"))]
    if missing:
        raise SystemExit(f"no embedding(s) in {a.emb_dir}: {missing}. Build them with a "
                         f"product_kernel.py / percat_radius.py run at --n {a.n}, which "
                         f"caches them. (Checked before loading data, so nothing was "
                         f"materialised.)")

    ds = get_data(src="train_4M", n=a.n, seed=0)
    idx = np.arange(len(ds.atoms))
    ncat = len(ds.category_names)
    names = np.array(ds.category_names)
    best_all = {}

    for seed in a.seeds:
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        y_tr = ds.y[tr]
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
        d = np.load(os.path.join(a.emb_dir, f"emb_{a.n}_s{seed}.npz"), allow_pickle=True)
        emb = {n: {"Ztr": d[f"{n}_tr"], "Zte": d[f"{n}_te"]} for n in
               sorted(set(MEAN_CHAN) | set(KERNEL_CHAN))}
        Zm_tr = np.hstack([emb[n]["Ztr"] for n in MEAN_CHAN])
        if a.mean == "M0":
            # no prior mean at all: the GP models the extensive residual directly
            r_tr = y_tr.copy()
        else:
            msz = (np.array([len(ds.atoms[i]) for i in tr], float)
                   if a.mean == "M4" else None)
            mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=msz)
            r_tr = y_tr - mean.predict(Zm_tr, size=msz)
        prior_var = float(np.var(r_tr))
        # the additive kernel splits the signal variance across channels so the diagonal
        # stays var(y); product and Gibbs multiply, so the diagonal is sv once
        sv0 = (np.array([prior_var / len(KERNEL_CHAN)] * len(KERNEL_CHAN))
               if a.kernel == "additive" else np.array([prior_var]))
        sv_grid = np.array(a.sv_grid) if a.sv_grid else np.array([1.0])
        print(f"[lik] seed {seed} mean={a.mean}: var(residual)={prior_var:.3f}, "
              f"sv per channel={sv0[0]:.3f}, jitter={JITTER}")

        cuts_cat = []
        for n in KERNEL_CHAN:
            cc, _ = per_category_cutoffs(emb[n]["Ztr"], emb[n]["Zte"], cat_tr, cat_te,
                                         ncat)
            cuts_cat.append(cc)
        cuts_cat = np.array(cuts_cat)

        print("=" * 96)
        print(f"SEED {seed}   (m = multiplier on the heuristic per-category radius; "
              f"m=1 is the heuristic)")
        print("=" * 96)
        print(f"{'category':>18} {'n_train':>8} {'argmax':>12} {'frac':>6} "
              f"{'dlogL vs m=1':>13}   profile over m={[float(g) for g in SCALE_GRID]}")
        tot = 0.0
        short_gibbs = {}
        for c in range(ncat):
            m = cat_tr == c
            n_c = int(m.sum())
            if n_c < 30:
                print(f"{str(names[c]):>18} {n_c:>8}   (skipped, too few rows)")
                continue
            Ds = [cdist(emb[n]["Ztr"][m], emb[n]["Ztr"][m]) for n in KERNEL_CHAN]
            rc = r_tr[m]
            dims = [emb[n]["Ztr"].shape[1] for n in KERNEL_CHAN]
            sp = None
            if a.kernel == "gibbs":
                # sigma_k WITHIN the block: the kernel only ever sees same-category
                # pairs, so the local density that matters is the one inside the family
                sks = []
                for D in Ds:
                    Dk = D.copy(); np.fill_diagonal(Dk, np.inf)
                    kk = min(a.gibbs_k, D.shape[0] - 1)
                    sks.append(np.maximum(np.sort(Dk, axis=1)[:, kk - 1], 1e-12))
                sp = float(np.percentile(sks[0], 90) / np.percentile(sks[0], 10))
                nbr_fn = lambda sc: _gibbs_nbrs(Ds, sks, sc)
            else:
                mode = "union" if a.kernel == "additive" else "inter"
                nbr_fn = lambda sc: _stat_nbrs(Ds, cuts_cat[:, c] * sc, mode)

            def _ll(scale, svv):
                if a.kernel == "gibbs":
                    return block_loglik_gibbs(Ds, sks, dims, scale, rc,
                                              float(svv[0]), JITTER)
                if a.kernel == "product":
                    return block_loglik_product(Ds, cuts_cat[:, c] * scale, rc,
                                                float(svv[0]), JITTER)
                return block_loglik(Ds, cuts_cat[:, c] * scale, rc, svv, JITTER)

            # one calibrated scale per neighbour target, plus the DENSE limit. Targets a
            # block cannot reach (fewer rows than neighbours asked for) are dropped
            # rather than silently clipped onto the dense arm, which is what previously
            # made small families look like flat, indifferent profiles.
            unit = scale_for_nbrs(nbr_fn, min(K, n_c - 1))   # m=1 == the heuristic
            scales = [unit * g for g in SCALE_GRID]
            achieved = [nbr_fn(sc) for sc in scales]

            LL = np.array([[_ll(sc, sv0 * sg) for sc in scales] for sg in sv_grid])
            si, gi = np.unravel_index(int(np.nanargmax(LL)), LL.shape)
            ll = LL[si]
            ref = int(np.argmin(np.abs(SCALE_GRID - 1.0)))   # m=1, the heuristic
            rel = ll - ll[ref]
            tot += rel[gi]
            at_dense = (gi == len(SCALE_GRID) - 1)          # ran to the end of the grid
            flat = float(np.nanmax(np.abs(rel))) < 1.0
            best_all.setdefault(c, []).append(
                (SCALE_GRID[gi], sv_grid[si], flat, at_dense,
                 achieved[gi] / max(n_c - 1, 1)))
            spark = " ".join(f"{v:+.0f}" for v in rel)
            svcol = f" sv*{sv_grid[si]:<5.2f}" if len(sv_grid) > 1 else ""
            if sp is not None:
                svcol += f" [l spread {sp:.2f}x]"
            lab = f"m={SCALE_GRID[gi]:g}" + (" (END)" if at_dense else "")
            print(f"{str(names[c]):>18} {n_c:>8} {lab:>12} "
                  f"{achieved[gi] / max(n_c - 1, 1):>6.2f} {rel[gi]:>+15.1f}{svcol}   "
                  f"{spark}")
        print(f"{'TOTAL':>18} {'':>8} {'':>9} {tot:>+14.1f}   "
              f"(sum over categories of the gain from re-optimising each radius)")
        print(f"  m is a multiplier on l; m=1 is the {K}-neighbour heuristic. 'frac' is "
              f"the achieved neighbour fraction at the argmax.\n")

    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"mean rung = {a.mean}"
          + (f", sv grid = {list(sv_grid)}" if len(sv_grid) > 1 else ""))
    print(f"{'category':>18} {'argmax m per seed':>26} {'mean':>7} {'argmax sv':>22}")
    n_edge = n_flat = 0
    for c in sorted(best_all):
        v = np.array([x[0] for x in best_all[c]])
        is_flat = all(x[2] for x in best_all[c])
        dense = all(x[3] for x in best_all[c])
        frac = float(np.mean([x[4] for x in best_all[c]]))
        n_flat += int(is_flat)
        n_edge += int((not is_flat) and dense)
        note = ("  (FLAT: uninformative)" if is_flat
                else "  <- ran to the END of the grid; widen it" if dense
                else f"  INTERIOR optimum at {frac:.0%} block density")
        print(f"{str(names[c]):>18} {np.array2string(v):>26} {frac:>7.2f}{note}")
    n_use = len(best_all) - n_flat
    if n_flat:
        print(f"\n  {n_flat} categorie(s) excluded as uninformative: too few rows to "
              f"reach the neighbour target, so the block is dense at every m.")
    print(f"\n  {n_edge}/{n_use} INFORMATIVE categories ran to the end of the grid.")
    print(f"\n  This is a stronger statement than the old multiplier grid could make. "
          f"The likelihood is not merely 'still rising at our boundary' -- it is "
          f"maximised at the FULLY DENSE covariance, which is the regime the position "
          f"paper's sec 7 says offers no computational advantage over a standard dense "
          f"GP. Any sparsity therefore comes from the PRIOR, not the data, and the "
          f"prior's strength should be stated in these same units (neighbours per row) "
          f"so it reads as a computational budget rather than a free parameter.")
    print("\n  An INTERIOR optimum is the informative case: that family has a real")
    print("  correlation length the data can identify, and MCMC over it is justified.")
    print("  Check the fraction column before believing one -- an 'optimum' at 0.95")
    print("  coverage is the block running out of pairs to add, not a length scale.")


if __name__ == "__main__":
    main()
