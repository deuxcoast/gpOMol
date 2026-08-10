"""Regression tests for GibbsWendlandKernel. Run directly: python scripts/test_gibbs_kernel.py

THE TEST THAT MATTERS IS THE FIRST ONE. A non-stationary kernel that does not reduce
EXACTLY to the stationary one when the length scale is constant is not a generalisation
of it, and every comparison against the existing arms would be measuring the difference
between two implementations rather than the effect of non-stationarity. The reduction is
exact and checkable: with l constant the Gibbs prefactor is identically 1 and the
effective support is l*sqrt(2), so the kernel must equal ProductWendlandKernel at
cutoff = l*sqrt(2) to floating-point precision.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wl_gp2scale.kernel import (ChannelSpec, GibbsWendlandKernel,  # noqa: E402
                                local_scale_k, with_gibbs_tags)
from product_kernel import ProductWendlandKernel                    # noqa: E402

RNG = np.random.default_rng(0)
NCAT, N, D1, D2 = 3, 400, 10, 10


def _data(n=N):
    Z1 = RNG.normal(size=(n, D1))
    Z2 = RNG.normal(size=(n, D2)) * 2.0
    cat = RNG.integers(0, NCAT, size=n)
    return Z1, Z2, cat


def _unit_scale(Z):
    """A constant local scale that puts the support at the MEDIAN pairwise distance.

    Not cosmetic. In 10 dimensions standard-normal points sit ~4.5 apart, so a naive
    l=1 gives a support of sqrt(2) and the Gram is the identity -- every density test
    then reads 1/n (the diagonal) at every c and passes without exercising anything.
    That is exactly how the first version of these tests passed vacuously."""
    from scipy.spatial.distance import pdist
    return float(np.median(pdist(Z))) / np.sqrt(2.0)


def test_reduces_to_stationary():
    """l constant -> prefactor == 1 and support == l*sqrt(2) -> the product kernel."""
    Z1, Z2, cat = _data()
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    ell = np.array([0.9, 2.3])                      # one constant l per channel
    sv = 3.7

    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    X = with_gibbs_tags([Z1, Z2], [np.full(len(Z1), ell[0]), np.full(len(Z1), ell[1])],
                        cat)
    hps = np.concatenate([[sv], np.ones(2 * NCAT)])   # c == 1 everywhere
    Kg = g(X, X, hps)

    specs = [ChannelSpec(0, D1, float(ell[0] * np.sqrt(2.0))),
             ChannelSpec(D1, D1 + D2, float(ell[1] * np.sqrt(2.0)))]
    p = ProductWendlandKernel(specs, device="cpu")
    Xp = np.hstack([Z1, Z2, cat.astype(float)[:, None]])
    Kp = p(Xp, Xp, [sv])

    err = np.abs(Kg - Kp).max()
    assert err < 1e-12, "reduction FAILED, max |Gibbs - product| = %.3e" % err
    print("  reduces to ProductWendlandKernel at l*sqrt(2): max diff %.2e" % err)


def test_psd_with_varying_ell():
    Z1, Z2, cat = _data(600)
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    sk1 = local_scale_k(Z1, Z1, cat, cat, k=10)
    sk2 = local_scale_k(Z2, Z2, cat, cat, k=10)
    spread = np.percentile(sk1, 90) / np.percentile(sk1, 10)
    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    X = with_gibbs_tags([Z1, Z2], [sk1, sk2], cat)
    hps = np.concatenate([[1.0], np.full(2 * NCAT, 4.0)])
    K = g(X, X, hps)
    dens = float((K != 0).mean())
    ev = np.linalg.eigvalsh(K)
    rel = ev[0] / ev[-1]
    assert dens > 0.02, "Gram too sparse (%.4f) to be a meaningful PSD test" % dens
    assert rel > -1e-8, "NOT PSD: relative min eigenvalue %.3e" % rel
    print("  PSD with l varying %.2fx at density %.3f: rel min eig %.2e"
          % (spread, dens, rel))


def test_category_mask():
    Z1, Z2, cat = _data(200)
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    u1, u2 = _unit_scale(Z1), _unit_scale(Z2)
    X = with_gibbs_tags([Z1, Z2], [np.full(200, u1), np.full(200, u2)], cat)
    K = g(X, X, np.concatenate([[1.0], np.full(2 * NCAT, 2.0)]))
    cross = K[cat[:, None] != cat[None, :]]
    assert np.all(cross == 0.0), "cross-category entries are not exactly zero"
    print("  cross-category block is exactly zero (%d entries)" % cross.size)


def test_c_scales_support():
    """Doubling c must double the effective support, hence never shrink the Gram."""
    Z1, Z2, cat = _data(300)
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    u1, u2 = _unit_scale(Z1), _unit_scale(Z2)
    X = with_gibbs_tags([Z1, Z2], [np.full(300, u1), np.full(300, u2)], cat)
    nz = []
    for c in (0.5, 1.0, 2.0):
        K = g(X, X, np.concatenate([[1.0], np.full(2 * NCAT, c)]))
        nz.append(float((K != 0).mean()))
    assert nz[0] < nz[1] < nz[2], "density is not STRICTLY monotone in c: %s" % nz
    assert nz[1] > 1.5 / 300, "Gram is essentially diagonal -- the test is vacuous"
    print("  density monotone in c: %s" % np.round(nz, 4).tolist())


def test_per_category_c():
    """A category given a larger c must get a denser block, and only that block."""
    Z1, Z2, cat = _data(300)
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    u1, u2 = _unit_scale(Z1), _unit_scale(Z2)
    X = with_gibbs_tags([Z1, Z2], [np.full(300, u1), np.full(300, u2)], cat)
    base = np.full((2, NCAT), 1.0)
    K0 = g(X, X, np.concatenate([[1.0], base.ravel()]))
    bumped = base.copy(); bumped[:, 1] = 2.0
    K1 = g(X, X, np.concatenate([[1.0], bumped.ravel()]))
    for c in range(NCAT):
        m = cat == c
        d0 = float((K0[np.ix_(m, m)] != 0).mean())
        d1 = float((K1[np.ix_(m, m)] != 0).mean())
        if c == 1:
            assert d1 > d0, "bumped category did not densify (%.4f -> %.4f)" % (d0, d1)
        else:
            assert d1 == d0, "category %d changed when only category 1 was bumped" % c
    print("  per-category c touches only its own block")


def test_hps_length_is_checked():
    chans = [ChannelSpec(0, D1, 0.0), ChannelSpec(D1, D1 + D2, 0.0)]
    g = GibbsWendlandKernel(channels=chans, n_cat=NCAT, device="cpu")
    assert g.n_hps() == 1 + 2 * NCAT
    try:
        g.unpack(np.ones(3))
    except ValueError:
        print("  wrong-length hps raises (n_hps = %d)" % g.n_hps())
        return
    raise AssertionError("a wrong-length hps vector was silently accepted")


if __name__ == "__main__":
    for fn in (test_reduces_to_stationary, test_psd_with_varying_ell,
               test_category_mask, test_c_scales_support, test_per_category_c,
               test_hps_length_is_checked):
        print(fn.__name__)
        fn()
    print("\nall passed")
