"""
variogram_screen.py
====================

A screening harness for molecular distance metrics and kernels, built around
the empirical semivariogram as the primary diagnostic.

Core idea
---------
One cached object -- a pairwise distance matrix D on a fixed subsample, plus a
target vector z -- yields all three quantities you care about:

  1. PREDICTIVE VALUE  : structured fraction (sill - nugget) / sill, from the
                         shape of the variogram. ~0 means d is blind to the
                         target (the flat-variogram / kNN-skill-zero failure).
  2. SPARSITY          : the variogram range IS the natural Wendland support
                         radius; sparsity ratio = P(d > range).
  3. COMPUTATIONAL COST: wall-clock to fill D (intrinsic metric cost), plus the
                         predicted downstream GP cost via the sparsity ratio.

Two tiers
---------
  TIER 1 (cheap gate) : variogram + sparsity + timing.  Run on every candidate.
  TIER 2 (expensive)  : GP leave-one-out with RMSE *and* CRPS.  Run only on
                        candidates that clear Tier 1.  CRPS is the real point --
                        calibrated uncertainty, not point accuracy.

Candidate contract
-------------------
A Candidate produces, from a list of molecule feature objects, EITHER a pairwise
distance matrix (kind="distance") OR a pairwise kernel/Gram matrix (kind="kernel").
Kernels are converted to the induced Hilbert distance so everything flows through
the identical variogram path. The SAME callable can be wrapped into a
gpCAM-compatible kernel(x1, x2, hps) for Tier 2 and production (see
`as_gpcam_kernel`), so no distance code is reimplemented downstream.

Run `python variogram_screen.py` for a self-contained demo on synthetic data
(informative metric vs. uninformative metric vs. random-distance negative
control). Then replace `load_subsample` with your OMol25 loader and register
your real candidates.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

# Optional deps -- harness degrades gracefully without them.
try:
    from scipy.optimize import curve_fit

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


CACHE_DIR = Path("./_variogram_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- #
# Candidate definition
# ----------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """A distance metric or kernel to screen.

    fn maps a sequence of molecule feature objects -> (N, N) matrix.
      - kind="distance": fn returns a pairwise distance matrix (>= 0, zero diag).
      - kind="kernel"  : fn returns a PSD Gram matrix; the harness converts it to
                         the induced Hilbert distance d^2 = kii + kjj - 2kij.
    normalize_kernel: cosine-normalize a kernel before inducing a distance.
        True  -> distance reflects *shape* similarity (size removed).
        False -> distance reflects shape + extensive size. For extensive targets
                 like total energy this re-introduces the size confound, so the
                 honest comparison is normalize=True PLUS the size-controlled
                 variogram (see run_screen).
    """

    name: str
    fn: Callable[[Sequence[Any]], np.ndarray]
    kind: Literal["distance", "kernel"] = "distance"
    normalize_kernel: bool = True

    def pairwise_distance(self, feats: Sequence[Any]) -> tuple[np.ndarray, float]:
        """Return (distance_matrix, seconds_to_compute)."""
        t0 = time.perf_counter()
        M = np.asarray(self.fn(feats), dtype=float)
        if self.kind == "kernel":
            M = kernel_to_distance(M, normalize=self.normalize_kernel)
        dt = time.perf_counter() - t0
        M = 0.5 * (M + M.T)  # enforce symmetry
        np.fill_diagonal(M, 0.0)
        return M, dt


def kernel_to_distance(K: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Induced Hilbert distance from a PSD kernel: d^2 = kii + kjj - 2kij.

    This distance is guaranteed Euclidean (negative type) because it comes from
    a PSD kernel -- which is precisely the property the raw Wasserstein metric
    lacks. Running it through Wendland-on-Euclidean restores the PD guarantee.
    """
    K = np.asarray(K, dtype=float)
    if normalize:
        d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
        K = K / np.outer(d, d)
    diag = np.diag(K)
    d2 = diag[:, None] + diag[None, :] - 2.0 * K
    return np.sqrt(np.clip(d2, 0.0, None))


# ----------------------------------------------------------------------------- #
# Empirical variogram
# ----------------------------------------------------------------------------- #
@dataclass
class Variogram:
    h: np.ndarray  # bin-center distances
    gamma: np.ndarray  # semivariance per bin
    counts: np.ndarray  # pairs per bin
    nugget: float
    sill: float
    vrange: float  # effective range = natural Wendland support radius
    structured_fraction: float
    target_var: float  # Var(z): the theoretical sill, for reference
    label: str = ""


def empirical_variogram(
    D: np.ndarray,
    z: np.ndarray,
    n_bins: int = 25,
    estimator: Literal["matheron", "cressie"] = "matheron",
    bin_mode: Literal["quantile", "equal"] = "quantile",
    pair_mask: np.ndarray | None = None,
    label: str = "",
) -> Variogram:
    """Compute the empirical semivariogram of target z under distances D.

    pair_mask: optional (N, N) boolean array; only True pairs are used. Use this
        for size-control -- e.g. mask to pairs in the same atom-count band so the
        variogram cannot earn structure from the size -> extensive-energy trend.
    """
    N = len(z)
    iu, ju = np.triu_indices(N, k=1)
    d = D[iu, ju]
    dz2 = (z[iu] - z[ju]) ** 2
    if pair_mask is not None:
        keep = pair_mask[iu, ju]
        d, dz2 = d[keep], dz2[keep]

    finite = np.isfinite(d) & np.isfinite(dz2)
    d, dz2 = d[finite], dz2[finite]
    if d.size < n_bins * 5:
        raise ValueError(f"Too few usable pairs ({d.size}) for {n_bins} bins.")

    if bin_mode == "quantile":
        edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(d.min(), d.max(), n_bins + 1)
    idx = np.clip(np.digitize(d, edges[1:-1]), 0, len(edges) - 2)

    nb = len(edges) - 1
    h = np.zeros(nb)
    gamma = np.zeros(nb)
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        sel = idx == b
        c = int(sel.sum())
        counts[b] = c
        if c == 0:
            h[b] = 0.5 * (edges[b] + edges[b + 1])
            continue
        h[b] = d[sel].mean()
        diffs = dz2[sel]
        if estimator == "matheron":
            gamma[b] = 0.5 * diffs.mean()
        else:  # Cressie-Hawkins robust estimator (outlier-resistant)
            m = (np.abs(diffs) ** 0.25).mean()
            gamma[b] = 0.5 * (m**4) / (0.457 + 0.494 / c)

    valid = counts > 0
    h, gamma, counts = h[valid], gamma[valid], counts[valid]

    nugget, sill, vrange = _fit_variogram_model(h, gamma, z)
    sf = float(np.clip((sill - nugget) / sill, 0.0, 1.0)) if sill > 0 else 0.0
    return Variogram(
        h, gamma, counts, nugget, sill, vrange, sf, float(np.var(z)), label
    )


def _fit_variogram_model(
    h, gamma, z, range_frac: float = 0.95, model: str = "empirical", smooth: bool = True
):
    """Estimate (nugget, sill, range) directly from the empirical variogram, so
    the numbers match what you read off the plot.

      nugget : gamma at the smallest-distance bin (the irreducible floor -- e.g.
               the conformational variance a connectivity-only descriptor cannot
               see). NOT pinned to zero.
      sill   : plateau height, median of the upper-distance third of bins (robust
               to the variogram overshooting on the way up).
      range  : the FIRST distance at which the variogram reaches `range_frac` of
               the climb from nugget to sill -- i.e. where it visually hits the
               plateau. This is the natural Wendland support radius.

    model="exponential" instead fits gamma = n + (s-n)(1-exp(-h/L)) and reports
    the classic 3L "practical range"; useful only when the variogram approaches
    its asymptote slowly. A sharply-plateauing empirical variogram is badly
    overestimated by the 3L rule, which is why "empirical" is the default.
    """
    g = np.asarray(gamma, dtype=float)
    nugget = float(g[0])  # empirical floor
    sill = float(np.median(g[max(1, len(g) * 2 // 3) :]))  # empirical plateau
    sill = max(sill, nugget + 1e-12)

    if model == "exponential" and _HAVE_SCIPY and len(h) >= 4:

        def _m(hh, n, s, L):
            return n + (s - n) * (1.0 - np.exp(-hh / L))

        try:
            p0 = [nugget, sill, np.median(h)]
            bounds = (
                [0, 0, 1e-9],
                [sill * 2 + 1e-9, np.var(z) * 4 + 1e-9, h.max() * 5],
            )
            popt, _ = curve_fit(_m, h, g, p0=p0, bounds=bounds, maxfev=10000)
            nugget, sill = float(popt[0]), float(popt[1])
            return nugget, sill, min(3.0 * float(popt[2]), float(h.max()))
        except Exception:
            pass

    # Empirical first-crossing range. Smooth (3-pt) only for crossing detection
    # so a single noisy bin near the threshold doesn't trip the range early.
    gg = (
        np.convolve(g, np.ones(3) / 3.0, mode="same") if (smooth and len(g) >= 3) else g
    )
    thresh = nugget + range_frac * (sill - nugget)
    above = np.where(gg >= thresh)[0]
    vrange = float(h[above[0]]) if above.size else float(h.max())
    return nugget, sill, vrange


# ----------------------------------------------------------------------------- #
# Sparsity from the same cached D
# ----------------------------------------------------------------------------- #
def sparsity_ratio(D: np.ndarray, support_radius: float) -> float:
    """Fraction of off-diagonal pairs a Wendland kernel of this radius zeroes."""
    N = D.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    d = D[iu, ju]
    d = d[np.isfinite(d)]
    return float(np.mean(d > support_radius))


# ----------------------------------------------------------------------------- #
# Caching: compute D once per (candidate, subsample), reuse for every readout
# ----------------------------------------------------------------------------- #
def _subsample_hash(feats: Sequence[Any]) -> str:
    try:
        blob = json.dumps([repr(f) for f in feats], sort_keys=True).encode()
    except Exception:
        blob = repr(feats).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def cached_distance(
    cand: Candidate, feats: Sequence[Any], use_cache: bool = True
) -> tuple[np.ndarray, float]:
    key = f"{cand.name}_{_subsample_hash(feats)}"
    npz = CACHE_DIR / f"{key}.npz"
    if use_cache and npz.exists():
        z = np.load(npz)
        return z["D"], float(z["seconds"])
    D, secs = cand.pairwise_distance(feats)
    if use_cache:
        np.savez(npz, D=D, seconds=secs)
    return D, secs


# ----------------------------------------------------------------------------- #
# Tier 1 screen
# ----------------------------------------------------------------------------- #
@dataclass
class ScreenResult:
    name: str
    structured_fraction: float  # predictive value (total-energy target)
    structured_fraction_sizectl: float  # predictive value, size-controlled
    vrange: float
    sparsity_ratio: float  # at the variogram range
    seconds_to_build_D: float
    n: int
    notes: str = ""
    _vg: Variogram | None = field(default=None, repr=False)
    _vg_sizectl: Variogram | None = field(default=None, repr=False)


def same_size_band_mask(sizes: np.ndarray, band: int = 1) -> np.ndarray:
    """True for pairs whose atom counts differ by <= band. Removes the size ->
    extensive-energy confound from the variogram."""
    diff = np.abs(sizes[:, None] - sizes[None, :])
    return diff <= band


def run_screen(
    candidates: list[Candidate],
    feats: Sequence[Any],
    z: np.ndarray,
    sizes: np.ndarray | None = None,
    size_band: int = 1,
    n_bins: int = 25,
    estimator: Literal["matheron", "cressie"] = "matheron",
    use_cache: bool = True,
    plot_dir: str | None = None,
) -> list[ScreenResult]:
    """Tier-1 screen. Same fixed subsample for every candidate -> apples-to-apples.
    Each variogram is computed twice: on the referenced target, and size-controlled.
    """
    z = np.asarray(z, dtype=float)
    results: list[ScreenResult] = []
    mask = (
        same_size_band_mask(np.asarray(sizes), size_band) if sizes is not None else None
    )

    for cand in candidates:
        D, secs = cached_distance(cand, feats, use_cache=use_cache)
        vg = empirical_variogram(
            D, z, n_bins=n_bins, estimator=estimator, label=f"{cand.name} (total)"
        )
        spars = sparsity_ratio(D, vg.vrange)

        if mask is not None:
            try:
                vg_sc = empirical_variogram(
                    D,
                    z,
                    n_bins=n_bins,
                    estimator=estimator,
                    pair_mask=mask,
                    label=f"{cand.name} (size-ctl)",
                )
                sf_sc = vg_sc.structured_fraction
            except ValueError:
                vg_sc, sf_sc = None, float("nan")  # too few same-size pairs
        else:
            vg_sc, sf_sc = None, float("nan")

        note = ""
        if vg.structured_fraction > 0.15 and (np.isnan(sf_sc) or sf_sc < 0.05):
            note = "STRUCTURE IS SIZE-ARTIFACT: collapses under size control"
        elif vg.structured_fraction < 0.05:
            note = "FLAT variogram: metric blind to target (fails gate)"
        elif spars < 0.5:
            note = "range too large: dense matrix, no gp2Scale advantage"

        results.append(
            ScreenResult(
                name=cand.name,
                structured_fraction=vg.structured_fraction,
                structured_fraction_sizectl=sf_sc,
                vrange=vg.vrange,
                sparsity_ratio=spars,
                seconds_to_build_D=secs,
                n=len(z),
                notes=note,
                _vg=vg,
                _vg_sizectl=vg_sc,
            )
        )

        if plot_dir and _HAVE_MPL:
            _plot_candidate(cand.name, D, vg, vg_sc, plot_dir)

    return results


def print_table(results: list[ScreenResult]) -> None:
    cols = [
        "candidate",
        "struct_frac",
        "struct_frac_sizectl",
        "range",
        "sparsity",
        "build_s",
    ]
    w = [max(len(cols[0]), max(len(r.name) for r in results)), 11, 19, 8, 9, 9]
    header = "  ".join(c.ljust(wi) for c, wi in zip(cols, w))
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x.structured_fraction):
        sc = (
            "  n/a"
            if np.isnan(r.structured_fraction_sizectl)
            else f"{r.structured_fraction_sizectl:.3f}"
        )
        row = "  ".join(
            [
                r.name.ljust(w[0]),
                f"{r.structured_fraction:.3f}".ljust(w[1]),
                sc.ljust(w[2]),
                f"{r.vrange:.3g}".ljust(w[3]),
                f"{r.sparsity_ratio:.3f}".ljust(w[4]),
                f"{r.seconds_to_build_D:.2f}".ljust(w[5]),
            ]
        )
        print(row)
        if r.notes:
            print(f"    -> {r.notes}")


def _plot_candidate(name, D, vg, vg_sc, plot_dir):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(vg.h, vg.gamma, "o-", label="total energy")
    if vg_sc is not None:
        ax1.plot(vg_sc.h, vg_sc.gamma, "s--", label="size-controlled")
    ax1.axhline(vg.target_var, color="gray", ls=":", label="Var(z) [sill]")
    ax1.axvline(vg.vrange, color="red", ls=":", label=f"range={vg.vrange:.2g}")
    ax1.set_xlabel("distance h")
    ax1.set_ylabel(r"$\gamma(h)$")
    ax1.set_title(f"{name}: variogram")
    ax1.legend(fontsize=8)

    iu, ju = np.triu_indices(D.shape[0], k=1)
    dd = D[iu, ju]
    dd = dd[np.isfinite(dd)]
    ax2.hist(dd, bins=40, color="steelblue", alpha=0.8)
    ax2.axvline(
        vg.vrange,
        color="red",
        ls=":",
        label=f"range (sparsity={np.mean(dd>vg.vrange):.2f})",
    )
    ax2.set_xlabel("pairwise distance")
    ax2.set_ylabel("count")
    ax2.set_title(f"{name}: distance distribution")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / f"{name}.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Tier 2 hook: gpCAM adapter + leave-one-out with RMSE and CRPS
# ----------------------------------------------------------------------------- #
def as_gpcam_kernel(cand: Candidate, wendland: bool = True):
    """Wrap a candidate's distance into a gpCAM kernel(x1, x2, hps) -> (N1,N2).
    x_data is a list of molecule feature objects (gpCAM allows arbitrary objects).
    This is the SAME distance code as Tier 1 -- production reuses it verbatim.

    hps layout:  hps[0] = signal variance,  hps[1] = support radius / length scale
    """

    def kernel(x1, x2, hps):
        # Build the cross-distance block by computing distance on the union.
        feats = list(x1) + list(x2)
        D, _ = cand.pairwise_distance(feats)
        n1 = len(x1)
        Dx = D[:n1, n1:]
        r = hps[1]
        if wendland:
            t = np.clip(Dx / r, 0.0, 1.0)
            base = ((1 - t) ** 8) * (35 * t**3 + 25 * t**2 + 8 * t + 1)
            base = np.where(Dx < r, base, 0.0)
        else:
            base = np.exp(-Dx / r)
        return hps[0] * base

    kernel.__name__ = f"kernel_{cand.name}"
    return kernel


def gp_loo_crps(cand: Candidate, feats, z, hps=(1.0, None), noise=1e-6, wendland=True):
    """Exact leave-one-out RMSE and CRPS for a Gaussian predictive distribution,
    computed from a single Cholesky via the standard LOO identities -- no GP
    refit per point. This is the Tier-2 test: CRPS is the calibrated-uncertainty
    number that is the actual thesis, not RMSE.

    Returns dict(rmse=..., crps=..., mean_pred_std=...).
    """
    z = np.asarray(z, dtype=float)
    N = len(z)
    kfn = as_gpcam_kernel(cand, wendland=wendland)
    sig, r = hps
    if r is None:  # default support radius = median pairwise distance
        D, _ = cand.pairwise_distance(feats)
        iu, ju = np.triu_indices(N, k=1)
        r = float(np.median(D[iu, ju]))
    K = kfn(feats, feats, np.array([sig, r])) + noise * np.eye(N)

    # LOO via inverse (fine at screening subsample sizes).
    Kinv = np.linalg.inv(K)
    alpha = Kinv @ z
    mu_loo = z - alpha / np.diag(Kinv)
    var_loo = 1.0 / np.diag(Kinv)
    var_loo = np.clip(var_loo, 1e-12, None)
    std_loo = np.sqrt(var_loo)

    err = z - mu_loo
    rmse = float(np.sqrt(np.mean(err**2)))
    crps = float(np.mean(_crps_gaussian(err, std_loo)))
    return dict(
        rmse=rmse, crps=crps, mean_pred_std=float(std_loo.mean()), support_radius=r
    )


def gram_loo_crps(
    K: np.ndarray, z: np.ndarray, noise_frac: float = 1e-3, autoscale: bool = True
):
    """Leave-one-out RMSE + CRPS for a GP whose covariance IS the given Gram
    matrix K used directly -- i.e. a dot-product / feature kernel, with NO
    distance step and NO compact support.

    This is the complement to the variogram gate. The variogram asks "is the
    target smooth in this DISTANCE?" (the question gp2Scale's stationary Wendland
    kernel cares about). This asks "is the target linear in these FEATURES?" A
    descriptor can fail the first and ace the second -- meaning it is informative
    but NOT in a gp2Scale-compatible (compact-support, distance-based) way.
    """
    K = np.asarray(K, dtype=float).copy()
    z = np.asarray(z, dtype=float)
    N = len(z)
    if autoscale:  # put K and noise on the scale of the target
        md = np.mean(np.diag(K))
        if md > 0:
            K *= np.var(z) / md
    A = K + noise_frac * np.var(z) * np.eye(N)
    Kinv = np.linalg.inv(A)
    alpha = Kinv @ z
    mu = z - alpha / np.diag(Kinv)
    var = np.clip(1.0 / np.diag(Kinv), 1e-12, None)
    err = z - mu
    rmse = float(np.sqrt(np.mean(err**2)))
    crps = float(np.mean(_crps_gaussian(err, np.sqrt(var))))
    return dict(rmse=rmse, crps=crps, mean_pred_std=float(np.sqrt(var).mean()))


def wendland_from_distance(D: np.ndarray, r: float, signal_var: float = 1.0):
    """Compactly-supported Wendland C2 kernel (gp2Scale Eq. 3) evaluated on a
    precomputed distance matrix D at support radius r. Exactly zero for d >= r,
    so the matrix is genuinely sparse; the diagonal is forced to signal_var."""
    D = np.asarray(D, dtype=float)
    t = np.clip(D / r, 0.0, 1.0)
    K = (1.0 - t) ** 8 * (35.0 * t**3 + 25.0 * t**2 + 8.0 * t + 1.0)
    K = np.where(D < r, K, 0.0)
    np.fill_diagonal(K, 1.0)
    return signal_var * K


def wendland_loo_crps(
    D: np.ndarray,
    r: float,
    z: np.ndarray,
    noise_frac: float = 1e-3,
    center: bool = True,
):
    """LOO RMSE / CRPS / coverage for a GP with a compactly-supported Wendland
    kernel of support radius r on distance matrix D, plus the matrix DENSITY at
    this radius (off-diagonal nonzero fraction -- the gp2Scale cost driver).

    Sweep r over a grid to trace the signal-vs-sparsity tradeoff directly:
    small r -> sparse but the kernel goes near-diagonal and predictions revert to
    the mean (CRPS rises); large r -> dense but best CRPS. The knee is the
    gp2Scale operating point. Replaces reading a single auto-range off the table.
    """
    D = np.asarray(D, dtype=float)
    z = np.asarray(z, dtype=float)
    N = len(z)
    sig2 = float(np.var(z))
    iu, ju = np.triu_indices(N, k=1)
    density = float(np.mean(D[iu, ju] < r))  # fraction of off-diag pairs in support
    K = wendland_from_distance(D, r, signal_var=sig2)
    A = K + noise_frac * sig2 * np.eye(N)
    try:
        Kinv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return dict(r=float(r), density=density, sparsity=1.0 - density, ok=False)
    zc = z - z.mean() if center else z
    dinv = np.diag(Kinv)
    err = (Kinv @ zc) / dinv  # LOO residual (constant-mean cancels)
    sd = np.sqrt(np.clip(1.0 / dinv, 1e-12, None))
    return dict(
        r=float(r),
        density=density,
        sparsity=1.0 - density,
        ok=True,
        rmse=float(np.sqrt(np.mean(err**2))),
        crps=float(np.mean(_crps_gaussian(err, sd))),
        coverage=float(np.mean(np.abs(err) <= 2 * sd)),
        mean_std=float(sd.mean()),
    )


def summarize_support_sweep(rows, tol: float = 0.10, min_gap_frac: float = 0.15):
    """Turn a support-radius sweep into an honest operating-point summary.

    The old rule ("sparsest r within 10% of the DENSE CRPS") is doubly wrong when
    the curve is U-shaped: the dense point is the overfit corner (worst
    calibration), and "within 10%" of it is satisfied almost everywhere, so the
    rule walks out to the degenerate sparsest point. Instead:

      * optimum  = argmin CRPS over the sweep (the real operating point).
      * reach    = sparsest radius whose CRPS is within `tol` of the OPTIMUM
                   (how far you can sparsify before leaving the basin).
      * meaningful = is the dense->optimum improvement bigger than min_gap_frac?
                   If not, the curve is too flat to claim a real optimum.
    """
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return None
    optimum = min(ok, key=lambda r: r["crps"])
    dense = max(ok, key=lambda r: r["density"])
    gap = (
        (dense["crps"] - optimum["crps"]) / dense["crps"] if dense["crps"] > 0 else 0.0
    )
    thresh = optimum["crps"] * (1.0 + tol)
    reach = min((r for r in ok if r["crps"] <= thresh), key=lambda r: r["density"])
    return dict(
        optimum=optimum,
        dense=dense,
        reach=reach,
        dense_to_opt_gap=gap,
        meaningful=gap >= min_gap_frac,
        tol=tol,
    )


def print_sweep_summary(s):
    if s is None:
        print("  (no valid sweep rows)")
        return
    o, d, r = s["optimum"], s["dense"], s["reach"]
    print(
        f"  OPTIMUM     : r={o['r']:.3f}  density={o['density']:.4f} "
        f"(sparsity {o['sparsity']:.4f})  CRPS={o['crps']:.3f}  cov={o['coverage']:.2f}"
    )
    print(
        f"  dense GP    : density={d['density']:.4f}  CRPS={d['crps']:.3f}  "
        f"cov={d['coverage']:.2f}   (dense->optimum CRPS gain {100*s['dense_to_opt_gap']:.1f}%)"
    )
    print(
        f"  reach (<= {int(s['tol']*100)}% above optimum): "
        f"density={r['density']:.4f} (sparsity {r['sparsity']:.4f})  CRPS={r['crps']:.3f}"
    )
    if not s["meaningful"]:
        print(
            "  NOTE: curve is nearly flat (small dense->optimum gain) -- the signal is"
        )
        print(
            "        shallow and the 'optimum' is weakly defined; treat with caution."
        )


def _crps_gaussian(err, sigma):
    """Closed-form CRPS for a Gaussian forecast N(mu, sigma^2) with z = err+mu.
    CRPS = sigma * [ w(2*Phi(w)-1) + 2*phi(w) - 1/sqrt(pi) ], w = err/sigma."""
    from math import pi, sqrt

    w = err / sigma
    Phi = 0.5 * (1 + _erf(w / np.sqrt(2)))
    phi = np.exp(-0.5 * w**2) / np.sqrt(2 * pi)
    return sigma * (w * (2 * Phi - 1) + 2 * phi - 1.0 / sqrt(pi))


def _erf(x):
    # vectorized erf without scipy
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return np.sign(x) * y


# ----------------------------------------------------------------------------- #
# Demo: synthetic data proving the variogram separates good/bad/random metrics
# ----------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(0)
    N = 400

    # "Molecules" are points in a latent 3D chemistry space; target = smooth
    # function of latent coords + size term + noise. One metric sees the latent
    # coords (informative), one sees a scrambled view (uninformative), one is
    # pure noise (negative control).
    latent = rng.normal(size=(N, 3))
    sizes = rng.integers(5, 30, size=N)
    z = (
        np.sin(latent[:, 0])
        + 0.5 * latent[:, 1] ** 2
        + 0.1 * sizes
        + rng.normal(scale=0.15, size=N)
    )

    feats = [
        dict(
            latent=latent[i], scrambled=rng.permutation(latent[i]), size=sizes[i], idx=i
        )
        for i in range(N)
    ]

    def informative(fs):
        X = np.array([f["latent"] for f in fs])
        return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

    def uninformative(fs):
        # distance driven only by size -> will look good on total energy,
        # must collapse under size control.
        s = np.array([f["size"] for f in fs], dtype=float)
        return np.abs(s[:, None] - s[None, :])

    def random_metric(fs):
        M = rng.random((len(fs), len(fs)))
        return M + M.T

    cands = [
        Candidate("informative", informative, kind="distance"),
        Candidate("size_only", uninformative, kind="distance"),
        Candidate("random_control", random_metric, kind="distance"),
    ]

    print("=== TIER 1: variogram screen ===\n")
    results = run_screen(
        cands,
        feats,
        z,
        sizes=sizes,
        size_band=1,
        plot_dir="./_variogram_plots",
        use_cache=False,
    )
    print_table(results)

    print("\n=== TIER 2: GP-LOO (RMSE + CRPS) for candidates clearing the gate ===\n")
    for r in results:
        if r.structured_fraction > 0.1 and not (
            np.isnan(r.structured_fraction_sizectl)
            or r.structured_fraction_sizectl < 0.05
        ):
            cand = next(c for c in cands if c.name == r.name)
            m = gp_loo_crps(cand, feats, z, hps=(float(np.var(z)), None))
            print(
                f"{r.name:16s}  RMSE={m['rmse']:.4f}  CRPS={m['crps']:.4f}  "
                f"<std>={m['mean_pred_std']:.4f}  r={m['support_radius']:.3f}"
            )
        else:
            print(f"{r.name:16s}  (did not clear Tier-1 gate)")

    if _HAVE_MPL:
        print("\nPlots written to ./_variogram_plots/")


if __name__ == "__main__":
    _demo()
