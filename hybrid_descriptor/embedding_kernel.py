"""
embedding_kernel.py
===================

Step 3 of the pipeline: reduce the standardised hybrid features to a low
dimension D in [15, 25], then define the compactly-supported kernel that gpCAM's
gp2Scale mode will use.

Two things are load-bearing here, and both are about positive-definiteness.

1. Why reduce to D ~ 15-25 (not more).
   Compactly-supported RBFs are NOT PD in arbitrarily high dimension. By
   Schoenberg's theorem the only radial functions PD on R^d for *every* d are
   global scale-mixtures of Gaussians (strictly positive, never compactly
   supported). Every Wendland function is PD only up to a finite maximal
   dimension. So the embedding dimension is not a free knob — pushing D too high
   can, on its own, break PD-ness of the Wendland Gram matrix. D=15 is the safe
   default; go to 25 only if the empirical PSD check below passes.

2. How "Mahalanobis" is realised.
   PCA rotates the feature space to decorrelate the axes. An ARD (per-axis
   length-scale) Wendland on the PCA coordinates is therefore a *diagonal*
   Mahalanobis metric in PCA space == a full Mahalanobis metric in the original
   feature space. The per-axis length scales are learned (MCMC), so the metric
   adapts to which combined WL/geometry/charge directions matter for the energy
   residual. No separate covariance matrix to estimate or invert.

   (We do NOT whiten in PCA: whitening fixes the per-axis scale, but ARD learns
   it from data. Whitening + ARD would just be redundant.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ----------------------------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------------------------


@dataclass
class FeatureReducer:
    """
    PCA via SVD. Fit on a subsample for scale; apply to everything.

    At OMol25 scale swap the exact SVD here for sklearn.decomposition.IncrementalPCA
    (streams minibatches, bounded memory) or sklearn.utils.extmath.randomized_svd.
    The transform is identical; only the fit differs.
    """

    n_components: int = 15
    mean_: np.ndarray = field(default=None, repr=False)
    components_: np.ndarray = field(default=None, repr=False)  # (D, D_raw)
    explained_variance_ratio_: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> "FeatureReducer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # economy SVD; rows of Vt are principal axes
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]
        var = (S**2) / (len(X) - 1)
        self.explained_variance_ratio_ = (var / var.sum())[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def retained_variance(self) -> float:
        """Fraction of variance kept by the top-D components. If this is low
        (<~0.8) AND the residual nugget is high, the reduction destroyed signal:
        raise n_components (re-checking PSD) or revisit the channels."""
        return float(self.explained_variance_ratio_.sum())


# ----------------------------------------------------------------------------
# ARD Wendland-Mahalanobis kernel (gp2Scale-compatible, compact support)
# ----------------------------------------------------------------------------
#
# Hyperparameter layout (shared vector across kernel/mean/noise in gpCAM):
#   hps[0]      = signal variance                       bounds ~ [1e-3, 10*var(resid)]
#   hps[1:D+1]  = per-PCA-axis length scales (support)  bounds ~ [1e-2, 2*range_axis]
# Total: D + 1 kernel hyperparameters. If you add a noise_function, its hps come
# AFTER these; document and extend hp_bounds accordingly.
#
# The length scales double as the compact-support radii: two molecules farther
# apart than the (anisotropic) support get EXACTLY zero covariance, which is what
# creates the sparse matrix gp2Scale exploits. Keep the upper bound modest —
# length scales that are too large destroy sparsity and defeat the framework.
#
# ============================================================================
# THE DIMENSION TRAP (read before choosing n_components)
# ============================================================================
# A Wendland function psi_{d0,k} is positive-definite ONLY on R^{d0} and lower.
# The Wendland used in the gp2Scale paper (Eq. 3, the degree-8 / cubic-poly form)
# is a d0 = 3 construction — it was built for 3-D spatial (climate) data. Applying
# a d0 = 3 Wendland to a D = 15-25 embedding is NOT guaranteed PD, and a passing
# empirical check on a small subsample does NOT prove PD-ness (a finite sample can
# simply miss the offending configuration; at full scale the solve then fails).
#
# The rule is:  embedding dimension  D  <=  Wendland design dimension  d0.
#
# Two clean ways to satisfy it:
#   (A) NATIVE, fast, GPU/sparse-optimised:  gpcam's wendland_anisotropic (d0 = 3).
#       Safe ONLY if you reduce to D <= 3. Compresses hard — verify retained
#       variance and that kNN skill at D=3 still beats WL before trusting it.
#   (B) EXPLICIT, dimension-correct:  wendland_psi(r, d0=D, k) below, which is a
#       genuine psi_{d0,k} with d0 = D, hence PD on R^D by Wendland's theorem.
#       This is the route that lets you keep D = 15-25 with a PD guarantee.
#
# There is a real tension: gp2Scale's native Wendland "wants" D=3, your chemical
# descriptor "wants" D=15-25. Resolving it IS part of validation — compare kNN
# skill at D=3 (native) vs D=15-25 (explicit) to see what resolution compression
# to 3 actually costs. Do not assume; measure.
# ============================================================================


def wendland_psi(r: np.ndarray, d0: int, k: int = 2) -> np.ndarray:
    """
    Wendland radial function psi_{d0,k}, normalised so psi(0)=1, with compact
    support on r in [0, 1]. PD on R^{d'} for every d' <= d0.

    Smoothness: C^{2k}. k=2 (C^4) is a good analog of the Matern-5/2 the position
    paper argues for (twice mean-square differentiable PES). Closed forms below
    are the standard minimal-degree Wendland functions; validated against the
    textbook cases psi_{3,1}=(1-r)^4(4r+1), psi_{3,2}=(1-r)^6(35r^2+18r+3)/3,
    psi_{3,3}=(1-r)^8(32r^3+25r^2+8r+1)/15.

    r may be any shape; values >= 1 return 0 (compact support enforced by clip).
    """
    l = d0 // 2 + k + 1
    t = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    s = 1.0 - t
    if k == 0:
        psi = s**l
    elif k == 1:
        psi = s ** (l + 1) * ((l + 1) * t + 1.0)
    elif k == 2:
        psi = s ** (l + 2) * ((l**2 + 4 * l + 3) * t**2 + (3 * l + 6) * t + 3.0) / 3.0
    elif k == 3:
        psi = (
            s ** (l + 3)
            * (
                (l**3 + 9 * l**2 + 23 * l + 15) * t**3
                + (6 * l**2 + 36 * l + 45) * t**2
                + (15 * l + 45) * t
                + 15.0
            )
            / 15.0
        )
    else:
        raise ValueError("k must be in {0,1,2,3}")
    return psi


def _aniso_scaled_radius(x1: np.ndarray, x2: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Anisotropic radial distance sqrt(sum_d ((x1_d - x2_d)/ls_d)^2) via the
    Gram expansion (O(N1*N2) memory, no 3-D broadcast — safe for large blocks)."""
    a = x1 / ls
    b = x2 / ls
    sa = np.einsum("ij,ij->i", a, a)
    sb = np.einsum("ij,ij->i", b, b)
    r2 = sa[:, None] + sb[None, :] - 2.0 * (a @ b.T)
    np.maximum(r2, 0.0, out=r2)
    return np.sqrt(r2)


def make_wendland_mahalanobis(
    dim: int, k: int = 2, backend: str = "explicit"
) -> Callable:
    """
    ARD Wendland-Mahalanobis kernel_function(x1, x2, hps) for gpCAM.

    backend="explicit" (default, recommended for D>3): dimension-correct
        psi_{dim,k} — PD on R^dim by construction. Use this to keep D=15-25.
    backend="gpcam": gpcam.kernels.wendland_anisotropic (d0=3, fast/sparse).
        Use ONLY when you have reduced to D<=3.

    hps[0]=signal variance, hps[1:dim+1]=per-axis length scales/support radii.
    """
    if backend == "gpcam":
        try:
            from gpcam.kernels import wendland_anisotropic
        except Exception as e:  # pragma: no cover
            wendland_anisotropic, _import_err = None, e

        def kernel(x1, x2, hps):
            if wendland_anisotropic is None:
                raise ImportError(f"gpcam unavailable: {_import_err}")
            return wendland_anisotropic(x1, x2, hps)

        if dim > 3:
            import warnings

            warnings.warn(
                f"backend='gpcam' Wendland is d0=3 but dim={dim}>3 — NOT PD-safe. "
                "Reduce to D<=3 or use backend='explicit'.",
                RuntimeWarning,
            )
    else:  # explicit, dimension-correct

        def kernel(x1, x2, hps):
            sig = hps[0]
            ls = np.asarray(hps[1 : dim + 1], dtype=float)
            r = _aniso_scaled_radius(np.asarray(x1, float), np.asarray(x2, float), ls)
            return sig * wendland_psi(r, d0=dim, k=k)

    kernel.dim = dim
    kernel.n_hps = dim + 1
    return kernel


def default_hp_bounds(X: np.ndarray, y_resid: np.ndarray) -> np.ndarray:
    """
    Sensible starting bounds for [signal_var, length_scales...] from the reduced
    embedding X (N, D) and the residual targets. Length-scale upper bounds are
    kept deliberately tight (2x per-axis range) to bias toward sparsity.
    """
    D = X.shape[1]
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges == 0] = 1.0
    sig_hi = 10.0 * float(np.var(y_resid)) if np.var(y_resid) > 0 else 10.0
    bounds = [[1e-3, max(sig_hi, 1e-2)]]
    for r in ranges:
        bounds.append([1e-2 * r, 2.0 * r])  # tight upper bound -> sparser matrix
    return np.array(bounds, dtype=float)


# ----------------------------------------------------------------------------
# Empirical PD guard — this IS the PD falsification check
# ----------------------------------------------------------------------------


def check_kernel_psd(
    kernel_fn: Callable, X_sample: np.ndarray, hps: np.ndarray, tol: float = 1e-8
) -> dict:
    """
    Build the Gram matrix on a subsample and inspect its spectrum. Run this
    BEFORE any full GP fit and before trusting a new descriptor/kernel/dimension.

    Returns a dict with min eigenvalue, whether it is PSD within tolerance, and
    the (near-)zero fraction of the Gram (a first look at achievable sparsity).

    Kill rule: if `min_eigenvalue` is materially negative (< -tol scaled by the
    matrix norm), the kernel is NOT PD at this dimension. Fixes, in order:
      1. lower FeatureReducer.n_components (e.g. 25 -> 15),
      2. use a higher-smoothness Wendland valid in more dimensions,
      3. only then consider a Matern-core * Wendland-taper product (note: the
         product is PD only if BOTH factors are PD in `dim` — it does not rescue a
         non-PD Wendland by itself).
    """
    X_sample = np.asarray(X_sample, dtype=float)
    K = kernel_fn(X_sample, X_sample, np.asarray(hps, dtype=float))
    K = 0.5 * (K + K.T)  # symmetrise away round-off
    eig = np.linalg.eigvalsh(K)
    scale = max(np.linalg.norm(K, ord=2), 1.0)
    min_eig = float(eig.min())
    near_zero = float(np.mean(np.abs(K) < tol))
    return {
        "min_eigenvalue": min_eig,
        "is_psd": bool(min_eig > -tol * scale),
        "gram_density": 1.0 - near_zero,  # fraction of |K_ij| above tol
        "n_sample": len(X_sample),
    }
