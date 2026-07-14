"""
kernel.py  (wl_gp2scale)
========================
The core deliverable: a gp2Scale-compatible, GPU block Wendland kernel that
NEVER materialises a dense covariance matrix.

gpCAM's gp2Scale mode calls ``kernel_function(x1, x2, hps)`` once per block, where
each block is at most ``gp2Scale_batch_size`` (10_000) rows. This module builds
each block on the GPU and returns ONLY the non-zeros as a ``scipy.sparse.coo_matrix``,
so the full covariance lives as distributed sparse blocks and the solve uses the
iterative/sparse path (conjugate gradient).

What the kernel returns (fvgp 4.8.3 contract)
---------------------------------------------
The kernel returns a DENSE block (numpy ndarray). This is not a regression from
"return only the non-zeros" -- it is how fvgp's gp2Scale extracts the non-zeros:
its worker-side wrapper ``kernel_function`` (fvgp/gp_prior.py:540-543) calls this
kernel, then does ``sparse.coo_matrix(k)`` ON THE WORKER and gathers only the COO
components (``data, rows, cols``) in global coordinates. So the dense block is a
transient local to the worker (the same ``torch.cdist`` tensor), the global matrix
is assembled from non-zeros only, and the dense N x N is never formed. Returning a
scipy.sparse block instead breaks PREDICTION: ``posterior_covariance`` uses the
kernel output directly as ``np.diag(kk)`` and as the CG right-hand side
(``_normalize_rhs`` -> ``np.asarray(vec, float64)``), both of which require a dense
array. The proven ``hybrid_descriptor`` kernel likewise returns dense.

Design, per the block-sparsity constraints
-------------------------------------------
1. Category block-sparsity. The embedding fed to the GP is (N, dim+1): ``dim``
   PLS coordinates plus one integer ``data_id`` category tag in the last column.
   If two blocks share no category, we return a pre-built all-zero block WITHOUT
   computing any distance (fvgp then stores zero non-zeros for it). Otherwise a
   per-pair mask zeroes cross-category entries (compact support already zeroes far
   pairs, so cross-category covariance is 0).
2. On-GPU distance + Wendland. Coordinates move to CUDA (or CPU fallback); the
   pairwise L2 distance is ``torch.cdist`` and the compact-support Wendland is
   applied elementwise on-device. The (<=10k x 10k) block is the only dense
   object; fvgp reduces it to COO on the worker before anything is gathered.
3. Compact support. psi(r) = (1-r)^4 (4r+1) for r<1 else 0 (the d0=3 Wendland C^2,
   matching the dense validation kernel). A dimension-correct d0=dim backend is
   available as a PD fallback if CG stalls in the 10-D embedding.

Positive-definiteness. psi_{3,2} is only guaranteed PD on R^3; on a 10-D embedding
we rely on compact support + a tight cutoff -> diagonal dominance -> practical PD,
plus minimal jitter (1e-6) and CG. ``check_kernel_psd`` falsifies this on a
subsample; if it fails, use ``backend="wendland_d0"`` (PD on R^dim by construction)
and/or tighten the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency at run time
    torch = None


# ----------------------------------------------------------------------------
# device / dtype helpers
# ----------------------------------------------------------------------------


def pick_device(device: Optional[str] = None) -> str:
    """Return an explicit torch device string. 'cuda' if available, else 'cpu'.
    Passing an explicit device wins (used by validation to force CPU)."""
    if device is not None:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------------
# Wendland radial functions (torch, on-device)
# ----------------------------------------------------------------------------


def _wendland32(t):
    """psi_{3,2}(r) with r already clipped to [0,1]:  (1-r)^4 (4r+1).
    This is the exact form used by the dense validation kernel."""
    s = 1.0 - t
    return s.pow(4) * (4.0 * t + 1.0)


def _wendland_d0(t, d0: int, k: int = 2):
    """Dimension-correct Wendland psi_{d0,k}(r), r clipped to [0,1]; PD on R^d0.
    Same closed forms as hybrid_descriptor/embedding_kernel.py, torch version.
    k=2 (C^4) is the default (Matern-5/2 analogue)."""
    l = d0 // 2 + k + 1
    s = 1.0 - t
    if k == 0:
        return s.pow(l)
    if k == 1:
        return s.pow(l + 1) * ((l + 1) * t + 1.0)
    if k == 2:
        return s.pow(l + 2) * (((l**2 + 4 * l + 3) * t.pow(2)) + (3 * l + 6) * t + 3.0) / 3.0
    if k == 3:
        return (
            s.pow(l + 3)
            * (
                (l**3 + 9 * l**2 + 23 * l + 15) * t.pow(3)
                + (6 * l**2 + 36 * l + 45) * t.pow(2)
                + (15 * l + 45) * t
                + 15.0
            )
            / 15.0
        )
    raise ValueError("k must be in {0,1,2,3}")


# ----------------------------------------------------------------------------
# kernel factory
# ----------------------------------------------------------------------------


@dataclass
class WLBlockKernel:
    """Callable ``kernel_function(x1, x2, hps)`` for gpCAM gp2Scale.

    Parameters
    ----------
    cutoff : float
        Compact-support radius on the embedding (from cutoff.recalibrate). Pairs
        farther than ``cutoff`` have exactly zero covariance -> sparsity.
    dim : int
        Number of embedding (PLS) coordinates. The covariance is computed on
        ``x[:, :dim]`` only.
    use_category_tag : bool
        If True, ``x[:, dim]`` is an integer category id used to skip/zero
        cross-category pairs. Set False for plain (unsorted, single-category) data.
    backend : {"wendland32", "wendland_d0"}
        "wendland32" = psi_{3,2} (matches the dense validation kernel, the default).
        "wendland_d0" = dimension-correct psi_{dim,2}, PD on R^dim (PD fallback).
    device : str | None
        Torch device; None -> cuda if available else cpu.
    dtype : str
        "float32" (default, GPU-friendly) or "float64" (tighter parity vs SciPy).
    cutoff_is_hp : bool
        If True, ``hps[1]`` overrides the cutoff (lets the optimiser tune support).

    hps layout: hps[0] = signal variance; hps[1] = cutoff (only if cutoff_is_hp).
    """

    cutoff: float
    dim: int = 10
    use_category_tag: bool = True
    backend: str = "wendland32"
    k: int = 2
    device: Optional[str] = None
    dtype: str = "float32"
    cutoff_is_hp: bool = False

    def __post_init__(self):
        if torch is None:
            raise ImportError("wl_gp2scale.kernel requires PyTorch.")
        self._device = pick_device(self.device)
        self._tdtype = torch.float32 if self.dtype == "float32" else torch.float64
        if self.backend not in ("wendland32", "wendland_d0"):
            raise ValueError("backend must be 'wendland32' or 'wendland_d0'")

    # -- split coordinates / category tag ------------------------------------
    def _split(self, x):
        x = np.asarray(x)
        if self.use_category_tag:
            coords = x[:, : self.dim]
            cats = x[:, self.dim].astype(np.int64)
        else:
            coords = x[:, : self.dim] if x.shape[1] > self.dim else x
            cats = None
        return coords, cats

    def _psi(self, t):
        if self.backend == "wendland32":
            return _wendland32(t)
        return _wendland_d0(t, d0=self.dim, k=self.k)

    # -- the gp2Scale entry point --------------------------------------------
    def __call__(self, x1, x2, hps):
        signal_var = float(hps[0])
        cutoff = float(hps[1]) if (self.cutoff_is_hp and len(hps) > 1) else self.cutoff

        c1, cats1 = self._split(x1)
        c2, cats2 = self._split(x2)
        n1, n2 = c1.shape[0], c2.shape[0]

        # 1. category block-skip: disjoint categories -> all-zero block, no
        #    distance computed (fvgp stores zero non-zeros for it).
        if cats1 is not None and cats2 is not None:
            if np.intersect1d(np.unique(cats1), np.unique(cats2)).size == 0:
                return np.zeros((n1, n2), dtype=np.float64)

        dev, td = self._device, self._tdtype
        a = torch.as_tensor(c1, dtype=td, device=dev)
        b = torch.as_tensor(c2, dtype=td, device=dev)

        # 2. on-GPU L2 distance + compact-support Wendland (worker-local dense block)
        D = torch.cdist(a, b)                       # (n1, n2)
        t = torch.clamp(D / cutoff, 0.0, 1.0)
        K = self._psi(t)
        K = torch.where(t < 1.0, K, torch.zeros_like(K))   # hard compact support
        if signal_var != 1.0:
            K = K * signal_var

        # 3. per-pair category mask (blocks that straddle a category boundary)
        if cats1 is not None and cats2 is not None:
            ca = torch.as_tensor(cats1, device=dev).view(-1, 1)
            cb = torch.as_tensor(cats2, device=dev).view(1, -1)
            K = torch.where(ca == cb, K, torch.zeros_like(K))

        # 4. return the dense block; fvgp's kernel_function extracts the non-zeros
        #    (sparse.coo_matrix(k)) on the worker before gathering. The exact zeros
        #    from compact support / category mask are dropped there -> sparse global.
        return K.to("cpu").double().numpy()


def make_wl_block_kernel(cutoff: float, **kw) -> WLBlockKernel:
    """Convenience factory returning a ready ``kernel_function(x1, x2, hps)``."""
    return WLBlockKernel(cutoff=cutoff, **kw)


# ----------------------------------------------------------------------------
# dense reference (for validation parity only) — NOT used at scale
# ----------------------------------------------------------------------------


def dense_wendland_reference(x1, x2, hps, cutoff, dim=None, metric="euclidean"):
    """Dense psi_{3,2} block via scipy.cdist, byte-for-byte matching
    descriptor_eval/gp_parity.py::wendland_kernel. Used only by validate.py to
    prove the sparse GPU kernel reproduces the dense CPU kernel."""
    from scipy.spatial.distance import cdist

    a = np.asarray(x1)[:, :dim] if dim else np.asarray(x1)
    b = np.asarray(x2)[:, :dim] if dim else np.asarray(x2)
    D = cdist(a, b, metric=metric)
    r = np.clip(D / cutoff, 0.0, 1.0)
    return float(hps[0]) * (1.0 - r) ** 4 * (4.0 * r + 1.0)


# ----------------------------------------------------------------------------
# PD falsification guard
# ----------------------------------------------------------------------------


def check_kernel_psd(kernel_fn: Callable, X_sample, hps, tol: float = 1e-8) -> dict:
    """Build the Gram matrix on a subsample and inspect its spectrum BEFORE the
    full fit. ``kernel_fn`` may return dense ndarray or scipy.sparse; both handled.

    Kill rule: if min eigenvalue is materially negative (< -tol * ||K||_2), the
    kernel is not PD at this dimension -> switch to backend='wendland_d0' and/or
    tighten the cutoff (do NOT paper over it with large jitter)."""
    X_sample = np.asarray(X_sample, dtype=float)
    K = kernel_fn(X_sample, X_sample, np.asarray(hps, dtype=float))
    if sp.issparse(K):
        K = K.toarray()
    K = np.asarray(K, dtype=float)
    K = 0.5 * (K + K.T)
    eig = np.linalg.eigvalsh(K)
    scale = max(np.linalg.norm(K, ord=2), 1.0)
    min_eig = float(eig.min())
    density = float(np.mean(np.abs(K) > tol))
    return {
        "min_eigenvalue": min_eig,
        "is_psd": bool(min_eig > -tol * scale),
        "gram_density": density,
        "n_sample": int(len(X_sample)),
    }
