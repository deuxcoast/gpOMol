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

    Passing an explicit device wins (used by validation to force CPU) -- but only
    where it is actually satisfiable. Asking for 'cuda' on a host with no GPU used to
    be honoured verbatim and then died deep inside torch's lazy init
    ("RuntimeError: No CUDA GPUs are available") on the first kernel evaluation. That
    is reachable in normal operation: gp2Scale evaluates training blocks on the dask
    workers, which sit on GPU nodes, but the prediction cross-covariance is computed
    in the DRIVER process, which need not be on one. Fall back instead of dying.
    """
    have_cuda = torch is not None and torch.cuda.is_available()
    if device is not None:
        if str(device).startswith("cuda") and not have_cuda:
            return "cpu"
        return device
    return "cuda" if have_cuda else "cpu"


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
        "float64" (default) or "float32". Keep float64: the Wendland Gram on this
        embedding is near-singular (compact support + duplicate molecules => cond
        ~1e9), so float32's ~1e-7 kernel error amplifies into a materially wrong
        solve. float32 is only safe if you have verified the conditioning.
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
    dtype: str = "float64"
    cutoff_is_hp: bool = False

    def __post_init__(self):
        if torch is None:
            raise ImportError("wl_gp2scale.kernel requires PyTorch.")
        self._tdtype = torch.float32 if self.dtype == "float32" else torch.float64
        if self.backend not in ("wendland32", "wendland_d0"):
            raise ValueError("backend must be 'wendland32' or 'wendland_d0'")

    @property
    def _device(self) -> str:
        """Resolved per process at call time -- see AdditiveWendlandKernel._device."""
        return pick_device(self.device)

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
        #
        # compute_mode is load-bearing, do NOT drop it. torch.cdist defaults to
        # "use_mm_for_euclid_dist_if_necessary", which computes the Gram expansion
        # ||a||^2 + ||b||^2 - 2 a.b. That suffers catastrophic cancellation for
        # identical / near-identical points and returns a NONZERO self-distance
        # (3e-5 in float32, 9e-10 even in float64), so psi(r) < 1 on the diagonal
        # and K[i,i] comes out below signal_var. Measured on the real embedding this
        # perturbs K by ~2e-4 relative -- harmless on its own, but cond(K) ~ 1e9 here
        # (compact support + duplicate molecules => near-singular Gram), so it
        # amplifies into a badly wrong solve: R^2 0.049 -> 0.027 with the sparse path
        # while the dense scipy.cdist path was fine. The direct mode matches scipy
        # exactly. It forgoes the matmul kernel, but D=10 makes the direct form cheap.
        D = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
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
# additive N-channel kernel (WL + geometry, later + charge)
# ----------------------------------------------------------------------------


@dataclass
class ChannelSpec:
    """One additive channel: the embedding columns ``[start:stop]``, its
    compact-support ``cutoff``, and its Wendland ``backend``/``k``."""

    start: int
    stop: int
    cutoff: float
    backend: str = "wendland32"
    k: int = 2


@dataclass
class AdditiveWendlandKernel:
    """Callable ``kernel_function(x1, x2, hps)`` for gp2Scale: a SUM of compact-support
    Wendland blocks, one per channel, each on its own column range with its own cutoff
    and its own signal variance. Generalises ``WLBlockKernel`` to k = sum_c sv_c *
    psi_c(||z_c - z'_c|| / cutoff_c); a single channel reproduces ``WLBlockKernel``
    exactly (same conditioning-critical numerics: donot_use_mm cdist, float64, hard
    compact support, per-pair category mask).

    The embedding fed to the GP is ``[z_0 | z_1 | ... | data_id]`` -- the category tag
    is the LAST column (index ``total_dim = channels[-1].stop``), so
    with_category_tag / sort_by_category are unchanged.

    hps layout: ``hps[:C]`` = the C per-channel signal variances; if
    ``cutoffs_are_hp`` then ``hps[C:2C]`` override the per-channel cutoffs (lets the
    optimiser tune support per channel). Signal variances are trained under --train.
    """

    channels: list  # list[ChannelSpec]
    use_category_tag: bool = True
    device: Optional[str] = None
    dtype: str = "float64"
    cutoffs_are_hp: bool = False

    def __post_init__(self):
        if torch is None:
            raise ImportError("wl_gp2scale.kernel requires PyTorch.")
        self.channels = [c if isinstance(c, ChannelSpec) else ChannelSpec(*c)
                         for c in self.channels]
        for c in self.channels:
            if c.backend not in ("wendland32", "wendland_d0"):
                raise ValueError("backend must be 'wendland32' or 'wendland_d0'")
        self._tdtype = torch.float32 if self.dtype == "float32" else torch.float64
        self._total_dim = self.channels[-1].stop

    @property
    def _device(self) -> str:
        """Resolved PER PROCESS at call time, NOT cached in ``__post_init__``.

        gp2Scale constructs this kernel on the driver and pickles it to the dask
        workers, so a device string decided in the constructor is one process's answer
        imposed on every other. Caching it meant the driver's verdict shipped to the
        workers: on a CPU-only driver the workers would silently lose their GPUs, and
        with an explicit ``--device cuda`` the driver instead inherited a 'cuda' it
        could not honour and crashed in ``predict``. Resolving here lets one object
        build training blocks on the worker GPUs and the prediction cross-covariance
        on whatever the driver has.
        """
        return pick_device(self.device)

    def _psi(self, t, ch: ChannelSpec):
        if ch.backend == "wendland32":
            return _wendland32(t)
        return _wendland_d0(t, d0=(ch.stop - ch.start), k=ch.k)

    def __call__(self, x1, x2, hps):
        C = len(self.channels)
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        n1, n2 = x1.shape[0], x2.shape[0]

        # category tag (last column); block-skip disjoint categories with no distance.
        cats1 = cats2 = None
        if self.use_category_tag:
            cats1 = x1[:, self._total_dim].astype(np.int64)
            cats2 = x2[:, self._total_dim].astype(np.int64)
            if np.intersect1d(np.unique(cats1), np.unique(cats2)).size == 0:
                return np.zeros((n1, n2), dtype=np.float64)

        dev, td = self._device, self._tdtype
        K = torch.zeros((n1, n2), dtype=td, device=dev)
        for c, ch in enumerate(self.channels):
            sv = float(hps[c])
            cutoff = (float(hps[C + c]) if (self.cutoffs_are_hp and len(hps) > C + c)
                      else ch.cutoff)
            a = torch.as_tensor(x1[:, ch.start:ch.stop], dtype=td, device=dev)
            b = torch.as_tensor(x2[:, ch.start:ch.stop], dtype=td, device=dev)
            # compute_mode is load-bearing (see WLBlockKernel.__call__): the mm
            # expansion returns nonzero self-distances on near-duplicate points and
            # corrupts the near-singular Gram.
            D = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
            t = torch.clamp(D / cutoff, 0.0, 1.0)
            Kc = self._psi(t, ch)
            Kc = torch.where(t < 1.0, Kc, torch.zeros_like(Kc))  # hard compact support
            if sv != 1.0:
                Kc = Kc * sv
            K = K + Kc

        # per-pair category mask (blocks straddling a category boundary)
        if cats1 is not None and cats2 is not None:
            ca = torch.as_tensor(cats1, device=dev).view(-1, 1)
            cb = torch.as_tensor(cats2, device=dev).view(1, -1)
            K = torch.where(ca == cb, K, torch.zeros_like(K))

        return K.to("cpu").double().numpy()


def make_additive_kernel(channels, **kw) -> AdditiveWendlandKernel:
    """Factory: ``channels`` = list of ChannelSpec (or (start, stop, cutoff[, backend,
    k]) tuples). Returns a ready ``kernel_function(x1, x2, hps)``."""
    return AdditiveWendlandKernel(channels=list(channels), **kw)


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


def check_kernel_diagonal(kernel_fn: Callable, X_sample, signal_var: float,
                          tol: float = 1e-10, hps=None) -> dict:
    """The Wendland diagonal must be EXACTLY signal_var: psi(0)=1 and the distance
    from a point to itself is exactly 0.

    A drifting diagonal means the distance backend is returning nonzero self-
    distances -- which is precisely what torch.cdist's default mm expansion does
    (3e-5 in float32, 9e-10 in float64). On this near-singular Gram (cond ~1e9) that
    silently corrupts the solve rather than erroring. Regression guard for the
    compute_mode / dtype fix; run it whenever the kernel's numerics change.

    This builds a ONE-ELEMENT ``hps`` and so only makes sense for a single-block
    kernel; pass ``hps=`` explicitly to override. The guards below exist because both
    failure modes against an additive kernel are unhelpful: C>=2 raises a bare
    IndexError from inside ``__call__``, and C==1 with ``cutoffs_are_hp=True`` silently
    falls back to the frozen ChannelSpec radius (kernel.py: ``len(hps) > C + c`` is
    False), so the check would pass while testing the wrong kernel."""
    n_ch = len(getattr(kernel_fn, "channels", None) or [])
    if hps is None and n_ch > 1:
        raise ValueError(
            f"check_kernel_diagonal builds a 1-element hps, but this kernel has "
            f"{n_ch} channels and reads hps[0:{n_ch}]. Use "
            f"validate.additive_kernel_guard (its case (c) checks diag == sum_c sv_c), "
            f"or pass hps= explicitly.")
    if hps is None and n_ch == 1 and getattr(kernel_fn, "cutoffs_are_hp", False):
        raise ValueError(
            "this kernel has cutoffs_are_hp=True and so needs hps of length 2 "
            "[signal_var, radius]; a 1-element hps would silently fall back to the "
            "frozen ChannelSpec.cutoff and test the wrong configuration. Pass hps=.")
    hps = np.asarray([signal_var] if hps is None else hps, dtype=float)
    K = kernel_fn(X_sample, X_sample, hps)
    if sp.issparse(K):
        K = K.toarray()
    d = np.diag(np.asarray(K, dtype=float))
    err = float(np.max(np.abs(d - signal_var)))
    scale = max(abs(signal_var), 1.0)
    return {
        "max_diag_err": err,
        "rel_diag_err": err / scale,
        "pass": bool(err <= tol * scale),
        "diag_min": float(d.min()),
        "diag_max": float(d.max()),
    }


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


# ============================ Gibbs (non-stationary) =========================


@dataclass
class GibbsWendlandKernel:
    """Non-stationary product kernel: a Gibbs construction with a WENDLAND shape.

        k(x,x') = sv * 1[cat(x)=cat(x')] * prod_c  P_c(x,x') * psi( d_c / sqrt(S_c) )

        S_c     = l_c(x)^2 + l_c(x')^2                         (effective support^2)
        P_c     = [ 2 l_c(x) l_c(x') / S_c ] ^ (dim_c / 2)     (the PSD prefactor)
        l_c(x)  = c[chan, cat(x)] * sigma_k_c(x)

    WHY THIS EXISTS. The stationary kernel asserts one radius and one amplitude
    everywhere, and that is measurably false here: residual variance spans 8.2x across
    chemical families and the median 10-NN distance spans 2.3x. Letting the length scale
    vary is not as simple as substituting l(x) into psi -- that is not a valid covariance
    function. The Gibbs / Paciorek-Schervish prefactor is what restores positive
    definiteness, by discounting covariance between points that DISAGREE about their
    length scale. It is the ratio of the geometric to the quadratic mean of the two
    scales, raised to the dimension: exactly 1 when they agree, falling to 0 as they
    diverge.

    COMPACT SUPPORT SURVIVES, which is the whole reason to use Wendland rather than the
    Gaussian this construction is usually written with: the prefactor is a positive
    scalar, so the kernel still vanishes beyond sqrt(l(x)^2 + l(x')^2) and the matrix
    stays sparse. That is what makes a non-stationary kernel affordable at gp2Scale sizes.

    ``dim_c`` MUST BE THE AMBIENT (embedding) DIMENSION OF THE CHANNEL, not the Wendland's
    design dimension. psi_{3,2} is "the d<=3 Wendland", so d=3 is the natural guess and it
    is badly wrong: measured on real data at a 10x length-scale spread, the prefactor at
    d=3 leaves a relative min-eigenvalue of -4.7e-2 (a jitter of 4.30 does NOT rescue
    that), at d=1 it is -0.35, with no prefactor -0.76, and only at the ambient d=10 does
    the Gram come out PSD to the float64 noise floor. See scripts/gibbs_psd_gate.py.

    POSITIVE DEFINITENESS IS VERIFIED NUMERICALLY, NOT PROVEN. Paciorek & Schervish's
    theorem needs the shape to be PD in R^d for every d, which Wendland functions are not.
    The gate script checks it directly at the operating point and reports how it fails
    when the prefactor is wrong; re-run it if the embedding dimension or the spread of l
    changes materially.

    INPUT LAYOUT.  ``[ z_0 | ... | z_{C-1} | sk_0 | ... | sk_{C-1} | data_id ]``
    -- the per-channel local scales ride along as extra columns because gp2Scale hands
    the kernel COORDINATE BLOCKS, never row indices, so there is nowhere else to look
    them up from. ``with_gibbs_tags`` builds this layout; the category tag stays LAST so
    ``sort_by_category`` is unchanged.

    hps layout: ``hps[0]`` = the single signal variance (the diagonal is a product of
    ones, so it enters once rather than per channel); ``hps[1:]`` = the (C, n_cat)
    multipliers c, flattened C-major. Putting them in hps rather than closing over them
    is what lets them be trained later at no cost now.
    """

    channels: list                     # list[ChannelSpec]; only start/stop are used
    n_cat: int = 1
    device: Optional[str] = None
    dtype: str = "float64"

    def __post_init__(self):
        if torch is None:
            raise ImportError("wl_gp2scale.kernel requires PyTorch.")
        self.channels = [c if isinstance(c, ChannelSpec) else ChannelSpec(*c)
                         for c in self.channels]
        self._tdtype = torch.float32 if self.dtype == "float32" else torch.float64
        self._emb_dim = self.channels[-1].stop          # end of the embedding block
        self._n_chan = len(self.channels)
        self._cat_col = self._emb_dim + self._n_chan    # scales sit in between

    @property
    def _device(self) -> str:
        # resolved per process at call time, never cached -- see AdditiveWendlandKernel
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def n_hps(self) -> int:
        return 1 + self._n_chan * self.n_cat

    def unpack(self, hps):
        """-> (signal_variance, c) with c shaped (C, n_cat)."""
        hps = np.asarray(hps, dtype=float).ravel()
        exp = self.n_hps()
        if hps.size != exp:
            raise ValueError("expected %d hps (1 sv + %d channels x %d categories), "
                             "got %d" % (exp, self._n_chan, self.n_cat, hps.size))
        return float(hps[0]), hps[1:].reshape(self._n_chan, self.n_cat)

    def __call__(self, x1, x2, hps):
        td, dev = self._tdtype, self._device
        x1, x2 = np.asarray(x1), np.asarray(x2)
        n1, n2 = x1.shape[0], x2.shape[0]
        cats1 = x1[:, self._cat_col].astype(np.int64)
        cats2 = x2[:, self._cat_col].astype(np.int64)
        if np.intersect1d(np.unique(cats1), np.unique(cats2)).size == 0:
            return np.zeros((n1, n2), dtype=np.float64)

        sv, cmat = self.unpack(hps)
        K = torch.ones((n1, n2), dtype=td, device=dev)
        for ci, ch in enumerate(self.channels):
            a = torch.as_tensor(x1[:, ch.start:ch.stop], dtype=td, device=dev)
            b = torch.as_tensor(x2[:, ch.start:ch.stop], dtype=td, device=dev)
            D = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
            # l = c[channel, category of the ROW] * that row's local scale. Exact rather
            # than approximate: every entry surviving the category mask has cats1==cats2.
            sk1 = torch.as_tensor(x1[:, self._emb_dim + ci], dtype=td, device=dev)
            sk2 = torch.as_tensor(x2[:, self._emb_dim + ci], dtype=td, device=dev)
            c_row = torch.as_tensor(cmat[ci][cats1], dtype=td, device=dev)
            c_col = torch.as_tensor(cmat[ci][cats2], dtype=td, device=dev)
            li = (c_row * sk1).clamp_min(1e-12).view(-1, 1)
            lj = (c_col * sk2).clamp_min(1e-12).view(1, -1)

            S = li * li + lj * lj
            dim_c = float(ch.stop - ch.start)          # AMBIENT dim -- see the docstring
            pref = (2.0 * li * lj / S).pow(0.5 * dim_c)
            t = torch.clamp(D / torch.sqrt(S), 0.0, 1.0)
            Kc = torch.where(t < 1.0, _wendland32(t), torch.zeros_like(t))
            K = K * pref * Kc

        K = K * sv
        ca = torch.as_tensor(cats1, device=dev).view(-1, 1)
        cb = torch.as_tensor(cats2, device=dev).view(1, -1)
        K = torch.where(ca == cb, K, torch.zeros_like(K))
        return K.to("cpu").double().numpy()


def local_scale_k(Z_ref, Z_query, cat_ref, cat_query, k=10, block=2048):
    """sigma_k(x) = distance to x's k-th nearest SAME-CATEGORY point in ``Z_ref``.

    Same-category because the kernel zeroes everything else, so the local density that
    matters is the one within the block. Blocked over queries: the full (n_query, n_ref)
    distance matrix is 51 GB at 200k and is never needed at once.

    For training points pass Z_query is Z_ref; the self-distance is dropped."""
    Z_ref = np.asarray(Z_ref, float)
    Z_query = np.asarray(Z_query, float)
    cat_ref = np.asarray(cat_ref)
    cat_query = np.asarray(cat_query)
    same_set = Z_query is Z_ref
    out = np.zeros(len(Z_query))
    for c in np.unique(cat_query):
        qi = np.where(cat_query == c)[0]
        ri = np.where(cat_ref == c)[0]
        if len(ri) == 0:
            out[qi] = np.nan
            continue
        kk = min(k, len(ri) - 1) if same_set else min(k, len(ri))
        kk = max(kk, 1)
        for s in range(0, len(qi), block):
            q = qi[s:s + block]
            D = np.linalg.norm(Z_query[q][:, None, :] - Z_ref[ri][None, :, :], axis=2)
            if same_set:
                for a, g in enumerate(q):
                    hit = np.where(ri == g)[0]
                    if hit.size:
                        D[a, hit[0]] = np.inf
            out[q] = np.partition(D, kk - 1, axis=1)[:, kk - 1]
    bad = ~np.isfinite(out) | (out <= 0)
    if bad.any():                       # singleton categories, exact duplicates
        out[bad] = np.nanmedian(out[~bad]) if (~bad).any() else 1.0
    return out


def with_gibbs_tags(Zs, scales, data_id):
    """Assemble ``[ z_0 | ... | z_{C-1} | sk_0 | ... | sk_{C-1} | data_id ]``.

    ``Zs`` and ``scales`` are per-channel lists, in the SAME order as the kernel's
    ``channels``. The category tag stays last so sort_by_category is unchanged."""
    Z = np.hstack([np.asarray(z, float) for z in Zs])
    S = np.column_stack([np.asarray(s, float) for s in scales])
    return np.hstack([Z, S, np.asarray(data_id, float)[:, None]])
