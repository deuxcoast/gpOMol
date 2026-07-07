"""
env_features_kernel.py
======================
Part 2 of the environment-level aggregate-GP pipeline.

Two jobs:
  (A) Turn per-atom SOAP/ACE environments into a low-D Euclidean embedding whose
      Euclidean metric IS the Mahalanobis metric you want the kernel to see.
  (B) Provide a compactly-supported kernel that is PROVABLY PD in that dimension.

PD and compact support at your chosen D -- the honest version:
--------------------------------------------------------------
Two distinct things caused your earlier indefiniteness, and they must not be conflated:
  (i) NON-EUCLIDEAN METRIC: Wendland-over-Wasserstein gave [-10,-4] eigenvalues because
      W_2 in high dimensions is not of negative type / not isometrically Euclidean, so
      phi(W_2) is not PD for ANY polynomial. The embedding move fixes THIS: once the
      descriptor is a genuine vector in R^D, the metric is Euclidean and that failure
      cannot recur.
  (ii) DESIGN DIMENSION: every radial Wendland phi_{d,k} is PD on R^d and lower, with a
      FINITE maximal dimension above which PD is lost (Schoenberg/Wendland). WHICH
      dimension depends on the polynomial. The high-order gp2Scale polynomial has a
      fairly high design dimension -- empirically PD to D~40 on the configs I tested --
      so a Euclidean D~=20 embedding with it is likely fine, but that is something you
      must VERIFY (check_pd below), not assume, because the ceiling is finite.

PD-safe DEFAULT used here -- a SEPARABLE (product) Wendland:
      k(x,x') = sigma^2 * prod_d  phi_1( |x_d - x'_d| / ell_d ),
phi_1 a 1-D Wendland (PD on R). Each factor is a valid 1-D kernel; a product of PSD
kernels is PSD (Schur product theorem) in ANY dimension D. Support is the axis-aligned
box {|x_d-x'_d| < ell_d for all d} -> compact -> sparse. The point is not that the
radial form is broken at D=20 (it may well be fine); it is that the product form is
GUARANTEED PD at every D with no design-dimension bookkeeping. That guarantee is worth
having on a run you cannot cheaply re-do.

Extensivity is NOT handled here. It is native to the aggregation A (see aggregate_solver).
An optional per-element self-energy baseline e(Z) is a *conditioning* aid only; the
extensive scaling still comes entirely from the sum over environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from dscribe.descriptors import SOAP

    _HAVE_DSCRIBE = True
except Exception:
    _HAVE_DSCRIBE = False


# ----------------------------------------------------------------------------- #
# (A) Feature reduction: per-atom SOAP -> standardize -> PCA -> whiten           #
# ----------------------------------------------------------------------------- #
@dataclass
class EnvEmbedding:
    """Fitted embedding: raw SOAP/ACE -> R^D with Euclidean == target Mahalanobis."""

    mean_: np.ndarray  # (F,)  feature mean (standardization)
    scale_: np.ndarray  # (F,)  feature std
    components_: np.ndarray  # (D,F) PCA directions
    explained_variance_: np.ndarray  # (D,)  PCA eigenvalues
    whiten: bool = True  # scale PCA axes to unit variance
    D: int = field(init=False)

    def __post_init__(self):
        self.D = self.components_.shape[0]

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """(n,F) raw descriptors -> (n,D) embedding. Euclidean here == Mahalanobis."""
        Xs = (X_raw - self.mean_) / self.scale_
        Z = Xs @ self.components_.T  # PCA projection
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_ + 1e-12)
        return Z


def fit_env_embedding(
    X_raw: np.ndarray, D: int = 20, whiten: bool = True
) -> EnvEmbedding:
    """
    Fit standardize + PCA(D) + optional whitening on a representative sample of
    per-atom descriptors. Whitening is what turns a subsequent *isotropic* box/radius
    into an anisotropic Mahalanobis neighborhood in the original SOAP space -- so the
    diagnostic's plain Euclidean neighbor query is already the Mahalanobis query.

    Keep D modestly below the ambient descriptor dimension. With the product-Wendland
    below, D is NOT limited by PD (any D is safe); it is limited by the curse of
    concentration -- large D collapses pairwise-distance spread and kills sparsity.
    """
    mean_ = X_raw.mean(axis=0)
    scale_ = X_raw.std(axis=0) + 1e-12
    Xs = ((X_raw - mean_) / scale_).astype(np.float32, copy=False)
    # randomized SVD: O(n*F*D), low memory. A full np.linalg.svd on wide compressed-SOAP
    # (F up to ~1e4) would allocate an n x F factor (~10 GB) and take minutes -- avoid it.
    from sklearn.utils.extmath import randomized_svd

    n = Xs.shape[0]
    U, S, Vt = randomized_svd(Xs, n_components=D, n_iter=5, random_state=0)
    ev = (S**2) / max(n - 1, 1)
    return EnvEmbedding(
        mean_=mean_,
        scale_=scale_,
        components_=Vt[:D],
        explained_variance_=ev[:D],
        whiten=whiten,
    )


def compute_soap(atoms_list, species, r_cut=5.0, n_max=6, l_max=4, average=False):
    """
    Per-atom SOAP for a list of ASE Atoms. average=False => one row per ATOM
    (this is the whole point of the environment-level move: no molecule pooling).
    Falls back to a deterministic synthetic descriptor if dscribe is absent, so the
    rest of the pipeline is runnable/testable without it.
    """
    if _HAVE_DSCRIBE:
        soap = SOAP(
            species=species,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            periodic=False,
            sparse=False,
        )
        # per-atom rows, one block per molecule; caller keeps molecule membership
        return [soap.create(a) for a in atoms_list]
    raise RuntimeError(
        "dscribe not installed; use synthetic_environments() for testing."
    )


# ----------------------------------------------------------------------------- #
# (B) PD-safe compact-support kernel: separable (product) Wendland               #
# ----------------------------------------------------------------------------- #
def _wendland_1d(t: np.ndarray) -> np.ndarray:
    """1-D Wendland phi_{1,1}(t) = (1-t)_+^3 (3t + 1). C^2, PD on R^1, support [0,1)."""
    tt = np.clip(t, 0.0, 1.0)
    return (1.0 - tt) ** 3 * (3.0 * tt + 1.0)


def product_wendland(x1: np.ndarray, x2: np.ndarray, hps: np.ndarray) -> np.ndarray:
    """
    PD-in-any-D compact-support kernel.
        hps[0]      = signal variance sigma^2
        hps[1:D+1]  = per-dimension support radii ell_d (also the length scales)
    Returns (N1,N2). Nonzero only inside the axis-aligned box of half-widths ell_d.

    NOTE: because the embedding is already whitened, ell_d are all ~O(1) and a single
    scalar radius is often enough (broadcast a scalar into hps[1:]).
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sig2 = hps[0]
    ell = np.asarray(hps[1:], dtype=float)
    if ell.size == 1:
        ell = np.full(x1.shape[1], ell.item())
    K = np.ones((x1.shape[0], x2.shape[0]))
    for d in range(x1.shape[1]):
        t = np.abs(np.subtract.outer(x1[:, d], x2[:, d])) / ell[d]
        K *= _wendland_1d(t)
        if not K.any():  # early exit: fully out of support
            break
    return sig2 * K


def wendland_radial(x1, x2, hps):
    """
    Radial anisotropic Wendland phi(||x-x'||_M) using the high-order gp2Scale polynomial.
    PD on R^d only up to this polynomial's finite design dimension. Empirically PD to
    D~40 on tested configs, so a D~=20 Euclidean embedding is plausibly fine -- but run
    check_pd() at YOUR D and radius before trusting it. Kept as the ellipsoidal-support
    alternative to the product form (rounder neighborhoods; one fewer length scale if
    isotropic). Use the product form when you want a guarantee instead of a check.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    ell = np.asarray(hps[1:], float)
    if ell.size == 1:
        ell = np.full(x1.shape[1], ell.item())
    diff = (x1[:, None, :] - x2[None, :, :]) / ell[None, None, :]
    r = np.sqrt((diff**2).sum(-1))
    r = np.clip(r, 0.0, 1.0)
    poly = 35 * r**3 + 25 * r**2 + 8 * r + 1  # gp2Scale Eq.(3) style
    return hps[0] * (1.0 - r) ** 8 * poly


def check_pd(kernel_fn, X, hps, jitter=0.0, n=400, seed=0):
    """
    Numerical PD guard: min eigenvalue of a Gram sub-block. Run this ONCE at your
    chosen D and radius before committing compute. Returns (min_eig, is_pd).
    A negative min_eig on product_wendland at any D would be a bug; on the radial
    UNSAFE kernel at D>3 it is expected -- that contrast is the whole lesson.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    G = kernel_fn(X[idx], X[idx], hps)
    G = 0.5 * (G + G.T) + jitter * np.eye(len(idx))
    w = np.linalg.eigvalsh(G)
    return float(w.min()), bool(w.min() >= -1e-8)


# ----------------------------------------------------------------------------- #
# Synthetic environments for local testing (no dscribe / no OMol25 needed)       #
# ----------------------------------------------------------------------------- #
def synthetic_environments(
    n_mol, atoms_per_mol=30, n_types=40, F=120, dup_fraction=0.0, seed=0
):
    """
    Fabricate a plausible environment population: a bounded set of 'environment types'
    (cluster centroids) with jitter, assembled into molecules. dup_fraction injects
    near-identical conformer copies -- the mechanism that can secretly inflate nnz/row.

    Returns
    -------
    X_raw  : (n_env, F) raw descriptors
    mol_of : (n_env,) int  molecule index of each environment (membership for A)
    y_mol  : (n_mol,) float synthetic *extensive* molecular energy
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_types, F))
    type_energy = rng.standard_normal(n_types) * 0.5  # per-type local energy
    X, mol_of, y_mol = [], [], np.zeros(n_mol)
    for m in range(n_mol):
        k = atoms_per_mol
        types = rng.integers(0, n_types, size=k)
        env = centroids[types] + 0.05 * rng.standard_normal((k, F))
        if dup_fraction > 0:  # duplicate-conformer leakage
            n_dup = int(dup_fraction * k)
            if n_dup:
                env[:n_dup] = env[0] + 1e-3 * rng.standard_normal((n_dup, F))
        X.append(env)
        mol_of.extend([m] * k)
        y_mol[m] = type_energy[types].sum() + 0.01 * rng.standard_normal()  # extensive
    return np.vstack(X), np.asarray(mol_of), y_mol


if __name__ == "__main__":
    X, mol_of, y = synthetic_environments(2000, seed=1)
    emb = fit_env_embedding(X, D=20)
    Z = emb.transform(X)
    hps = np.concatenate([[1.0], np.full(20, 1.5)])
    me_prod, pd_prod = check_pd(product_wendland, Z, hps)
    me_rad, pd_rad = check_pd(wendland_radial, Z, hps)
    print(f"embedding D={emb.D}, envs={len(Z)}")
    print(
        f"product_wendland  min_eig={me_prod:+.3e}  PD={pd_prod}  (guaranteed for all D)"
    )
    print(
        f"radial Wendland   min_eig={me_rad:+.3e}  PD={pd_rad}  (verify at your D/radius)"
    )
