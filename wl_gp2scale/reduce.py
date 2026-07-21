"""
reduce.py  (wl_gp2scale)
========================
Supervised dimensionality reduction of the sparse WL feature matrix to a 10-D
embedding, WITHOUT ever densifying it.

Why supervised PLS (not TruncatedSVD or a two-stage shortcut): the energy signal
is diffuse across many low-correlation WL columns that only PLS's multivariate
projection captures; unsupervised / prescreened reductions were measured to lose
0.1-0.4 R^2. So we keep full PLS and make it scale by streaming.

How it stays sparse. Centring/scaling would normally subtract a dense column mean
and densify X. Instead we keep X sparse and fold the per-column weight into two
matvecs of an implicit operator ``Xtilde = (X - 1 mu^T) diag(1/std)``, where ``std``
is the chosen scaling's weight (pareto=sqrt(col-std) by DEFAULT, standard=col-std,
center=1 -- see SparsePLS.scaling; the grid picked pareto):

    Xtilde @ r       = X @ (r/std)          - (m . r) * 1_n
    Xtilde^T @ t     = (X^T @ t) / std      - (sum t) * m         ,  m = mu/std

SIMPLS (de Jong 1993) for a single response then needs only these matvecs and a
p-dimensional deflation basis -- X is never densified and never deflated in place.
Each component is a couple of O(nnz) sparse matvecs; 10 components over 200k rows
is cheap.

Scaling: NATURAL (not unit-norm) scores. The stored rotation is the SIMPLS unit
weight ``w = S/||S||``, so ``transform`` returns the natural score ``t = Xtilde w``.
Because the per-column weighting is a fixed population statistic, ``var(t_a) =
w_a^T Cov(Xtilde) w_a`` is a population quantity -- N-invariant. Textbook SIMPLS instead normalises
each score to unit NORM (``t/||t||``), whose per-sample scale is ~1/sqrt(N); that
made pairwise distances (hence the compact-support cutoff and every length-scale
hyperparameter) shrink with N and forced a per-N recalibration. Natural scaling
removes that N-dependence so hyperparameters transfer from small runs to 200k.
Consequence: components now have UNEQUAL scale (leading dims spread wider), so an
isotropic distance is dominated by the leading PLS dims -- see the truncated-R^2
diagnostic (``truncated_r2_curve``) for whether the tail carries signal that an
ARD (per-dim length-scale) kernel would need to reach.

Gate. ``batch_pls_r2`` densifies a SMALL validation slice (8k/20k only) and fits
sklearn ``PLSRegression`` so validate.py can assert streaming == batch R^2 before
committing to the 200k run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


# ----------------------------- column statistics ---------------------------


def _col_stats(X):
    """Column means and (population) stds of a sparse matrix without densifying."""
    n = X.shape[0]
    mu = np.asarray(X.mean(axis=0)).ravel()
    ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = np.maximum(ex2 - mu**2, 0.0)
    std = np.sqrt(var)
    std[std == 0.0] = 1.0
    return mu, std


# ----------------------------- streaming SIMPLS ----------------------------


@dataclass
class SparsePLS:
    """Streaming SIMPLS on a sparse X. Fitted state is the p x A rotation matrix
    (plus the standardisation stats), so transform is a single sparse matmul.

    Rotations are UNIT SIMPLS weights, so ``transform`` returns natural (not
    unit-norm) scores -- an N-invariant embedding scale (see module docstring).

    ``scaling`` sets the per-column pre-weighting folded into the implicit operator.
    DEFAULT is "pareto", chosen from a scaling x min_count x n_components held-out
    OLS grid (3 seeds, 20k):
      * "pareto" (DEFAULT): divide by sqrt(std). Best robustness/signal trade-off in
        the grid -- ~0.34 held-out R^2 at min_count=2 with NO toxic tail and
        essentially no min_count sensitivity. Ties full standardisation's best
        (~0.36, but that needs min_count>=10 and goes NEGATIVE at low min_count).
      * "standard": z-score, divide by std -> unit-variance columns. On a sparse WL
        count matrix this INFLATES a feature present in k of N molecules to a spike of
        height ~sqrt(N/k) (independent of its count), so rare features get
        high-variance cross-covariance estimates that greedy SIMPLS overfits. With
        min_count=2 (the production prune) this is ANTI-PREDICTIVE (R^2 < 0 by dim 10).
      * "center": subtract mean only (std:=1). Fully min_count-insensitive (confirms
        /std is the root cause) but underfits (~0.25) by under-weighting mid-freq
        signal. WL columns are already same-unit per-atom counts, so standardisation
        was never needed for commensurability.
    """

    n_components: int = 10
    scaling: str = "pareto"
    mu_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    m_: np.ndarray = field(default=None, repr=False)          # mu/std
    R_: np.ndarray = field(default=None, repr=False)          # p x A rotations
    y_mean_: float = 0.0

    # implicit standardised matvecs (no densify) --------------------------------
    def _fwd(self, X, r):
        # Xtilde @ r
        return X.dot(r / self.std_) - float(self.m_ @ r)

    def _bwd(self, X, t):
        # Xtilde^T @ t
        return (X.T.dot(t)) / self.std_ - float(t.sum()) * self.m_

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        self.mu_, std = _col_stats(X)
        if self.scaling == "center":
            std = np.ones_like(std)               # center only (no /std)
        elif self.scaling == "pareto":
            std = np.sqrt(std)                    # milder than full standardisation
        elif self.scaling != "standard":
            raise ValueError("scaling must be 'standard', 'center', or 'pareto'")
        self.std_ = std
        self.m_ = self.mu_ / self.std_
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        p = X.shape[1]
        A = min(self.n_components, p)

        S = self._bwd(X, yc)                      # p : Xtilde^T yc
        V = np.zeros((p, A))                      # deflation basis (orthonormal)
        R = np.zeros((p, A))                      # rotations (UNIT weights)
        for a in range(A):
            nrm = np.linalg.norm(S)
            if nrm < 1e-12:
                R = R[:, :a]
                break
            w = S / nrm                           # unit weight direction (STORED)
            t = self._fwd(X, w)                   # n : natural scores  Xtilde @ w
            tn = np.linalg.norm(t)
            if tn < 1e-12:
                R = R[:, :a]
                break
            # Unit-norm score is used ONLY to build the deflation basis. v = pl/||pl||
            # is scale-invariant, so normalising t here does not change V or the
            # deflation -- it only keeps the loadings at O(1) for numerics.
            t_unit = t / tn
            pl = self._bwd(X, t_unit)             # p : loading Xtilde^T t_unit
            if a > 0:
                pl = pl - V[:, :a] @ (V[:, :a].T @ pl)   # orthogonalise vs basis
            vn = np.linalg.norm(pl)
            v = pl / vn if vn > 1e-12 else pl
            V[:, a] = v
            S = S - v * float(v @ S)              # deflate cross-product
            S = S - V[:, : a + 1] @ (V[:, : a + 1].T @ S)  # full re-orthogonalise
            # Store the UNIT weight (not w/tn). transform then yields the natural
            # score t = Xtilde @ w, whose per-sample scale is a population quantity
            # (var = w^T Corr(X) w), N-INVARIANT -- so the cutoff/hyperparameters
            # transfer across N. (The old w/tn stored unit-NORM scores, per-sample
            # scale ~1/sqrt(N), which is why the cutoff had to be recalibrated per N.)
            R[:, a] = w
        self.R_ = R
        return self

    def transform(self, X):
        """Return the (N, A) supervised embedding. One sparse @ dense(p x A) matmul
        plus a rank-1 correction -- stays sparse-friendly."""
        X = sp.csr_matrix(X)
        Rs = self.R_ / self.std_[:, None]         # fold 1/std into rotations
        offset = self.m_ @ self.R_                # (A,)  from the centering term
        return X.dot(Rs) - offset[None, :]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ----------------------------- parity helpers ------------------------------


def regression_r2(Z_tr, y_tr, Z_te, y_te):
    """Held-out R^2 of an OLS fit on the low-D embedding -- a direct measure of how
    much supervised signal the embedding retains (used on both streaming and batch
    embeddings so the comparison is apples-to-apples)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    lr = LinearRegression().fit(Z_tr, y_tr)
    return float(r2_score(y_te, lr.predict(Z_te)))


def truncated_r2_curve(Z_tr, y_tr, Z_te, y_te, dims=None):
    """Held-out OLS R^2 using only the FIRST k embedding dims, for k in ``dims``.

    This is the decision instrument for isotropic-vs-ARD Wendland. With natural
    (Option-1) scaling the leading PLS dims dominate an isotropic L2 distance, so
    the kernel effectively sees only the top few components. If R^2 has already
    saturated by then, an isotropic cutoff loses nothing and we keep it simple. If
    R^2 keeps climbing through the tail dims, that later signal is exactly what an
    isotropic distance drowns out -> ARD (per-dim length-scales) is warranted.

    Returns a list of (k, r2) and prints the curve plus each dim's marginal gain and
    the per-component score std (the anisotropy that makes this question live)."""
    Z_tr = np.asarray(Z_tr, float)
    Z_te = np.asarray(Z_te, float)
    A = Z_tr.shape[1]
    if dims is None:
        dims = sorted({1, 2, 3, 5, A} & set(range(1, A + 1)))
    stds = Z_tr.std(axis=0)
    print(f"[trunc] per-component score std (train): {np.round(stds, 4)}")
    print(f"[trunc]   std ratio dim1/dim{A} = {stds[0] / max(stds[-1], 1e-30):.1f}x "
          f"(anisotropy an isotropic cutoff would weight by)")
    curve, prev = [], None
    for k in dims:
        r2 = regression_r2(Z_tr[:, :k], y_tr, Z_te[:, :k], y_te)
        gain = "" if prev is None else f"  (+{r2 - prev:.4f} vs previous)"
        print(f"[trunc]   dims 1..{k:>2}: R^2 = {r2:.4f}{gain}")
        curve.append((int(k), float(r2)))
        prev = r2
    return curve


def batch_pls_r2(X_tr, y_tr, X_te, y_te, n_components=10, scaling="pareto"):
    """Reference: dense pre-scale + sklearn PLSRegression on a SMALL slice.
    Returns (embedding-OLS R^2, PLSRegression.score R^2). Only for validate.py.

    ``scaling`` MUST match the SparsePLS scaling under test or the parity gate is
    meaningless (it would compare two different embeddings). Applies the same
    per-column weight as SparsePLS: pareto=/sqrt(std), standard=/std, center=none."""
    from sklearn.cross_decomposition import PLSRegression

    X_tr = np.asarray(X_tr.todense()) if sp.issparse(X_tr) else np.asarray(X_tr)
    X_te = np.asarray(X_te.todense()) if sp.issparse(X_te) else np.asarray(X_te)
    mu = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    if scaling == "center":
        w = np.ones_like(std)
    elif scaling == "pareto":
        w = np.sqrt(std)
    elif scaling == "standard":
        w = std
    else:
        raise ValueError("scaling must be 'standard', 'center', or 'pareto'")
    Xs_tr = (X_tr - mu) / w
    Xs_te = (X_te - mu) / w
    pls = PLSRegression(n_components=n_components, scale=False).fit(Xs_tr, y_tr)
    Z_tr, Z_te = pls.transform(Xs_tr), pls.transform(Xs_te)
    return regression_r2(Z_tr, y_tr, Z_te, y_te), float(pls.score(Xs_te, y_te))


# ----------------------------- linear prior mean ---------------------------


@dataclass
class LinearEmbeddingMean:
    """OLS linear mean ``m(z) = b0 + b^T z`` on the embedding, for use as a GP PRIOR
    MEAN (Approach A / "detrend"): the GP models the residual ``y - m(z)`` and
    predictions add ``m(z*)`` back. This is exactly gp2Scale's posterior-mean equation
    (Noack et al. 2025, Eq. 2) with a linear m -- means are an explicitly supported
    gp2Scale customization and are orthogonal to the compact-support/sparse-K machinery
    (the mean never touches K).

    Why: the compact-support Wendland reverts to the prior mean where a test point has
    no in-support neighbour. With the default zero mean that is 0 (mean reversion, ~45%
    of test points here); with a linear mean it reverts to the OLS prediction instead,
    so the GP inherits the linear/OLS accuracy as a floor and adds local corrections
    where coverage exists. PLS also bakes a global linear trend into the embedding, so
    removing it leaves the (near-stationary) residual the compact-support kernel is
    actually good at -- the use case gp2Scale is designed for.

    Equivalent to passing ``m(z*)`` as fvgp's ``prior_mean_function`` (the "inside the
    GP" route, needed only when the mean is trained jointly). For frozen-hyperparameter
    predict the point predictions are identical, and the manual detrend cannot perturb
    the sparse solve.
    """

    coef_: np.ndarray = field(default=None, repr=False)   # (d,)
    intercept_: float = 0.0

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        H = np.hstack([np.ones((len(Z), 1)), Z])
        beta, *_ = np.linalg.lstsq(H, y, rcond=None)
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, Z):
        return self.intercept_ + np.asarray(Z, dtype=float) @ self.coef_
