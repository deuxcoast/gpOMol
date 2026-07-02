"""
extensive_mean.py
=================

Step 1 of the hybrid-GP pipeline: strip the *extensive* part of the DFT total
energy with a cheap linear model, so the Gaussian process only ever regresses
the *intensive* residual.

Design decision (pre-subtraction, not a gpCAM mean function)
------------------------------------------------------------
gpCAM lets you pass a `mean_function(x, hps)` whose parameters are trained
jointly with the kernel. We deliberately DO NOT do that for the element
references. Instead we fit them once by (ridge-regularised) least squares over
the whole training set and subtract them, feeding the GP the residual as its
`y_data`.

Reasons:
  * Element references fit on 1-2M molecules are pinned to numerical precision;
    their posterior uncertainty is negligible, so folding them into the GP's
    hyperparameter vector buys nothing and costs ~n_elements extra dimensions in
    an already-expensive MCMC.
  * A single OLS solve is O(N * n_elements^2) once, versus re-evaluating the mean
    inside every likelihood call.
  * CRITICAL for honest diagnostics: every semivariogram / kNN / skill number in
    diagnostics.py must be computed on the residual. Pre-subtraction makes the
    residual an explicit array you can hand to those checks, so the sparse-kernel
    machinery can never be silently credited with error the mean removed. (This is
    the "earlier gain was a mean-function artifact" failure mode, guarded against.)

Extensivity story
-----------------
Total energy is extensive: E(M) ~ sum over atoms. The leading term is
sum_Z n_Z(M) * eps_Z. Removing it leaves a residual that is *weakly* extensive at
most (roughly ~ number of bonds); the optional composition columns below soak up
the next-order size dependence. What the GP then sees is intensive enough that a
compact-support stationary-ish kernel is not being asked to track an unbounded
quantity — which is the whole point.

If you later find residual size-dependence in the semivariogram (a nugget that
scales with molecule size), add columns to `extra_feature_fn` (bond count, ring
count, n_atoms**2) rather than reaching for a fancier kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class ExtensiveEnergyModel:
    """
    Linear extensive-energy model:  E_ext(M) = intercept + sum_Z n_Z eps_Z + w . extra(M)

    Parameters
    ----------
    fit_intercept : bool
        Include a constant offset. Harmless for residual modelling; absorbs a
        global baseline. Set False for a strict "empty molecule -> 0" convention.
    ridge : float
        L2 regularisation on the coefficients (NOT the intercept). Small but
        non-zero guards against near-collinear / rare-element columns. 1e-6 is a
        reasonable default at OMol25 scale.
    extra_feature_fn : callable or None
        Optional map (Z_list, extra_context) -> np.ndarray of *low-order*
        composition features appended to the element-count block, e.g. bond count.
        Keep this short; it is a prior mean, not a model of the physics.
    """

    fit_intercept: bool = True
    ridge: float = 1e-6
    extra_feature_fn: Optional[Callable[[Sequence[int], object], np.ndarray]] = None

    # learned state
    elements_: np.ndarray = field(default=None, repr=False)  # sorted unique Z
    element_index_: dict = field(default=None, repr=False)  # Z -> column
    coef_: np.ndarray = field(default=None, repr=False)  # element + extra coeffs
    intercept_: float = 0.0
    n_extra_: int = 0

    # ------------------------------------------------------------------ design
    def _design(
        self,
        Z_lists: Sequence[Sequence[int]],
        extra_context: Optional[Sequence[object]] = None,
    ) -> np.ndarray:
        """Build the [n_molecules, n_elements (+ extra)] design matrix."""
        n = len(Z_lists)
        counts = np.zeros((n, len(self.elements_)), dtype=float)
        for i, Zs in enumerate(Z_lists):
            for Z in Zs:
                j = self.element_index_.get(int(Z))
                if j is not None:  # unseen element -> ignored here
                    counts[i, j] += 1.0

        if self.extra_feature_fn is not None:
            ctx = extra_context if extra_context is not None else [None] * n
            extra = np.vstack(
                [
                    np.atleast_1d(self.extra_feature_fn(Zs, c))
                    for Zs, c in zip(Z_lists, ctx)
                ]
            )
            self.n_extra_ = extra.shape[1]
            X = np.hstack([counts, extra])
        else:
            self.n_extra_ = 0
            X = counts
        return X

    # --------------------------------------------------------------------- fit
    def fit(
        self,
        Z_lists: Sequence[Sequence[int]],
        y_total: np.ndarray,
        extra_context: Optional[Sequence[object]] = None,
    ) -> "ExtensiveEnergyModel":
        """
        Z_lists : list of per-molecule atomic-number sequences (len N)
        y_total : (N,) DFT total energies
        """
        y_total = np.asarray(y_total, dtype=float).ravel()
        all_Z = sorted({int(Z) for Zs in Z_lists for Z in Zs})
        self.elements_ = np.array(all_Z, dtype=int)
        self.element_index_ = {Z: j for j, Z in enumerate(all_Z)}

        X = self._design(Z_lists, extra_context)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Ridge normal equations, but do NOT penalise the intercept column.
        p = X.shape[1]
        R = self.ridge * np.eye(p)
        if self.fit_intercept:
            R[0, 0] = 0.0
        beta = np.linalg.solve(X.T @ X + R, X.T @ y_total)

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    # ----------------------------------------------------------------- predict
    def predict(
        self,
        Z_lists: Sequence[Sequence[int]],
        extra_context: Optional[Sequence[object]] = None,
    ) -> np.ndarray:
        """Extensive energy prediction (N,)."""
        X = self._design(Z_lists, extra_context)
        return self.intercept_ + X @ self.coef_

    def residual(
        self,
        Z_lists: Sequence[Sequence[int]],
        y_total: np.ndarray,
        extra_context: Optional[Sequence[object]] = None,
    ) -> np.ndarray:
        """Intensive residual y_total - E_ext  (this is the GP's y_data)."""
        y_total = np.asarray(y_total, dtype=float).ravel()
        return y_total - self.predict(Z_lists, extra_context)

    # ------------------------------------------------------------- diagnostics
    def reference_energies(self) -> dict:
        """Return {Z: eps_Z} learned references, for sanity-checking against
        known atomic energies (a cheap correctness check before anything else)."""
        return {int(Z): float(self.coef_[j]) for Z, j in self.element_index_.items()}
