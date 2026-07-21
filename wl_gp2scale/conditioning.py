"""
conditioning.py  (wl_gp2scale)
==============================
Measure WHY the sparse solve is slow, and pick the noise variance from numbers
instead of a blind jitter sweep.

The gp2Scale solve is CG on ``(K + sigma^2 I) alpha = y``. CG iterations scale like
sqrt(cond), and for a near-duplicate-riddled Gram ``cond ~ lambda_max(K) / sigma^2``:
near-identical molecules give near-identical rows -> near-zero eigenvalues of K, so
the jitter/noise ``sigma^2`` floors the smallest eigenvalue and sets the condition
number. This module quantifies all of that on the actual embedding:

  * near_duplicate_stats  -- how many molecules are near-identical, and the biggest
    cluster (the source of the near-singular block).
  * nugget_from_duplicates -- the descriptor-aliasing noise: 0.5 <(y_i - y_j)^2> over
    near-duplicate pairs = the variogram NUGGET = the statistically correct sigma^2.
  * sparse_wendland_gram + spectrum -- lambda_max(K) (and lambda_min) via eigsh.
  * cg_iterations -- ACTUAL scipy-CG iteration count + residual at each candidate
    sigma^2, so the noise is chosen from a measured cost curve.

Everything is cheap (GP-free, subsampled) so it runs on a laptop.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, eigsh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# ----------------------------- near-duplicate structure --------------------


def near_duplicate_stats(Z, eps, cat=None):
    """Molecules within ``eps`` of another (near-identical descriptors). Same-category
    only if ``cat`` given (matching the block kernel). Returns count of near-dup pairs,
    number of points in some near-dup, largest connected cluster, and the fraction."""
    Z = np.asarray(Z, float)
    n = len(Z)
    tree = cKDTree(Z)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if cat is not None and len(pairs):
        cat = np.asarray(cat)
        pairs = pairs[cat[pairs[:, 0]] == cat[pairs[:, 1]]]
    if len(pairs) == 0:
        return {"n_pairs": 0, "n_points_in_dup": 0, "largest_cluster": 1,
                "frac_in_dup": 0.0, "eps": float(eps)}
    g = sparse.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels)
    in_dup = sizes[labels] > 1
    return {
        "n_pairs": int(len(pairs)),
        "n_points_in_dup": int(in_dup.sum()),
        "largest_cluster": int(sizes.max()),
        "frac_in_dup": float(in_dup.mean()),
        "eps": float(eps),
    }


def nugget_from_duplicates(Z, y, eps, cat=None):
    """The descriptor-aliasing noise: 0.5 * mean((y_i - y_j)^2) over near-duplicate
    pairs (distance < eps). This is the variogram nugget gamma(0+) -- the irreducible
    spread of the target among molecules the descriptor cannot distinguish, hence the
    statistically correct observation-noise variance ``sigma^2``. Returns None if there
    are no near-duplicate pairs to estimate it from."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).ravel()
    tree = cKDTree(Z)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if cat is not None and len(pairs):
        cat = np.asarray(cat)
        pairs = pairs[cat[pairs[:, 0]] == cat[pairs[:, 1]]]
    if len(pairs) == 0:
        return None
    dy = y[pairs[:, 0]] - y[pairs[:, 1]]
    return float(0.5 * np.mean(dy**2))


# ----------------------------- sparse Gram + spectrum ----------------------


def sparse_wendland_gram(Z, cutoff, signal_var=1.0, cat=None):
    """Assemble the compact-support Wendland psi_{3,2} Gram as a sparse CSR, using a
    KD-tree for the in-support pairs and the same-category mask the kernel applies.
    Diagonal is ``signal_var``; off-diagonals ``signal_var * (1-r)^4 (4r+1)``."""
    Z = np.asarray(Z, float)
    n = len(Z)
    tree = cKDTree(Z)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)
        r = d / cutoff
        w = signal_var * (1.0 - r) ** 4 * (4.0 * r + 1.0)
        if cat is not None:
            cat = np.asarray(cat)
            keep = cat[pairs[:, 0]] == cat[pairs[:, 1]]
            pairs, w = pairs[keep], w[keep]
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        vals = np.concatenate([w, w, np.full(n, float(signal_var))])
    else:
        rows = cols = np.arange(n)
        vals = np.full(n, float(signal_var))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def spectrum(K, want_min=True):
    """Largest and (optionally) smallest eigenvalue of a sparse symmetric K via eigsh.
    lambda_min of a near-duplicate Gram is ~0 by construction; if the SA solve does not
    converge we return None for it (lambda_max alone drives the cond estimate)."""
    lam_max = float(eigsh(K, k=1, which="LA", return_eigenvectors=False,
                          maxiter=5000)[0])
    lam_min = None
    if want_min:
        try:
            lam_min = float(eigsh(K, k=1, which="SA", return_eigenvectors=False,
                                  maxiter=5000, tol=1e-4)[0])
        except Exception:
            lam_min = None
    return lam_max, lam_min


# ----------------------------- measured CG cost ----------------------------


def cg_iterations(K, sigma2, rhs, tol=1e-5, maxiter=None, M=None):
    """Actually run scipy CG on (K + sigma2 I) x = rhs and count iterations. Returns
    (iters, rel_residual, converged). This is the empirical cost curve the noise is
    chosen from -- it mirrors what fvgp's sparseCG(pre) does."""
    n = K.shape[0]
    A = (K + sigma2 * sparse.eye(n, format="csr")).tocsr()
    rhs = np.asarray(rhs, float).ravel()
    count = {"n": 0}

    def _cb(xk):
        count["n"] += 1

    x, info = cg(A, rhs, rtol=tol, maxiter=maxiter, M=M, callback=_cb)
    resid = float(np.linalg.norm(A @ x - rhs) / (np.linalg.norm(rhs) or 1.0))
    return count["n"], resid, bool(info == 0)


# ----------------------------- orchestration -------------------------------


def conditioning_report(Z, y, cutoff, cat=None, signal_var=None,
                        noise_grid=(1e-6, 1e-3, 1e-2, 1e-1, 1.0),
                        dup_eps=None, cg_tol=1e-5, cg_maxiter=5000, seed=0):
    """Full conditioning picture at a cutoff: near-duplicate structure, the nugget
    (recommended noise), lambda_max/min, and a measured cond + CG-iteration curve over
    ``noise_grid`` (with the nugget inserted). Returns a dict; caller prints it."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).ravel()
    n = len(Z)
    sv = float(signal_var) if signal_var is not None else float(np.var(y))
    if dup_eps is None:
        dup_eps = 0.05 * cutoff        # "near-identical" = well inside the support

    dup = near_duplicate_stats(Z, dup_eps, cat=cat)
    nugget = nugget_from_duplicates(Z, y, dup_eps, cat=cat)

    K = sparse_wendland_gram(Z, cutoff, signal_var=sv, cat=cat)
    lam_max, lam_min = spectrum(K)

    rng = np.random.default_rng(seed)
    rhs = rng.standard_normal(n)       # representative RHS for the CG cost probe
    lmin0 = lam_min if (lam_min is not None and lam_min > 0) else 0.0
    grid = sorted(set(list(noise_grid) + ([nugget] if nugget else [])))
    curve = []
    for s2 in grid:
        cond = (lam_max + s2) / (lmin0 + s2)   # eig(K+s2 I) = eig(K)+s2
        iters, resid, conv = cg_iterations(K, s2, rhs, tol=cg_tol, maxiter=cg_maxiter)
        curve.append({"sigma2": float(s2), "cond": float(cond),
                      "sqrt_cond": float(np.sqrt(cond)), "cg_iters": int(iters),
                      "cg_resid": resid, "converged": conv,
                      "is_nugget": bool(nugget is not None and np.isclose(s2, nugget))})
    return {
        "n": n, "cutoff": float(cutoff), "signal_var": sv, "dup_eps": float(dup_eps),
        "duplicates": dup, "nugget": nugget, "nnz": int(K.nnz),
        "lambda_max": lam_max, "lambda_min": lam_min, "curve": curve,
    }
