"""
aggregate_uq.py
==============
Part 4 -- predictive variance for the aggregate GP.

For a query molecule with environments {env_j*} and aggregate s* = sum_j f(env_j*):

    Var[s*] = a_*^T K_{**} a_*  -  k_*^T (K_mol + sigma^2 I)^{-1} k_*,      (*)
      a_*  = ones(n_star)                       (sum over the query's own environments)
      K_** = K_env among the query's OWN environments        (n_star x n_star, tiny)
      k_*  = A_train (K_{train,*} a_*)  in R^M              (aggregate cross-cov)

Why pooling "reintroduces density": the second term needs (K_mol + sigma^2 I)^{-1} k_*,
and K_mol = A K_env A^T is dense, so you cannot store or reuse a dense factorization.
Everything stays matrix-free through the same operator as training.

THE NON-OBVIOUS HAZARD (measured, not assumed): CATASTROPHIC CANCELLATION.
--------------------------------------------------------------------------
For a molecule the model predicts WELL, the two terms in (*) are both large and nearly
equal (prior aggregate variance ~ n_star^2 * per-pair covariance, tens; posterior ~ 0.01),
so Var[s*] is a small difference of large numbers. To get one correct digit of Var you
need the reduction term accurate to ~1e-4 relative. Consequences:
  * The EXACT method (one accurate solve, tol<=1e-8) is reliable -- verified to ~0.5% here.
  * LOVE / low-rank-inverse caches are NOT reliable here at modest rank: their O(1e-2)
    relative error on the reduction is amplified ~1000x and routinely overshoots the prior
    (variance clips to 0). Use LOVE only at high rank AND validate against exact on a
    sample; otherwise prefer accurate solves with a SHARED preconditioner (below).
  * For a NOVEL molecule (outside training support) there is no cancellation: posterior
    ~= prior, correctly large. Cancellation is worst exactly where the answer is smallest.
"""

from __future__ import annotations

import numpy as np
from aggregate_solver import aggregate_operator, jacobi_preconditioner
from scipy.sparse.linalg import cg, minres


# --------------------------- exact, per query -------------------------------- #
def predictive_variance_exact(
    A_train,
    K_env,
    sigma2,
    Kcross,
    K_star_star,
    a_star,
    method="minres",
    tol=1e-9,
    maxiter=5000,
):
    """
    Exact Var[s*] via one matrix-free solve. Use a TIGHT tol (default 1e-9) because of
    the cancellation above -- a loose solve silently corrupts the small posterior variance.
      Kcross      : sparse (n_train x n_star) train-vs-query env kernel (compact support)
      K_star_star : dense (n_star x n_star) query-vs-query env kernel (tiny)
      a_star      : (n_star,) ones
    """
    prior = float(a_star @ (K_star_star @ a_star))
    k_star = A_train @ (Kcross @ a_star)  # M
    op = aggregate_operator(A_train, K_env, sigma2)
    Minv = jacobi_preconditioner(A_train, K_env, sigma2)
    solver = minres if method == "minres" else cg
    z, _ = solver(op, k_star, M=Minv, rtol=tol, maxiter=maxiter)
    reduction = float(k_star @ z)
    return max(prior - reduction, 0.0), prior, reduction


# ------------- amortized across queries: shared preconditioner --------------- #
def predictive_variance_batch(
    A_train,
    K_env,
    sigma2,
    Kcross_list,
    Kss_list,
    a_star_list,
    method="minres",
    tol=1e-9,
    maxiter=5000,
):
    """
    Variances for MANY query molecules. The rigorous, cancellation-safe amortization:
    build the operator + preconditioner ONCE, then do one accurate solve per query
    (warm-startable, few iterations under a good preconditioner). This trades LOVE's
    O(r)/query for O(iter*nnz)/query but is numerically trustworthy for aggregate UQ.
    Returns list of (var, prior, reduction).
    """
    op = aggregate_operator(A_train, K_env, sigma2)
    Minv = jacobi_preconditioner(A_train, K_env, sigma2)  # built once, reused
    solver = minres if method == "minres" else cg
    out = []
    for Kcross, Kss, a_star in zip(Kcross_list, Kss_list, a_star_list):
        prior = float(a_star @ (Kss @ a_star))
        k_star = A_train @ (Kcross @ a_star)
        z, _ = solver(op, k_star, M=Minv, rtol=tol, maxiter=maxiter)
        red = float(k_star @ z)
        out.append((max(prior - red, 0.0), prior, red))
    return out


# ---------------------- LOVE-style Lanczos cache (validate!) ----------------- #
def build_love_cache(A_train, K_env, sigma2, rank=100, probe=None, seed=0):
    """
    One-time Lanczos on (K_mol + sigma^2 I): R (M x r) with (K_mol+sigma^2 I)^{-1} ~ R R^T.
    Correctness notes learned the hard way:
      * With full reorthogonalization the recurrence coefficients no longer form the
        tridiagonal; recompute T = Q^T (op Q) directly.
      * Floor Ritz values at sigma2 (operator's spectrum is >= sigma2).
    Even so, see the module header: for aggregate variance this cache must be validated
    against predictive_variance_exact because of cancellation. Prefer GPyTorch LOVE at
    high rank for production. Kept here as the O(r)/query option when you have verified
    it is accurate enough for YOUR prior/posterior magnitude ratio.
    """
    rng = np.random.default_rng(seed)
    M = A_train.shape[0]
    op = aggregate_operator(A_train, K_env, sigma2)
    q = rng.standard_normal(M) if probe is None else probe.copy()
    q = q / np.linalg.norm(q)
    Q = np.zeros((M, rank))
    q_prev = np.zeros(M)
    b = 0.0
    r_eff = rank
    for k in range(rank):
        Q[:, k] = q
        w = op.matvec(q)
        a = float(q @ w)
        w = w - a * q - b * q_prev
        w -= Q[:, : k + 1] @ (Q[:, : k + 1].T @ w)
        b = float(np.linalg.norm(w))
        if b < 1e-10:
            r_eff = k + 1
            break
        q_prev = q
        q = w / b
    Q = Q[:, :r_eff]
    T = Q.T @ np.column_stack([op.matvec(Q[:, j]) for j in range(r_eff)])
    T = 0.5 * (T + T.T)
    wT, VT = np.linalg.eigh(T)
    wT = np.clip(wT, sigma2, None)
    return Q @ (VT / np.sqrt(wT))


def predictive_variance_love(R, A_train, K_env, Kcross, K_star_star, a_star):
    """Cheap Var[s*] = a_*^T K_** a_* - ||R^T k_*||^2 (validate against exact -- see header)."""
    prior = float(a_star @ (K_star_star @ a_star))
    k_star = A_train @ (Kcross @ a_star)
    return max(prior - float(np.sum((R.T @ k_star) ** 2)), 0.0), prior


if __name__ == "__main__":
    import scipy.sparse as sp
    from aggregate_solver import build_A, build_Kenv_from_graph
    from env_features_kernel import (
        fit_env_embedding,
        product_wendland,
        synthetic_environments,
    )
    from scipy.spatial import cKDTree

    X, mol_of, y = synthetic_environments(300, atoms_per_mol=18, seed=5)
    n_mol = len(y)
    emb = fit_env_embedding(X, D=12)
    Z = emb.transform(X)
    hps = np.concatenate([[1.0], np.full(12, 2.5)])
    sigma2 = 1e-3
    A = build_A(mol_of, n_mol)
    Kenv = build_Kenv_from_graph(Z, product_wendland, hps, radius=2.5)

    star_env = np.where(mol_of == 0)[0]
    Zs = Z[star_env]
    a_star = np.ones(len(star_env))
    tree = cKDTree(Z)
    rows, cols, vals = [], [], []
    for js, zj in enumerate(Zs):
        cs = np.asarray(tree.query_ball_point(zj, r=2.5), np.int64)
        rows.append(cs)
        cols.append(np.full(len(cs), js))
        vals.append(product_wendland(zj[None], Z[cs], hps).ravel())
    Kcross = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(Z), len(star_env)),
    ).tocsr()
    Kss = product_wendland(Zs, Zs, hps)

    var_ex, pri, red = predictive_variance_exact(A, Kenv, sigma2, Kcross, Kss, a_star)
    Kmol = (A @ Kenv @ A.T).toarray() + sigma2 * np.eye(n_mol)
    k_star = A @ (Kcross @ a_star)
    var_dense = float(a_star @ (Kss @ a_star) - k_star @ np.linalg.solve(Kmol, k_star))
    print(
        f"prior={pri:.4f}  reduction={red:.4f}  (cancellation ratio prior/Var = {pri/max(var_ex,1e-12):.0f}x)"
    )
    print(
        f"Var exact ={var_ex:.6e}   dense oracle={var_dense:.6e}   |diff|={abs(var_ex-var_dense):.2e}"
    )

    batch = predictive_variance_batch(A, Kenv, sigma2, [Kcross], [Kss], [a_star])
    print(f"Var batch ={batch[0][0]:.6e}   (shared-preconditioner accurate solve)")
    print(
        "LOVE: convergence toward exact (rank -> M); note overshoot under cancellation"
    )
    for r in [100, 200]:
        R = build_love_cache(A, Kenv, sigma2, rank=r, probe=y)
        v, _ = predictive_variance_love(R, A, Kenv, Kcross, Kss, a_star)
        print(f"   rank={r:4d}  LOVE Var={v:.6e}  |LOVE-exact|={abs(v-var_ex):.2e}")
