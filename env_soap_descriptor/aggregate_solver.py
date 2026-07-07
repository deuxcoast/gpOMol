"""
aggregate_solver.py
==================
Part 3 -- matrix-free training/inference for the linear-functional GP
    y_M = sum_{i in M} f(env_i) + eps,     f ~ GP(0, K_env),   K_env compact-support/sparse.

Marginal:            y ~ N(0, K_mol + sigma^2 I),   K_mol := A K_env A^T   (M x M).
Effective operator:  K_mol is DENSE even when K_env is sparse (two molecules couple if
                     any of their environments are within support). So we NEVER form it.
Everything is a matvec through the three sparse objects A, K_env, A^T.

This is deliberately outside vanilla gpCAM: GPOptimizer models one observation per input.
Here observations are molecule totals, so we reuse gp2Scale's *kernel* and its distributed
COO->CSR assembly to build the sparse K_env, then drive our own MINRES/CG on the aggregate
operator. The solver primitives (sparseMINRES / sparseCG) are the same family gp2Scale uses.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, minres


# --------------------- build the sparse operands ----------------------------- #
def build_A(mol_of, n_mol):
    """Sum-pooling A (M x n_env), CSR. Row m has 1s at the columns of m's environments."""
    n_env = len(mol_of)
    return sp.csr_matrix(
        (np.ones(n_env), (mol_of, np.arange(n_env))), shape=(n_mol, n_env)
    )


def build_Kenv_from_graph(Z, kernel_fn, hps, radius, block=20_000):
    """
    Assemble sparse K_env (n_env x n_env, CSR) by evaluating `kernel_fn` only on the
    radius-`radius` neighbor pairs. On the cluster this is exactly gp2Scale's job:
    Dask workers each own a row-block, build the local neighbor graph, evaluate the
    compact kernel on those pairs, and return COO; the host concatenates to CSR.
    Here: a single-node reference implementation using cKDTree, block by block.
    """
    from scipy.spatial import cKDTree

    n = len(Z)
    tree = cKDTree(Z)
    rows, cols, vals = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        # neighbor lists for this row-block against the whole tree
        nbrs = tree.query_ball_point(Z[s:e], r=radius)
        for local_i, cs in enumerate(nbrs):
            i = s + local_i
            cs = np.asarray(cs, dtype=np.int64)
            if cs.size == 0:
                continue
            kvals = kernel_fn(Z[i : i + 1], Z[cs], hps).ravel()  # (len(cs),)
            nz = kvals != 0.0
            rows.append(np.full(nz.sum(), i))
            cols.append(cs[nz])
            vals.append(kvals[nz])
    if not rows:
        return sp.csr_matrix((n, n))
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    return 0.5 * (K + K.T)  # symmetrize (guards against query asymmetry)


# --------------------- the matrix-free effective operator -------------------- #
def aggregate_operator(A, K_env, sigma2):
    """
    LinearOperator implementing (A K_env A^T + sigma^2 I) @ v  for v in R^M, via:
        t = A^T v      (scatter molecule values to environments;  O(nnz(A)))
        u = K_env t    (sparse matvec;                            O(nnz(K_env)))
        w = A u        (sum environment values back to molecules; O(nnz(A)))
        return w + sigma2 * v
    Never instantiates K_mol. Cost per iteration is dominated by nnz(K_env) -- which is
    precisely the quantity the Part-1 diagnostic decides is linear-or-not in n_env.
    """
    M = A.shape[0]

    def mv(v):
        t = A.T @ v
        u = K_env @ t
        w = A @ u
        return w + sigma2 * v

    return LinearOperator((M, M), matvec=mv, rmatvec=mv, dtype=float)


def jacobi_preconditioner(A, K_env, sigma2):
    """
    M^{-1} ~ 1/diag(K_mol + sigma^2 I), computed WITHOUT forming K_mol:
        diag(K_mol)_m = sum_{i,j in m} K_env[i,j] = ((A K_env) .* A).sum(axis=1).
    A @ K_env has ~ (envs/mol)*(nnz/row) nonzeros per molecule row -- cheap.
    Duplicate-conformer blocks make K_mol ill-conditioned; Jacobi is a first defense,
    a pivoted-Cholesky / Nystrom-on-centroids preconditioner is the next step if CG stalls.
    """
    AK = A @ K_env  # M x n_env, sparse
    diag_kmol = np.asarray(AK.multiply(A).sum(axis=1)).ravel()
    d = diag_kmol + sigma2
    d = np.where(d > 0, d, sigma2)
    inv = 1.0 / d
    M = A.shape[0]
    return LinearOperator((M, M), matvec=lambda v: inv * v, dtype=float)


# --------------------------- solve + predict --------------------------------- #
def solve_alpha(A, K_env, y, sigma2, method="minres", tol=1e-6, maxiter=1000):
    """alpha = (K_mol + sigma^2 I)^{-1} y, matrix-free. Returns (alpha, info)."""
    op = aggregate_operator(A, K_env, sigma2)
    Minv = jacobi_preconditioner(A, K_env, sigma2)
    solver = minres if method == "minres" else cg
    alpha, info = solver(op, y, M=Minv, rtol=tol, maxiter=maxiter)
    return alpha, info


def predict_mean(alpha, A_train, Kcross, a_star):
    """
    Posterior mean of the QUERY molecule total  s* = sum_j f(env_j*).
        mu* = k_*^T alpha,   k_* = A_train (K_{train,*} a_star)   in R^M.
    Kcross : sparse (n_train x n_star) train-vs-query environment kernel (compact support).
    a_star : (n_star,) ones (sum over the query's own environments).
    """
    w = Kcross @ a_star  # n_train: cross-cov of each train env to the query aggregate
    k_star = A_train @ w  # M: pooled to training molecules
    return float(k_star @ alpha), k_star


if __name__ == "__main__":
    # end-to-end smoke test on synthetic data, comparing matrix-free solve to a dense oracle
    from env_features_kernel import (
        fit_env_embedding,
        product_wendland,
        synthetic_environments,
    )

    X, mol_of, y = synthetic_environments(400, atoms_per_mol=20, seed=3)
    n_mol = len(y)
    emb = fit_env_embedding(X, D=15)
    Z = emb.transform(X)
    hps = np.concatenate([[1.0], np.full(15, 2.5)])
    sigma2 = 1e-3

    A = build_A(mol_of, n_mol)
    Kenv = build_Kenv_from_graph(Z, product_wendland, hps, radius=2.5)
    print(f"n_env={len(Z)}  nnz(K_env)={Kenv.nnz}  density={Kenv.nnz/len(Z)**2:.2e}")

    alpha, info = solve_alpha(A, Kenv, y, sigma2, method="minres")
    # dense oracle
    Kmol = (A @ Kenv @ A.T).toarray() + sigma2 * np.eye(n_mol)
    alpha_dense = np.linalg.solve(Kmol, y)
    rel = np.linalg.norm(alpha - alpha_dense) / np.linalg.norm(alpha_dense)
    print(f"MINRES info={info}  ||alpha_mf - alpha_dense||/||.|| = {rel:.2e}")
