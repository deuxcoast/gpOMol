r"""
gp_wendland_sweep.py
====================
Step 2 of the pipeline: an EXACT Gaussian process with a compactly supported
Wendland kernel at a series of FIXED support radii, on the referenced-energy
target, using the Option-4 Wasserstein distance between molecules.

Purpose
-------
Map the *density-vs-accuracy tradeoff*. For each fixed support radius rho we
record (covariance-matrix density, held-out energy error, uncertainty
calibration). This is the manual sweep that scouts the landscape BEFORE the
non-stationary MCMC run -- it tells you whether the high-sparsity regime is also
accurate, and gives the dense Matern baseline as an accuracy ceiling.

This is NOT the production model: the radius is fixed by hand (not sampled by
MCMC) and stationary (one global rho, not rho(x)). The non-stationary MCMC
kernel in gpCAM should do at least as well -- so these numbers are a
conservative floor.

Two backends (`--backend`)
--------------------------
  numpy  : transparent exact GP (closed-form). Fully self-contained, prints the
           kernel-matrix min eigenvalue so the positive-definiteness issue
           (Wendland-over-Wasserstein is not guaranteed PD) is VISIBLE.
  gpcam  : the production GPOptimizer with a custom precomputed-distance kernel.
Run both once at the same settings to confirm they agree.

Inputs
------
  * the referencing artifact from reference_energy.py (--ref-npz): provides
    r_train/r_test and train_idx/test_idx (the residual target, index-keyed).
  * the OMol25 dataset (--src), to compute Option-4 representations for those
    same indices and build the Wasserstein distance matrix.

Requirements
------------
    pip install pot ase scipy scikit-learn matplotlib fairchem-core gpcam

Usage
-----
    python gp_wendland_sweep.py --src ../train_4M \
        --ref-npz ./referencing/20260617_130856_reference.npz \
        --n-cap 600 --backend numpy
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist, squareform


# ----------------------------------------------------------------------
# Option-4 representation + Wasserstein distance (kept identical to the
# diagnostic, so the GP sees the same metric)
# ----------------------------------------------------------------------
def distance_profile_representation(atoms, n_quantiles=16):
    R = atoms.get_positions()
    Z = atoms.get_atomic_numbers().astype(float)
    n = len(Z)
    probs = np.linspace(0.0, 1.0, n_quantiles)
    if n == 1:
        return Z, np.zeros((1, n_quantiles))
    D = squareform(pdist(R))
    profiles = np.empty((n, n_quantiles))
    for i in range(n):
        d_i = np.delete(D[i], i)
        d_i.sort()
        profiles[i] = np.quantile(d_i, probs)
    return Z, profiles


def molecule_wasserstein(repA, repB, alpha=1.0, beta=1.0, method="emd", reg=0.5):
    import ot

    ZA, qA = repA
    ZB, qB = repB
    geo = cdist(qA, qB, metric="euclidean")
    typ = (ZA[:, None] != ZB[None, :]).astype(float)
    C = np.ascontiguousarray(alpha * typ + beta * geo)
    a = np.full(len(ZA), 1.0 / len(ZA))
    b = np.full(len(ZB), 1.0 / len(ZB))
    if method == "emd":
        return float(ot.emd2(a, b, C))
    return float(ot.sinkhorn2(a, b, C, reg, method="sinkhorn_stabilized"))


def build_distance_matrix(reps, alpha=1.0, beta=1.0, method="emd", reg=0.5):
    S = len(reps)
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            D[i, j] = D[j, i] = molecule_wasserstein(
                reps[i], reps[j], alpha, beta, method, reg
            )
        if (i + 1) % 50 == 0:
            print(f"  distance rows {i + 1}/{S}", flush=True)
    return D


def nearest_pd(K, noise_floor):
    """Project onto nearest PD matrix, flooring eigenvalues at the noise level
    (NOT near zero) to keep the matrix well-conditioned."""
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    min_eig = float(w.min())
    n_clipped = int(np.sum(w < noise_floor))
    w = np.clip(w, noise_floor, None)  # floor at noise, not 1e-6*max
    return (V * w) @ V.T, min_eig, n_clipped


# ----------------------------------------------------------------------
# Kernels (operate on a precomputed distance block)
# ----------------------------------------------------------------------
def wendland(r, rho):
    """Compactly supported Wendland kernel (gp2Scale Eq. 3 form). k=1 at r=0,
    exactly 0 for r >= rho."""
    t = np.clip(r / rho, 0.0, 1.0)
    return (1 - t) ** 8 * (35 * t**3 + 25 * t**2 + 8 * t + 1)


def matern52(r, ell):
    """Dense Matern-5/2 (never exactly zero) -- the accuracy-ceiling baseline."""
    s = np.sqrt(5.0) * r / ell
    return (1 + s + s**2 / 3.0) * np.exp(-s)


# ----------------------------------------------------------------------
# Transparent exact GP (constant prior mean, matches gpCAM's default)
# ----------------------------------------------------------------------
def gp_numpy(D, train_pos, test_pos, y_train, corr_fn, signal_var, noise):
    mu = float(np.mean(y_train))
    yc = y_train - mu
    K = signal_var * corr_fn(D[np.ix_(train_pos, train_pos)])
    K_pd, min_eig, n_clipped = nearest_pd(K, noise_floor=noise)  # floor at noise
    if min_eig < 0:
        print(
            f"    PD repair: min eig {min_eig:.2e}, clipped {n_clipped}/{len(K)} eigenvalues"
        )
    Ks = signal_var * corr_fn(D[np.ix_(test_pos, train_pos)])
    c, low = cho_factor(K_pd)
    alpha = cho_solve((c, low), yc)
    mean = Ks @ alpha + mu
    v = cho_solve((c, low), Ks.T)
    var = np.clip(signal_var - np.einsum("ij,ji->i", Ks, v), 0.0, None)
    return mean, var, min_eig


# ----------------------------------------------------------------------
# gpCAM backend (production tool, same precomputed distances via index kernel)
# ----------------------------------------------------------------------
def gp_gpcam(D, train_pos, test_pos, y_train, rho, signal_var, noise):
    from gpcam import GPOptimizer

    def kernel(x1, x2, hps):
        i = x1[:, 0].astype(int)
        j = x2[:, 0].astype(int)
        return hps[0] * wendland(D[np.ix_(i, j)], rho)

    x_tr = train_pos.reshape(-1, 1).astype(float)
    x_te = test_pos.reshape(-1, 1).astype(float)
    gp = GPOptimizer(
        x_tr,
        y_train,
        init_hyperparameters=np.array([signal_var]),
        kernel_function=kernel,
        noise_variances=np.full(len(y_train), noise),
    )
    # no train() -> rho and signal variance stay fixed (the fixed-radius regime)
    mean = gp.posterior_mean(x_te)["m(x)"]
    var = gp.posterior_covariance(x_te)["v(x)"]
    return mean, var


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def metrics(mean, var, y_true, noise):
    err = mean - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    sd = np.sqrt(var + noise)  # predictive sd for observations
    z = err / sd
    cov95 = float(np.mean(np.abs(z) <= 1.96))  # want ~0.95 if calibrated
    z2 = float(np.mean(z**2))  # want ~1.0 if calibrated
    return mae, rmse, cov95, z2


def density(D, rho):
    off = D[np.triu_indices(D.shape[0], k=1)]
    return float((off <= rho).mean())


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--ref-npz", required=True, help="output of reference_energy.py")
    p.add_argument(
        "--n-cap",
        type=int,
        default=600,
        help="cap molecules (OT distance matrix is O(n^2)!)",
    )
    p.add_argument("--quantiles", type=int, default=16)
    p.add_argument("--backend", choices=["numpy", "gpcam"], default="numpy")
    p.add_argument(
        "--noise-frac",
        type=float,
        default=0.05,
        help="obs noise sd as a fraction of target sd",
    )
    p.add_argument(
        "--dist-cache",
        default=None,
        help="path to load/save the Wasserstein matrix (.npy)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # --- load referencing artifact ---
    ref = np.load(args.ref_npz, allow_pickle=True)
    train_idx, test_idx = ref["train_idx"], ref["test_idx"]
    r_train, r_test = ref["r_train"], ref["r_test"]

    # --- cap size, preserving train/test membership ---
    rng = np.random.default_rng(args.seed)
    if len(train_idx) + len(test_idx) > args.n_cap:
        frac = args.n_cap / (len(train_idx) + len(test_idx))
        n_tr = max(2, int(len(train_idx) * frac))
        n_te = max(1, int(len(test_idx) * frac))
        tr_sel = rng.choice(len(train_idx), n_tr, replace=False)
        te_sel = rng.choice(len(test_idx), n_te, replace=False)
        train_idx, r_train = train_idx[tr_sel], r_train[tr_sel]
        test_idx, r_test = test_idx[te_sel], r_test[te_sel]
    n_tr, n_te = len(train_idx), len(test_idx)
    print(f"using {n_tr} train + {n_te} test molecules")

    # union order: train first, then test -> positions into D
    union_idx = np.concatenate([train_idx, test_idx])
    train_pos = np.arange(n_tr)
    test_pos = np.arange(n_tr, n_tr + n_te)

    # --- distance matrix (load cache or compute) ---
    if args.dist_cache and os.path.exists(args.dist_cache):
        D = np.load(args.dist_cache)
        assert D.shape[0] == len(union_idx), "cache size mismatch; delete it"
        print(f"loaded distance matrix from {args.dist_cache}")
    else:
        from reference_energy import load_dataset

        dataset = load_dataset(args.src)
        print("computing Option-4 representations...")
        reps = [
            distance_profile_representation(dataset.get_atoms(int(i)), args.quantiles)
            for i in union_idx
        ]
        print(f"computing {len(reps)*(len(reps)-1)//2:,} Wasserstein distances...")
        D = build_distance_matrix(reps)
        if args.dist_cache:
            np.save(args.dist_cache, D)
            print(f"saved distance matrix -> {args.dist_cache}")

    y_train, y_test = r_train, r_test
    signal_var = float(np.var(y_train))
    noise = (args.noise_frac * float(np.std(y_train))) ** 2
    print(
        f"signal var = {signal_var:.3f},  noise var = {noise:.4f} eV^2,  "
        f"target sd = {np.std(y_train):.3f} eV"
    )

    # --- dense Matern baseline (accuracy ceiling) ---
    ell = float(np.median(D[np.triu_indices(len(D), 1)]))
    bmean, bvar, bmineig = gp_numpy(
        D, train_pos, test_pos, y_train, lambda r: matern52(r, ell), signal_var, noise
    )
    bmae, brmse, bcov, bz2 = metrics(bmean, bvar, y_test, noise)
    print(
        f"\n[dense Matern baseline]  MAE={bmae:.3f}  RMSE={brmse:.3f}  "
        f"cov95={bcov:.2f}  z^2={bz2:.2f}  (min eig {bmineig:.1e})"
    )

    # --- Wendland sweep over support radii ---
    off = D[np.triu_indices(len(D), 1)]
    rhos = [(q, float(np.percentile(off, q))) for q in (50, 60, 75, 90, 95, 99)]
    print("\n--- Wendland sweep (fixed radius) ---")
    print(
        f"{'pct':>4} {'rho':>8} {'density':>9} {'MAE':>8} {'RMSE':>8} "
        f"{'cov95':>7} {'z^2':>7} {'min_eig':>10}"
    )
    rows = []
    for q, rho in rhos:
        dens = density(D, rho)
        if args.backend == "numpy":
            mean, var, mineig = gp_numpy(
                D,
                train_pos,
                test_pos,
                y_train,
                lambda r: wendland(r, rho),
                signal_var,
                noise,
            )
        else:
            mean, var = gp_gpcam(
                D, train_pos, test_pos, y_train, rho, signal_var, noise
            )
            mineig = float("nan")
        mae, rmse, cov, z2 = metrics(mean, var, y_test, noise)
        rows.append((q, rho, dens, mae, rmse, cov, z2, mineig))
        print(
            f"{q:>4} {rho:>8.3f} {dens:>9.4f} {mae:>8.3f} {rmse:>8.3f} "
            f"{cov:>7.2f} {z2:>7.2f} {mineig:>10.1e}"
        )

    # --- plot MAE vs density, with the dense baseline as the ceiling ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./graphs/png", exist_ok=True)
    dens_arr = [r[2] for r in rows]
    mae_arr = [r[3] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(dens_arr, mae_arr, "o-", label="Wendland (compact)")
    plt.axhline(bmae, ls="--", color="gray", label=f"dense Matern ceiling ({bmae:.2f})")
    for q, _, d, m, *_ in rows:
        plt.annotate(f"{q}%", (d, m), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("covariance-matrix density (fraction of pairs kept)")
    plt.ylabel("held-out MAE (eV)")
    plt.title("Density vs accuracy: how sparse can we be and stay accurate?")
    plt.legend()
    plt.tight_layout()
    fn = f"./graphs/png/{ts}_density_vs_accuracy_{args.backend}.png"
    plt.savefig(fn, dpi=130)
    print(f"\nsaved -> {fn}")


if __name__ == "__main__":
    main()
