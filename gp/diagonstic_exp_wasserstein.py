"""
diagnose_exp_wasserstein.py
===========================

Diagnostic for the "exponential kernel over Wasserstein distance" stopgap.

The exponential (Laplacian) kernel  k(W) = sigma_f^2 * exp(-W / ell)  is positive
definite *iff* the distance W is of negative type. That holds for W1 on the line
and 1-D W2, but is NOT guaranteed for optimal transport over distributions in R^d
(d >= 2) -- which is the regime the Option-4 outer transport actually lives in. So
"proven PSD" is true for the toy case and an open empirical question for our data.

This script answers two *separate* questions on a precomputed subsample:

  (A) IS-IT-VALID  -- Is exp(-W/ell) actually PD on OUR W, across lengthscales?
                      (eigenvalue check; same kind you ran for Wendland/Matern)

  (B) IS-THERE-SIGNAL -- Does the Wasserstein metric carry the energy signal AT ALL?
                      Measured with KERNEL-FREE readouts, so the answer does not
                      depend on whether any kernel happens to be PD. This is the
                      thing that is currently confounded with the PD failure.

Decoupling (A) from (B) is the point. The four outcomes map to four decisions:

  PD + signal      -> exp kernel is a usable stopgap; sliced-W for the scalable build
  indefinite + signal -> metric is good, kernel is the problem; need a PD-by-construction
                         metric (sliced-Wasserstein). An exp kernel will NOT save you.
  PD + no signal   -> you'd get a valid GP that predicts nothing. Reconsider descriptor.
  indefinite + no signal -> Wasserstein over Option-4 is a dead end for energy. Pivot.

Inputs (from the existing pipeline)
-----------------------------------
  --dist  W.npy   : (N, N) symmetric Wasserstein distance matrix, zero diagonal
                    (output of option4_wasserstein.py)
  --energy y.npy  : (N,) per-element-referenced total energies in eV
                    (output of reference_energy.py)

Self-test
---------
  python diagnose_exp_wasserstein.py --demo
      Builds a synthetic dataset with a Euclidean W (so exp kernel IS PD) and a
      genuine smooth energy signal. The diagnostic should report PD + strong signal.
      To see the null, rerun with --demo --shuffle-energy : same geometry, energies
      randomly permuted, so the readout should report PD + NO signal. This confirms
      the readout actually discriminates signal from its absence before you trust it
      on real data.
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- #
# Data loading / synthetic self-test
# ----------------------------------------------------------------------------- #
def load_data(dist_path, energy_path):
    W = np.load(dist_path)
    y = np.load(energy_path).astype(float).ravel()
    if W.shape[0] != W.shape[1]:
        sys.exit(f"Distance matrix must be square; got {W.shape}.")
    if W.shape[0] != y.shape[0]:
        sys.exit(f"Size mismatch: W is {W.shape[0]}x{W.shape[0]}, y has {y.shape[0]}.")
    # Symmetrize defensively and zero the diagonal.
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W, y


def make_demo(n=400, dim=5, noise=0.15, shuffle_energy=False, seed=0):
    """Euclidean W (exp kernel provably PD) + a smooth latent-coordinate energy.

    Returns W, y plus an 'expectation' string describing what the diagnostic
    SHOULD report, so the self-test is interpretable.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    # Euclidean pairwise distances -> exp(-W/ell) is PD by Schoenberg.
    diff = X[:, None, :] - X[None, :, :]
    W = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(W, 0.0)
    # Smooth function of the latent coordinates = genuine signal in the metric.
    w = rng.normal(size=dim)
    y = np.tanh(X @ w) + 0.3 * (X[:, 0] ** 2) + noise * rng.normal(size=n)
    expectation = "PD (Euclidean) + STRONG signal"
    if shuffle_energy:
        y = rng.permutation(y)
        expectation = "PD (Euclidean) + NO signal (energies permuted)"
    return W, y, expectation


# ----------------------------------------------------------------------------- #
# (A) Positive-definiteness diagnostic
# ----------------------------------------------------------------------------- #
def lengthscale_grid(W, factors=(0.25, 0.5, 1.0, 2.0, 4.0)):
    iu = np.triu_indices_from(W, k=1)
    med = np.median(W[iu])
    return med, [f * med for f in factors]


def pd_diagnostic(W, lengthscales):
    """For each ell, build the unit-variance exp kernel and inspect its spectrum.

    The exp kernel has unit diagonal, so scaling by sigma_f^2 scales every
    eigenvalue equally and never changes the SIGN. The scale-free measure of
    indefiniteness is therefore lambda_min / lambda_max -- comparable across
    kernels regardless of signal variance (unlike a raw lambda_min, which is why
    the Wendland '-4 to -10 vs sigma^2~36' numbers need the ratio to compare).
    """
    rows = []
    for ell in lengthscales:
        K = np.exp(-W / ell)
        evals = np.linalg.eigvalsh(K)  # ascending, symmetric
        lmin, lmax = float(evals[0]), float(evals[-1])
        neg = evals[evals < 0]
        rows.append(
            {
                "ell": ell,
                "lambda_min": lmin,
                "lambda_max": lmax,
                "ratio": lmin / lmax if lmax > 0 else np.nan,
                "n_negative": int(neg.size),
                "neg_mass": float(-neg.sum()),  # total negative eigen-mass
                "trace": float(evals.sum()),
            }
        )
    return rows


def classify_pd(rows, tol=1e-8):
    """A kernel counts as 'effectively PD' if its worst relative eigenvalue across
    the lengthscale grid is within numerical tolerance of zero."""
    worst = min(r["ratio"] for r in rows)
    if worst >= -tol:
        return "PD", worst
    if worst >= -1e-2:
        return "NEAR-PD", worst
    return "INDEFINITE", worst


# ----------------------------------------------------------------------------- #
# (B1) Kernel-free signal readout: distance vs energy gap
# ----------------------------------------------------------------------------- #
def spearman(a, b):
    """Spearman rho without a scipy dependency: Pearson on ranks."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def distance_energy_readout(W, y, max_pairs=2_000_000, n_bins=12, seed=0):
    """If the metric carries signal, molecule pairs that are CLOSE in W should
    have SMALL energy gaps. Quantify with rank correlation between W and |dE|,
    plus a binned mean-|dE| curve that should rise with distance."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(W, k=1)
    wv = W[iu]
    dE = np.abs(y[iu[0]] - y[iu[1]])

    if wv.size > max_pairs:
        sel = rng.choice(wv.size, size=max_pairs, replace=False)
        wv_s, dE_s = wv[sel], dE[sel]
    else:
        wv_s, dE_s = wv, dE

    rho = spearman(wv_s, dE_s)

    # Binned curve over distance quantiles (robust to W's scale/shape).
    edges = np.quantile(wv, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(wv, edges) - 1, 0, n_bins - 1)
    bin_mid = 0.5 * (edges[:-1] + edges[1:])
    bin_mean = np.array(
        [dE[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)]
    )
    # Lift = how much the energy gap grows from nearest to farthest bin.
    valid = ~np.isnan(bin_mean)
    lift = (
        (bin_mean[valid][-1] / bin_mean[valid][0]) if bin_mean[valid][0] > 0 else np.nan
    )
    return {"spearman": rho, "bin_mid": bin_mid, "bin_mean": bin_mean, "lift": lift}


# ----------------------------------------------------------------------------- #
# (B2) Kernel-free signal readout: k-NN regression in the metric
# ----------------------------------------------------------------------------- #
def knn_readout(W, y, ks=(1, 2, 3, 5, 10, 20)):
    """Leave-one-out: predict each energy from the mean of its k nearest
    neighbours in W. Skill score = 1 - MAE_knn / MAE_baseline, where the baseline
    is the leave-one-out global mean. Skill > 0 means the metric beats 'predict
    the average'; this is the single most decisive signal number."""
    n = W.shape[0]
    order = np.argsort(W, axis=1)  # nearest first; col 0 is self (W=0)
    neighbors = order[:, 1 : max(ks) + 1]  # drop self

    # Leave-one-out mean baseline.
    total = y.sum()
    loo_mean = (total - y) / (n - 1)
    mae_base = np.mean(np.abs(y - loo_mean))

    results = []
    for k in ks:
        pred = y[neighbors[:, :k]].mean(axis=1)
        mae = float(np.mean(np.abs(y - pred)))
        results.append({"k": k, "mae": mae, "skill": 1.0 - mae / mae_base})
    best = max(results, key=lambda r: r["skill"])
    return {"baseline_mae": float(mae_base), "per_k": results, "best": best}


# ----------------------------------------------------------------------------- #
# (B3) Optional GP confirmation: exp-kernel leave-one-out (mean + calibration)
# ----------------------------------------------------------------------------- #
def gp_loo_readout(W, y, ell, noise_frac=0.05, psd_tol=1e-8):
    """Exact GP leave-one-out via the Rasmussen-Williams shortcut (Eq. 5.12).

    Reports LOO MAE and 2-sigma credible-interval coverage. If exp(-W/ell) is
    indefinite we add the *minimal* jitter to reach PSD and FLAG it -- a large
    required jitter is itself the diagnostic (it means the patch drowns the
    signal, exactly the failure mode from the Wendland attempts)."""
    yc = y - y.mean()
    sigf2 = float(np.var(yc))
    K = sigf2 * np.exp(-W / ell)

    evals = np.linalg.eigvalsh(K)
    lmin = float(evals[0])
    jitter = 0.0
    flag = "clean (kernel PD at this lengthscale)"
    if lmin < -psd_tol:
        jitter = -lmin + 1e-6 * sigf2
        frac = jitter / sigf2
        flag = (
            f"jitter={jitter:.3g} added to reach PSD "
            f"(= {frac:.1%} of signal var -- "
            f"{'tolerable' if frac < 0.2 else 'LARGE: patch likely drowns signal'})"
        )

    noise = noise_frac**2 * sigf2
    Ky = K + (jitter + noise) * np.eye(W.shape[0])
    try:
        Kinv = np.linalg.inv(Ky)
    except np.linalg.LinAlgError:
        return {"ok": False, "flag": "inversion failed"}

    alpha = Kinv @ yc
    dinv = np.diag(Kinv)
    mu_loo = yc - alpha / dinv
    var_loo = 1.0 / dinv
    resid = yc - mu_loo
    mae = float(np.mean(np.abs(resid)))
    sd = np.sqrt(np.clip(var_loo, 0, None))
    coverage = float(np.mean(np.abs(resid) <= 2.0 * sd))  # target ~0.95
    return {
        "ok": True,
        "ell": ell,
        "mae": mae,
        "coverage": coverage,
        "noise_frac": noise_frac,
        "flag": flag,
    }


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def make_plots(rows, de, knn, outdir):
    # 1. lambda_min/lambda_max vs lengthscale
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot([r["ell"] for r in rows], [r["ratio"] for r in rows], "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("lengthscale  ell  (units of median W)")
    ax.set_ylabel(r"$\lambda_{\min} / \lambda_{\max}$")
    ax.set_title(
        "Positive-definiteness of exp(-W/ell)\n(negative => not a valid kernel)"
    )
    ax.set_xscale("log")
    fig.tight_layout()
    p1 = f"{outdir}/diag_eigen_vs_lengthscale.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 2. binned distance-energy curve
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(de["bin_mid"], de["bin_mean"], "o-")
    ax.set_xlabel("Wasserstein distance (bin midpoint)")
    ax.set_ylabel("mean |energy gap|  (eV)")
    ax.set_title(
        f"Distance vs energy gap\nSpearman rho = {de['spearman']:.3f}  "
        f"(rising curve => signal)"
    )
    fig.tight_layout()
    p2 = f"{outdir}/diag_distance_energy.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # 3. kNN skill vs k
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ks = [r["k"] for r in knn["per_k"]]
    sk = [r["skill"] for r in knn["per_k"]]
    ax.plot(ks, sk, "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("k (neighbours in W)")
    ax.set_ylabel("skill = 1 - MAE/MAE_baseline")
    ax.set_title("k-NN-in-metric signal\n(skill > 0 => metric beats the mean)")
    fig.tight_layout()
    p3 = f"{outdir}/diag_knn_skill.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    return [p1, p2, p3]


# ----------------------------------------------------------------------------- #
# Readout
# ----------------------------------------------------------------------------- #
def print_readout(
    n, median_w, rows, pd_verdict, worst_ratio, de, knn, gp, expectation=None
):
    L = []
    add = L.append
    add("=" * 72)
    add("  EXP-KERNEL / WASSERSTEIN DIAGNOSTIC READOUT")
    add("=" * 72)
    add(f"  subsample size        : N = {n}  ({n*(n-1)//2:,} pairs)")
    add(f"  median pairwise W     : {median_w:.4g}")
    if expectation:
        add(f"  [self-test] expected  : {expectation}")
    add("")
    add("-" * 72)
    add("  (A) IS THE EXP KERNEL PD ON OUR DATA?")
    add("-" * 72)
    add(
        f"  {'ell/median':>11} {'lambda_min':>12} {'lambda_max':>12} "
        f"{'min/max':>10} {'n_neg':>7} {'neg_mass':>10}"
    )
    for r in rows:
        add(
            f"  {r['ell']/median_w:>11.2f} {r['lambda_min']:>12.4g} "
            f"{r['lambda_max']:>12.4g} {r['ratio']:>10.3e} "
            f"{r['n_negative']:>7d} {r['neg_mass']:>10.4g}"
        )
    add("")
    add(
        f"  VERDICT (A): {pd_verdict}   (worst lambda_min/lambda_max = {worst_ratio:.2e})"
    )
    add("")
    add("-" * 72)
    add("  (B) DOES THE METRIC CARRY THE ENERGY SIGNAL?  [kernel-free]")
    add("-" * 72)
    add(f"  B1  distance vs |energy gap|")
    add(
        f"        Spearman rho        : {de['spearman']:.3f}   "
        f"(>0 wanted; closer pairs => smaller gap)"
    )
    add(f"        gap lift near->far  : {de['lift']:.2f}x")
    add(f"  B2  k-NN-in-metric (leave-one-out)")
    add(f"        baseline MAE (mean) : {knn['baseline_mae']:.4g} eV")
    for r in knn["per_k"]:
        add(f"        k={r['k']:<3d}  MAE={r['mae']:.4g} eV   skill={r['skill']:+.3f}")
    b = knn["best"]
    add(f"        best: k={b['k']}  skill={b['skill']:+.3f}")
    if gp and gp.get("ok"):
        add(f"  B3  exp-kernel GP (leave-one-out, ell=median)")
        add(f"        LOO MAE             : {gp['mae']:.4g} eV")
        add(f"        2-sigma coverage    : {gp['coverage']:.2f}   (target ~0.95)")
        add(f"        PD handling         : {gp['flag']}")
    add("")

    # Verdicts -> decision
    signal = de["spearman"] > 0.1 and knn["best"]["skill"] > 0.05
    sig_txt = "SIGNAL PRESENT" if signal else "NO USABLE SIGNAL"
    add("-" * 72)
    add(f"  VERDICT (B): {sig_txt}")
    add("-" * 72)
    add("")
    add("  DECISION:")
    if pd_verdict in ("PD", "NEAR-PD") and signal:
        add("    -> Exp kernel is a usable STOPGAP on this data, and the metric")
        add("       tracks energy. Use it for interim results, and pursue")
        add("       sliced-Wasserstein for the COMPACTLY-SUPPORTED (scalable) build,")
        add("       since exp(-W/ell) is dense and gives no gp2Scale advantage.")
    elif pd_verdict == "INDEFINITE" and signal:
        add("    -> The METRIC is good but the kernel is the problem. An exp kernel")
        add("       won't rescue this on our data. Move to a PD-by-construction")
        add("       metric: sliced-Wasserstein (1-D W2 = L2 of quantile functions,")
        add("       which your sorted Option-4 profiles already are).")
    elif signal is False and pd_verdict in ("PD", "NEAR-PD"):
        add("    -> You'd get a VALID GP that predicts little. The bottleneck is the")
        add("       descriptor/metric, not the kernel. Reconsider the representation")
        add("       before more kernel engineering.")
    else:
        add("    -> Wasserstein over Option-4 looks like a dead end for energy here:")
        add("       neither valid nor predictive. Pivot the descriptor (e.g. WL graph")
        add("       distance) and re-run the sparsity + signal diagnostics first.")
    add("")
    add("  NOTE: exp(-W/ell) has full (dense) support -- this is a SIGNAL/VALIDITY")
    add("  probe, not a candidate for the production gp2Scale kernel.")
    add("=" * 72)
    print("\n".join(L))


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dist", help="path to (N,N) Wasserstein distance matrix .npy")
    ap.add_argument("--energy", help="path to (N,) referenced energies .npy (eV)")
    ap.add_argument("--outdir", default=".", help="directory for output PNGs")
    ap.add_argument("--no-gp", action="store_true", help="skip the GP LOO readout (B3)")
    ap.add_argument("--demo", action="store_true", help="run synthetic self-test")
    ap.add_argument(
        "--shuffle-energy",
        action="store_true",
        help="(demo) permute energies to test the null / no-signal case",
    )
    ap.add_argument("--demo-n", type=int, default=400, help="(demo) sample size")
    args = ap.parse_args()

    expectation = None
    if args.demo:
        W, y, expectation = make_demo(n=args.demo_n, shuffle_energy=args.shuffle_energy)
    else:
        if not (args.dist and args.energy):
            ap.error("provide --dist and --energy, or use --demo")
        W, y = load_data(args.dist, args.energy)

    n = W.shape[0]
    median_w, grid = lengthscale_grid(W)

    rows = pd_diagnostic(W, grid)
    pd_verdict, worst_ratio = classify_pd(rows)
    de = distance_energy_readout(W, y)
    knn = knn_readout(W, y)
    gp = None if args.no_gp else gp_loo_readout(W, y, ell=median_w)

    paths = make_plots(rows, de, knn, args.outdir)
    print_readout(n, median_w, rows, pd_verdict, worst_ratio, de, knn, gp, expectation)
    print("\nplots written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
