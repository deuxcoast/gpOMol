#!/usr/bin/env python
"""
additive_screen.py  (descriptor_eval, two-channel additive kernel screen)
=========================================================================
Does a 3D geometry+charge channel lift the WL-graph ceiling? Screens the
additive Wendland

    k(x, x') = sv * [ w_wl * psi(||z_wl - z'_wl|| / c_wl)
                    + w_g  * psi(||z_g  - z'_g || / c_g ) ]

against the SAME held-out split and shared residual target as gp_parity, where
z_wl and z_g are the SEPARATE PLS embeddings of the WL-graph descriptor
(features.WLFeaturizer) and the geometry descriptor (geometry.GeometryFeaturizer).
Each channel keeps its own compact-support radius (c_wl, c_g), set per channel
from a percentile of that channel's train distances, so the two live on their own
natural scales (exactly the block-sparse structure wl_gp2scale would use). The
signal variance is FROZEN (sv = Var(y_tr), split w_wl + w_g = 1) per Marcus's
"get it working, train later"; the diagonal stays at Var(y) so it is comparable
to the single-channel screens.

Reports three numbers on one split -- WL-only, GEOM-only, WL+GEOM -- so the LIFT
(does additive beat the WL 0.239@10k / 0.56@200k ceiling, and by how much?) is
read off directly. If it lifts here, promote k_geom to a second block-sparse
Wendland in wl_gp2scale and rerun the scaling ladder.

Run from inside descriptor_eval/. Reuses gp_parity for data, featurizers, PLS,
and the single-channel GP so there is one source of truth.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import gp_parity as gpp


# ------------------------------ additive kernel -----------------------------


def make_additive_kernel(d_wl, c_wl, c_geom, w_wl, w_geom, metric="euclidean"):
    """Factory for the two-block additive Wendland. The GP input is the
    concatenation [z_wl | z_geom]; the kernel splits it at d_wl, applies a
    compact-support Wendland psi(r)=(1-r)^4 (4r+1) to each block with its own
    cutoff, and returns their weighted sum scaled by the signal variance hps[0]."""

    def kernel(x1, x2, hps):
        sv = hps[0]
        r1 = np.clip(cdist(x1[:, :d_wl], x2[:, :d_wl], metric=metric) / c_wl, 0.0, 1.0)
        r2 = np.clip(cdist(x1[:, d_wl:], x2[:, d_wl:], metric=metric) / c_geom, 0.0, 1.0)
        k = w_wl * (1.0 - r1) ** 4 * (4.0 * r1 + 1.0)
        k += w_geom * (1.0 - r2) ** 4 * (4.0 * r2 + 1.0)
        return sv * k

    return kernel


def make_wendland(cutoff, metric="euclidean"):
    """Single-block compact-support Wendland kernel (for the single-channel
    baselines), same psi and hps[0]=signal_var convention as the additive one."""

    def kernel(x1, x2, hps):
        r = np.clip(cdist(x1, x2, metric=metric) / cutoff, 0.0, 1.0)
        return hps[0] * (1.0 - r) ** 4 * (4.0 * r + 1.0)

    return kernel


# ------------------------------ linear prior mean ---------------------------


def fit_linear_mean(X_tr, y_tr):
    """OLS (with intercept) of y on the embedding -- the descriptor_eval analogue of
    wl_gp2scale's `--prior-mean linear`. The GP then models the RESIDUAL y - m(z) and
    the mean is added back at predict, so under-covered test points fall back to the
    OLS fit instead of reverting to a constant (the mean-reversion that caps the
    constant-mean GP at R^2~0.1). Returns the coefficient vector [intercept, w...]."""
    A = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return beta


def apply_linear_mean(beta, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ beta


def run_channel(name, X_tr, y_tr, X_te, y_te, kernel, jitter, prior_mean):
    """Fit -> predict one channel (or the additive pair) with an optional linear
    prior mean. With prior_mean='linear' the GP models the OLS residual on the same
    embedding X and the mean is added back; 'none' reproduces the constant-mean GP
    (gpcam's default) on y directly. signal_var = Var(residual)."""
    if prior_mean == "linear":
        beta = fit_linear_mean(X_tr, y_tr)
        m_tr, m_te = apply_linear_mean(beta, X_tr), apply_linear_mean(beta, X_te)
        ols_r2 = float(r2_score(y_te, m_te))
    else:
        m_tr = np.zeros(len(y_tr))
        m_te = np.zeros(len(y_te))
        ols_r2 = None
    r_tr = y_tr - m_tr
    sv = float(np.var(r_tr))
    mean_r, var, jit = fit_additive(X_tr, r_tr, X_te, kernel, sv, jitter=jitter)
    mean = mean_r + m_te
    r2 = float(r2_score(y_te, mean))
    rmse = float(np.sqrt(np.mean((mean - y_te) ** 2)))
    tag = f"  (OLS-mean R^2={ols_r2:.3f})" if ols_r2 is not None else ""
    print(f"[additive] {name}  R^2 = {r2:.3f}  RMSE = {rmse:.4g}  "
          f"(sv={sv:.3g}, jitter {jit:g}){tag}")
    return dict(r2=r2, rmse=rmse, mean=mean, var=var, jitter=jit, ols_r2=ols_r2)


def _predict_additive(X_tr, y_tr, X_te, kernel, signal_var, jit):
    """One additive-Wendland GP fit + predict at a fixed diagonal noise. Raises
    np.linalg.LinAlgError if the covariance is not PD at this jitter."""
    from gpcam import GPOptimizer

    gp = GPOptimizer(
        x_data=X_tr,
        y_data=y_tr,
        init_hyperparameters=np.array([signal_var]),
        kernel_function=kernel,
        noise_variances=jit * np.ones(len(y_tr)),
    )
    mean = gpp._first(gp.posterior_mean(X_te), ["f(x)", "m(x)"])
    var = gpp._first(
        gp.posterior_covariance(X_te, variance_only=True),
        ["v(x)", "S(x)", "variance"],
    )
    return mean, np.maximum(var, 0.0)


def fit_additive(X_tr, y_tr, X_te, kernel, signal_var, jitter=None, jitter_exps=range(-6, 0)):
    """Fit the frozen-hyperparameter additive Wendland GP. With `jitter` set, one
    fixed-noise shot; otherwise the escalating jitter loop gp_parity uses. Returns
    (mean, var, jitter)."""
    if jitter is not None:
        print(f"[additive][jitter] single-shot fixed jitter = {jitter:g}")
        mean, var = _predict_additive(X_tr, y_tr, X_te, kernel, signal_var, jitter)
        return mean, var, jitter
    for e in jitter_exps:
        jit = 10.0**e
        try:
            print(f"[additive][jitter] trying diagonal noise = {jit:.0e} ...")
            mean, var = _predict_additive(X_tr, y_tr, X_te, kernel, signal_var, jit)
            print(f"[additive][jitter] SUCCESS at {jit:.0e}")
            return mean, var, jit
        except np.linalg.LinAlgError:
            print(f"[additive][jitter] Cholesky failed at {jit:.0e}; escalating x10")
    raise RuntimeError("additive covariance not PD-regularizable up to 1e-1")


# ------------------------------ main ----------------------------------------


def cutoff_for_neighbors(Z_tr, Z_te, target, metric):
    """Per-channel compact-support radius TUNED to a neighbour COUNT (the sparsity
    lever from the working recipe -- median ~60 in-support neighbours), not a shared
    percentile: the cutoff = median over test points of the distance to the target-th
    nearest train point, so a typical point sees ~target train neighbours. Percentile
    is the wrong knob because the WL and geometry embeddings have differently-shaped
    distance distributions."""
    D = cdist(Z_te, Z_tr, metric=metric)
    k = min(int(target), D.shape[1] - 1)
    return float(np.median(np.partition(D, k, axis=1)[:, k]))


def median_neighbors(Z_tr, Z_te, cutoff, metric):
    return float(np.median((cdist(Z_te, Z_tr, metric=metric) < cutoff).sum(axis=1)))


def _cutoff(Z, pct, metric):
    return float(np.percentile(pdist(Z, metric=metric), pct))


def main():
    ap = argparse.ArgumentParser(description="Two-channel (WL + geometry) additive Wendland screen")
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--pls-components", type=int, default=10, help="PLS dims PER channel")
    ap.add_argument("--scaling", default="pareto", choices=["standard", "pareto", "center"])
    ap.add_argument(
        "--target-neighbors",
        type=int,
        default=60,
        help="tune each channel's cutoff to this median in-support neighbour count "
        "(the recipe sweet spot); ignored if --cutoff-pct is given",
    )
    ap.add_argument(
        "--cutoff-pct",
        type=float,
        default=None,
        help="OVERRIDE: fixed percentile of each channel's train distances "
        "(bypasses --target-neighbors; the old dense knob)",
    )
    ap.add_argument(
        "--w-geom",
        type=float,
        default=0.5,
        help="additive weight on the geometry block (WL block gets 1 - w_geom)",
    )
    ap.add_argument(
        "--prior-mean",
        default="linear",
        choices=["linear", "none"],
        help="linear = OLS prior mean on the embedding (models residual, adds back; "
        "the wl_gp2scale --prior-mean linear fix for mean-reversion); none = "
        "constant-mean GP (reproduces the earlier low-R^2 regime)",
    )
    ap.add_argument("--geom-channels", default="rdf,angle,torsion,elec")
    ap.add_argument("--geom-top-k", type=int, default=6)
    ap.add_argument("--geom-r-max", type=float, default=6.0)
    ap.add_argument("--charge-key", default="lowdin_charges")
    ap.add_argument("--wl-depth", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=None, help="fixed jitter (skip escalation)")
    ap.add_argument("--subsample", type=int, default=None, help="use only first N train mols")
    a = ap.parse_args()

    metric = gpp.METRIC_ALIASES.get(a.metric.lower(), a.metric.lower())
    geom_channels = tuple(c.strip() for c in a.geom_channels.split(",") if c.strip())

    # --- data + split (identical to gp_parity) ------------------------------
    atoms, y = gpp.build_atoms()
    a_tr, a_te, y_tr, y_te = train_test_split(
        atoms, y, test_size=gpp.TEST_FRACTION, random_state=gpp.RANDOM_STATE
    )
    if a.subsample is not None and a.subsample < len(a_tr):
        a_tr, y_tr = a_tr[: a.subsample], y_tr[: a.subsample]
        print(f"[subsample] train -> {len(a_tr)} (test held fixed at {len(y_te)})")
    print(f"[split] train {len(y_tr)}  test {len(y_te)}  var(y)={np.var(y_tr):.4g}")

    # --- per-channel featurize -> scale -> PLS ------------------------------
    Xw_tr, Xw_te, fw = gpp.featurize_wl(
        a_tr, a_te, wl_depth=a.wl_depth, min_count=a.min_count, scaling=a.scaling
    )
    Zw_tr, Zw_te, ew = gpp._reduce(Xw_tr, y_tr, Xw_te, "pls", a.pls_components, 200, 300)

    Xg_tr, Xg_te, fg = gpp.featurize_geometry(
        a_tr, a_te, channels=geom_channels, top_k=a.geom_top_k,
        r_max=a.geom_r_max, charge_key=a.charge_key, scaling=a.scaling,
    )
    Zg_tr, Zg_te, eg = gpp._reduce(Xg_tr, y_tr, Xg_te, "pls", a.pls_components, 200, 300)
    print(f"[embed] WL {ew} (D={fw.n_features_})   GEOM {eg} (D={fg.n_features_})")

    results = {}
    print(f"[additive] prior_mean={a.prior_mean}")

    # --- per-channel tuned cutoffs (same for baselines + additive) ----------
    if a.cutoff_pct is not None:
        c_wl = _cutoff(Zw_tr, a.cutoff_pct, metric)
        c_geom = _cutoff(Zg_tr, a.cutoff_pct, metric)
        tune = f"pct={a.cutoff_pct}"
    else:
        c_wl = cutoff_for_neighbors(Zw_tr, Zw_te, a.target_neighbors, metric)
        c_geom = cutoff_for_neighbors(Zg_tr, Zg_te, a.target_neighbors, metric)
        tune = f"target_nbrs={a.target_neighbors}"
    nb_wl = median_neighbors(Zw_tr, Zw_te, c_wl, metric)
    nb_g = median_neighbors(Zg_tr, Zg_te, c_geom, metric)
    print(f"[additive] tuned cutoffs ({tune}): c_wl={c_wl:.4g} (median {nb_wl:.0f} nbrs)  "
          f"c_geom={c_geom:.4g} (median {nb_g:.0f} nbrs)")

    # --- single-channel baselines ------------------------------------------
    for name, Ztr, Zte, c in [
        ("WL-only", Zw_tr, Zw_te, c_wl),
        ("GEOM-only", Zg_tr, Zg_te, c_geom),
    ]:
        print(f"\n[additive] === {name} (cutoff={c:.4g}) ===")
        res = run_channel(
            name, Ztr, y_tr, Zte, y_te,
            make_wendland(c, metric), a.jitter, a.prior_mean,
        )
        results[name] = res["r2"]

    # --- additive channel ---------------------------------------------------
    print(f"\n[additive] === WL+GEOM (additive Wendland, w_geom={a.w_geom}) ===")
    X_tr = np.hstack([Zw_tr, Zg_tr])
    X_te = np.hstack([Zw_te, Zg_te])
    kernel = make_additive_kernel(
        Zw_tr.shape[1], c_wl, c_geom, 1.0 - a.w_geom, a.w_geom, metric
    )
    t0 = time.perf_counter()
    res = run_channel(
        "WL+GEOM", X_tr, y_tr, X_te, y_te, kernel, a.jitter, a.prior_mean,
    )
    results["WL+GEOM"] = res["r2"]
    print(f"[additive] (additive fit {time.perf_counter()-t0:.1f}s)")

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 56)
    print(f"{'channel':<14}{'R^2':>10}")
    for k in ("WL-only", "GEOM-only", "WL+GEOM"):
        print(f"{k:<14}{results[k]:>10.3f}")
    lift = results["WL+GEOM"] - results["WL-only"]
    print("-" * 56)
    print(f"lift of adding geometry over WL-only:  {lift:+.3f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
