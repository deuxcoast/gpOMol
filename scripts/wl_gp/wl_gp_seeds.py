"""Firm up the one positive signal: h=4 RAW COUNTS beat the radius kernel on R^2 at seed 42
by +0.0023. Is that real or one seed?

Everything else in the arm table is decided by margins far larger than seed noise (the GP's
sigma* ranking is 3x worse than the radius kernel's, at every WL configuration). This is
the only number small enough to need replication, and the comparison is PAIRED -- same
split, same 800 test points, same M4 prior mean on the same cached PLS embeddings, only
the kernel differs -- so the large between-seed spread in R^2 (0.808 at seed 42 against a
0.729 three-seed mean) is common-mode and cancels.

The radius arms already exist as arm_s{7,123}_k200.npz, so only the WL arm is re-run.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")
N, NTE, JITTER, DEPTH = 20_000, 800, 4.30, 4
MEAN_CHAN = ("wl", "geom", "strain")
SOLVE = dict(linalg_mode="sparseCG", solve_maxiter=2000, solve_tol=1e-6,
             batch_size=10_000, compute_device="cpu", device="cpu")

import importlib.util
_s = importlib.util.spec_from_file_location("wl_gp", os.path.join(SCR, "wl_gp.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
WLSubtreeKernel, build_phi = _m.WLSubtreeKernel, _m.build_phi


def run_seed(seed):
    from distributed import Client
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean
    from wl_gp2scale.pipeline import (build_gp, predict, release_gp, sort_by_category,
                                      SolveWarningCounter)
    ds = get_data(src="train_4M", n=N, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    y_tr, y_te = ds.y[tr], ds.y[te]
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    size_tr = np.array([len(ds.atoms[i]) for i in tr], float)
    size_te = np.array([len(ds.atoms[i]) for i in te], float)
    sel = np.random.default_rng(0).choice(len(y_te), NTE, replace=False)

    d = np.load(os.path.join(SCR, f"emb_20000_s{seed}.npz"), allow_pickle=True)
    Zm_tr = np.hstack([d[f"{n}_tr"] for n in MEAN_CHAN])
    Zm_te = np.hstack([d[f"{n}_te"] for n in MEAN_CHAN])
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=size_tr)
    r_tr = y_tr - mean.predict(Zm_tr, size=size_tr)
    prior_var = float(np.var(r_tr))

    Phi, _ = build_phi([ds.atoms[i] for i in tr], [ds.atoms[i] for i in te], DEPTH)
    ntr = len(tr)
    cats_all = np.concatenate([cat_tr, cat_te])
    kern = WLSubtreeKernel(Phi, cats_all, normalize=False, use_category_tag=True)
    A = Phi[:ntr]
    sv = prior_var / float(np.median(np.asarray(A.multiply(A).sum(axis=1)).ravel()) or 1.0)

    X = np.column_stack([np.arange(len(cats_all), dtype=float), cats_all])
    cl = Client(n_workers=4, threads_per_worker=1, processes=False,
                dashboard_address=None, silence_logs=50)
    t0 = time.time()
    try:
        Xs, ys, _ = sort_by_category(X[:ntr], r_tr)
        with SolveWarningCounter() as wc:
            gp, _ = build_gp(Xs, ys, None, None, cl, kernel_override=kern,
                             signal_var=[sv], jitter=JITTER, **SOLVE)
            m_gp, v_gp = predict(gp, X[ntr:][sel], batch=200, variance=True)
        del gp
        release_gp(cl)
    finally:
        cl.close()
    wc.report(n_evals=NTE, label=f"wl h4 raw s{seed}")
    mu = mean.predict(Zm_te, size=size_te)[sel] + m_gp
    ref = np.load(os.path.join(ARMS, f"arm_s{seed}_k200.npz"))
    err = ref["y"] - mu
    return (r2_score(ref["y"], mu),
            float(spearmanr(np.sqrt(np.maximum(v_gp, 0)), np.abs(err)).statistic),
            r2_score(ref["y"], ref["mu"]),
            float(spearmanr(np.sqrt(ref["v_gp"]), np.abs(ref["err"])).statistic),
            time.time() - t0)


def main():
    print(f"{'seed':>6} {'WL h=4 raw R2':>14} {'radius R2':>10} {'dR2':>9} "
          f"{'WL rho':>9} {'radius rho':>11} {'drho':>9}")
    print("-" * 76)
    rows = []
    for seed in (42, 7, 123):
        if seed == 42:                       # already measured in wl_gp.py
            w_r2, w_rho, r_r2, r_rho = 0.8102, 0.0067, 0.8079, 0.2088
        else:
            w_r2, w_rho, r_r2, r_rho, t = run_seed(seed)
        rows.append((seed, w_r2, r_r2, w_rho, r_rho))
        print(f"{seed:>6} {w_r2:>14.4f} {r_r2:>10.4f} {w_r2-r_r2:>+9.4f} "
              f"{w_rho:>+9.4f} {r_rho:>+11.4f} {w_rho-r_rho:>+9.4f}")
    d_r2 = np.array([r[1] - r[2] for r in rows])
    d_rho = np.array([r[3] - r[4] for r in rows])
    for nm, dv in (("R^2", d_r2), ("rho(sigma*,|err|)", d_rho)):
        se = dv.std(ddof=1) / np.sqrt(len(dv))
        print(f"\n  paired WL - radius, {nm}: mean {dv.mean():+.4f}  SE {se:.4f}  "
              f"({'significant' if abs(dv.mean()) > 2*se else 'NOT significant'} at 2 SE)")


if __name__ == "__main__":
    main()
