"""The radius grid, scored on CALIBRATION instead of R^2.

WHY. Every support-radius decision in this project was made on held-out R^2. The
calibration study (CALIBRATION.md Step 3) found the GP's sigma* beaten on every
uncertainty metric by the distance to the nearest same-category training point, and
proposed a mechanism: at a 200-neighbour radius the median test point has 458 in-support
neighbours, so sigma* is nearly constant and cannot rank anything, while dist-to-NN
measures local sparsity directly -- and local sparsity is what the outliers are made of.
If that mechanism is right, a TIGHTER radius should make sigma* responsive and it should
close on (or overtake) dist-to-NN, at some cost in R^2.

THE MEASUREMENT. For each split seed, featurise once, then for each target neighbour
count K in the grid recompute ONLY the per-channel cutoff and rebuild the GP. The
embeddings, the prior mean, the test subset and the two GP-free baselines are therefore
IDENTICAL across K within a seed, so the K arms are paired and the only thing moving is
the radius. dist-to-NN does not depend on K at all, which makes
`Spearman(sigma*) - Spearman(dist-to-NN)` a paired per-seed difference -- the statistic
[[channel-placement-criterion]] argued for after WL turned out to have std 0.076 at 20k.

Stage B writes one .npz per (seed, K) arm before scoring anything, so a crash 80 minutes
in does not cost the arms that already solved. Scoring lives in score.py and reads only
those files.

fvgp traps this run is exposed to, and what is done about each (see [[fvgp-gotchas]]):
  * sparse_krylov_mode="block" silently returns the PRIOR variance for whole blocks of
    test points. We never pass `args`, so fvgp's default "single" applies -- asserted at
    startup rather than assumed.
  * posterior_covariance(add_noise=False) is the default and pipeline.predict does not
    override it, so v_gp is the LATENT variance. The observation variance is formed here
    as v_gp + var(mean params) + jitter, and both are scored.
  * an unconverged Krylov solve is a warning seen ONCE per process. SolveWarningCounter
    plus a FINITE solve_maxiter makes it visible and bounded.
  * one live gp2Scale GP per dask client, and the third build on one client has died
    before. Fresh Client per arm.
"""
import os, sys, time, json
import numpy as np
from sklearn.model_selection import train_test_split

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")
os.makedirs(ARMS, exist_ok=True)

# ---------------------------------------------------------------- frozen configuration
N            = 20_000
DATA_SEED    = 0            # which 20k molecules (frozen across the whole project)
SEEDS        = (42, 7, 123)  # train/test split seeds -- the stability axis
# 60/200/500 is the grid the plan asked for. 15 and 30 are added because the plan's
# hypothesis is "TIGHTER makes sigma* responsive", and at K=60 the median test point
# still keeps ~55 in-support neighbours -- if the trend runs the right way at 60 the
# answer would sit outside the grid. The tight arms are also the cheapest (0.1 s/point
# against 0.8 at K=500), so the resolution is nearly free.
KGRID        = (15, 30, 60, 200, 500)
NTE          = int(os.environ.get("NTE", 800))   # test points carrying a posterior
                            # VARIANCE -- one linear solve EACH, so this is the arm's cost
SMOKE        = os.environ.get("SMOKE", "")       # e.g. SMOKE=1 NTE=40 for a path check
# 4.30 matches the Step 3 headline table, so (seed 42, K=200) is a reproduction check on
# this whole pipeline rather than a fresh number nobody can compare. It is NOT the
# production 0.78; the additive kernel is jitter-insensitive for R^2 but not for
# calibration, so the jitter is held FIXED across arms and its level is a caveat, not a
# variable. jitter_check.py re-runs the extremes of the grid at 0.78.
JITTER       = 4.30
KERNEL_CHAN  = ("wl", "geom")               # blocks in the additive kernel
MEAN_CHAN    = ("wl", "geom", "strain")     # channels the linear prior mean sees
FEAT = dict(depth=3, bond_mult=1.1, geom_top_k=10, pls=10, scaling="pareto",
            min_count=2, perceiver="ase", geom_r_max=6.0,
            geom_channels=("rdf", "angle", "torsion", "elec"),
            charge_key="lowdin_charges", diag_sample=5000)
SOLVE = dict(linalg_mode="sparseCG", solve_maxiter=2000, solve_tol=1e-6,
             batch_size=10_000, compute_device="cpu", device="cpu")


def _assert_single_krylov():
    """fvgp's block Krylov mode returns the prior variance for entire blocks of test
    points with zero warnings -- the defect CALIBRATION.md documents. We never enable it,
    but 'we never enable it' is exactly what was believed while it was on."""
    from fvgp.gp_lin_alg import _resolve_krylov_mode
    mode = _resolve_krylov_mode(None)
    assert mode == "single", f"fvgp krylov mode is {mode!r}; block CG is BROKEN"
    print(f"[guard] fvgp sparse krylov mode = {mode!r}")


def featurise(seed):
    """Embeddings for one split seed, cached. target_neighbors is passed as 0 here on
    purpose: build_channels' only use of it is to derive each channel's cutoff, and this
    experiment derives a DIFFERENT cutoff per arm. Caching a cutoff would invite reusing
    the wrong one."""
    from wl_gp2scale.data import get_data
    from wl_gp2scale.pipeline import build_channels
    from distributed import Client

    cache = os.path.join(SCR, f"emb_{N}_s{seed}.npz")
    ds = get_data(src="train_4M", n=N, seed=DATA_SEED)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    meta = dict(
        tr=tr, te=te, y_tr=ds.y[tr], y_te=ds.y[te],
        cat_tr=ds.data_id[tr], cat_te=ds.data_id[te],
        size_tr=np.array([len(ds.atoms[i]) for i in tr], float),
        size_te=np.array([len(ds.atoms[i]) for i in te], float),
        cat_names=np.array(ds.category_names),
    )
    names = sorted(set(KERNEL_CHAN) | set(MEAN_CHAN))
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        emb = {n: {"Ztr": d[f"{n}_tr"], "Zte": d[f"{n}_te"]} for n in names}
        print(f"[feat] seed {seed}: reused {cache}")
        return emb, meta

    t0 = time.time()
    cl = Client(n_workers=4, threads_per_worker=1, processes=False,
                dashboard_address=None, silence_logs=50)
    try:
        emb = build_channels(
            [ds.atoms[i] for i in tr], meta["y_tr"], meta["cat_tr"],
            [ds.atoms[i] for i in te], meta["cat_te"], set(names), client=cl,
            target_neighbors=0, cutoff_pct=25.0, **FEAT)
    finally:
        cl.close()
    np.savez(cache, **{f"{n}_{s}": emb[n][f"Z{s}"] for n in emb for s in ("tr", "te")})
    print(f"[feat] seed {seed}: built + cached in {time.time()-t0:.0f}s")
    return emb, meta


def run_arm(seed, K, emb, meta, sel):
    """One (seed, K) arm: recompute the cutoffs at K, build the additive GP, take the
    posterior variance on the NTE selected test points, and write the raw arrays."""
    from distributed import Client
    from wl_gp2scale.cutoff import cutoff_for_neighbors, sparsity_report
    from wl_gp2scale.reduce import LinearEmbeddingMean
    from wl_gp2scale.pipeline import (build_gp, predict, release_gp, sort_by_category,
                                      with_category_tag, SolveWarningCounter)

    out = os.path.join(ARMS, f"arm_s{seed}_k{K}{'_smoke' if SMOKE else ''}.npz")
    if os.path.exists(out):
        print(f"[arm] seed={seed} K={K}: {out} exists, skipping")
        return

    y_tr, y_te = meta["y_tr"], meta["y_te"]
    cat_tr, cat_te = meta["cat_tr"], meta["cat_te"]
    size_tr, size_te = meta["size_tr"], meta["size_te"]

    # ---- prior mean: identical across K by construction (no cutoff anywhere in it)
    Zm_tr = np.hstack([emb[n]["Ztr"] for n in MEAN_CHAN])
    Zm_te = np.hstack([emb[n]["Zte"] for n in MEAN_CHAN])
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=size_tr)
    r_tr = y_tr - mean.predict(Zm_tr, size=size_tr)
    prior_var = float(np.var(r_tr))

    # ---- cutoffs AT THIS K, and how many test points the K-th neighbour even exists for
    cuts, short = {}, {}
    for n in KERNEL_CHAN:
        cuts[n] = cutoff_for_neighbors(
            emb[n]["Ztr"], emb[n]["Zte"], K, dim=emb[n]["Ztr"].shape[1],
            data_id_tr=cat_tr, data_id_te=cat_te, sample=FEAT["diag_sample"])
    # cutoff_for_neighbors masks cross-category distances to inf and then DROPS the
    # non-finite k-th distances, so a category with fewer than K train rows contributes
    # nothing to the median that sets the radius. Record which, and how much of the test
    # set that is -- at K=500 it is three categories.
    counts = np.bincount(cat_tr, minlength=len(meta["cat_names"]))
    short_cats = np.flatnonzero(counts <= K)
    short["cats"] = short_cats
    short["frac_test"] = float(np.isin(cat_te, short_cats).mean())

    # ---- the additive GP on the residual
    Ztr = np.hstack([emb[n]["Ztr"] for n in KERNEL_CHAN])
    Zte_s = np.hstack([emb[n]["Zte"] for n in KERNEL_CHAN])[sel]
    specs, off = [], 0
    for n in KERNEL_CHAN:
        d = emb[n]["Ztr"].shape[1]
        specs.append((off, off + d, float(cuts[n])))
        off += d
    for n in KERNEL_CHAN:
        sparsity_report(emb[n]["Ztr"], cuts[n], dim=emb[n]["Ztr"].shape[1],
                        data_id=cat_tr, sample=FEAT["diag_sample"])

    C = len(KERNEL_CHAN)
    cl = Client(n_workers=4, threads_per_worker=1, processes=False,
                dashboard_address=None, silence_logs=50)
    t0 = time.time()
    try:
        Xs, ys, _ = sort_by_category(with_category_tag(Ztr, cat_tr), r_tr)
        with SolveWarningCounter() as wc:
            gp, _ = build_gp(Xs, ys, None, None, cl, channels=specs,
                             signal_var=[prior_var / C] * C, jitter=JITTER, **SOLVE)
            t_build = time.time() - t0
            print(f"[arm] seed={seed} K={K} build={t_build:.1f}s; "
                  f"{NTE} variance solves starting")
            m_gp, v_gp = predict(gp, with_category_tag(Zte_s, cat_te[sel]),
                                 batch=200, variance=True, verbose=True)
        del gp
        release_gp(cl)
    finally:
        cl.close()
    wc.report(n_evals=NTE, label=f"s{seed}k{K}")   # returns the FRACTION; keep the count
    n_warn = int(wc.total)
    t_arm = time.time() - t0

    # ---- assemble the prediction and everything the scorer needs
    mu = mean.predict(Zm_te, size=size_te)[sel] + m_gp
    v_mean = mean.predict_var(Zm_te, size=size_te)[sel]
    err = y_te[sel] - mu

    # in-support same-category train neighbours per TEST point, per channel. This is the
    # mechanism the whole experiment is about: if sigma* is flat because coverage is
    # total, this is the number that has to come down as K does.
    from scipy.spatial.distance import cdist
    nbr = {}
    for n in KERNEL_CHAN:
        D = cdist(emb[n]["Zte"][sel], emb[n]["Ztr"])
        D[cat_te[sel][:, None] != cat_tr[None, :]] = np.inf
        nbr[n] = (D < cuts[n]).sum(axis=1)

    np.savez(out, seed=seed, K=K, jitter=JITTER, sel=sel,
             mu=mu, err=err, y=y_te[sel], cat=cat_te[sel],
             v_gp=v_gp, v_mean=v_mean, m_gp=m_gp,
             cutoffs=np.array([cuts[n] for n in KERNEL_CHAN]),
             chan=np.array(KERNEL_CHAN),
             nbr=np.stack([nbr[n] for n in KERNEL_CHAN]),
             short_cats=short["cats"], short_frac_test=short["frac_test"],
             prior_var=prior_var, n_warn=n_warn, t_arm=t_arm,
             cat_names=meta["cat_names"])
    print(f"[arm] seed={seed} K={K} DONE in {t_arm:.0f}s  cutoffs="
          f"{[round(cuts[n],4) for n in KERNEL_CHAN]}  warnings={n_warn}  -> {out}")


def baselines(seed, emb, meta, sel):
    """The two GP-free sigma candidates. Neither depends on K, so they are computed once
    per seed and shared by every arm -- which is what makes the K comparison paired."""
    out = os.path.join(ARMS, f"base_s{seed}.npz")
    if os.path.exists(out):
        print(f"[base] seed {seed}: exists")
        return
    from scipy.spatial.distance import cdist
    y_tr, cat_tr, cat_te = meta["y_tr"], meta["cat_tr"], meta["cat_te"]
    size_tr, size_te = meta["size_tr"], meta["size_te"]
    tr, te = meta["tr"], meta["te"]
    ncat = len(meta["cat_names"])

    Zm_tr = np.hstack([emb[n]["Ztr"] for n in MEAN_CHAN])
    from wl_gp2scale.reduce import LinearEmbeddingMean
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=size_tr)
    r_tr = y_tr - mean.predict(Zm_tr, size=size_tr)

    # sigma from log n + category, fitted on the TRAIN residual (known at test time)
    D_tr, D_te = np.eye(ncat)[cat_tr], np.eye(ncat)[cat_te]
    V_tr = np.hstack([np.ones((len(tr), 1)), np.log(size_tr)[:, None], D_tr[:, 1:]])
    V_te = np.hstack([np.ones((len(te), 1)), np.log(size_te)[:, None], D_te[:, 1:]])
    bv, *_ = np.linalg.lstsq(V_tr, np.log(r_tr ** 2 + 1e-8), rcond=None)
    sd_cheap = np.exp(0.5 * (V_te @ bv))[sel]

    # distance to the nearest same-category train point, on the KERNEL embedding
    Ztr = np.hstack([emb[n]["Ztr"] for n in KERNEL_CHAN])
    Zte_s = np.hstack([emb[n]["Zte"] for n in KERNEL_CHAN])[sel]
    Dn = cdist(Zte_s, Ztr)
    Dn[cat_te[sel][:, None] != cat_tr[None, :]] = np.inf
    dnn = Dn.min(axis=1)
    np.savez(out, seed=seed, sel=sel, sd_cheap=sd_cheap, dnn=dnn)
    print(f"[base] seed {seed}: written")


def main():
    _assert_single_krylov()
    seeds = [int(s) for s in sys.argv[1:]] or list(SEEDS)
    print(f"[run] seeds={seeds} K={KGRID} NTE={NTE} jitter={JITTER} "
          f"kernel={KERNEL_CHAN} mean={MEAN_CHAN}")
    for seed in seeds:
        emb, meta = featurise(seed)
        # same positional draw every seed; the molecules differ because the split does
        sel = np.random.default_rng(0).choice(len(meta["y_te"]), NTE, replace=False)
        baselines(seed, emb, meta, sel)
        for K in KGRID:
            run_arm(seed, K, emb, meta, sel)
    print("[run] all arms complete")


if __name__ == "__main__":
    main()
