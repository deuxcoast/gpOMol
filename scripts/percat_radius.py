"""Per-category support radius, scored on UQ. The cheap discrete version of gp2Scale's
non-stationary rho(x).

WHY THIS AND NOT A GLOBAL RADIUS. Median 10-nearest-neighbour distance WITHIN a chemical
family varies 2.3x across families (0.88 in biomolecules to 2.02 in spice), so one global
cutoff is simultaneously too wide for the dense families and too tight for the sparse
ones. gp2Scale's prescribed answer is a spatially varying rho(x) sampled by MCMC under a
sparsity-preferring prior; a per-category radius is its discrete approximation and needs
no training at all, which makes it the right thing to try first. If a sensibly-set
per-category radius does nothing, MCMC over the same parameterisation will not rescue it.

IT IS ALSO PSD FOR FREE. `kernel.py` zeroes cross-category covariance unconditionally, so
after sort_by_category the Gram is exactly block diagonal. A block diagonal matrix whose
blocks are each PSD is PSD, and each block here is an ordinary Wendland Gram on its own
points with its own radius. No Paciorek-Schervish prefactor, no validity question.

THE CONFOUND THIS CONTROLS FOR. Per-category radii change the OVERALL density as well as
its allocation, and RADIUS_CALIBRATION.md established that density alone moves both R^2
and sigma*'s ranking skill (both rise monotonically with it). So an unmatched per-category
arm that looks better may only be denser. `--match-density` rescales every category radius
by one global constant until realised train density matches the global-radius arm, which
isolates the SHAPE of the allocation from its scale. Run both.

SELF-CONTAINED ON PURPOSE. `--arm global` rebuilds the baseline (one global cutoff at the
200-neighbour target) through the SAME code path as the per-category arms, and embeddings
are cached under `cache/` rather than a temp directory. An earlier version of this
comparison relied on result files and a featurisation cache that lived only in a scratch
directory; the scratch was cleaned and the baseline had to be rebuilt from nothing. Arms
that are supposed to be paired should not depend on state that can disappear
independently of the repository.

    python scripts/percat_radius.py --arm global   --seeds 42 7 123
    python scripts/percat_radius.py --arm percat   --seeds 42 7 123
    python scripts/percat_radius.py --arm matched  --seeds 42 7 123
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl_gp2scale.kernel import AdditiveWendlandKernel, ChannelSpec  # noqa: E402

N, NTE, JITTER, K = 20_000, 800, 4.30, 200
KERNEL_CHAN = ("wl", "geom")
MEAN_CHAN = ("wl", "geom", "strain")
SOLVE = dict(linalg_mode="sparseCG", solve_maxiter=2000, solve_tol=1e-6,
             batch_size=10_000, compute_device="cpu", device="cpu")
# production featurisation (wl_config_for_200k / RADIUS_CALIBRATION.md)
FEAT = dict(depth=3, bond_mult=1.1, geom_top_k=10, pls=10, scaling="pareto",
            min_count=2, perceiver="ase", geom_r_max=6.0,
            geom_channels=("rdf", "angle", "torsion", "elec"),
            charge_key="lowdin_charges", diag_sample=5000)


def load_or_build_embeddings(ds, tr, te, cat_tr, cat_te, seed, emb_dir,
                             n_total=N, workers=4, scheduler_file=None):
    """Cached per-seed PLS embeddings. `target_neighbors=0` because every arm derives its
    OWN cutoffs downstream; caching a cutoff here would invite reusing the wrong one."""
    from wl_gp2scale.pipeline import build_channels
    os.makedirs(emb_dir, exist_ok=True)
    path = os.path.join(emb_dir, f"emb_{n_total}_s{seed}.npz")
    names = sorted(set(MEAN_CHAN) | set(KERNEL_CHAN))
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"[emb] reused {path}")
        return {n: {"Ztr": d[f"{n}_tr"], "Zte": d[f"{n}_te"]} for n in names}
    t0 = time.time()
    if scheduler_file:
        from wl_gp2scale.pipeline import connect_dask
        cl = connect_dask(scheduler_file, n_workers=workers)
    else:
        from distributed import Client
        cl = Client(n_workers=workers, threads_per_worker=1, processes=False,
                    dashboard_address=None, silence_logs=50)
    try:
        emb = build_channels([ds.atoms[i] for i in tr], ds.y[tr], cat_tr,
                             [ds.atoms[i] for i in te], cat_te, set(names),
                             client=cl, target_neighbors=0, cutoff_pct=25.0, **FEAT)
    finally:
        cl.close()
    np.savez(path, **{f"{n}_{s}": emb[n][f"Z{s}"] for n in emb for s in ("tr", "te")})
    print(f"[emb] built + cached {path} in {time.time() - t0:.0f}s")
    return emb


class PerCategoryAdditiveKernel(AdditiveWendlandKernel):
    """AdditiveWendlandKernel with a PER-CATEGORY cutoff per channel.

    The only difference from the parent is the line that forms ``t``: the scalar
    ``ch.cutoff`` becomes a lookup ``cut[c][cats1]``, broadcast down the rows. Using the
    ROW category is exact rather than approximate, because every entry that survives the
    cross-category mask has cats1 == cats2, and every entry that does not is zeroed a few
    lines later regardless of which radius produced it.
    """

    cutoffs_by_cat: np.ndarray = None      # (C, n_categories)

    def __call__(self, x1, x2, hps):
        import torch
        C = len(self.channels)
        x1, x2 = np.asarray(x1), np.asarray(x2)
        n1, n2 = x1.shape[0], x2.shape[0]
        cats1 = x1[:, self._total_dim].astype(np.int64)
        cats2 = x2[:, self._total_dim].astype(np.int64)
        if np.intersect1d(np.unique(cats1), np.unique(cats2)).size == 0:
            return np.zeros((n1, n2), dtype=np.float64)

        dev, td = self._device, self._tdtype
        K_ = torch.zeros((n1, n2), dtype=td, device=dev)
        for c, ch in enumerate(self.channels):
            sv = float(hps[c])
            cut_row = torch.as_tensor(self.cutoffs_by_cat[c][cats1], dtype=td,
                                      device=dev).view(-1, 1)
            a = torch.as_tensor(x1[:, ch.start:ch.stop], dtype=td, device=dev)
            b = torch.as_tensor(x2[:, ch.start:ch.stop], dtype=td, device=dev)
            D = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
            t = torch.clamp(D / cut_row, 0.0, 1.0)
            Kc = self._psi(t, ch)
            Kc = torch.where(t < 1.0, Kc, torch.zeros_like(Kc))
            if sv != 1.0:
                Kc = Kc * sv
            K_ = K_ + Kc

        ca = torch.as_tensor(cats1, device=dev).view(-1, 1)
        cb = torch.as_tensor(cats2, device=dev).view(1, -1)
        K_ = torch.where(ca == cb, K_, torch.zeros_like(K_))
        return K_.to("cpu").double().numpy()


def per_category_cutoffs(Ztr, Zte, cat_tr, cat_te, ncat, target=K):
    """Radius per category: the median over that category's test points of the distance
    to its target-th nearest SAME-CATEGORY train point. Identical policy to the global
    `cutoff_for_neighbors`, applied within each category instead of pooled.

    Small categories cannot supply `target` same-category neighbours (at 20k: spice ~180
    train rows, orbnet_denali ~198, rgd ~302). Rather than drop them, as the global policy
    silently does, K is reduced to n_c - 1 and the reduction is reported."""
    cuts, short = np.zeros(ncat), {}
    for c in range(ncat):
        mtr, mte = cat_tr == c, cat_te == c
        n_c = int(mtr.sum())
        if n_c < 2 or mte.sum() == 0:
            cuts[c] = np.nan
            continue
        k = min(int(target), n_c - 1)
        if k < target:
            short[c] = (n_c, k)
        D = cdist(Zte[mte], Ztr[mtr])
        cuts[c] = float(np.median(np.partition(D, k - 1, axis=1)[:, k - 1]))
    if np.isnan(cuts).any():                 # empty categories: harmless, never queried
        cuts[np.isnan(cuts)] = np.nanmedian(cuts)
    return cuts, short


def train_density(Ztr, cat_tr, cuts_by_cat, chan_slices, sample=5000, seed=0):
    """Realised fraction of non-zero entries, over a sampled band of rows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Ztr), size=min(sample, len(Ztr)), replace=False)
    nz = np.zeros((len(idx), len(Ztr)), bool)
    for c, (s, e) in enumerate(chan_slices):
        D = cdist(Ztr[idx, s:e], Ztr[:, s:e])
        nz |= D < cuts_by_cat[c][cat_tr[idx]][:, None]
    nz &= cat_tr[idx][:, None] == cat_tr[None, :]
    return float(nz.mean()), float(np.median(nz.sum(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--emb-dir", default="cache", help="cache for emb_{N}_s{seed}.npz")
    ap.add_argument("--n", type=int, default=N, help="dataset size (200000 at scale)")
    ap.add_argument("--nte", type=int, default=NTE,
                    help="test points carrying a posterior VARIANCE -- one linear solve "
                         "each, the dominant cost at scale")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None,
                    help="connect to an existing dask cluster instead of a local client")
    ap.add_argument("--device", default="cpu", help="OUR kernel's torch device")
    ap.add_argument("--diag-sample", type=int, default=5000,
                    help="rows in the driver-side density band; lower to ~1000 at 200k")
    ap.add_argument("--mean", default="M1", choices=["M1", "M4"],
                    help="prior mean rung. M1 = linear on the embedding; M4 adds the "
                         "size x embedding interaction. M4 is stronger (+0.09 R^2) but "
                         "leaves the kernel far less to do (+0.022 vs +0.040) and had "
                         "the WORST UQ gap of any rung, so which one production should "
                         "use is an open question, not a default.")
    ap.add_argument("--out", default="cache/percat")
    ap.add_argument("--arm", default="percat", choices=["global", "percat", "matched"],
                    help="global = one cutoff at the 200-neighbour target (the "
                         "baseline); percat = per-category radii at the same target; "
                         "matched = per-category radii rescaled by one constant so "
                         "realised density equals the global arm's, which isolates the "
                         "allocation SHAPE from its overall density")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from distributed import Client
    from fvgp.gp_lin_alg import _resolve_krylov_mode
    from wl_gp2scale.cutoff import cutoff_for_neighbors
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean
    from wl_gp2scale.pipeline import (build_gp, predict, release_gp, sort_by_category,
                                      with_category_tag, SolveWarningCounter)
    assert _resolve_krylov_mode(None) == "single", "fvgp block CG is broken"

    ds = get_data(src="train_4M", n=a.n, seed=0)
    idx = np.arange(len(ds.atoms))
    ncat = len(ds.category_names)
    names = np.array(ds.category_names)

    for seed in a.seeds:
        tag = a.arm + ("" if a.mean == "M1" else "-M4")
        out = os.path.join(a.out, f"{tag}_s{seed}.npz")
        if os.path.exists(out):
            print(f"[percat] {out} exists, skipping")
            continue

        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        y_tr, y_te = ds.y[tr], ds.y[te]
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
        size_te = np.array([len(ds.atoms[i]) for i in te], float)
        size_tr = np.array([len(ds.atoms[i]) for i in tr], float)
        sel = np.random.default_rng(0).choice(len(y_te), a.nte, replace=False)

        emb = load_or_build_embeddings(ds, tr, te, cat_tr, cat_te, seed, a.emb_dir,
                                       n_total=a.n, workers=a.workers,
                                       scheduler_file=a.scheduler_file)
        Zm_tr = np.hstack([emb[n]["Ztr"] for n in MEAN_CHAN])
        Zm_te = np.hstack([emb[n]["Zte"] for n in MEAN_CHAN])
        Zk_tr = np.hstack([emb[n]["Ztr"] for n in KERNEL_CHAN])
        Zk_te = np.hstack([emb[n]["Zte"] for n in KERNEL_CHAN])

        # M1: linear on the embedding, NO size interaction (matches ladder rung M1)
        msz = size_tr if a.mean == "M4" else None
        tsz = size_te if a.mean == "M4" else None
        mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=msz)
        r_tr = y_tr - mean.predict(Zm_tr, size=msz)
        prior_var = float(np.var(r_tr))

        specs, slices, off = [], [], 0
        cuts_cat, shorts = [], {}
        cuts_glob = []
        for n in KERNEL_CHAN:
            dd = emb[n]["Ztr"].shape[1]
            cg = cutoff_for_neighbors(emb[n]["Ztr"], emb[n]["Zte"], K, dim=dd,
                                      data_id_tr=cat_tr, data_id_te=cat_te, sample=a.diag_sample)
            cc, sh = per_category_cutoffs(emb[n]["Ztr"], emb[n]["Zte"], cat_tr, cat_te,
                                          ncat)
            cuts_glob.append(cg); cuts_cat.append(cc); shorts.update(sh)
            specs.append(ChannelSpec(off, off + dd, float(cg)))
            slices.append((off, off + dd)); off += dd
        cuts_cat = np.array(cuts_cat)

        cuts_global_bcast = np.array([[c] * ncat for c in cuts_glob])
        dens_g, nbr_g = train_density(Zk_tr, cat_tr, cuts_global_bcast, slices)
        if a.arm == "global":                 # baseline: one cutoff everywhere
            cuts_cat = cuts_global_bcast
        dens_p, nbr_p = train_density(Zk_tr, cat_tr, cuts_cat, slices)
        scale = 1.0
        if a.arm == "matched":
            lo, hi = 0.3, 3.0
            for _ in range(24):                     # bisect on one global multiplier
                mid = 0.5 * (lo + hi)
                dm, _ = train_density(Zk_tr, cat_tr, cuts_cat * mid, slices)
                if dm > dens_g:
                    hi = mid
                else:
                    lo = mid
            scale = 0.5 * (lo + hi)
            cuts_cat = cuts_cat * scale
            dens_p, nbr_p = train_density(Zk_tr, cat_tr, cuts_cat, slices)
        print(f"[percat] seed {seed} arm={tag}: global density {dens_g:.4e} "
              f"(median {nbr_g:.0f} nbrs) | this arm {dens_p:.4e} "
              f"(median {nbr_p:.0f})" + (f"  [rescaled x{scale:.3f}]"
                                         if a.arm == "matched" else ""))
        for c in range(ncat):
            r = cuts_cat[:, c] / np.array(cuts_glob)
            print(f"    {names[c]:<18} rho/rho_global = "
                  f"{np.array2string(np.round(r, 3))}"
                  + (f"   (only {shorts[c][0]} train rows, K={shorts[c][1]})"
                     if c in shorts else ""))

        kern = PerCategoryAdditiveKernel(channels=specs, use_category_tag=True,
                                         device=a.device, dtype="float64")
        kern.cutoffs_by_cat = cuts_cat

        C = len(specs)
        if a.scheduler_file:
            from wl_gp2scale.pipeline import connect_dask
            cl = connect_dask(a.scheduler_file, n_workers=a.workers)
        else:
            cl = Client(n_workers=a.workers, threads_per_worker=1, processes=False,
                        dashboard_address=None, silence_logs=50)
        t0 = time.time()
        try:
            Xs, ys, _ = sort_by_category(with_category_tag(Zk_tr, cat_tr), r_tr)
            with SolveWarningCounter() as wc:
                gp, _ = build_gp(Xs, ys, None, None, cl, kernel_override=kern,
                                 signal_var=[prior_var / C] * C, jitter=JITTER, **SOLVE)
                m_gp, v_gp = predict(gp, with_category_tag(Zk_te[sel], cat_te[sel]),
                                     batch=200, variance=True)
            del gp
            release_gp(cl)
        finally:
            cl.close()
        wc.report(n_evals=a.nte, label=f"{tag}-s{seed}")

        mu = mean.predict(Zm_te, size=tsz)[sel] + m_gp
        err = y_te[sel] - mu
        # baselines, formed exactly as the ladder formed them for rung M1
        Dn = cdist(Zk_te[sel], Zk_tr)
        Dn[cat_te[sel][:, None] != cat_tr[None, :]] = np.inf
        dnn = Dn.min(axis=1)
        D_tr, D_te = np.eye(ncat)[cat_tr], np.eye(ncat)[cat_te]
        V_tr = np.hstack([np.ones((len(tr), 1)), np.log(size_tr)[:, None], D_tr[:, 1:]])
        V_te = np.hstack([np.ones((len(te), 1)), np.log(size_te)[:, None], D_te[:, 1:]])
        bv, *_ = np.linalg.lstsq(V_tr, np.log(r_tr ** 2 + 1e-8), rcond=None)
        sd_cheap = np.exp(0.5 * (V_te @ bv))[sel]

        np.savez(out, seed=seed, rung=a.mean, arm=tag, y=y_te[sel], mu=mu, err=err,
                 v_gp=v_gp, dnn=dnn, sd_cheap=sd_cheap, cat=cat_te[sel],
                 cuts_cat=cuts_cat, cuts_glob=np.array(cuts_glob), scale=scale,
                 dens_global=dens_g, dens_percat=dens_p, prior_var=prior_var,
                 ols_r2=r2_score(y_te[sel], mean.predict(Zm_te, size=tsz)[sel]),
                 n_warn=int(wc.total), cat_names=names)
        print(f"[percat] seed {seed} {tag}: GP_R2={r2_score(y_te[sel], mu):.4f} "
              f"({time.time() - t0:.0f}s) -> {out}\n")


if __name__ == "__main__":
    main()
