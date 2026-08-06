"""Multiplicative kernel: k = k_wl * k_geom, against the additive k_wl + k_geom.

THE MECHANISM BEING TESTED, and it is about UQ rather than accuracy. Under the ADDITIVE
kernel a molecule counts as covered if it has neighbours close in graph topology OR in
geometry, so the support is a UNION. Under the PRODUCT it needs neighbours close in BOTH,
so the support is an INTERSECTION. Coverage is exactly what the posterior variance
measures, so a stricter notion of coverage is a real mechanism for making sigma* respond
to genuine novelty -- the first such mechanism in this project that is not just "move the
radius", which RADIUS_CALIBRATION.md showed does not work.

Chemically it is also the more defensible statement: two molecules with the same bond graph
but different conformations should not be called similar, and the additive kernel currently
says they are.

PSD comes free from the Schur product theorem (the elementwise product of PSD matrices is
PSD), so no new theory is needed. Signal variance enters once, not per channel: the
diagonal is sv * psi(0) * psi(0) = sv, against the additive kernel's sv_wl + sv_geom.

THE RADIUS POLICY IS THE WHOLE DIFFICULTY. Intersection support is drastically sparser
than union at the same per-channel radii -- if the channels were independent, 3.9e-2 would
become ~4.9e-4. Worse, holding each channel at a fixed neighbour count makes the product's
density fall as ~1/N^2, so the matrix would go essentially diagonal at 200k. Both channel
radii therefore have to be widened until the PRODUCT hits the target. That is a knob we
control, which is precisely what the WL subtree kernel lacked when its density turned out
to be scale-invariant.

ARMS. `matched` bisects one common multiplier on both channel radii until realised train
density equals the additive baseline's, so the product is compared at EQUAL COST and the
only thing differing is the shape of the support. The unrescaled ("natural") density is
reported without fitting, since at ~8 neighbours per row it would be dominated by mean
reversion and would not test the mechanism.

BASELINE: cache/percat/global_s{seed}.npz -- additive kernel, one global radius at the
200-neighbour target, same M1 mean, same jitter, same 800 test points, same seeds.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wl_gp2scale.kernel import ChannelSpec, _wendland32  # noqa: E402
from percat_radius import load_or_build_embeddings  # noqa: E402

N, NTE, JITTER, K = 20_000, 800, 4.30, 200
KERNEL_CHAN = ("wl", "geom")
MEAN_CHAN = ("wl", "geom", "strain")
SOLVE = dict(linalg_mode="sparseCG", solve_maxiter=2000, solve_tol=1e-6,
             batch_size=10_000, compute_device="cpu", device="cpu")


class ProductWendlandKernel:
    """k(x,x') = sv * prod_c psi(d_c / rho_c), zeroed across categories.

    Mirrors AdditiveWendlandKernel's structure and device handling; the only change is
    that channel blocks MULTIPLY instead of summing, and there is a single signal
    variance because the diagonal is a product of ones rather than a sum."""

    def __init__(self, channels, cats=None, device="cpu", dtype="float64",
                 cutoffs_by_cat=None):
        self.channels = [c if isinstance(c, ChannelSpec) else ChannelSpec(*c)
                         for c in channels]
        self.device, self.dtype = device, dtype
        self._total_dim = self.channels[-1].stop
        # (C, n_categories) or None. When set, each channel's radius is looked up by the
        # ROW's category, which is exact rather than approximate: every entry surviving
        # the cross-category mask has cats1 == cats2, and every entry that does not is
        # zeroed below regardless of which radius produced it.
        self.cutoffs_by_cat = cutoffs_by_cat

    def __call__(self, x1, x2, hps):
        import torch
        td = torch.float64 if self.dtype == "float64" else torch.float32
        x1, x2 = np.asarray(x1), np.asarray(x2)
        n1, n2 = x1.shape[0], x2.shape[0]
        cats1 = x1[:, self._total_dim].astype(np.int64)
        cats2 = x2[:, self._total_dim].astype(np.int64)
        if np.intersect1d(np.unique(cats1), np.unique(cats2)).size == 0:
            return np.zeros((n1, n2), dtype=np.float64)

        dev = self.device
        Kp = torch.ones((n1, n2), dtype=td, device=dev)
        for ci, ch in enumerate(self.channels):
            a = torch.as_tensor(x1[:, ch.start:ch.stop], dtype=td, device=dev)
            b = torch.as_tensor(x2[:, ch.start:ch.stop], dtype=td, device=dev)
            D = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
            if self.cutoffs_by_cat is None:
                cut = float(ch.cutoff)
            else:
                cut = torch.as_tensor(self.cutoffs_by_cat[ci][cats1], dtype=td,
                                      device=dev).view(-1, 1)
            t = torch.clamp(D / cut, 0.0, 1.0)
            Kc = torch.where(t < 1.0, _wendland32(t), torch.zeros_like(t))
            Kp = Kp * Kc                       # INTERSECTION support
        Kp = Kp * float(hps[0])
        ca = torch.as_tensor(cats1, device=dev).view(-1, 1)
        cb = torch.as_tensor(cats2, device=dev).view(1, -1)
        Kp = torch.where(ca == cb, Kp, torch.zeros_like(Kp))
        return Kp.to("cpu").double().numpy()


def distance_bands(Ztr, slices, sample=5000, seed=0):
    """Per-channel dense (sample x n_train) distance blocks, computed ONCE.

    The radius bisection evaluates density ~26 times; recomputing these blocks each pass
    is invisible at 20k (0.6 GB, seconds) and fatal at 200k (6.4 GB per channel, minutes
    per pass). Compute once, re-threshold cheaply."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Ztr), size=min(sample, len(Ztr)), replace=False)
    return idx, [cdist(Ztr[idx, s:e], Ztr[:, s:e]) for s, e in slices]


def support_stats(idx, Ds, cat_tr, cuts, mode):
    """Realised density, median neighbours per row and isolated fraction, from
    precomputed bands. mode='union' is the additive kernel's support, 'inter' the
    product's. ``cuts`` is (C,) per channel or (C, n_categories) for the per-category
    variant, where the ROW's category selects the radius."""
    cuts = np.asarray(cuts)
    nz = None
    for c, D in enumerate(Ds):
        thr = cuts[c] if cuts.ndim == 1 else cuts[c][cat_tr[idx]][:, None]
        near = D < thr
        nz = near if nz is None else (nz | near if mode == "union" else nz & near)
    nz &= cat_tr[idx][:, None] == cat_tr[None, :]
    deg = nz.sum(axis=1) - 1                    # drop the self entry
    return float(nz.mean()), float(np.median(deg)), float(np.mean(deg <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--emb-dir", default="cache")
    ap.add_argument("--mean", default="M1", choices=["M1", "M4"],
                    help="prior mean rung. M1 = linear on the embedding; M4 adds the "
                         "size x embedding interaction. M4 is stronger (+0.09 R^2) but "
                         "leaves the kernel far less to do (+0.022 vs +0.040) and had "
                         "the WORST UQ gap of any rung, so which one production should "
                         "use is an open question, not a default.")
    ap.add_argument("--out", default="cache/product")
    ap.add_argument("--n", type=int, default=N, help="dataset size (200000 at scale)")
    ap.add_argument("--nte", type=int, default=NTE,
                    help="test points carrying a posterior VARIANCE. One linear solve "
                         "EACH against the full system, so this is the dominant cost at "
                         "scale: budget it, do not raise it casually at 200k")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None,
                    help="connect to an existing dask cluster instead of a local client")
    ap.add_argument("--device", default="cpu", help="OUR kernel's torch device")
    ap.add_argument("--diag-sample", type=int, default=5000,
                    help="rows in the driver-side density band. Each channel holds a "
                         "dense (sample x n_train) float64 block ON THE DRIVER: 0.6 GB at "
                         "20k but 6.4 GB at 200k, per channel. Lower to ~1000 at scale")
    ap.add_argument("--percat", action="store_true",
                    help="per-category radii INSIDE the product, i.e. combine the two "
                         "levers that each moved the UQ gap on their own (+0.076 for "
                         "intersection support, +0.024 for per-category allocation). "
                         "They act on different things -- one changes WHAT COUNTS as a "
                         "neighbour, the other WHERE support is allocated -- so whether "
                         "they add, overlap or fight is an empirical question.")
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

    for seed in a.seeds:
        tag = ("prodpercat" if a.percat else "product") + ("" if a.mean == "M1" else "-M4")
        out = os.path.join(a.out, f"{tag}_s{seed}.npz")
        if os.path.exists(out):
            print(f"[prod] {out} exists"); continue

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

        msz = size_tr if a.mean == "M4" else None
        tsz = size_te if a.mean == "M4" else None
        mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=msz)
        r_tr = y_tr - mean.predict(Zm_tr, size=msz)
        prior_var = float(np.var(r_tr))

        base_cuts, slices, off = [], [], 0
        for n in KERNEL_CHAN:
            dd = emb[n]["Ztr"].shape[1]
            base_cuts.append(cutoff_for_neighbors(
                emb[n]["Ztr"], emb[n]["Zte"], K, dim=dd, data_id_tr=cat_tr,
                data_id_te=cat_te, sample=5000))
            slices.append((off, off + dd)); off += dd
        base_cuts = np.array(base_cuts)
        glob_cuts = base_cuts.copy()
        if a.percat:
            from percat_radius import per_category_cutoffs
            pc = [per_category_cutoffs(emb[n]["Ztr"], emb[n]["Zte"], cat_tr, cat_te,
                                       ncat)[0] for n in KERNEL_CHAN]
            base_cuts = np.array(pc)                 # (C, ncat)

        bidx, bands = distance_bands(Zk_tr, slices, sample=a.diag_sample)
        d_add, nb_add, iso_add = support_stats(bidx, bands, cat_tr, glob_cuts, "union")
        d_nat, nb_nat, iso_nat = support_stats(bidx, bands, cat_tr, base_cuts, "inter")
        print(f"[prod] seed {seed}: additive(union) density {d_add:.4e} "
              f"(median {nb_add:.0f} nbrs, {iso_add:.1%} isolated)")
        print(f"[prod] seed {seed}: product(inter) UNRESCALED density {d_nat:.4e} "
              f"(median {nb_nat:.0f} nbrs, {iso_nat:.1%} isolated) "
              f"-> {d_add / max(d_nat, 1e-12):.0f}x sparser than additive")

        lo, hi = 1.0, 8.0                        # widen both channels until densities match
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            dm, _, _ = support_stats(bidx, bands, cat_tr, base_cuts * mid, "inter")
            if dm > d_add:
                hi = mid
            else:
                lo = mid
        s = 0.5 * (lo + hi)
        cuts = base_cuts * s
        d_p, nb_p, iso_p = support_stats(bidx, bands, cat_tr, cuts, "inter")
        del bands
        print(f"[prod] seed {seed}: matched at scale x{s:.3f} -> density {d_p:.4e} "
              f"(median {nb_p:.0f} nbrs, {iso_p:.1%} isolated)")

        if a.percat:
            specs = [ChannelSpec(sl[0], sl[1], float(np.median(c)))
                     for sl, c in zip(slices, cuts)]
            kern = ProductWendlandKernel(specs, device=a.device, dtype="float64",
                                         cutoffs_by_cat=cuts)
        else:
            specs = [ChannelSpec(sl[0], sl[1], float(c))
                     for sl, c in zip(slices, cuts)]
            kern = ProductWendlandKernel(specs, device=a.device, dtype="float64")

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
                                 signal_var=[prior_var], jitter=JITTER, **SOLVE)
                m_gp, v_gp = predict(gp, with_category_tag(Zk_te[sel], cat_te[sel]),
                                     batch=200, variance=True)
            del gp
            release_gp(cl)
        finally:
            cl.close()
        wc.report(n_evals=a.nte, label=f"prod-s{seed}")

        mu = mean.predict(Zm_te, size=tsz)[sel] + m_gp
        err = y_te[sel] - mu
        Dn = cdist(Zk_te[sel], Zk_tr)
        Dn[cat_te[sel][:, None] != cat_tr[None, :]] = np.inf
        dnn = Dn.min(axis=1)
        D_tr, D_te = np.eye(ncat)[cat_tr], np.eye(ncat)[cat_te]
        V_tr = np.hstack([np.ones((len(tr), 1)), np.log(size_tr)[:, None], D_tr[:, 1:]])
        V_te = np.hstack([np.ones((len(te), 1)), np.log(size_te)[:, None], D_te[:, 1:]])
        bv, *_ = np.linalg.lstsq(V_tr, np.log(r_tr ** 2 + 1e-8), rcond=None)
        sd_cheap = np.exp(0.5 * (V_te @ bv))[sel]

        np.savez(out, seed=seed, arm=tag, rung=a.mean, y=y_te[sel], mu=mu, err=err,
                 v_gp=v_gp, dnn=dnn, sd_cheap=sd_cheap, cat=cat_te[sel],
                 cuts_cat=(cuts if cuts.ndim == 2 else
                           np.repeat(cuts[:, None], ncat, axis=1)),
                 cuts_glob=glob_cuts, scale=s,
                 dens_global=d_add, dens_percat=d_p, dens_natural=d_nat,
                 med_nbrs=nb_p, isolated=iso_p, prior_var=prior_var,
                 ols_r2=r2_score(y_te[sel], mean.predict(Zm_te, size=tsz)[sel]),
                 n_warn=int(wc.total), cat_names=np.array(ds.category_names))
        print(f"[prod] seed {seed} {tag}: GP_R2={r2_score(y_te[sel], mu):.4f} "
              f"({time.time() - t0:.0f}s) -> {out}\n")


if __name__ == "__main__":
    main()
