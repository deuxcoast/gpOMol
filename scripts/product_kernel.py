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
    ap.add_argument("--no-variance", action="store_true",
                    help="skip the posterior VARIANCE and report accuracy only, on the "
                         "FULL test set. Variance is one linear solve per test point "
                         "and is ~99%% of the wall time at 200k; the mean is one solve "
                         "total. Use this for scaling/accuracy rungs, and validate UQ "
                         "at 20k where the variance is affordable. Arms written this "
                         "way carry has_variance=0 and the scorer omits them from the "
                         "UQ tables instead of ranking their NaNs.")
    ap.add_argument("--kernel-chan", default="wl,geom",
                    help="channels multiplied INTO THE KERNEL, comma separated. The "
                         "default wl,geom is the pair the +0.068 result was measured "
                         "on, but those two agree about neighbourhoods 12x more than "
                         "independence -- they are one view described twice, which is "
                         "why the intersection is only ~6x sparser than the union and "
                         "caps how far the radii can be widened at matched density. "
                         "Measured lift at the K=200 operating point: geom x wl 12.2, "
                         "geom x strain 9.7, strain x wl 1.0 (independent). The prior "
                         "MEAN always keeps all of MEAN_CHAN regardless of this flag.")
    ap.add_argument("--percat", action="store_true",
                    help="per-category radii INSIDE the product, i.e. combine the two "
                         "levers that each moved the UQ gap on their own (+0.076 for "
                         "intersection support, +0.024 for per-category allocation). "
                         "They act on different things -- one changes WHAT COUNTS as a "
                         "neighbour, the other WHERE support is allocated -- so whether "
                         "they add, overlap or fight is an empirical question.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    kchan = tuple(c.strip() for c in a.kernel_chan.split(",") if c.strip())
    unknown = set(kchan) - set(MEAN_CHAN)
    if unknown:
        raise SystemExit(f"--kernel-chan: unknown channel(s) {sorted(unknown)}; "
                         f"embeddings are built for {sorted(set(MEAN_CHAN))}")
    if len(kchan) < 2:
        raise SystemExit("--kernel-chan needs >=2 channels; a product of one channel "
                         "is just that channel and the comparison is meaningless")
    # Non-default channel sets get their own arm tag, or they overwrite the wl,geom
    # results sitting in the same --out and the scorer cannot tell them apart.
    chan_tag = "" if kchan == KERNEL_CHAN else "-" + "".join(kchan)

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
        tag = (("prodpercat" if a.percat else "product") + chan_tag
               + ("" if a.mean == "M1" else "-M4"))
        out = os.path.join(a.out, f"{tag}_s{seed}.npz")
        if os.path.exists(out):
            # The filename carries the ARM and SEED but not N, so a cheap 40k shakedown
            # and the real 200k run collide in one --out dir: the 200k run would skip,
            # and the scorer would report the 40k numbers as the 200k result. Cheap to
            # detect, expensive to notice afterwards.
            prev = np.load(out, allow_pickle=True)
            n_prev = int(prev["n"]) if "n" in prev.files else None
            if n_prev is not None and n_prev != a.n:
                raise SystemExit(
                    f"[prod] ABORT: {out} already holds an n={n_prev:,} result but this "
                    f"run is n={a.n:,}. These do not belong in one --out dir -- the "
                    f"scorer cannot tell them apart. Use a different --out (e.g. "
                    f"cache/product_{a.n // 1000}k) or delete the old file."
                )
            # An accuracy-only arm and a UQ arm are DIFFERENT measurements that land on
            # the same filename, so matching n is not enough: without this, the cheap
            # --no-variance rung silently blocks the expensive variance rung that is the
            # actual experiment, and the run exits 0 having done nothing.
            v_prev = (int(prev["has_variance"]) if "has_variance" in prev.files else 1)
            if v_prev != int(not a.no_variance):
                raise SystemExit(
                    f"[prod] ABORT: {out} holds an "
                    f"{'accuracy-only (--no-variance)' if not v_prev else 'UQ'} result "
                    f"but this run is "
                    f"{'accuracy-only (--no-variance)' if a.no_variance else 'UQ'}. "
                    f"Same filename, different measurement. Use a separate --out (e.g. "
                    f"{a.out}_uq) so both survive."
                )
            print(f"[prod] {out} exists (n={n_prev if n_prev else '?'}), skipping")
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
        Zk_tr = np.hstack([emb[n]["Ztr"] for n in kchan])
        Zk_te = np.hstack([emb[n]["Zte"] for n in kchan])

        msz = size_tr if a.mean == "M4" else None
        tsz = size_te if a.mean == "M4" else None
        mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=msz)
        r_tr = y_tr - mean.predict(Zm_tr, size=msz)
        prior_var = float(np.var(r_tr))

        # THE DENSITY TARGET IS THE BASELINE'S, NOT THIS ARM'S. Every arm is compared
        # against the additive union over KERNEL_CHAN (percat_radius.py --arm global),
        # so that union is the density to match -- otherwise a wl,strain arm matches
        # itself to union(wl,strain), lands at a different density from the baseline,
        # and the comparison is confounded by exactly the thing density matching exists
        # to remove (RADIUS_CALIBRATION: R^2 and sigma*'s ranking BOTH rise with
        # density). Identical to the old behaviour when kchan == KERNEL_CHAN.
        need = tuple(dict.fromkeys(KERNEL_CHAN + kchan))       # ordered unique
        cut_by, band_slices, off = {}, {}, 0
        for n in need:
            dd = emb[n]["Ztr"].shape[1]
            # sample MUST match percat_radius.py's, not just for driver RAM (6.4 GB at
            # 200k with the old hardcoded 5000, plus a partition copy) but for the
            # EXPERIMENT: this is the global radius the additive baseline actually
            # runs at, and it sets the density target the product arm bisects onto. A
            # different sample here gives a different radius, so the two arms would no
            # longer be density-matched -- silently, since each script's own numbers
            # stay self-consistent.
            cut_by[n] = cutoff_for_neighbors(
                emb[n]["Ztr"], emb[n]["Zte"], K, dim=dd, data_id_tr=cat_tr,
                data_id_te=cat_te, sample=a.diag_sample)
            band_slices[n] = (off, off + dd); off += dd

        slices = [(0, 0)] * len(kchan)                # offsets into Zk_tr (kchan only)
        off = 0
        for i, n in enumerate(kchan):
            dd = emb[n]["Ztr"].shape[1]
            slices[i] = (off, off + dd); off += dd

        base_cuts = np.array([cut_by[n] for n in kchan])
        glob_cuts = base_cuts.copy()
        tgt_cuts = np.array([cut_by[n] for n in KERNEL_CHAN])
        if a.percat:
            from percat_radius import per_category_cutoffs
            pc = [per_category_cutoffs(emb[n]["Ztr"], emb[n]["Zte"], cat_tr, cat_te,
                                       ncat)[0] for n in kchan]
            base_cuts = np.array(pc)                 # (C, ncat)

        Zb_tr = np.hstack([emb[n]["Ztr"] for n in need])
        gb = a.diag_sample * len(Zb_tr) * 8 / 1e9
        print(f"[prod] seed {seed}: {len(tr):,} train / {len(te):,} test; computing "
              f"{len(need)} distance bands {need} for the radius bisection "
              f"({a.diag_sample} x {len(Zb_tr):,} = {gb:.1f} GB each, on the DRIVER) ...",
              flush=True)
        tband = time.time()
        bidx, all_bands = distance_bands(Zb_tr, [band_slices[n] for n in need],
                                         sample=a.diag_sample)
        del Zb_tr
        by_name = dict(zip(need, all_bands))
        bands = [by_name[n] for n in kchan]                    # this arm's channels
        tgt_bands = [by_name[n] for n in KERNEL_CHAN]          # the baseline's channels
        print(f"[prod] seed {seed}: bands ready in {time.time() - tband:.0f}s", flush=True)
        d_add, nb_add, iso_add = support_stats(bidx, tgt_bands, cat_tr, tgt_cuts, "union")
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
        del bands, tgt_bands, all_bands, by_name
        print(f"[prod] seed {seed}: matched at scale x{s:.3f} -> density {d_p:.4e} "
              f"(median {nb_p:.0f} nbrs, {iso_p:.1%} isolated)")
        # DENSITY MATCHING IS THE COMPARISON. The bracket [1, 8] was set at 20k, where
        # x1.59 sufficed; intersection density falls faster with N than union does, so
        # the required scale GROWS and the bracket can saturate. A saturated bisection
        # returns hi and quietly leaves the product arm sparser than the baseline,
        # which is exactly the confound the matching exists to remove -- and it would
        # read as "the product kernel is worse at scale".
        rel = abs(d_p - d_add) / max(d_add, 1e-12)
        if rel > 0.05:
            raise SystemExit(
                f"[prod] ABORT seed {seed}: density matching FAILED -- product "
                f"{d_p:.4e} vs additive target {d_add:.4e} ({rel:.1%} off) at scale "
                f"x{s:.3f} in bracket [1, 8]. The arms would not be density-matched. "
                f"Widen the bracket (hi) and rerun; do not score this."
            )

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
            print(f"[prod] seed {seed}: building the GP on {len(Xs):,} train rows "
                  f"({a.workers} workers, kernel on {a.device}) ...", flush=True)
            with SolveWarningCounter() as wc:
                tb = time.time()
                gp, _ = build_gp(Xs, ys, None, None, cl, kernel_override=kern,
                                 signal_var=[prior_var], jitter=JITTER, **SOLVE)
                if a.no_variance:
                    # The posterior MEAN reuses the precomputed KVinvY, so it costs ONE
                    # solve in total no matter how many test points. Variance is what
                    # costs one solve EACH. With variance off, evaluating on the full
                    # test set is therefore free and gives a much tighter R^2 (40k
                    # points instead of 800 at 200k) -- this is the whole speed
                    # difference between this and dim_sweep.
                    print(f"[prod] seed {seed}: GP built in {time.time() - tb:.0f}s. "
                          f"Mean only on ALL {len(te):,} test points (one solve total; "
                          f"--no-variance, so no UQ columns).", flush=True)
                    m_gp, v_gp = predict(gp, with_category_tag(Zk_te, cat_te),
                                         batch=2000, variance=False, verbose=True)
                else:
                    print(f"[prod] seed {seed}: GP built in {time.time() - tb:.0f}s. Now "
                          f"{a.nte} posterior variances = {a.nte} SEPARATE LINEAR SOLVES "
                          f"against the full system -- the dominant cost at this N.",
                          flush=True)
                    m_gp, v_gp = predict(gp, with_category_tag(Zk_te[sel], cat_te[sel]),
                                         batch=100, variance=True, verbose=True)
            del gp
            release_gp(cl)
        finally:
            cl.close()
        wc.report(n_evals=(len(te) if a.no_variance else a.nte), label=f"prod-s{seed}")

        # which test points this arm reports on: all of them under --no-variance,
        # otherwise the nte-point variance subset
        ev = np.arange(len(y_te)) if a.no_variance else sel
        mu = mean.predict(Zm_te, size=tsz)[ev] + m_gp
        err = y_te[ev] - mu
        if a.no_variance:
            # dist-to-NN is a (n_eval x n_train) dense block -- 40k x 160k = 51 GB at
            # 200k. It is a UQ baseline and there is no UQ here, so skip it rather
            # than sample it and half-report an arm the scorer would then rank.
            dnn = np.full(len(ev), np.nan)
            sd_cheap = np.full(len(ev), np.nan)
        else:
            Dn = cdist(Zk_te[sel], Zk_tr)
            Dn[cat_te[sel][:, None] != cat_tr[None, :]] = np.inf
            dnn = Dn.min(axis=1)
            D_tr, D_te = np.eye(ncat)[cat_tr], np.eye(ncat)[cat_te]
            V_tr = np.hstack([np.ones((len(tr), 1)), np.log(size_tr)[:, None],
                              D_tr[:, 1:]])
            V_te = np.hstack([np.ones((len(te), 1)), np.log(size_te)[:, None],
                              D_te[:, 1:]])
            bv, *_ = np.linalg.lstsq(V_tr, np.log(r_tr ** 2 + 1e-8), rcond=None)
            sd_cheap = np.exp(0.5 * (V_te @ bv))[sel]

        np.savez(out, seed=seed, arm=tag, rung=a.mean, n=a.n,
                 nte=(len(te) if a.no_variance else a.nte),
                 has_variance=int(not a.no_variance),
                 diag_sample=a.diag_sample, kernel_chan=np.array(kchan),
                 y=y_te[ev], mu=mu, err=err,
                 v_gp=v_gp, dnn=dnn, sd_cheap=sd_cheap, cat=cat_te[ev],
                 cuts_cat=(cuts if cuts.ndim == 2 else
                           np.repeat(cuts[:, None], ncat, axis=1)),
                 cuts_glob=glob_cuts, scale=s,
                 dens_global=d_add, dens_percat=d_p, dens_natural=d_nat,
                 med_nbrs=nb_p, isolated=iso_p, prior_var=prior_var,
                 ols_r2=r2_score(y_te[ev], mean.predict(Zm_te, size=tsz)[ev]),
                 n_warn=int(wc.total), cat_names=np.array(ds.category_names))
        print(f"[prod] seed {seed} {tag}: GP_R2={r2_score(y_te[ev], mu):.4f} "
              f"on {len(ev):,} test pts "
              f"({time.time() - t0:.0f}s) -> {out}\n")


if __name__ == "__main__":
    main()
