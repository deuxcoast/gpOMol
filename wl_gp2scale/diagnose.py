"""
diagnose.py  (wl_gp2scale)
==========================
Two diagnostics for the gp2Scale path, sharing one embedding build:

  --mode bisect  (default)  Attribute a predictive-accuracy failure to the kernel,
                            the gp2Scale assembly, the solver, or block handling.
  --mode sweep              Sweep the cutoff percentile and report R^2 vs realised
                            density vs PROJECTED memory at the target N, so the
                            production cutoff is chosen from data rather than
                            extrapolation.

------------------------------------------------------------------------------
mode=bisect
------------------------------------------------------------------------------
Holds the embedding fixed and varies ONE thing at a time:

  (a) dense scipy cdist kernel + Cholesky      <- the gp_parity.py reference
  (b) our GPU block kernel  + Cholesky         <- kernel correct? (no gp2Scale)
  (c) gp2Scale, 1 block,   sparseLU            <- assembly correct? (direct solve)
  (d) gp2Scale, 1 block,   sparseCG            <- solver?
  (e) gp2Scale, k blocks,  sparseLU            <- multi-block assembly?
  (f) gp2Scale, k blocks,  sparseCG

Read by comparing R^2 / max|Δ| against (a):
  * (b) differs            -> the kernel itself is wrong.
  * (c) differs            -> gp2Scale ASSEMBLY is wrong (block masking / mirroring).
  * (d) differs, (c) fine  -> the sparseCG solver path is wrong.
  * only multi-block differs -> block-count / range handling.

This is how the torch.cdist mm-expansion + float32 bug was found: every gp2Scale
case was IDENTICALLY wrong across both solvers and both block counts, which
exonerated the solver and pointed at the covariance matrix itself.

`batch_size` interacts with N: fvgp uses num_batches = N // batch_size and falls
back to a single range when that is 0 (gp_prior.py::_ranges), so batch_size >= N
means "one big block".

------------------------------------------------------------------------------
mode=sweep
------------------------------------------------------------------------------
The cutoff percentile is a DENSITY knob and does not transfer across N: the
in-support fraction of pairs is ~pct/100 (times P(same category), since the kernel
zeroes cross-category pairs), so the NEIGHBOUR COUNT scales with N. Measured at 20k:
pct=25 -> ~818 neighbours/point and R^2=0.1188; the same pct=25 at 196k would mean
~10,000 neighbours/point, ~1.96e9 nnz and a ~70 GB driver-side assembly peak.

Density is a fraction of pairs, so density(pct) measured here projects to the target
N as nnz = density * N_target^2. That projection -- not the R^2 column -- is what
sizes the run. R^2 here is measured at --ntr training points with a dense Cholesky,
so treat it as a RELATIVE comparison across pct, not an absolute prediction of the
200k R^2 (more training data raises it).

Memory is driver-side: fvgp's gp2Scale distributes kernel EVALUATION but gathers the
COO components and assembles ONE scipy CSR on the client (gp_prior.py:294-306).
"""

from __future__ import annotations

import argparse

import numpy as np


def build_argparser():
    ap = argparse.ArgumentParser(
        description="gp2Scale diagnostics: bisect / cutoff sweep / informative radius")
    ap.add_argument("--mode", choices=["bisect", "sweep", "radius", "variogram"],
                    default="bisect")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=6000, help="molecules to load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2,
                    help="2 matches descriptor_eval/gp_parity.py")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--pls", type=int, default=10)
    ap.add_argument("--cutoff-pct", type=float, default=25.0,
                    help="bisect mode: the cutoff percentile to test at")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="absolute compact-support radius; overrides --cutoff-pct")
    ap.add_argument("--ntr", type=int, default=3000, help="train rows for the test")
    ap.add_argument("--nte", type=int, default=3000,
                    help="test rows (more -> less noisy per-bin RMSE in radius mode)")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scheduler-file", default=None)
    # sweep-only
    ap.add_argument("--pcts", default="25,10,5,2,1,0.5",
                    help="sweep mode: comma-separated cutoff percentiles")
    ap.add_argument("--target-n", type=int, default=196_000,
                    help="sweep mode: training-set size to project memory to")
    ap.add_argument("--cond", action="store_true",
                    help="sweep mode: also report cond(KV) (eigvalsh, slow)")
    # radius-only
    ap.add_argument("--nbins", type=int, default=10,
                    help="radius mode: nn-distance bins (deciles by default)")
    ap.add_argument("--nn-sizes", default="",
                    help="radius mode: comma-separated train sizes for the nn-distance "
                         "scaling fit; default is a log ladder up to the full split")
    # variogram-only (also used for the cross-check in radius mode)
    ap.add_argument("--vario-sample", type=int, default=5000,
                    help="variogram: rows to subsample for the pairwise semivariance")
    ap.add_argument("--vario-bins", type=int, default=20,
                    help="variogram: number of equal-count distance bins")
    ap.add_argument("--sill-frac", type=float, default=0.95,
                    help="variogram: gamma/sill fraction defining the effective range")
    ap.add_argument("--same-cat", dest="same_cat", action="store_true", default=True,
                    help="variogram: use only same-category pairs (default; faithful "
                         "to the block kernel)")
    ap.add_argument("--no-same-cat", dest="same_cat", action="store_false",
                    help="variogram: use all pairs regardless of category")
    ap.add_argument("--plot-dir", default="diagnostics",
                    help="directory for PNG plots (variogram/radius modes)")
    ap.add_argument("--no-plot", dest="plot", action="store_false", default=True,
                    help="skip writing PNG plots")
    ap.add_argument("--out", default=None,
                    help="variogram mode: optional .npz of the curve arrays")
    return ap


def _load_embedding(args):
    """Load molecules and build the supervised 10-D embedding once (shared)."""
    from sklearn.model_selection import train_test_split

    from .data import get_data
    from .pipeline import WLGPPipeline

    ds = get_data(src=args.src, n=args.n, seed=args.seed)
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=42)

    pipe = WLGPPipeline(depth=args.depth, min_count=args.min_count,
                        pls_components=args.pls, cutoff_percentile=args.cutoff_pct,
                        cutoff_abs=args.cutoff)
    Z_tr = pipe.fit([ds.atoms[i] for i in tr], ds.y[tr], ds.data_id[tr], client=None)
    Z_te = pipe.transform([ds.atoms[i] for i in te], client=None)

    ntr = min(args.ntr, len(Z_tr))
    nte = min(args.nte, len(Z_te))
    # Keep the FULL train split too: the GP fits are capped by dense Cholesky, but
    # the nearest-neighbour analysis is only a cdist and should use every training
    # molecule available (it is the density that is under study).
    return {
        "Z_tr": Z_tr[:ntr], "y_tr": ds.y[tr][:ntr], "cat_tr": ds.data_id[tr][:ntr],
        "Z_te": Z_te[:nte], "y_te": ds.y[te][:nte], "cat_te": ds.data_id[te][:nte],
        "Z_tr_full": Z_tr, "y_tr_full": ds.y[tr], "cat_tr_full": ds.data_id[tr],
        "dim": pipe.dim_, "cutoff": pipe.cutoff_, "ntr": ntr, "nte": nte,
    }


# ---------------------------------------------------------------- sweep mode


def run_sweep(args, E):
    """R^2 vs realised density vs projected memory, per cutoff percentile."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.spatial.distance import pdist
    from sklearn.metrics import r2_score

    from .kernel import make_wl_block_kernel
    from .pipeline import with_category_tag

    Z_tr, y_tr, Z_te, y_te = E["Z_tr"], E["y_tr"], E["Z_te"], E["y_te"]
    dim, ntr = E["dim"], E["ntr"]
    sv = float(np.var(y_tr))
    # real category tags: the production kernel zeroes cross-category pairs, so the
    # realised density (and hence the memory projection) must account for them
    Xtr = with_category_tag(Z_tr, E["cat_tr"])
    Xte = with_category_tag(Z_te, E["cat_te"])
    hps = np.array([sv])

    d = pdist(Z_tr[: min(3000, ntr)])
    pcts = [float(p) for p in args.pcts.split(",")]
    tn = args.target_n

    print(f"\n[sweep] ntr={ntr} nte={E['nte']} dim={dim} signal_var={sv:.3f} "
          f"jitter={args.jitter:g}  projecting memory to N={tn:,}")
    print(f"[sweep] R2 is measured at ntr={ntr} (dense Cholesky) -> use it to compare "
          f"pct RELATIVELY; more training data raises the absolute value.\n")
    hdr = (f"{'pct':>6}{'cutoff':>9}{'density':>10}{'nbr@ntr':>9}{'nbr@N':>9}"
           f"{'R2':>9}{'PD':>4}{'nnz@N':>14}{'CSR GB':>9}{'peak GB':>9}")
    if args.cond:
        hdr += f"{'cond':>10}"
    print(hdr); print("-" * len(hdr))

    for pct in pcts:
        cut = float(np.percentile(d, pct))
        kern = make_wl_block_kernel(cut, dim=dim, use_category_tag=True,
                                    device=args.device, dtype="float64")
        K = np.asarray(kern(Xtr, Xtr, hps))
        nz = K != 0
        density = float(nz.mean())
        nbrs = float(np.median(nz.sum(axis=1)))
        KV = K + args.jitter * np.eye(ntr)
        try:
            alpha = cho_solve(cho_factor(KV), y_tr)
            ks = np.asarray(kern(Xte, Xtr, hps))
            r2 = r2_score(y_te, ks @ alpha)
            pd_ok = "yes"
        except Exception:
            r2, pd_ok = float("nan"), "NO"
        # project to the target N: density is a fraction of pairs, so it carries over
        nnz_t = density * float(tn) ** 2
        csr_gb = nnz_t * 12 / 1e9
        # driver assembly peak: upper-tri COO (data+row+col, 8B each), mirrored, ->CSR
        peak_gb = (nnz_t / 2) * 24 * 2 / 1e9 + csr_gb
        # neighbours/point AT THE TARGET N -- this is the regime that must be
        # preserved (nbr@ntr is only what this small slice happens to see)
        nbr_t = density * float(tn)
        line = (f"{pct:>6}{cut:>9.4f}{density:>10.3e}{nbrs:>9.0f}{nbr_t:>9,.0f}"
                f"{r2:>+9.4f}{pd_ok:>4}{nnz_t:>14,.0f}{csr_gb:>9.1f}{peak_gb:>9.1f}")
        if args.cond:
            ev = np.linalg.eigvalsh(KV)
            line += f"{ev.max() / max(ev.min(), 1e-300):>10.1e}"
        print(line)

    print(f"\n[sweep] How to read this:")
    print(f"  * 'nbr@N' (= density * {tn:,}) is the regime that transfers across N,")
    print(f"    NOT 'pct'. The validated 20k run had ~818 neighbours/point at R²=0.1188;")
    print(f"    pick the pct whose nbr@N lands near that, then sanity-check R² here.")
    print(f"  * 'peak GB' is DRIVER-side and must fit the driver node's RAM")
    print(f"    (Perlmutter ~256 GB, minus dask/python headroom). It is the binding")
    print(f"    constraint, and it is why pct=25 does not survive the jump to 200k.")
    print(f"  * R² is at ntr={ntr}: compare pct values RELATIVELY, do not read it as")
    print(f"    the 200k R² (more training data raises it).")


# --------------------------------------------------------------- radius mode


# ------------------------------------------------------------- variogram mode


def _variogram_range(Z, y, cat, args):
    """Semivariogram + effective range on one embedding. Returns (curve, range)."""
    from .radius import range_from_variogram, semivariogram

    vg = semivariogram(Z, y, sample=args.vario_sample, n_bins=args.vario_bins,
                       cat=cat, seed=0)
    rng = range_from_variogram(vg["lag"], vg["gamma"], vg["sill"],
                               sill_frac=args.sill_frac)
    return vg, rng


def run_variogram(args, E):
    """GP-FREE radius picker: semivariogram of y over embedding distance -> the
    correlation length (range) where gamma reaches the sill. Ties that SIGNAL radius
    to kernel CONDITIONING via sparsity_report, and shows the absolute range is
    N-invariant (the justification for retiring per-N --cutoff-pct for signal)."""
    from .cutoff import sparsity_report

    Z, y = E["Z_tr_full"], E["y_tr_full"]
    cat = E["cat_tr_full"] if args.same_cat else None
    dim = E["dim"]
    tag = "same-category pairs" if args.same_cat else "all pairs"

    vg, rng = _variogram_range(Z, y, cat, args)
    sill = vg["sill"]
    print(f"\n[vario] semivariogram on {len(Z):,} train molecules, {tag}, "
          f"{args.vario_sample} sampled rows ({vg['n_pairs']:,} pairs), "
          f"dim={dim}")
    print(f"[vario] sill (gamma plateau) = {sill:.3f}   "
          f"(total Var(y) = {np.var(y):.3f})")
    hdr = f"{'lag h':>11}{'gamma(h)':>11}{'gamma/sill':>12}{'pairs':>12}"
    print(hdr); print("-" * len(hdr))
    for h, g, c in zip(vg["lag"], vg["gamma"], vg["count"]):
        print(f"{h:>11.5f}{g:>11.3f}{g/sill:>12.3f}{c:>12,}")

    if rng is None:
        print(f"\n[vario] gamma never reaches {args.sill_frac:.0%} of the sill within "
              f"the sampled distances -> no decorrelation resolved. The correlation "
              f"length is larger than the data spans (or the signal is representational,"
              f" not spatial). Rerun at larger --n or inspect the curve/plot.")
    if rng is not None:
        print(f"\n[vario] effective RANGE = {rng:.5f} embedding units "
              f"(gamma reaches {args.sill_frac:.0%} of sill here).")
        print(f"[vario]   Recommended cutoff = the range itself, no multiplier "
              f"(--cutoff {rng:.5f}): the Wendland tapering to zero at the range is the")
        print(f"[vario]   intended behaviour -- it EXCLUDES the poorly-correlated shell and")
        print(f"[vario]   keeps the kernel sparse/well-conditioned. Raise --cutoff only if")
        print(f"[vario]   a sweep shows higher R^2 without wrecking conditioning.")

        # tie the SIGNAL radius to CONDITIONING: density / neighbours at the cutoff.
        print(f"\n[vario] kernel density at cutoff={rng:.5f} (= range):")
        rep = sparsity_report(Z, rng, dim=dim, data_id=cat)
        med = rep["median_neighbors"]
        if med > 200:
            print(f"[vario]   median {med:.0f} in-support neighbours is HIGH -> the signal "
                  f"radius is denser than is well-conditioned; the density guard "
                  f"(--cutoff-pct) is the binding constraint, tighten the cutoff.")
        elif med < 10:
            print(f"[vario]   median {med:.0f} in-support neighbours is LOW -> risk of mean "
                  f"reversion; this radius may be too tight for good coverage.")

    # Trend / non-stationarity check. A supervised PLS embedding orients y linearly
    # along the leading axes, so y is NON-stationary in embedding space: the variogram
    # keeps rising past the sill instead of plateauing, and the range over-estimates
    # the LOCAL predictive radius. Flag it so the range is not over-trusted.
    overshoot = float(np.max(vg["gamma"])) / sill if sill > 0 else float("nan")
    if overshoot > 1.5:
        print(f"\n[vario] NOTE: gamma overshoots the sill ({overshoot:.1f}x at the far "
              f"lag) -> y has a TREND in the embedding (expected: PLS orients y linearly "
              f"along the leading axes). The variogram does not cleanly plateau, so the "
              f"range OVER-estimates the local predictive radius. Cross-check with "
              f"`--mode radius` (RMSE-cliff R_inf is the more reliable cutoff basis "
              f"here), or run the variogram on the OLS-detrended residual for a clean "
              f"local range.")

    # N-invariance: recompute the range on subsamples of the training pool. Under the
    # current natural+pareto scaling this should be ~CONSTANT in absolute units (it was
    # ~1/sqrt(N) under the old unit-norm scaling -- see radius.py docstring).
    print(f"\n[vario] N-invariance of the range (subsampled training pool):")
    print(f"{'n_train':>9}{'range':>11}")
    rng2 = np.random.default_rng(0)
    n = len(Z)
    for m in sorted({max(500, n // 4), n // 2, n}):
        sub = rng2.choice(n, size=m, replace=False)
        cat_sub = cat[sub] if cat is not None else None
        _, r_m = _variogram_range(Z[sub], y[sub], cat_sub, args)
        rs = f"{r_m:.5f}" if r_m is not None else "  n/a"
        print(f"{m:>9,}{rs:>11}")
    print(f"[vario] a ~constant absolute range across n confirms the radius transfers "
          f"across N -> pick it once (percentile no longer needed for SIGNAL).")

    if args.plot:
        from . import plots
        sub = (f"{tag}, n={len(Z):,}, sill={sill:.2f}"
               + (f", range={rng:.4f}" if rng is not None else ""))
        p = plots.plot_semivariogram(vg["lag"], vg["gamma"], sill, rng,
                                     out_dir=args.plot_dir, subtitle=sub)
        print(f"\n[vario] wrote {p}")
    if args.out:
        np.savez(args.out, lag=vg["lag"], gamma=vg["gamma"], count=vg["count"],
                 sill=sill, range=(np.nan if rng is None else rng))
        print(f"[vario] wrote {args.out}")


# ---------------------------------------------------------------- radius mode


def run_radius(args, E):
    """Informative radius + a density-only prediction of R^2 at --target-n."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.spatial.distance import cdist, pdist
    from scipy.stats import percentileofscore
    from sklearn.metrics import r2_score

    from .kernel import make_wl_block_kernel
    from .pipeline import with_category_tag
    from .radius import (frac_within, nn_scaling_exponent, predict_r2_at_n,
                         rmse_vs_nn_distance)

    Z_tr, y_tr, Z_te, y_te = E["Z_tr"], E["y_tr"], E["Z_te"], E["y_te"]
    Z_tr_full = E["Z_tr_full"]
    dim, ntr, cut = E["dim"], E["ntr"], E["cutoff"]
    sv = float(np.var(y_tr))
    tn = args.target_n

    # 1. fit the GP (dense Cholesky) and predict, to measure RMSE vs nn-distance
    kern = make_wl_block_kernel(cut, dim=dim, use_category_tag=True,
                                device=args.device, dtype="float64")
    Xtr = with_category_tag(Z_tr, E["cat_tr"])
    Xte = with_category_tag(Z_te, E["cat_te"])
    hps = np.array([sv])
    K = np.asarray(kern(Xtr, Xtr, hps)) + args.jitter * np.eye(ntr)
    alpha = cho_solve(cho_factor(K), y_tr)
    y_pred = np.asarray(kern(Xte, Xtr, hps)) @ alpha
    r2_now = r2_score(y_te, y_pred)

    nn_now = cdist(Z_te, Z_tr).min(axis=1)
    curve = rmse_vs_nn_distance(nn_now, y_te, y_pred, n_bins=args.nbins)
    base = curve["baseline"]

    print(f"\n[radius] ntr={ntr} nte={E['nte']} dim={dim} cutoff={cut:.5f} "
          f"signal_var={sv:.3f}  measured R²={r2_now:+.4f}")
    print(f"[radius] baseline RMSE (predict the mean) = {base:.3f}\n")
    hdr = f"{'bin':>4}{'%test':>7}{'med nn-dist':>13}{'RMSE':>9}{'RMSE/base':>11}  informative?"
    print(hdr); print("-" * (len(hdr) + 4))
    for i, (d, r, f) in enumerate(zip(curve["bin_median_nn"], curve["bin_rmse"],
                                      curve["bin_frac"])):
        ratio = r / base
        flag = "yes" if ratio < 0.9 else ("~" if ratio < 1.0 else "NO (>= mean)")
        print(f"{i:>4}{100*f:>6.0f}%{d:>13.5f}{r:>9.3f}{ratio:>11.2f}  {flag}")

    n_good = int((curve["bin_rmse"] < 0.9 * base).sum())
    print(f"\n[radius] {n_good}/{len(curve['bin_rmse'])} bins are informative "
          f"(RMSE < 0.9*baseline). R_inf below is the per-bin cliff at 0.9*baseline on a "
          f"median-smoothed profile (each bin is only nte/{args.nbins} points). The "
          f"CUMULATIVE curve (shown for context) stays low well past the cliff because "
          f"near points dominate the aggregate -- it is NOT used for the radius:")
    print(f"{'nn-dist <=':>12}{'% test':>8}{'cum RMSE':>10}{'/base':>8}")
    for d, c in zip(curve["cum_nn"], curve["cum_rmse"]):
        pc = 100 * float(np.mean(nn_now <= d))
        print(f"{d:>12.5f}{pc:>7.0f}%{c:>10.3f}{c/base:>8.2f}")

    R_inf = curve["radius"]

    # cross-check R_inf against the GP-FREE semivariogram range: two estimates of one
    # correlation length (a-posteriori error cliff vs a-priori signal decorrelation).
    _vg, vg_rng = _variogram_range(
        E["Z_tr_full"], E["y_tr_full"],
        E["cat_tr_full"] if args.same_cat else None, args)
    _ratio = (vg_rng / R_inf) if (vg_rng and R_inf) else None
    print(f"\n[radius] cross-check: semivariogram range = "
          f"{'n/a' if vg_rng is None else f'{vg_rng:.5f}'} (a-priori, pairwise "
          f"y-decorrelation)  vs  RMSE-cliff R_inf = "
          f"{'n/a' if R_inf is None else f'{R_inf:.5f}'} (a-posteriori, local "
          f"predictive radius).")
    if _ratio and _ratio > 2.0:
        print(f"[radius]   range is {_ratio:.1f}x R_inf -> y is NON-stationary (a trend "
              f"in the embedding, expected from supervised PLS), so the variogram range "
              f"over-estimates the local radius. Trust R_inf for the cutoff here.")

    if args.plot:
        from . import plots
        p = plots.plot_rmse_vs_nn(
            curve, out_dir=args.plot_dir,
            subtitle=f"ntr={ntr}, dim={dim}, cutoff={cut:.4f}")
        print(f"[radius] wrote {p}")

    if R_inf is None:
        print("\n[radius] Even the nearest test points do not beat 0.9*baseline -> the")
        print("[radius] error is flat in neighbour distance. That is REPRESENTATIONAL,")
        print("[radius] not a density problem: more molecules will not help, and 200k")
        print("[radius] is not justified on this evidence. (If R² here is also ~0 or")
        print("[radius] negative, first check the embedding is not simply too weak at")
        print("[radius] this --n; rerun at --n 20000 before concluding anything.)")
        return

    # 2. the recommended cutoff IS R_inf (no multiplier). The scale is N-invariant now,
    #    so use the absolute radius directly via --cutoff.
    dp = pdist(Z_tr[: min(2500, ntr)])
    pct_inf = float(percentileofscore(dp, R_inf))
    inside = frac_within(nn_now, R_inf)
    print(f"\n[radius] informative radius R_inf = {R_inf:.5f} (embedding units)")
    print(f"[radius]   = the {pct_inf:.3f}th percentile of pairwise distances "
          f"(the absolute value transfers across N; percentile shown for context)")
    print(f"[radius]   {100*inside:.1f}% of test molecules are inside it at ntr={ntr:,}; "
          f"the other {100*(1-inside):.1f}% are predicted WORSE than the mean.")
    # Cutoff = R_inf, no multiplier: R_inf is where neighbours stop carrying signal, so
    # the Wendland tapering to zero exactly there is the intended behaviour -- it
    # EXCLUDES the poorly-correlated points (Marcus's guidance) and keeps the kernel as
    # sparse / well-conditioned as possible. The Wendland does down-weight the mid-range
    # (psi(0.5)=0.19), so if you want more weight out toward R_inf, just raise --cutoff a
    # little and re-check R^2 + conditioning -- do NOT bake in a fixed 2x multiplier
    # (that pulls the poorly-correlated (R_inf, 2*R_inf) shell back in and roughly
    # quadruples the neighbour count).
    print(f"\n[radius]   -> --cutoff {R_inf:.5f}   (= R_inf, no multiplier; overrides "
          f"--cutoff-pct)")
    print(f"[radius]      Adjust upward only if a sweep shows higher R^2 without wrecking")
    print(f"[radius]      conditioning (cutoff.sparsity_report at the chosen radius).")

    # 3. how fast does nn-distance shrink as the train set grows? (embedding FIXED,
    #    so this isolates density from any change in the representation)
    if args.nn_sizes:
        sizes = [int(s) for s in args.nn_sizes.split(",")]
    else:
        top = len(Z_tr_full)
        sizes = sorted({int(top / (2 ** k)) for k in range(6)} | {top})
        sizes = [s for s in sizes if s >= 250]
    slope, d_eff, sizes, meds, nn_full = nn_scaling_exponent(Z_te, Z_tr_full, sizes)
    print(f"\n[radius] nn-distance vs train size (embedding held fixed):")
    print(f"{'n_train':>9}{'median nn':>12}{'% inside R_inf':>16}")
    for n, m in zip(sizes, meds):
        idx = np.random.default_rng(0).choice(len(Z_tr_full), size=n, replace=False)
        f_in = frac_within(cdist(Z_te, Z_tr_full[idx]).min(axis=1), R_inf)
        print(f"{n:>9,}{m:>12.5f}{100*f_in:>15.1f}%")
    print(f"[radius] log-log slope = {slope:+.4f}  ->  nn ~ n^({slope:.3f}), "
          f"effective dimension d_eff ~ {d_eff:.1f}")

    # 4. project to the target N and predict R^2 from the density shift alone
    print(f"\n[radius] projected to N_train={tn:,} (density effect ONLY, embedding fixed):")
    print(f"{'n_train':>9}{'median nn':>12}{'% inside R_inf':>16}{'pred R²':>10}")
    # ntr MUST be in this list: its row is the self-check (scale factor 1, so the
    # prediction is just the measured data re-binned and should reproduce r2_now).
    for n in sorted(set(list(sizes) + [ntr, tn])):
        sc = (n / ntr) ** slope
        nn_s = nn_now * sc
        p = predict_r2_at_n(curve, nn_now, ntr, n, slope, y_te)
        f_in = frac_within(nn_s, R_inf)
        tag = "  <- measured" if n == ntr else ("  <- TARGET" if n == tn else "")
        print(f"{n:>9,}{p['median_nn_scaled']:>12.5f}{100*f_in:>15.1f}%"
              f"{p['pred_r2']:>+10.4f}{tag}")
    print(f"\n[radius] sanity: predicted R² at n={ntr:,} should be close to the "
          f"measured {r2_now:+.4f} (it is the same data, re-binned).")
    print("[radius] This is a FLOOR: it holds the embedding fixed, so it counts only")
    print("[radius] the density gain. The representation also improves with N (the PLS")
    print("[radius] probe went -0.266 at 16k train to +0.1212 at 40k), which this")
    print("[radius] estimate deliberately ignores.")


# --------------------------------------------------------------- bisect mode


def run_bisect(args, E):
    from distributed import Client
    from gpcam import GPOptimizer
    from sklearn.metrics import r2_score

    from .kernel import dense_wendland_reference, make_wl_block_kernel
    from .pipeline import (_first, build_gp, connect_dask, predict, release_gp,
                           with_category_tag)

    Z_tr, y_tr, Z_te, y_te = E["Z_tr"], E["y_tr"], E["Z_te"], E["y_te"]
    dim, ntr, nte, cut = E["dim"], E["ntr"], E["nte"], E["cutoff"]
    sv = float(np.var(y_tr))
    # single dummy category so the mask is inert and we compare the sparse
    # machinery against the dense reference in isolation
    Xtr = with_category_tag(Z_tr, np.zeros(ntr))
    Xte = with_category_tag(Z_te, np.zeros(nte))
    jit = args.jitter

    print(f"\n[diag] ntr={ntr} nte={nte} dim={dim} cutoff={cut:.5f} "
          f"signal_var={sv:.3f} jitter={jit:g}\n")
    hdr = f"{'case':<44}{'R2':>9}{'max|d| vs (a)':>16}{'corr':>8}"
    print(hdr); print("-" * len(hdr))

    def row(label, m, m_ref=None):
        r2 = r2_score(y_te, m)
        if m_ref is None:
            print(f"{label:<44}{r2:>+9.4f}{'--':>16}{'--':>8}")
        else:
            d = float(np.max(np.abs(m - m_ref)))
            c = float(np.corrcoef(m, m_ref)[0, 1])
            print(f"{label:<44}{r2:>+9.4f}{d:>16.3e}{c:>8.4f}")

    def dref(x1, x2, hps):
        return dense_wendland_reference(x1, x2, hps, cut, dim=None)

    g = GPOptimizer(x_data=Z_tr, y_data=y_tr, init_hyperparameters=np.array([sv]),
                    kernel_function=dref, noise_variances=jit * np.ones(ntr))
    m_ref = _first(g.posterior_mean(Z_te), ["f(x)", "m(x)"])
    row("(a) dense scipy cdist + Cholesky [REF]", m_ref)

    kern = make_wl_block_kernel(cut, dim=dim, use_category_tag=True,
                                device=args.device, dtype="float64")
    g2 = GPOptimizer(x_data=Xtr, y_data=y_tr, init_hyperparameters=np.array([sv]),
                     kernel_function=kern, noise_variances=jit * np.ones(ntr))
    row("(b) our kernel + Cholesky (no gp2Scale)",
        _first(g2.posterior_mean(Xte), ["f(x)", "m(x)"]), m_ref)

    if args.scheduler_file:
        client = connect_dask(args.scheduler_file, n_workers=args.workers)
    else:
        client = Client(n_workers=args.workers, threads_per_worker=1)
        client.wait_for_workers(args.workers)

    one_block = ntr * 2
    multi = max(1, ntr // 3)
    for label, bs, mode in [
        ("(c) gp2Scale 1 block   + sparseLU", one_block, "sparseLU"),
        ("(d) gp2Scale 1 block   + sparseCG", one_block, "sparseCG"),
        ("(e) gp2Scale ~3 blocks + sparseLU", multi, "sparseLU"),
        ("(f) gp2Scale ~3 blocks + sparseCG", multi, "sparseCG"),
    ]:
        try:
            gp, _ = build_gp(Xtr, y_tr, cut, dim, client, signal_var=sv, jitter=jit,
                             batch_size=bs, backend="wendland32",
                             compute_device="cpu",  # see pipeline.build_gp
                             device=args.device, linalg_mode=mode)
            m, _v = predict(gp, Xte)
            row(f"{label} (nb={ntr // bs})", m, m_ref)
            del gp
        except Exception as e:
            print(f"{label:<44}{'ERROR':>9}  {type(e).__name__}: {str(e)[:60]}")
        release_gp(client)

    client.close()
    print("\n[diag] Interpretation (legend, not assertions):")
    print("  (b) != (a)            -> the block kernel is wrong")
    print("  (c) != (a)            -> gp2Scale ASSEMBLY is wrong (masking/mirroring)")
    print("  (d) != (c)            -> the sparseCG solver path is wrong")
    print("  (e)/(f) != (c)/(d)    -> block-count / range handling is wrong")


def main():
    args = build_argparser().parse_args()
    E = _load_embedding(args)
    if args.mode == "sweep":
        run_sweep(args, E)
    elif args.mode == "radius":
        run_radius(args, E)
    elif args.mode == "variogram":
        run_variogram(args, E)
    else:
        run_bisect(args, E)


if __name__ == "__main__":
    main()
