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
    ap.add_argument("--mode", choices=["bisect", "sweep", "radius"], default="bisect")
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
    ap.add_argument("--ntr", type=int, default=3000, help="train rows for the test")
    ap.add_argument("--nte", type=int, default=1000, help="test rows for the test")
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
                        pls_components=args.pls, cutoff_percentile=args.cutoff_pct)
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

    n_good = int((curve["bin_rmse"] < base).sum())
    print(f"\n[radius] {n_good}/{len(curve['bin_rmse'])} bins beat the baseline. "
          f"Per-bin RMSE is noisy (each bin is only nte/{args.nbins} points), so the "
          f"radius below comes from the CUMULATIVE curve:")
    print(f"{'nn-dist <=':>12}{'% test':>8}{'cum RMSE':>10}{'/base':>8}")
    for d, c in zip(curve["cum_nn"], curve["cum_rmse"]):
        pc = 100 * float(np.mean(nn_now <= d))
        print(f"{d:>12.5f}{pc:>7.0f}%{c:>10.3f}{c/base:>8.2f}")

    R_inf = curve["radius"]
    if R_inf is None:
        print("\n[radius] Even the nearest test points do not beat 0.9*baseline -> the")
        print("[radius] error is flat in neighbour distance. That is REPRESENTATIONAL,")
        print("[radius] not a density problem: more molecules will not help, and 200k")
        print("[radius] is not justified on this evidence. (If R² here is also ~0 or")
        print("[radius] negative, first check the embedding is not simply too weak at")
        print("[radius] this --n; rerun at --n 20000 before concluding anything.)")
        return

    # 2. express the radius as a percentile -- the scale-free, transferable form
    dp = pdist(Z_tr[: min(2500, ntr)])
    pct_inf = float(percentileofscore(dp, R_inf))
    pct_2inf = float(percentileofscore(dp, 2.0 * R_inf))
    inside = frac_within(nn_now, R_inf)
    print(f"\n[radius] informative radius R_inf = {R_inf:.5f} (embedding units)")
    print(f"[radius]   = the {pct_inf:.3f}th percentile of pairwise distances "
          f"(percentile is the scale-free, transferable form)")
    print(f"[radius]   {100*inside:.1f}% of test molecules are inside it at ntr={ntr:,}; "
          f"the other {100*(1-inside):.1f}% are predicted WORSE than the mean.")
    # Do NOT set cutoff = R_inf. The Wendland tapers INSIDE the cutoff -- psi(0.5)
    # = 0.19, psi(0.75) = 0.04 -- so with cutoff = R_inf a neighbour sitting at
    # R_inf gets exactly zero weight and one at R_inf/2 only 19%, i.e. we would be
    # discarding neighbours that demonstrably carry signal. Setting cutoff = 2*R_inf
    # puts psi=0.19 at R_inf and ~0 beyond it: full weight on the near neighbours,
    # meaningful weight out to the edge of the informative zone, negligible past it.
    print(f"\n[radius]   -> --cutoff-pct {pct_2inf:.2f}   (cutoff = 2*R_inf = "
          f"{2*R_inf:.5f})")
    print(f"[radius]      The Wendland tapers inside the cutoff (psi(0.5)=0.19), so")
    print(f"[radius]      cutoff=R_inf (pct {pct_inf:.2f}) would zero out neighbours that")
    print(f"[radius]      still carry signal. 2*R_inf places psi=0.19 exactly at R_inf.")

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
    else:
        run_bisect(args, E)


if __name__ == "__main__":
    main()
