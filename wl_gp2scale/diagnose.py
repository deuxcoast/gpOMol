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
    ap = argparse.ArgumentParser(description="gp2Scale diagnostics: bisect / cutoff sweep")
    ap.add_argument("--mode", choices=["bisect", "sweep"], default="bisect")
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
    return {
        "Z_tr": Z_tr[:ntr], "y_tr": ds.y[tr][:ntr], "cat_tr": ds.data_id[tr][:ntr],
        "Z_te": Z_te[:nte], "y_te": ds.y[te][:nte], "cat_te": ds.data_id[te][:nte],
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
                             compute_device=("gpu" if args.device == "cuda" else "cpu"),
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
    else:
        run_bisect(args, E)


if __name__ == "__main__":
    main()
