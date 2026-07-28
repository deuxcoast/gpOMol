"""
dim_sweep.py  (wl_gp2scale)
===========================
Channel comparison on the ACTUAL gp2Scale Wendland kernel: for each requested
channel mode -- ``wl`` (graph), ``geom`` (3D geometry+charge), ``additive``
(k = k_WL + k_geom) -- report held-out predictive R^2 at the current N. This is the
ladder driver for the 40k/80k/200k comparison established in descriptor_eval, where
geometry dominates at small N but WL scales faster, so the two are expected to
converge near ~200k (exactly where the additive kernel should pay off).

For each split seed the two embeddings are built ONCE (WLGPPipeline for the graph
channel, GeometryPipeline for the geometry channel), then each requested mode reuses
them -- no re-featurisation. Each channel gets its OWN compact-support cutoff tuned to
``--target-neighbors`` (the validated sparsity lever; percentile mis-tracks the median
neighbour count on the clustered embedding, see cutoff.cutoff_for_neighbors). The
additive kernel sums the two blocks; ``--prior-mean linear`` detrends on the
CONCATENATED embedding (gp2Scale Eq. 2). Per-channel signal variances are frozen at
var(y)/n_channels by default (diagonal = var(y)); ``--train`` optimises them by
marginal likelihood (the principled auto-weighting -- avoids the equal-split dilution
seen at 10k).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .cutoff import cutoff_for_neighbors, recalibrate, sparsity_report
from .pipeline import (GeometryPipeline, WLGPPipeline, build_gp, predict,
                       release_gp, sort_by_category, train_hyperparameters,
                       with_category_tag)
from .reduce import LinearEmbeddingMean, regression_r2

# numeric tag for the --out rows. "additive" is not a channel: it is the SUM of every
# single channel requested in the same run.
CHANNEL_CODE = {"wl": 0, "geom": 1, "strain": 2, "elec": 3, "additive": 4}
SINGLE = ("wl", "geom", "strain", "elec")
# geometry-derived channels and the geometry sub-channels each is built from. "geom"
# takes whatever --geom-channels says; the others are single sub-channels promoted to
# first-class channels so they can be given their own embedding, their own length scale,
# or routed to the prior mean instead of the kernel.
GEOM_MODES = {"geom": None, "strain": ("strain",), "elec": ("elec",)}


def _channel_cutoff(Z_tr, Z_te, cat_tr, cat_te, args):
    """Per-channel compact-support radius: tuned to ~--target-neighbors (default), else
    the --cutoff-pct percentile of this channel's own pairwise distances."""
    if args.target_neighbors:
        return cutoff_for_neighbors(
            Z_tr, Z_te, args.target_neighbors, dim=Z_tr.shape[1],
            data_id_tr=cat_tr, data_id_te=cat_te,
        )
    c, _ = recalibrate(Z_tr, percentile=args.cutoff_pct, dim=Z_tr.shape[1])
    return c


def _measure_nugget(chan, y_tr, cat_tr, eps_frac=0.05, min_pairs=30,
                    ladder=(0.02, 0.05, 0.10, 0.20, 0.35)):
    """Variogram nugget for THIS channel set -- the statistically correct noise.

    ``--jitter`` is the observation-noise variance (``pipeline.build_gp`` passes it
    straight to ``noise_variances``), and the default 1e-6 is what produced the
    historical CG stall: near-duplicate molecules give near-identical Gram rows, so
    the smallest eigenvalue -- and hence the condition number -- is set by sigma^2.
    The nugget ``0.5<(y_i - y_j)^2>`` over pairs the descriptor cannot distinguish is
    the right value, and it is a property of the embedding, so it has to be measured
    per channel set and per N rather than carried over.

    Two details specific to the ADDITIVE kernel:

    * a row of ``K = sum_c K_c`` is only near-duplicated when the points are close in
      EVERY channel, so the nugget belongs to the CONCATENATED embedding. Measuring on
      one channel over-counts duplicates and over-estimates the noise (conservative,
      but it costs accuracy);
    * the channels have different support radii (0.30 vs 1.7 at 20k), so the
      concatenation is scaled by each channel's own cutoff first. In those units one
      support radius is 1.0 in every channel, and ``eps_frac`` means "within this
      fraction of the support radius in each channel" (hence the sqrt(C) scaling of
      the combined Euclidean radius).

    Returns ``(nugget, report)``; ``nugget`` is None when no near-duplicate pairs
    exist, in which case the caller should keep its configured jitter.
    """
    from .conditioning import near_duplicate_stats, nugget_from_duplicates

    Zn = np.hstack([c["Ztr"] / max(float(c["cutoff"]), 1e-12) for c in chan])
    scale = np.sqrt(len(chan))
    var_y = float(np.var(y_tr))
    rows = []
    for f in sorted(set(ladder) | {eps_frac}):
        eps = f * scale
        st = near_duplicate_stats(Zn, eps, cat=cat_tr)
        ng = nugget_from_duplicates(Zn, y_tr, eps, cat=cat_tr)
        rows.append((f, st["n_pairs"], st["frac_in_dup"], st["largest_cluster"], ng))
    print(f"[nugget] eps_frac  pairs  frac_in_dup  max_cluster   nugget  nugget/var(y)")
    for f, npair, frac, big, ng in rows:
        s = f"{ng:>8.3f}{ng / var_y:>14.4f}" if ng is not None else f"{'--':>8}{'--':>14}"
        print(f"[nugget] {f:>8.3f}{npair:>7,}{frac:>13.3f}{big:>13}{s}")
    # A nugget averaged over a handful of pairs is noise, not a variance component.
    # Walk UP the ladder from the requested eps until there are enough pairs to
    # estimate from; a wider eps biases the estimate upward (it admits genuinely
    # distinguishable pairs), which errs toward more noise -- the safe direction for
    # conditioning.
    chosen, used = None, None
    for f, npair, _, _, ng in rows:
        if f < eps_frac or ng is None:
            continue
        if npair >= min_pairs:
            chosen, used = ng, f
            break
    if chosen is None:
        best = max((r for r in rows if r[4] is not None), key=lambda r: r[1], default=None)
        print(f"[nugget] fewer than {min_pairs} near-duplicate pairs at every eps "
              f"(most was {best[1] if best else 0}); NOT trusting the estimate, "
              f"keeping the configured --jitter. This channel set separates the data, "
              f"so a large nugget should not be needed -- but check CG converges.")
    else:
        note = "" if used == eps_frac else f" (widened from {eps_frac} for pair count)"
        print(f"[nugget] using sigma^2 = {chosen:.4f} "
              f"({chosen / var_y:.4f} x var(y) = {var_y:.3f}) at eps_frac={used}{note}")
    return chosen, rows


def _gp_r2_channels(chan, y_tr, cat_tr, y_te, cat_te, client, args, mean_chan=None,
                    jitter=None):
    """Build a frozen (or --train'd) additive gp2Scale GP over the given channels and
    return held-out R^2. ``chan`` is a list of {"Ztr","Zte","cutoff"} dicts; the kernel
    sums one Wendland block per channel on its own column range/cutoff.

    ``mean_chan`` (default: same as ``chan``) is the channel list the LINEAR PRIOR MEAN
    is fit on, and it need not match the kernel's. That asymmetry is the point: a
    prior-mean term costs NO Gram block, adds NO neighbours, and cannot worsen
    conditioning, yet it applies to EVERY test point -- including the ones with no
    in-support neighbour, which is exactly where a compact-support kernel reverts to the
    mean. For a channel whose effect on the energy is close to additive-and-global
    (strain is a sum of harmonic bond terms, i.e. an energy estimate), the mean is the
    cheaper home; the kernel only earns its block if the relationship is local/nonlinear.
    """
    from sklearn.metrics import r2_score

    Ztr = np.hstack([c["Ztr"] for c in chan])
    Zte = np.hstack([c["Zte"] for c in chan])
    specs, off = [], 0                                # (start, stop, cutoff) per channel
    for c in chan:
        d = c["Ztr"].shape[1]
        specs.append((off, off + d, float(c["cutoff"])))
        off += d

    if getattr(args, "prior_mean", "none") == "linear":
        mc = mean_chan if mean_chan is not None else chan
        Mtr = np.hstack([c["Ztr"] for c in mc])
        Mte = np.hstack([c["Zte"] for c in mc])
        mean = LinearEmbeddingMean().fit(Mtr, y_tr)
        y_fit = y_tr - mean.predict(Mtr)              # GP models the residual
    else:
        mean, y_fit, Mte = None, y_tr, None

    Xtr = with_category_tag(Ztr, cat_tr)
    Xtr, y_fit_s, _ = sort_by_category(Xtr, y_fit)
    Xte = with_category_tag(Zte, cat_te)
    C = len(specs)
    var = float(np.var(y_fit))
    signal_var = [var / C] * C                        # equal split -> diagonal = var(y)

    t0 = time.time()
    gp, _ = build_gp(
        Xtr, y_fit_s, None, None, client,
        channels=specs, signal_var=signal_var,
        jitter=(args.jitter if jitter is None else jitter),
        batch_size=args.batch_size,
        compute_device="cpu", device=args.device, linalg_mode=args.linalg,
        cg_maxiter=args.cg_maxiter, cg_tol=args.cg_tol,
    )
    if getattr(args, "train", False):
        bounds = [[1e-6, 10.0 * var]] * C             # per-channel signal-var bounds
        hps = train_hyperparameters(gp, bounds, max_iter=args.train_iter)
        print(f"[sweep]   trained signal vars -> {np.round(np.asarray(hps), 5)}")
    t_build = time.time() - t0
    # Printed BEFORE predict on purpose: this timing IS the measurement a scaling rung
    # exists to produce, and it used to be lost whenever predict raised (a driver with
    # no GPU took down an otherwise complete solve, and the jitter arm it belonged to
    # had to be re-run from featurisation).
    print(f"[sweep]   build(+solve){'+train' if args.train else ''}={t_build:.1f}s")
    m, _ = predict(gp, Xte, batch=args.pred_batch, variance=False)
    if mean is not None:
        m = m + mean.predict(Mte)                     # add the linear mean back
    r2 = float(r2_score(y_te, m))
    del gp
    release_gp(client)
    return r2


def _ols_r2(chan, y_tr, y_te):
    """Cross-check: held-out OLS R^2 on the concatenated embedding (cutoff-free)."""
    Ztr = np.hstack([c["Ztr"] for c in chan])
    Zte = np.hstack([c["Zte"] for c in chan])
    return regression_r2(Ztr, y_tr, Zte, y_te)


def _restrict(ds, args):
    """Optionally narrow the loaded subset before anything is featurised.

    ``--subsets`` keeps only the named ``data_id``s. ``--require-perceiver`` drops
    rows a perceiver cannot handle -- `rdkit_bonds` (xyz2mol) fails on ~70% of
    train_4M, so any arm that uses it must be compared on the rows where it works,
    and EVERY arm must then see those same rows or the comparison is not paired.
    """
    from dataclasses import replace

    keep = np.ones(len(ds.atoms), dtype=bool)
    if args.subsets:
        want = set(args.subsets)
        names = ds.category_names
        keep &= np.array([names[c] in want for c in ds.data_id])
        print(f"[sweep] --subsets {sorted(want)}: {keep.sum():,} of {len(keep):,} rows")
    if args.require_perceiver:
        from .wl_features import build_graph

        ok = np.zeros(len(ds.atoms), dtype=bool)
        for i, a in enumerate(ds.atoms):
            if not keep[i]:
                continue
            try:
                adj, _ = build_graph(a, args.bond_mult, args.require_perceiver)
                ok[i] = sum(len(x) for x in adj) > 0
            except Exception:
                ok[i] = False
        print(f"[sweep] --require-perceiver {args.require_perceiver}: "
              f"{ok.sum():,} of {keep.sum():,} kept rows are usable")
        keep &= ok
    if keep.all():
        return ds
    idx = np.flatnonzero(keep)
    return replace(ds, atoms=[ds.atoms[i] for i in idx], y=ds.y[idx],
                   data_id=ds.data_id[idx])


def run(args, client):
    from sklearn.model_selection import train_test_split

    from .data import get_data

    ds = get_data(src=args.src, n=args.n, seed=args.data_seed)
    ds = _restrict(ds, args)
    modes = args.channels
    singles = [m for m in SINGLE if m in modes]
    if "additive" in modes and not singles:
        raise SystemExit("--channels additive needs at least one of wl / geom / strain "
                         "to add together")
    mean_only = [m for m in (args.mean_channels or []) if m not in singles]
    # mean-only channels still need their embedding built, just not a kernel block
    need = set(singles) | set(args.mean_channels or [])
    rows, nuggets = [], []
    print(f"[sweep] config: channels={modes} scaling={args.scaling} "
          f"prior_mean={args.prior_mean} target_nbrs={args.target_neighbors} "
          f"train={args.train} pls={args.pls} jitter={args.jitter:g} linalg={args.linalg}")

    for seed in args.seeds:
        print(f"\n########## split seed {seed} ##########")
        idx = np.arange(len(ds))
        tr, te = train_test_split(idx, test_size=args.test_size, random_state=seed)
        atoms_tr = [ds.atoms[i] for i in tr]
        atoms_te = [ds.atoms[i] for i in te]
        y_tr, y_te = ds.y[tr], ds.y[te]
        cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]

        emb = {}   # channel name -> {"Ztr","Zte","cutoff"}
        if "wl" in need:
            pw = WLGPPipeline(depth=args.depth, min_count=args.min_count,
                              pls_components=args.pls, cutoff_percentile=None,
                              scaling=args.scaling, vocab_sample=0,
                              cutoff_mult=args.bond_mult,
                              perceiver=(args.wl_perceiver or args.perceiver))
            Zw_tr = pw.fit(atoms_tr, y_tr, cat_tr, client=client)
            Zw_te = pw.transform(atoms_te, client=client)
            emb["wl"] = {"Ztr": Zw_tr, "Zte": Zw_te,
                         "cutoff": _channel_cutoff(Zw_tr, Zw_te, cat_tr, cat_te, args)}
        # geom and strain are SEPARATE channels on purpose. Their features are not
        # commensurable -- the histogram channels are O(1) per-atom frequencies while
        # strain is a sum of squared deviations -- and pareto scaling preserves scale by
        # design, so folding strain into the geometry PLS lets it dominate the embedding
        # (measured: geom GP R^2 0.317 -> 0.064, cutoff 2.4 -> 180). Own embedding, own
        # length scale: exactly what the additive kernel exists for.
        for name, sub in GEOM_MODES.items():
            if name not in need:
                continue
            chans = tuple(args.geom_channels) if sub is None else sub
            pg = GeometryPipeline(top_k=args.geom_top_k, channels=chans,
                                  r_max=args.geom_r_max, pls_components=args.pls,
                                  cutoff_percentile=None, scaling=args.scaling,
                                  charge_key=args.charge_key,
                                  cutoff_mult=args.bond_mult,
                                  perceiver=args.perceiver)
            Z_tr = pg.fit(atoms_tr, y_tr, cat_tr, client=client)
            Z_te = pg.transform(atoms_te, client=client)
            emb[name] = {"Ztr": Z_tr, "Zte": Z_te,
                         "cutoff": _channel_cutoff(Z_tr, Z_te, cat_tr, cat_te, args)}

        mean_chan = ([emb[m] for m in args.mean_channels]
                     if args.mean_channels else None)
        for mode in modes:
            chan = ([emb[m] for m in singles] if mode == "additive" else [emb[mode]])
            for c in chan:
                sparsity_report(c["Ztr"], c["cutoff"], dim=c["Ztr"].shape[1], data_id=cat_tr)
            ols = _ols_r2(mean_chan if mean_chan else chan, y_tr, y_te)
            # measured PER MODE: the additive kernel's duplicate structure is not the
            # same as any single channel's, so one global nugget would be wrong
            jit = args.jitter
            if args.measure_nugget:
                print(f"[nugget] seed={seed} mode={mode}")
                ng, _ = _measure_nugget(chan, y_tr, cat_tr, args.nugget_eps,
                                        args.nugget_min_pairs)
                if ng is not None:
                    jit = ng
            nuggets.append((seed, CHANNEL_CODE[mode], jit))
            gp_r2 = _gp_r2_channels(chan, y_tr, cat_tr, y_te, cat_te, client, args,
                                    mean_chan=mean_chan, jitter=jit)
            cuts = "+".join(f"{c['cutoff']:.3f}" for c in chan)
            extra = f"  mean+={'+'.join(mean_only)}" if mean_only else ""
            print(f"[sweep] seed={seed} mode={mode:>8}  cutoffs={cuts}  "
                  f"jitter={jit:.4g}  OLS_R2={ols:.4f}  GP_R2={gp_r2:.4f}{extra}")
            rows.append((seed, CHANNEL_CODE[mode], ols, gp_r2))

    print("\n================= SUMMARY =================")
    print(f"{'seed':>5} {'mode':>9} {'OLS_R2':>8} {'GP_R2':>8}")
    inv = {v: k for k, v in CHANNEL_CODE.items()}
    for s, code, ols, gp in rows:
        print(f"{s:>5} {inv[int(code)]:>9} {ols:>8.4f} {gp:>8.4f}")
    print("\n--- GP_R2 across seeds, by channel ---")
    for mode in modes:
        vals = np.array([gp for (s, code, ols, gp) in rows if int(code) == CHANNEL_CODE[mode]])
        if len(vals):
            print(f"  {mode:>8}: GP_R2 mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"n={len(vals)}  values={np.round(vals, 4)}")
    if args.out:
        np.savez(args.out, rows=np.array(rows, dtype=float),
                 channels=np.array(modes), seeds=np.array(args.seeds),
                 channel_code=np.array([CHANNEL_CODE[m] for m in modes]),
                 # (seed, channel_code, jitter actually used) -- so a later run at a
                 # different N can be compared against the noise it was solved with
                 nuggets=np.array(nuggets, dtype=float))
        print(f"\n[sweep] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="wl_gp2scale channel comparison (wl | geom | additive)")
    ap.add_argument("--src", default="train_4M")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--data-seed", type=int, default=0, help="subset draw seed (frozen in cache)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                    help="TRAIN/TEST split seeds (the stability axis)")
    ap.add_argument("--channels", nargs="+", default=["wl", "geom", "additive"],
                    choices=["wl", "geom", "strain", "elec", "additive"],
                    help="channel modes to evaluate (embeddings built once, reused). "
                         "'additive' = the sum of every SINGLE channel also listed, so "
                         "'--channels wl geom strain additive' scores each alone and all "
                         "three summed. Keep 'strain' separate from --geom-channels: its "
                         "features are not commensurable with the histogram channels.")
    ap.add_argument("--pls", type=int, default=10, help="PLS components per channel")
    ap.add_argument("--scaling", default="pareto",
                    choices=["pareto", "standard", "center"],
                    help="SparsePLS column pre-weighting (default pareto)")
    ap.add_argument("--prior-mean", default="linear", choices=["none", "linear"],
                    help="'linear' fits OLS on the (concatenated) embedding, GPs the "
                         "residual, adds it back (gp2Scale Eq. 2; fixes mean reversion)")
    ap.add_argument("--mean-channels", nargs="+", default=None,
                    choices=["wl", "geom", "strain", "elec"],
                    help="channels the LINEAR PRIOR MEAN is fit on (default: whatever "
                         "the kernel uses). Listing a channel here but NOT in --channels "
                         "puts it in the mean ONLY -- no Gram block, no neighbour "
                         "requirement, applies to every test point. Use it to ask "
                         "whether a channel's effect is additive-and-global (mean is "
                         "enough) or local (needs a kernel block): e.g. "
                         "--channels wl geom additive --mean-channels wl geom strain")
    ap.add_argument("--target-neighbors", type=int, default=60,
                    help="per-channel cutoff tuned to this median in-support neighbour "
                         "count (0 -> fall back to --cutoff-pct)")
    ap.add_argument("--cutoff-pct", type=float, default=25.0,
                    help="fallback percentile per channel when --target-neighbors 0")
    ap.add_argument("--train", action="store_true",
                    help="optimise per-channel signal variances by marginal likelihood "
                         "(default frozen at var(y)/n_channels)")
    ap.add_argument("--train-iter", type=int, default=50)
    # geometry channel
    ap.add_argument("--geom-channels", default="rdf,angle,torsion,elec",
                    help="comma-separated subset of rdf,angle,torsion,elec")
    ap.add_argument("--geom-top-k", type=int, default=6)
    ap.add_argument("--geom-r-max", type=float, default=6.0)
    ap.add_argument("--charge-key", default="lowdin_charges")
    # WL channel
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    # BOND PERCEPTION, not the kernel's compact support (that is --cutoff-pct /
    # --target-neighbors). This is the multiplier on ASE's covalent radii that
    # decides which atom pairs count as bonded, and it is shared by the WL, geometry
    # and strain channels. data_exploration REPORT.md §11 measured what it does to
    # the perceived graph: at 1.2 (the historical default) 88% of same-molecule
    # conformer families receive DIFFERENT WL graphs, 41% of carbons get more than
    # four bonds, and the median perceived ring count is 7; at 1.1 those fall to 62%
    # and 2. Sweeping it is the experiment that finding argued for.
    ap.add_argument("--bond-mult", type=float, default=1.2,
                    help="covalent-radius multiplier for BOND PERCEPTION "
                         "(featurisation), not the kernel cutoff")
    # WHICH perceiver, as opposed to how wide the ASE one is. "rdkit_ctd" ignores
    # --bond-mult (it has no such knob). See data_exploration/REPORT.md §16.
    ap.add_argument("--perceiver", default="ase",
                    choices=["ase", "rdkit_ctd", "rdkit_bonds"],
                    help="bond perception backend for every graph-derived channel")
    # so an arm can change the WL graph while holding the geometry channel fixed --
    # otherwise two things move at once and the comparison means nothing
    ap.add_argument("--wl-perceiver", default=None,
                    choices=["ase", "rdkit_ctd", "rdkit_bonds"],
                    help="override the perceiver for the WL channel only")
    ap.add_argument("--subsets", nargs="+", default=None,
                    help="keep only these data_id values")
    ap.add_argument("--require-perceiver", default=None,
                    choices=["rdkit_ctd", "rdkit_bonds"],
                    help="drop rows this perceiver cannot handle (paired arms)")
    # solver / cluster
    ap.add_argument("--jitter", type=float, default=1e-6,
                    help="observation-noise variance sigma^2 (goes straight to "
                         "noise_variances). The 1e-6 default is the setting that "
                         "caused the historical CG stall -- prefer --measure-nugget")
    ap.add_argument("--measure-nugget", action="store_true",
                    help="measure the variogram nugget on the built embedding and use "
                         "it as sigma^2 instead of --jitter (per mode, per seed)")
    ap.add_argument("--nugget-eps", type=float, default=0.05,
                    help="near-duplicate radius as a fraction of each channel's own "
                         "compact-support cutoff")
    ap.add_argument("--nugget-min-pairs", type=int, default=30,
                    help="minimum near-duplicate pairs before the nugget estimate is "
                         "trusted; eps is widened until reached, else --jitter stands")
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--pred-batch", type=int, default=2000)
    ap.add_argument("--linalg", default="sparseCG")
    ap.add_argument("--cg-maxiter", type=int, default=None,
                    help="cap CG iterations so an ill-conditioned split fails fast")
    ap.add_argument("--cg-tol", type=float, default=None, help="CG relative tolerance")
    ap.add_argument("--device", default="cuda", help="OUR kernel's torch device")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None)
    ap.add_argument("--out", default=None, help="optional .npz of the summary rows")
    args = ap.parse_args()
    args.geom_channels = [c.strip() for c in args.geom_channels.split(",") if c.strip()]

    if args.scheduler_file:
        from .pipeline import connect_dask
        client = connect_dask(args.scheduler_file, n_workers=args.workers)
    else:
        from distributed import Client
        client = Client(n_workers=args.workers, threads_per_worker=1)
        client.wait_for_workers(args.workers)
        print(f"[sweep] local dask: {args.workers} workers")

    try:
        run(args, client)
    finally:
        client.close()
    print("[sweep] done.")


if __name__ == "__main__":
    main()
