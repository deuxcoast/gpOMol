"""
train_mcmc.py  (wl_gp2scale)
===========================
Train the compact-support radii as KERNEL HYPERPARAMETERS, the way gp2Scale intends.

WHY THIS EXISTS. Everywhere else in this package the support radius is a frozen
preprocessing constant, chosen by a policy: pick the radius that yields ~N median
in-support neighbours. The gp2Scale paper (arXiv 2512.06143) treats it as a
hyperparameter -- sec. 4.4 names it outright, "one additional hyperparameter: the radius
of the neighbors, or the number of neighbors (which may vary as a function of the input
space)" -- and its discussion defines the method AGAINST our approach: sparsity should be
*discovered*, not "induced by user-specified choices, such as the number of inducing
points or neighbors".

That distinction is measurable here, not academic. At 20k, moving the frozen policy from
60 to 200 neighbours took the additive kernel's contribution over the linear mean from
+0.0202 to +0.0281 -- a 39% gain from a constant nobody had fitted -- and R^2 was still
rising at 200.

WHY IT IS A SEPARATE MODULE. `dim_sweep` is a frozen-hyperparameter sweep: many short
solves producing a table of R^2. This is one long chain producing a trace, a checkpoint
and a posterior. It also needs instrumentation wrapped around the whole of `gp.train`.
The two share `pipeline.build_channels` so the embeddings are provably identical -- the
headline result is a learned radius compared against dim_sweep's neighbour-count grid,
and that comparison is only paired if both sides featurise the same way.

FOUR THINGS ABOUT TRAINING UNDER gp2Scale THAT ARE EASY TO GET WRONG, all handled here:

1. `max_iter` is `n_updates`: MH sweeps, one likelihood evaluation per proposal BLOCK per
   sweep, each a full sparse covariance rebuild. gp2Scale forces `method='mcmc'` and
   raises on the gradient path, so there is no alternative.
2. `gp.train` returns the componentwise median of the LAST 1% of the trace. At
   n_updates=400 that is four samples. We read `gp.mcmc_info["max x"]` instead.
3. An unconverged Krylov solve is a *warning*, seen at most once per process, and its
   result flows into the likelihood regardless. See `pipeline.SolveWarningCounter`.
4. A radius that walks to its upper bound means the memory budget picked the answer, not
   the likelihood. `TrainResult.at_bound` makes that a reportable failure rather than a
   number someone quotes later.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from .conditioning import sparse_wendland_gram
from .cutoff import cutoff_for_neighbors, sparsity_report
from .pipeline import (SolveWarningCounter, build_channels, build_gp, predict,
                       release_gp, sort_by_category, train_hyperparameters,
                       with_category_tag)
from .radius import range_from_variogram, semivariogram
from .reduce import LinearEmbeddingMean


@dataclass
class RadiusTrainConfig:
    """Everything the driver needs. Defaults are the measured production values."""

    n: int = 20_000
    seed: int = 42                     # train/test split seed
    data_seed: int = 0
    src: str = "train_4M"
    test_size: float = 0.2
    channels: tuple = ("wl", "geom")           # kernel blocks
    mean_channels: tuple = ("wl", "geom", "strain")   # linear prior mean

    # featurisation (settled; see wl-config memory / REPORT.md sec.15)
    depth: int = 3
    bond_mult: float = 1.1
    geom_top_k: int = 10
    pls: int = 10
    min_count: int = 2
    diag_sample: int = 5000

    # MCMC
    n_updates: int = 400               # MH sweeps; 2 blocks -> ~2x likelihood evals
    seed_neighbors: int = 200          # radius the chain STARTS from
    r_bounds_span: float = 4.0         # multiplicative half-span around the centre
    budget_gb: float = 40.0            # driver-side CSR budget -> radius upper bound

    # solver. NOTE solve_tol=1e-6, not fvgp's 1e-5 default: measured, MINRES at 1e-5
    # lands 3.15e-3 off the dense reference (above our 1e-3 parity bar) while reporting
    # convergence and emitting zero warnings. At 1e-6 it matches CG (6.4e-5 vs 4.4e-5).
    linalg: str = "sparseMINRES"
    solve_tol: float = 1e-6
    solve_maxiter: int = 2000          # MUST be finite -- see SolveWarningCounter
    logdet_rtol: float = 0.01
    logdet_lanczos_degree: int = 20
    jitter: float = 0.78               # additive kernel is jitter-insensitive (measured)
    warn_fail_frac: float = 0.05

    # plumbing
    batch_size: int = 10_000
    pred_batch: int = 2000
    device: str = "cuda"
    burn_in_frac: float = 0.3
    checkpoint: str | None = None
    out: str | None = None
    # observability. A chain is hours long and the ONLY per-sample hook fvgp leaves us
    # is the prior (run_in_every_iteration is not forwarded by GPtraining.train), so
    # progress reporting lives there. 20 proposals ~= 10 sweeps at 2 blocks, matching
    # fvgp's own info cadence so the two interleave rather than fight.
    log_every: int = 20


@dataclass
class TrainResult:
    hps_map: np.ndarray = None        # gp.mcmc_info["max x"] -- the one to use
    hps_median: np.ndarray = None     # fvgp's return (last 1% of trace)
    hps_post: np.ndarray = None       # our median over trace[burn_in:]
    trace: np.ndarray = None
    loglik: np.ndarray = None
    bounds: np.ndarray = None
    at_bound: dict = field(default_factory=dict)
    accept: dict = field(default_factory=dict)
    warn_counts: dict = field(default_factory=dict)
    n_likelihood_evals: int = 0
    sec_per_eval: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    def valid(self):
        """Is the experiment interpretable? A radius pinned to a bound is not a result."""
        return not any(v for k, v in self.at_bound.items() if k.startswith("r"))


# --------------------------------------------------------------------------- bounds


def radius_bounds(chan, y_resid_tr, cat_tr, cfg, names=None):
    """Per-channel [lo, hi] for the radii, plus the signal-variance bounds.

    Returns ``(bounds, meta)`` with ``bounds`` shaped (2C, 2): rows [0:C] are the signal
    variances, rows [C:2C] the radii, matching AdditiveWendlandKernel's hps layout.

    Four numbers per channel, and each is there for a reason:

    * **centre** -- the variogram range on the RESIDUAL. Not on raw ``y``: with a linear
      prior mean the GP models ``y - m(z)``, so a variogram of raw ``y`` on a PLS
      embedding measures the linear trend PLS was built to capture, not what the kernel
      sees. ``range_from_variogram`` returns None when gamma never reaches the sill,
      which is itself informative (no decorrelation within sampled distances -> the
      signal is representational rather than spatial); we fall back to the 60-neighbour
      radius and say so.
    * **upper** -- the memory budget, inverted through ``cutoff_for_neighbors`` so the
      constraint is expressed in the units that actually bind. At 20k this does not bind
      at all (~83k neighbours' worth of budget for a 16k-row problem), so the upper bound
      there is physics (span x centre); at 200k it does bind, and the run must say so.
    * **lower** -- ``max(r_5, centre/4)``. Below ~5 neighbours K degenerates toward
      diagonal, and the hazard is not that this fits badly but that it fits *cheaply*: a
      near-diagonal K + sigma^2 I is well-conditioned with an easy log-determinant, so the
      marginal likelihood can genuinely prefer it. That is a failure mode the chain will
      find if we let it.
    * **anchors** at 5/60/200/500 neighbours, reported so the learned radius can be read
      on the same axis as the frozen grid.
    """
    names = list(names or chan.keys()) if isinstance(chan, dict) else list(names)
    C = len(names)
    N = len(y_resid_tr)
    var_resid = float(np.var(y_resid_tr))
    lo, hi, meta = np.zeros(C), np.zeros(C), {}

    # split the CSR budget across channels: an entry of the additive kernel is nonzero
    # when EITHER channel is in support, so nnz <= sum_c nnz_c is the conservative bound.
    nnz_budget_c = cfg.budget_gb * 1e9 / (12.0 * max(C, 1))
    n_target_c = max(int(nnz_budget_c / max(N, 1)), 10)

    for i, name in enumerate(names):
        c = chan[name] if isinstance(chan, dict) else chan[i]
        Ztr, Zte = c["Ztr"], c["Zte"]
        d = Ztr.shape[1]
        kw = dict(dim=d, data_id_tr=cat_tr, data_id_te=None, sample=cfg.diag_sample)
        anchors = {}
        for k in (5, 60, cfg.seed_neighbors, 500):
            anchors[k] = cutoff_for_neighbors(Ztr, Zte, k, dim=d, sample=cfg.diag_sample)
        sv = semivariogram(Ztr, y_resid_tr, sample=cfg.diag_sample, cat=cat_tr)
        r_var = range_from_variogram(sv["lag"], sv["gamma"], sv["sill"])
        centre = r_var if r_var else anchors[60]
        # If the budget allows more neighbours than there ARE points, memory is not the
        # binding constraint -- the dataset is. Distinguish the two, or a tiny run
        # reports "budget-limited" when it is nothing of the sort.
        budget_binds_here = n_target_c < N - 1
        r_hi_budget = cutoff_for_neighbors(Ztr, Zte, min(n_target_c, N - 1),
                                           dim=d, sample=cfg.diag_sample)
        lo[i] = max(anchors[5], centre / 4.0)
        hi[i] = (min(r_hi_budget, centre * cfg.r_bounds_span) if budget_binds_here
                 else centre * cfg.r_bounds_span)
        if not (hi[i] > lo[i]):                 # degenerate box -> widen off the centre
            hi[i] = max(lo[i] * 2.0, centre * cfg.r_bounds_span)
        sp = sparsity_report(Ztr, lo[i], dim=d, data_id=cat_tr, sample=cfg.diag_sample)
        meta[name] = {
            "anchors": anchors, "variogram_range": r_var, "centre": centre,
            "lo": float(lo[i]), "hi": float(hi[i]),
            "budget_neighbors": n_target_c, "r_hi_budget": float(r_hi_budget),
            "budget_binds": bool(budget_binds_here
                                 and r_hi_budget < centre * cfg.r_bounds_span),
            "frac_zero_at_lo": float(sp["frac_zero_neighbor"]),
        }
        print(f"[bounds] {name:>7}: variogram_range="
              f"{'None (no decorrelation seen -> using r_60)' if r_var is None else f'{r_var:.4f}'}"
              f"  centre={centre:.4f}  box=[{lo[i]:.4f}, {hi[i]:.4f}]")
        print(f"[bounds] {name:>7}: anchors nbrs->r " +
              "  ".join(f"{k}->{v:.3f}" for k, v in anchors.items()) +
              (f"   budget allows ~{n_target_c:,} nbrs (r<={r_hi_budget:.3f})"
               f"{'  <-- BINDS' if meta[name]['budget_binds'] else ''}"
               if budget_binds_here else
               f"   budget allows ~{n_target_c:,} nbrs > N={N:,}, so memory does not "
               f"bind at this scale (upper bound is {cfg.r_bounds_span}x the centre)"))

    bounds = np.vstack([
        np.column_stack([np.full(C, 1e-6), np.full(C, 10.0 * var_resid)]),  # signal vars
        np.column_stack([lo, hi]),                                          # radii
    ])
    assert bounds.shape == (2 * C, 2)
    assert np.all(bounds[:, 0] < bounds[:, 1]), f"degenerate bounds:\n{bounds}"
    assert np.all(bounds[C:, 0] > 0), "a zero radius gives D/0 -> nan in the kernel"
    return bounds, meta


# ------------------------------------------------------------------------ proposals


def make_proposals(bounds, C):
    """Two block-Metropolis proposals: signal variances, and radii.

    This is the paper's own "block MCMC". fvgp's default is a SINGLE block over every
    hyperparameter with one adapted step size and one shared acceptance target, which
    couples quantities whose likelihood responses differ by orders of magnitude -- a
    signal variance and a support radius do not move the objective on the same scale, and
    a step size that suits one starves the other. Per-component scaling is already
    handled (each std comes from its own bound width), so blocking is about decoupling
    the *adaptation*, not the units.

    Two blocks, not 2C: cost per sweep is linear in block count, since each block costs a
    likelihood evaluation and therefore a full sparse covariance rebuild.
    """
    from fvgp.gp_mcmc import ProposalDistribution

    lb, ub = np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1]
    std = 0.2 * (ub - lb) / np.sqrt(12.0)        # fvgp's own formula (gp_mcmc.py:84)
    i_sv, i_r = np.arange(C), np.arange(C, 2 * C)
    return [
        ProposalDistribution(i_sv, init_prop_Sigma=np.diag(std[i_sv] ** 2), ID="signal_var"),
        ProposalDistribution(i_r, init_prop_Sigma=np.diag(std[i_r] ** 2), ID="radius"),
    ]


# ---------------------------------------------------------------------- the prior


def nnz_budget_prior(meta):
    """MCMC prior that also enforces the memory budget, logs the trace, and checkpoints.

    Three jobs in one callable, because fvgp gives us exactly one per-sample hook: the
    prior is evaluated BEFORE the likelihood and a -inf return skips the solve entirely
    (gp_mcmc.py:196-199), and `run_in_every_iteration` is not forwarded by
    GPtraining.train, so there is nowhere else to stand.

    1. The box check fvgp would do anyway.
    2. The JOINT nnz constraint, which per-channel box bounds cannot express: the
       additive kernel's cost is driven by the union of the channels' supports, so a
       proposal can sit inside every per-channel bound and still be unaffordable.
       Estimated driver-side on a subsample via `conditioning.sparse_wendland_gram` --
       milliseconds against a covariance build costing tens of seconds -- and rejected
       for free before any worker is asked to do anything.
    3. Progress reporting, trace + warning-count logging, and the checkpoint write.
       This is the ONLY place any of that can happen: a chain runs for hours, and
       everything else in this module prints either before it starts or after it ends.
    """
    C = meta["C"]

    def prior(theta, bounds, args=None):
        theta = np.asarray(theta, float)
        bounds = np.asarray(bounds, float)
        inside = bool(np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])))
        nnz = np.nan
        if inside:
            nnz = estimate_union_nnz(theta[C:], meta)
        rec = (time.time(), theta.copy(), float(nnz),
               dict(meta["counter"].counts) if meta.get("counter") else {})
        meta["log"].append(rec)
        n = len(meta["log"])
        if meta.get("checkpoint") and n % 25 == 0:
            _write_checkpoint(meta)
        if meta.get("log_every") and n % meta["log_every"] == 0:
            _progress(meta, n, theta, nnz)
        if not inside:
            meta["n_box_rejects"] += 1
            return -np.inf
        if np.isfinite(nnz) and nnz > meta["nnz_budget"]:
            meta["n_budget_rejects"] += 1
            return -np.inf
        return 0.0

    return prior


def _progress(meta, n, theta, nnz):
    """One line per ``log_every`` proposals, from inside the prior.

    Reports what actually goes wrong on a long chain: the radii drifting toward a bound,
    the nnz creeping toward the memory budget, solves quietly failing to converge, and
    the cost per evaluation rising as the radius grows (so the ETA from the first few
    minutes is not the ETA for the run). ``flush=True`` because this is typically
    redirected to a file and a stalled run must not look like a quiet one.
    """
    C, total = meta["C"], meta["n_total_proposals"]
    el = time.time() - meta["t0"]
    per = el / max(n, 1)
    eta = per * max(total - n, 0)
    warns = sum(meta["counter"].counts.values()) if meta.get("counter") else 0
    r = np.asarray(theta[C:], float)
    lo, hi = meta["bounds"][C:, 0], meta["bounds"][C:, 1]
    frac = (r - lo) / np.maximum(hi - lo, 1e-12)      # 0 = at lower bound, 1 = at upper
    print(f"[mcmc] {n:>5}/{total} ({100.0*n/total:4.1f}%)  "
          f"elapsed {el/60:6.1f}m  {per:5.1f}s/prop  eta {eta/60:6.1f}m  "
          f"r={np.array2string(r, precision=4, floatmode='fixed')} "
          f"(pos in box {np.array2string(frac, precision=2, floatmode='fixed')})  "
          # nnz is only computed for in-box proposals; out-of-box ones are rejected
          # before the estimate, so report that rather than a bare nan
          f"nnz={'out-of-box' if not np.isfinite(nnz) else f'{nnz:.3g}'}"
          f"/{meta['nnz_budget']:.3g}  "
          f"rej box/budget={meta['n_box_rejects']}/{meta['n_budget_rejects']}  "
          f"noconv={warns}", flush=True)
    if frac.max() > 0.95 or frac.min() < 0.05:
        print("[mcmc]   ^ a radius is sitting against its bound; if it stays there the "
              "BOUND chose it, not the likelihood (see at_bound in the summary)",
              flush=True)


def estimate_union_nnz(radii, meta):
    """Driver-side estimate of the additive kernel's nnz at these radii.

    Unions the per-channel sparsity patterns on a shared subsample and rescales by
    (N/m)^2. Union, not sum: an entry is nonzero when ANY channel is in support, so
    summing would double-count the pairs that are close in both -- and in this data the
    channels are far from independent.
    """
    m, N = meta["m_sub"], meta["N"]
    pat = None
    for i, r in enumerate(radii):
        G = sparse_wendland_gram(meta["Zsub"][i], float(r), 1.0, cat=meta["cat_sub"])
        Gb = (G != 0)
        # .maximum(), not +: on boolean sparse, addition is not a union (it upcasts and
        # can double-count), whereas elementwise max is exactly "nonzero in either".
        pat = Gb if pat is None else pat.maximum(Gb)
    return float(pat.nnz) * (N / m) ** 2 if pat is not None else 0.0


def _write_checkpoint(meta):
    import numpy as _np

    log = meta["log"]
    keys = sorted(log[-1][3]) if log and log[-1][3] else []
    _np.savez(meta["checkpoint"],
              theta=_np.array([r[1] for r in log], dtype=float),
              t=_np.array([r[0] for r in log], dtype=float),
              nnz=_np.array([r[2] for r in log], dtype=float),
              # cumulative non-convergence per proposal: the DELTA between consecutive
              # rows attributes failures to the specific likelihood evaluation that
              # produced them, which is the only per-sample attribution available.
              warn_counts=_np.array([[r[3].get(k, 0) for k in keys] for r in log],
                                    dtype=float),
              warn_keys=_np.array(keys),
              bounds=meta["bounds"], n_budget_rejects=meta["n_budget_rejects"],
              n_box_rejects=meta["n_box_rejects"])


# ------------------------------------------------------------------------- driver


def train_radii(chan, y_tr, cat_tr, client, cfg, names=None):
    """Run the block-MCMC over [signal variances, radii]. Returns a TrainResult."""
    names = list(names or (chan.keys() if isinstance(chan, dict) else range(len(chan))))
    C = len(names)
    specs, off = [], 0
    Ztr_list = []
    for name in names:
        c = chan[name] if isinstance(chan, dict) else chan[name]
        d = c["Ztr"].shape[1]
        specs.append((off, off + d, float(c["cutoff"])))
        Ztr_list.append(c["Ztr"])
        off += d
    Ztr = np.hstack(Ztr_list)

    # the GP models the residual after the linear prior mean, exactly as dim_sweep does
    mean = LinearEmbeddingMean().fit(Ztr, y_tr)
    y_resid = y_tr - mean.predict(Ztr)

    bounds, bmeta = radius_bounds(chan, y_resid, cat_tr, cfg, names=names)
    Xtr = with_category_tag(Ztr, cat_tr)
    Xtr, y_fit_s, order = sort_by_category(Xtr, y_resid)

    # seed the radii at the requested neighbour count, so the chain starts on the grid's
    # axis rather than wherever the frozen policy happened to sit
    seed_r = np.array([bmeta[n]["anchors"][cfg.seed_neighbors] for n in names])
    seed_r = np.clip(seed_r, bounds[C:, 0], bounds[C:, 1])
    specs = [(s, e, float(r)) for (s, e, _), r in zip(specs, seed_r)]
    sv0 = np.full(C, float(np.var(y_resid)) / C)
    init_hps = np.concatenate([sv0, seed_r])

    m_sub = min(cfg.diag_sample, len(Ztr))
    sub = np.random.default_rng(0).choice(len(Ztr), size=m_sub, replace=False)
    meta = {
        "C": C, "N": len(Ztr), "m_sub": m_sub,
        "Zsub": [(chan[n]["Ztr"] if isinstance(chan, dict) else chan[n]["Ztr"])[sub]
                 for n in names],
        "cat_sub": np.asarray(cat_tr)[sub],
        "nnz_budget": cfg.budget_gb * 1e9 / 12.0,
        "log": [], "n_budget_rejects": 0, "n_box_rejects": 0, "bounds": bounds,
        "checkpoint": cfg.checkpoint, "counter": None,
        "log_every": cfg.log_every, "t0": time.time(),
        # 2 proposal blocks -> 2 prior calls per sweep; drives the progress ETA
        "n_total_proposals": 2 * cfg.n_updates,
    }

    gp, _ = build_gp(
        Xtr, y_fit_s, None, None, client, channels=specs, signal_var=sv0,
        cutoffs_are_hp=True, jitter=cfg.jitter, batch_size=cfg.batch_size,
        compute_device="cpu", device=cfg.device, linalg_mode=cfg.linalg,
        solve_maxiter=cfg.solve_maxiter, solve_tol=cfg.solve_tol,
        logdet_rtol=cfg.logdet_rtol, logdet_lanczos_degree=cfg.logdet_lanczos_degree,
    )
    print(f"[train] {C} channels, {2*C} hyperparameters, init={np.round(init_hps, 4)}")
    print(f"[train] {cfg.n_updates} sweeps x 2 blocks ~= {2*cfg.n_updates} likelihood "
          f"evaluations, each a full sparse covariance rebuild")
    print(f"[train] progress every {cfg.log_every} proposals; fvgp reports f(x) every "
          f"10 sweeps. Run with `python -u` so both reach the log promptly.")

    res = TrainResult(bounds=bounds, diagnostics=bmeta)
    props = make_proposals(bounds, C)   # kept, so jump_trace is readable afterwards
    meta["t0"] = t0 = time.time()
    with SolveWarningCounter() as counter:
        meta["counter"] = counter
        hps_median = train_hyperparameters(
            # info=True: fvgp prints the log-likelihood every 10 sweeps. It is the one
            # quantity the prior cannot report (the prior runs BEFORE the likelihood),
            # and a flat or wildly swinging f(x) is the first sign of a bad chain.
            gp, bounds, max_iter=cfg.n_updates, info=True,
            init_hyperparameters=init_hps,
            mcmc_prop_distrs=props,
            mcmc_prior=nnz_budget_prior(meta),
        )
    wall = time.time() - t0
    # both rejection kinds short-circuit BEFORE the likelihood, so neither cost a solve
    n_evals = max(len(meta["log"]) - meta["n_budget_rejects"]
                  - meta["n_box_rejects"], 1)

    info = getattr(gp, "mcmc_info", {}) or {}
    trace = np.asarray(info.get("x", np.empty((0, 2 * C))), dtype=float)
    burn = int(cfg.burn_in_frac * len(trace))
    res.hps_median = np.asarray(hps_median, float)
    res.hps_map = np.asarray(info.get("max x", hps_median), float)
    res.hps_post = (np.median(trace[burn:], axis=0) if len(trace) > burn
                    else res.hps_median)
    res.trace, res.loglik = trace, np.asarray(info.get("f(x)", []), dtype=float)
    res.n_likelihood_evals = n_evals
    res.sec_per_eval = wall / n_evals
    res.warn_counts = dict(counter.counts)
    # Per-block acceptance. The adapter targets 0.234; far below means the proposal is
    # too bold and the chain is stuck, far above means it is crawling. Reported per
    # block because that asymmetry is the whole reason for blocking in the first place.
    for p in props:
        jt = np.asarray(getattr(p, "jump_trace", []), dtype=float).ravel()
        b = int(cfg.burn_in_frac * len(jt))
        res.accept[p.ID] = float(jt[b:].mean()) if len(jt) > b else float("nan")
    print(f"[train] wall {wall/60:.1f}m, {n_evals} likelihood evals "
          f"({meta['n_box_rejects']} box + {meta['n_budget_rejects']} budget rejects "
          f"cost no solve), {res.sec_per_eval:.1f}s/eval")
    print("[train] acceptance (target 0.234): "
          + "  ".join(f"{k}={v:.3f}" for k, v in res.accept.items()))
    for k, v in res.accept.items():
        if np.isfinite(v) and not (0.10 <= v <= 0.60):
            print(f"[train]   ^ the {k} block is {'crawling' if v > 0.6 else 'stuck'} "
                  f"at {v:.2f}: steps are too "
                  f"{'small' if v > 0.6 else 'bold'}, so it is exploring very little of "
                  f"its range. Treat its posterior as unconverged regardless of "
                  f"n_updates.")
    counter.report(n_evals=n_evals, fail_frac=cfg.warn_fail_frac, label="train")

    # a radius pinned to a bound means the BUDGET chose it, not the likelihood
    tol = 1e-3 * (bounds[:, 1] - bounds[:, 0])
    labels = [f"sv_{n}" for n in names] + [f"r_{n}" for n in names]
    for k, lab in enumerate(labels):
        v = res.hps_map[k]
        res.at_bound[lab] = ("lo" if v <= bounds[k, 0] + tol[k] else
                             "hi" if v >= bounds[k, 1] - tol[k] else None)
    del gp
    release_gp(client)
    return res


def report(res, names, cfg, chan=None, cat_tr=None):
    """Print the learned radii in BOTH units -- absolute, and median neighbours, so the
    result lands on the same axis as dim_sweep's neighbour-count grid."""
    C = len(names)
    print("\n" + "=" * 74)
    print(f"{'hyperparameter':<16} {'MAP':>10} {'median':>10} {'burn-in':>10} "
          f"{'bounds':>22} {'at bound':>9}")
    print("=" * 74)
    labels = [f"sv_{n}" for n in names] + [f"r_{n}" for n in names]
    for k, lab in enumerate(labels):
        b = res.bounds[k]
        print(f"{lab:<16} {res.hps_map[k]:>10.4f} {res.hps_median[k]:>10.4f} "
              f"{res.hps_post[k]:>10.4f} {f'[{b[0]:.3f}, {b[1]:.3f}]':>22} "
              f"{str(res.at_bound[lab] or '-'):>9}")
    if chan is not None:
        print()
        for i, n in enumerate(names):
            r = float(res.hps_map[C + i])
            sp = sparsity_report(chan[n]["Ztr"], r, dim=chan[n]["Ztr"].shape[1],
                                 data_id=cat_tr, sample=cfg.diag_sample)
            print(f"[learned] {n:>7}: r*={r:.4f} -> median {sp['median_neighbors']:.0f} "
                  f"neighbours (frozen grid runs 60 / 200 / 500)")
    print(f"\n[train] {res.n_likelihood_evals} likelihood evals, "
          f"{res.sec_per_eval:.1f}s each")
    if not res.valid():
        pinned = [k for k, v in res.at_bound.items() if v and k.startswith("r")]
        print(f"\n*** EXPERIMENT NOT INTERPRETABLE: {pinned} pinned to a bound. ***")
        print("    A radius at its UPPER bound means the memory budget chose it, not the")
        print("    likelihood -- the conclusion is 'widen the grid', not 'training works'.")
        print("    At its LOWER bound the chain found the degenerate near-diagonal mode,")
        print("    where an easy log-determinant beats actual signal. Do not quote it.")
    spread = float(np.max(np.abs(res.hps_map - res.hps_post)))
    if spread > 0.1 * float(np.mean(np.abs(res.hps_map) + 1e-12)):
        print(f"\n[train] WARNING: MAP and burn-in median differ by {spread:.4g}; "
              f"the chain has probably not mixed -- raise --n-updates.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    cfg = RadiusTrainConfig()
    for f, v in vars(cfg).items():
        if isinstance(v, tuple):
            ap.add_argument(f"--{f.replace('_','-')}", nargs="+", default=list(v))
        elif isinstance(v, bool):
            ap.add_argument(f"--{f.replace('_','-')}", action="store_true")
        else:
            t = type(v) if v is not None else str
            ap.add_argument(f"--{f.replace('_','-')}", type=t, default=v)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scheduler-file", default=None)
    a = ap.parse_args()
    cfg = RadiusTrainConfig(**{k: (tuple(v) if isinstance(getattr(cfg, k, None), tuple)
                                   else v)
                               for k, v in vars(a).items()
                               if k not in ("workers", "scheduler_file")})

    from sklearn.model_selection import train_test_split

    from .data import get_data
    from .pipeline import connect_dask

    client = connect_dask(a.scheduler_file, a.workers)
    ds = get_data(src=cfg.src, n=cfg.n, seed=cfg.data_seed)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed)
    names = list(cfg.channels)
    chan = build_channels(
        [ds.atoms[i] for i in tr], ds.y[tr], ds.data_id[tr],
        [ds.atoms[i] for i in te], ds.data_id[te],
        set(names) | set(cfg.mean_channels), client=client,
        pls=cfg.pls, depth=cfg.depth, min_count=cfg.min_count, bond_mult=cfg.bond_mult,
        geom_top_k=cfg.geom_top_k, target_neighbors=cfg.seed_neighbors,
        diag_sample=cfg.diag_sample,
    )
    res = train_radii(chan, ds.y[tr], ds.data_id[tr], client, cfg, names=names)
    report(res, names, cfg, chan=chan, cat_tr=ds.data_id[tr])
    if cfg.out:
        np.savez(cfg.out, hps_map=res.hps_map, hps_median=res.hps_median,
                 hps_post=res.hps_post, trace=res.trace, loglik=res.loglik,
                 bounds=res.bounds, names=np.array(names),
                 sec_per_eval=res.sec_per_eval)
        print(f"[train] wrote {cfg.out}")


if __name__ == "__main__":
    main()
