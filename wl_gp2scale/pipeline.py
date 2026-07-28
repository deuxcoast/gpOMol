"""
pipeline.py  (wl_gp2scale)
==========================
Glue that turns molecules into a distributed gp2Scale GP fit/predict:

    load -> WL featurize (sparse) -> supervised reduce (10-D) -> tag+sort by
    category -> recalibrate cutoff -> GPOptimizer(gp2Scale, sparseCG) -> predict

Supervised steps (vocab, PLS, cutoff) are fit on TRAIN only and the frozen
transforms applied to TEST -- no leakage.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .cutoff import cutoff_for_neighbors, recalibrate
from .data import stratified_sample_indices
from .geometry_features import SparseGeometryFeaturizer
from .kernel import make_additive_kernel, make_wl_block_kernel
from .reduce import SparsePLS
from .wl_features import SparseWLFeaturizer


# ----------------------------- embedding pipeline --------------------------


@dataclass
class WLGPPipeline:
    depth: int = 3
    min_count: int = 5
    pls_components: int = 10
    scaling: str = "pareto"  # SparsePLS column pre-weighting (grid-chosen; see reduce)
    cutoff_percentile: float = 25.0  # None => skip cutoff calibration (caller supplies
                                     # it, e.g. dim_sweep sweeps --cutoffs); avoids a
                                     # wasted/confusing recalibrate print in that path
    cutoff_abs: float = None  # absolute compact-support radius; if set, OVERRIDES the
                              # percentile (the scale is now N-invariant, so an absolute
                              # radius from the variogram/R_inf transfers across N)
    vocab_sample: int = 0  # 0 = fit vocab on ALL train (no OOV); >0 = stratified cap
    cutoff_mult: float = 1.2
    perceiver: str = "ase"
    # fitted state
    featurizer: SparseWLFeaturizer = field(default=None, repr=False)
    reducer: SparsePLS = field(default=None, repr=False)
    cutoff_: float = None
    dim_: int = None

    def fit(self, atoms, y, data_id, client=None, chunk=500):
        """Fit vocab + supervised PLS on TRAIN, recalibrate cutoff. Returns the train
        embedding Z (N, pls_components) to avoid recompute.

        Vocabulary scope: fitting on a subsample leaves train labels out-of-vocabulary
        (they get DROPPED), which throws away signal -- descriptor_eval/gp_parity.py
        fits on all of train and so has 0% train OOV. We therefore use ALL training
        molecules unless vocab_sample is smaller than the train set, and warn when a
        subsample is actually in force."""
        if self.vocab_sample and self.vocab_sample < len(atoms):
            sidx = stratified_sample_indices(np.asarray(data_id), self.vocab_sample)
            fit_atoms = [atoms[i] for i in sidx]
            print(
                f"[pipe] fitting WL vocab on {len(sidx):,} stratified sample molecules "
                f"of {len(atoms):,} train -> expect NONZERO train OOV (dropped signal). "
                f"Raise vocab_sample to >= n_train for parity with gp_parity.py."
            )
        else:
            fit_atoms = atoms
            print(f"[pipe] fitting WL vocab on ALL {len(atoms):,} training molecules")
        self.featurizer = SparseWLFeaturizer(
            depth=self.depth, min_count=self.min_count, cutoff_mult=self.cutoff_mult,
            perceiver=self.perceiver,
        ).fit(fit_atoms)

        X_tr = self.featurizer.transform(atoms, client=client, chunk=chunk)
        self.reducer = SparsePLS(
            n_components=self.pls_components, scaling=self.scaling
        ).fit(X_tr, y)
        Z_tr = self.reducer.transform(X_tr)
        self.dim_ = Z_tr.shape[1]
        if self.cutoff_abs is not None:
            self.cutoff_ = float(self.cutoff_abs)
            from scipy.spatial.distance import pdist
            from scipy.stats import percentileofscore
            samp = Z_tr[: min(2500, len(Z_tr)), : self.dim_]
            pct = float(percentileofscore(pdist(samp), self.cutoff_))
            print(f"[pipe] using ABSOLUTE cutoff {self.cutoff_:.5f} "
                  f"(~{pct:.2f}th pairwise-distance pctile here; --cutoff-pct ignored)")
        elif self.cutoff_percentile is not None:
            self.cutoff_, _ = recalibrate(
                Z_tr, percentile=self.cutoff_percentile, dim=self.dim_
            )
        else:
            self.cutoff_ = None   # caller supplies the cutoff (e.g. dim_sweep sweeps it)
        return Z_tr

    def transform(self, atoms, client=None, chunk=500):
        X = self.featurizer.transform(atoms, client=client, chunk=chunk)
        return self.reducer.transform(X)


@dataclass
class GeometryPipeline:
    """Geometry+charge channel counterpart to WLGPPipeline (same fit/transform
    interface): fit the geometry element vocab + natural-scaled SparsePLS on TRAIN,
    recalibrate the compact-support cutoff, return the train embedding. Reuses
    SparsePLS (natural scaling -> N-invariant cutoff) and cutoff.recalibrate so the
    geometry channel behaves like the WL channel in the additive kernel."""

    top_k: int = 6
    channels: tuple = ("rdf", "angle", "torsion", "elec")
    r_max: float = 6.0
    charge_key: str = "lowdin_charges"
    pls_components: int = 10
    scaling: str = "pareto"
    cutoff_percentile: float = 25.0
    cutoff_abs: float = None
    cutoff_mult: float = 1.2
    perceiver: str = "ase"
    # fitted state
    featurizer: SparseGeometryFeaturizer = field(default=None, repr=False)
    reducer: SparsePLS = field(default=None, repr=False)
    cutoff_: float = None
    dim_: int = None

    def fit(self, atoms, y, data_id=None, client=None, chunk=500):
        print(f"[geom-pipe] fitting geometry channel on {len(atoms):,} training molecules")
        self.featurizer = SparseGeometryFeaturizer(
            top_k=self.top_k, channels=tuple(self.channels), r_max=self.r_max,
            charge_key=self.charge_key, cutoff_mult=self.cutoff_mult,
            perceiver=self.perceiver,
        ).fit(atoms)
        X_tr = self.featurizer.transform(atoms, client=client, chunk=chunk)
        self.reducer = SparsePLS(
            n_components=self.pls_components, scaling=self.scaling
        ).fit(X_tr, y)
        Z_tr = self.reducer.transform(X_tr)
        self.dim_ = Z_tr.shape[1]
        if self.cutoff_abs is not None:
            self.cutoff_ = float(self.cutoff_abs)
        elif self.cutoff_percentile is not None:
            self.cutoff_, _ = recalibrate(
                Z_tr, percentile=self.cutoff_percentile, dim=self.dim_
            )
        else:
            self.cutoff_ = None
        return Z_tr

    def transform(self, atoms, client=None, chunk=500):
        X = self.featurizer.transform(atoms, client=client, chunk=chunk)
        return self.reducer.transform(X)


# ----------------------------- category tag / sort -------------------------


def with_category_tag(Z, data_id):
    """Append the integer category id as the last column (the kernel reads it)."""
    return np.hstack([np.asarray(Z, float), np.asarray(data_id, float)[:, None]])


def sort_by_category(Z_tagged, y):
    """Contiguous category blocks (stable) so most off-diagonal gp2Scale blocks are
    single-category and get skipped. Returns (Z_sorted, y_sorted, order)."""
    order = np.argsort(Z_tagged[:, -1], kind="stable")
    return Z_tagged[order], np.asarray(y)[order], order


# ----------------------------- Dask connection -----------------------------


def connect_dask(scheduler_file=None, n_workers=16, poll_timeout=1800,
                 worker_timeout=300):
    """Connect to the Perlmutter scheduler file (poll until it appears) or start a
    local Client(). Waits for n_workers before returning."""
    from distributed import Client

    if scheduler_file is None:
        scheduler_file = os.path.join(
            os.environ.get("SCRATCH", "."), "scheduler_file_gpOmol.json"
        )
    if os.environ.get("SCRATCH") or os.path.exists(scheduler_file):
        t0 = time.time()
        while not os.path.isfile(scheduler_file):
            if time.time() - t0 > poll_timeout:
                raise TimeoutError(f"scheduler file never appeared: {scheduler_file}")
            time.sleep(2)
        # The scheduler file lives in $SCRATCH and PERSISTS across allocations. If it
        # is left over from a previous allocation, its address points at a node that
        # is gone, and Client() dies with a 30s timeout + a deep tornado traceback.
        # Catch that and say what actually happened.
        try:
            client = Client(scheduler_file=scheduler_file)
        except (OSError, TimeoutError) as e:
            try:
                import json
                addr = json.load(open(scheduler_file)).get("address", "?")
            except Exception:
                addr = "?"
            raise RuntimeError(
                f"could not reach the Dask scheduler at {addr} (from {scheduler_file}). "
                f"This scheduler_file is almost certainly STALE -- left from a previous "
                f"allocation whose scheduler no longer exists. In THIS allocation, "
                f"(re)launch the cluster first: `./launch-dask-conda.sh {n_workers}` "
                f"(it rm's the stale file and writes a fresh one), wait for the workers "
                f"to register, then rerun."
            ) from None
        print(f"[dask] connected via {scheduler_file}")
    else:
        client = Client()
        print("[dask] started a local cluster (no scheduler file)")
    if n_workers:
        print(f"[dask] waiting for {n_workers} workers ...")
        # Bounded wait. client.wait_for_workers() defaults to timeout=None, i.e. it
        # blocks FOREVER if the cluster was launched with fewer workers than asked
        # for -- which silently burns the allocation (4 GPU nodes) while looking busy.
        # The common cause is a mismatch: `./launch-dask-conda.sh 4` against
        # `--workers 16`. Fail in seconds with the actual counts instead.
        try:
            client.wait_for_workers(n_workers, timeout=worker_timeout)
        except Exception:
            have = len(client.nthreads())
            raise RuntimeError(
                f"only {have} of {n_workers} workers registered after "
                f"{worker_timeout}s. The cluster's worker count must match: "
                f"`./launch-dask-conda.sh {n_workers}` (and salloc -n {n_workers}) "
                f"vs --workers {n_workers}. Currently {have} are up."
            ) from None
    # client.nthreads() is a live RPC to the scheduler. client.scheduler_info() reads
    # a CACHED identity that can lag right after wait_for_workers returns -- it once
    # reported "5 workers ready" on a healthy 16-worker cluster (the scheduler log
    # showed all 16 registered and none removed), which read as a cluster failure and
    # cost a round of debugging. wait_for_workers returning is the authoritative fact.
    n_live = len(client.nthreads())
    print(f"[dask] {n_live} workers ready")
    return client


# ----------------------------- GP fit / predict ----------------------------


def _first(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k]).ravel()
    raise KeyError(f"none of {keys} in gpCAM keys {list(d)}")


def require_imate():
    """gpcam 8.4.1 / fvgp 4.8.3 import `imate` inside the gp2Scale constructor
    (for the randomised log-determinant), so it is REQUIRED to even instantiate a
    gp2Scale GPOptimizer -- not only for training. It is NOT in requirements.txt.
    Fail early with a clear message instead of a deep traceback."""
    try:
        import imate  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "gp2Scale requires `imate`, which is not installed. Install it in the "
            "`gpomol` venv (`pip install imate`) and validate on Perlmutter before "
            "running. This is a hard requirement of gpcam 8.4.1's gp2Scale path, "
            "not optional-for-training."
        ) from e


def build_gp(
    X_tr,
    y_tr,
    cutoff,
    dim,
    client,
    signal_var=None,
    jitter=1e-6,
    batch_size=10_000,
    backend="wendland32",
    linalg_mode="sparseCG",
    compute_device="cpu",
    device=None,
    dtype="float64",
    cutoff_is_hp=False,
    cutoffs_are_hp=False,
    logdet_rtol=0.5,
    logdet_lanczos_degree=None,
    solve_maxiter=None,
    solve_tol=None,
    cg_maxiter=None,
    cg_tol=None,
    logdet_verbose=False,
    args=None,
    channels=None,
):
    """Construct the gp2Scale GPOptimizer with the sparse GPU block kernel.

    compute_device vs device -- these are DIFFERENT knobs, keep them apart:
      * `device` is OUR kernel's torch device. Set it to "cuda" to build the blocks
        on the GPU. This is where the GPU actually earns its keep.
      * `compute_device` is fvgp's. It selects fvgp's own linear algebra (dense torch
        paths we never touch, since we are sparse) and -- the trap -- whether imate
        runs its logdet on the GPU:
            gpu = compute_device == "gpu" and _imate_gpu_enabled(args)   # gp_lin_alg.py:1027
        `_imate_gpu_enabled` checks whether TORCH/cupy have CUDA, NOT whether imate
        was built with it. A pip-installed imate has no CUDA support, so on a GPU node
        compute_device="gpu" green-lights a backend that then dies with
            ImportError: This package has not been compiled with GPU support
        Hence the default "cpu": it costs nothing (the kernel still runs on `device`,
        and the sparse solve is scipy/CPU regardless) and avoids the broken path.
        Only pass "gpu" if imate was rebuilt with USE_CUDA=1.

    dtype defaults to float64 on purpose: this Gram is near-singular (cond ~1e9),
    so float32 kernel error amplifies into a wrong solve. An earlier version of this
    function did not forward dtype at all, silently running the kernel in float32
    while the dense reference ran float64 -- that alone moved R^2 from 0.049 to 0.027.

    cutoff_is_hp vs cutoffs_are_hp -- these address DIFFERENT kernels and must not be
    conflated: `cutoff_is_hp` is the single-block `WLBlockKernel`'s flag (hps[1]),
    `cutoffs_are_hp` is the additive kernel's (hps[C:2C]). Passing one where the other
    is meant silently trains nothing.

    ``cutoffs_are_hp=True`` makes the compact-support radii TRAINABLE, which is what the
    gp2Scale paper (arXiv 2512.06143 sec. 4.4) actually prescribes -- "one additional
    hyperparameter: the radius of the neighbors, or the number of neighbors". Our
    default is still the frozen radius chosen by the neighbour-count policy, which the
    paper's own discussion names as the thing competing methods get wrong (sparsity
    "induced by user-specified choices, such as the number of inducing points or
    neighbors" rather than discovered). Measured at 20k, moving the frozen radius from
    60 to 200 neighbours grew the kernel's contribution over the linear mean by 39%,
    so the frozen policy is leaving signal on the table.

    solve_maxiter / solve_tol are solver-agnostic. They used to be `cg_maxiter` /
    `cg_tol` and emitted ONLY `sparse_cg_*`, which meant the fail-fast iteration cap was
    silently INERT under MINRES -- fvgp's MINRES path reads `sparse_minres_maxiter` /
    `sparse_minres_tol` (gp_lin_alg.py:1059-1063) and never looks at the CG keys. Since
    an unconverged Krylov solve is only a WARNING in fvgp (gp_lin_alg.py:1071, 1152) and
    the result flows into the likelihood regardless, an inert cap is a correctness bug,
    not just a performance one. The old names remain as deprecated aliases.
    """

    require_imate()
    from gpcam import GPOptimizer

    if channels is not None:
        # Additive N-channel kernel k = sum_c sv_c psi_c. `channels` is a list of
        # ChannelSpec (or (start, stop, cutoff[, backend, k]) tuples); `signal_var` is
        # the per-channel signal-variance vector (default: var(y) split equally, so the
        # diagonal stays var(y)). hps = the C signal variances; cutoffs stay frozen.
        kern = make_additive_kernel(
            channels, use_category_tag=True, device=device, dtype=dtype,
            cutoffs_are_hp=cutoffs_are_hp,
        )
        if signal_var is None:
            init_hps = np.full(len(channels), float(np.var(y_tr)) / len(channels))
        else:
            init_hps = np.asarray(signal_var, dtype=float).ravel()
        if cutoffs_are_hp:
            # Seed the radii from the ChannelSpecs themselves, so the FIRST likelihood
            # evaluation is exactly the frozen configuration and the chain starts at the
            # known-good point rather than somewhere fvgp invented. `channels` may hold
            # ChannelSpec objects or raw (start, stop, cutoff) tuples (kernel.py:299).
            r0 = np.array([float(c.cutoff if hasattr(c, "cutoff") else c[2])
                           for c in channels])
            init_hps = np.concatenate([init_hps, r0])
    else:
        kern = make_wl_block_kernel(
            cutoff,
            dim=dim,
            use_category_tag=True,
            backend=backend,
            device=device,
            dtype=dtype,
            cutoff_is_hp=cutoff_is_hp,
        )
        sv = float(signal_var) if signal_var is not None else float(np.var(y_tr))
        init_hps = np.array([sv, cutoff]) if cutoff_is_hp else np.array([sv])

    # fvgp will NOT catch a length error here. `out_of_bounds` (gp.py:1917) loops
    # range(len(x)) indexing bounds[i], so a too-long hps raises a bare IndexError deep
    # in training and a too-short one silently leaves init short while gpMCMC's prior
    # broadcast-misbehaves. Assert at construction, where the message can be useful.
    _expect = (2 if cutoffs_are_hp else 1) * len(channels) if channels is not None \
        else (2 if cutoff_is_hp else 1)
    assert len(init_hps) == _expect, (
        f"init_hps has length {len(init_hps)}, expected {_expect} "
        f"(channels={None if channels is None else len(channels)}, "
        f"cutoffs_are_hp={cutoffs_are_hp}, cutoff_is_hp={cutoff_is_hp}). "
        f"hyperparameter_bounds passed to train() must match this length."
    )

    # fvgp computes log|KV| in the constructor unconditionally (GPkv._refresh), but
    # predict-only never READS it (only gp_marginal_likelihood.py does). Measured, it
    # is a small part of the constructor -- ~9s of 167s at 20k, since the near-singular
    # alpha CG solve dominates -- so we do NOT try to eliminate it (an earlier monkey-
    # patch that stubbed it to 0.0 was removed: fragile, and not worth ~4% of wall
    # time). We do keep it CHEAP the clean way: fvgp reads args["random_logdet_error_
    # rtol"], and a loose value makes imate's stochastic-Lanczos stop at its floor
    # (min_num_samples=10) instead of refining a number we discard. Tighten to 0.01
    # under --train, which actually uses the value.
    _args = dict(args or {})
    _args.setdefault("random_logdet_error_rtol", float(logdet_rtol))
    if logdet_lanczos_degree is not None:
        # The only other exposed lever on the stochastic-Lanczos logdet; min_num_samples,
        # max_num_samples and orthogonalize are hard-coded (gp_lin_alg.py:1039-1041).
        # Matters for a trainable radius: log|KV| grows with r_0, so the ABSOLUTE SLQ
        # error grows with it too, biasing the MH accept/reject in a radius-dependent way.
        _args.setdefault("random_logdet_lanczos_degree", int(logdet_lanczos_degree))
    # Solve instrumentation / fail-fast: cap the Krylov solver so an ill-conditioned split
    # fails fast (fvgp warns "CG/MINRES not successful") instead of grinding for 20 min.
    #
    # Emit under EVERY key any solver reads rather than branching on linalg_mode. This is
    # safe, not sloppy: _resolve_krylov_maxiter (gp_lin_alg.py:893-901) returns on the
    # solver-specific key BEFORE consulting the generic sparse_krylov_maxiter, and CG's
    # tolerance chain checks sparse_cg_tol FIRST (gp_lin_alg.py:1095). So the extra keys
    # cannot change what CG resolves -- they only stop the cap being silently dropped when
    # the mode is MINRES.
    _maxiter = solve_maxiter if solve_maxiter is not None else cg_maxiter
    _tol = solve_tol if solve_tol is not None else cg_tol
    if _maxiter is not None:
        _args.setdefault("sparse_cg_maxiter", int(_maxiter))
        _args.setdefault("sparse_minres_maxiter", int(_maxiter))
        _args.setdefault("sparse_krylov_maxiter", int(_maxiter))
    if _tol is not None:
        _args.setdefault("sparse_cg_tol", float(_tol))
        _args.setdefault("sparse_minres_tol", float(_tol))
    if logdet_verbose:
        _args.setdefault("random_logdet_verbose", True)
        _args.setdefault("random_logdet_print_info", True)

    gp = GPOptimizer(
        x_data=np.asarray(X_tr, float),
        y_data=np.asarray(y_tr, float),
        init_hyperparameters=init_hps,
        noise_variances=jitter * np.ones(len(y_tr)),
        compute_device=compute_device,
        kernel_function=kern,
        gp2Scale=True,
        gp2Scale_batch_size=batch_size,
        dask_client=client,
        linalg_mode=linalg_mode,
        args=_args,
    )
    return gp, kern


def release_gp(client):
    """Free the active gp2Scale GP before building another on the SAME dask client.

    fvgp 4.8.3 forbids two live gp2Scale GPs per client (a `WeakValueDictionary`
    guard keyed by client.id; scatter refcount race). Call between sequential fits,
    AFTER dropping your own reference to the previous GP (``del gp``). This clears
    the registry entry and flushes pending scatter releases on the workers. For
    truly independent runs, prefer a fresh client per GP."""
    import gc

    gc.collect()
    try:
        from fvgp.gp import _GP_INSTANCES_PER_CLIENT
        _GP_INSTANCES_PER_CLIENT.pop(client.id, None)
    except Exception:
        pass
    try:
        client.run(lambda: None)  # flush pending scatter releases on workers
    except Exception:
        pass


def predict(gp, X_te, batch=None, variance=True, verbose=False):
    """Posterior mean and (optionally) variance on the test embedding, in batches.

    Batching is not cosmetic at 200k. fvgp builds the cross-covariance
    k = kernel(x_data, x_pred) DENSE (gp_posterior.py:185); at 196k train x 4k test
    that is ~6.3 GB in one allocation. Batching bounds it to n_train x batch.

    Cost asymmetry -- read this before choosing a test-set size:
      * posterior_mean uses the PRECOMPUTED KVinvY (A = k.T @ KVinvY), so it costs
        ONE solve in total no matter how many test points. Cheap once k is bounded.
      * posterior_covariance calls KVsolve(k), i.e. ONE SOLVE PER TEST POINT against
        the full N x N system. At N=196k that is hours-to-days for a few thousand
        test points, and batching does NOT reduce the total work (only peak memory).

    So at 200k: keep the variance test set small (hundreds), or pass variance=False
    and take mean-only on the full test set.
    """
    X_te = np.asarray(X_te, float)
    n = len(X_te)
    bs = int(batch) if batch else n
    ms, vs = [], []
    for s in range(0, n, bs):
        xb = X_te[s : s + bs]
        ms.append(_first(gp.posterior_mean(xb), ["f(x)", "m(x)"]))
        if variance:
            vs.append(
                _first(gp.posterior_covariance(xb, variance_only=True),
                       ["v(x)", "S(x)", "variance"])
            )
        if verbose:
            print(f"[predict]   {min(s + bs, n)}/{n}")
    m = np.concatenate(ms)
    v = np.maximum(np.concatenate(vs), 0.0) if variance else np.full(n, np.nan)
    return m, v


#: Geometry-derived channels and the geometry sub-channels each is built from. "geom"
#: takes whatever ``geom_channels`` says; the others are single sub-channels promoted to
#: first-class channels so they can be given their own embedding, their own length scale,
#: or routed to the prior mean instead of the kernel (strain, in production).
GEOM_SUBCHANNELS = {"geom": None, "strain": ("strain",), "elec": ("elec",)}


def build_channels(atoms_tr, y_tr, cat_tr, atoms_te, cat_te, names, *,
                   client=None, chunk=500, pls=10, scaling="pareto",
                   depth=3, min_count=2, bond_mult=1.2, perceiver="ase",
                   wl_perceiver=None, geom_channels=("rdf", "angle", "torsion", "elec"),
                   geom_top_k=6, geom_r_max=6.0, charge_key="lowdin_charges",
                   target_neighbors=60, cutoff_pct=25.0, diag_sample=5000):
    """Featurise the requested channels and pick each one's compact-support radius.

    Returns ``{name: {"Ztr", "Zte", "cutoff"}}`` -- the input the additive kernel and the
    linear prior mean both consume.

    THIS EXISTS TO BE SHARED. It was lifted verbatim out of ``dim_sweep.run`` so that the
    frozen neighbour-count sweep and the MCMC radius training build their embeddings
    through ONE code path. Those two produce the headline comparison of this work -- a
    learned radius against a grid over neighbour counts -- and that comparison is only
    paired if the embeddings are identical. Two independent copies of thirty lines of
    featurisation would drift, and the drift would look like a scientific result.

    Everything is a keyword argument, deliberately: the ``dim_sweep`` originals took the
    argparse ``Namespace`` and so could not be called from anywhere else.

    Note ``names`` may include channels destined for the PRIOR MEAN only (strain, in the
    production configuration). They still need an embedding; they just never get a Gram
    block. Their cutoff is computed and unused, which is cheap and keeps the return shape
    uniform.
    """
    emb = {}
    need = set(names)

    def _cut(Z_tr, Z_te):
        if target_neighbors:
            return cutoff_for_neighbors(
                Z_tr, Z_te, target_neighbors, dim=Z_tr.shape[1],
                data_id_tr=cat_tr, data_id_te=cat_te, sample=diag_sample)
        c, _ = recalibrate(Z_tr, percentile=cutoff_pct, dim=Z_tr.shape[1],
                           sample=diag_sample)
        return c

    if "wl" in need:
        pw = WLGPPipeline(depth=depth, min_count=min_count, pls_components=pls,
                          cutoff_percentile=None, scaling=scaling, vocab_sample=0,
                          cutoff_mult=bond_mult,
                          perceiver=(wl_perceiver or perceiver))
        Ztr = pw.fit(atoms_tr, y_tr, cat_tr, client=client, chunk=chunk)
        Zte = pw.transform(atoms_te, client=client, chunk=chunk)
        emb["wl"] = {"Ztr": Ztr, "Zte": Zte, "cutoff": _cut(Ztr, Zte)}

    # geom and strain are SEPARATE channels on purpose. Their features are not
    # commensurable -- the histogram channels are O(1) per-atom frequencies while strain
    # is a sum of squared deviations -- and pareto scaling preserves scale by design, so
    # folding strain into the geometry PLS lets it dominate the embedding (measured: geom
    # GP R^2 0.317 -> 0.064, cutoff 2.4 -> 180). Own embedding, own length scale: exactly
    # what the additive kernel exists for.
    for name, sub in GEOM_SUBCHANNELS.items():
        if name not in need:
            continue
        chans = tuple(geom_channels) if sub is None else sub
        pg = GeometryPipeline(top_k=geom_top_k, channels=chans, r_max=geom_r_max,
                              pls_components=pls, cutoff_percentile=None,
                              scaling=scaling, charge_key=charge_key,
                              cutoff_mult=bond_mult, perceiver=perceiver)
        Ztr = pg.fit(atoms_tr, y_tr, cat_tr, client=client, chunk=chunk)
        Zte = pg.transform(atoms_te, client=client, chunk=chunk)
        emb[name] = {"Ztr": Ztr, "Zte": Zte, "cutoff": _cut(Ztr, Zte)}

    missing = need - set(emb)
    if missing:
        raise ValueError(f"unknown channel(s) {sorted(missing)}; "
                         f"known: ['wl'] + {sorted(GEOM_SUBCHANNELS)}")
    return emb


class SolveWarningCounter:
    """Count fvgp's silent solver failures across a long run.

    THE HAZARD. When a Krylov solve does not converge, fvgp does not raise -- it calls
    ``warnings.warn`` ("MINRES not successful" / "CG not successful") and returns the
    UNCONVERGED vector, which then flows into the likelihood as if it were a solution.
    Worse, ``warnings.simplefilter("once", UserWarning)`` is installed at import time in
    four separate fvgp modules, so you see the message AT MOST ONCE PER PROCESS. Over a
    few hundred MCMC sweeps that means one line of output standing in for an unknown
    number of corrupted likelihood evaluations.

    We hit the CG twin of this already: a 20k arm at jitter 1e-6 returned R^2 = -1388,
    which looked like a statistical result and was actually a solve truncated at its
    iteration cap.

    THE MECHANISM. ``simplefilter("always")`` calls ``warnings._filters_mutated()``,
    which bumps the global filter version and thereby invalidates every module's
    ``__warningregistry__`` -- that is what makes an already-seen message fire again.
    We then swap in a counting ``showwarning`` hook rather than using
    ``catch_warnings(record=True)``, which would retain a warning object per event over
    hundreds of solves when all we want is a tally. Both the hook and the filter list
    are restored on exit.

    Pair this with a FINITE ``solve_maxiter``. With the default ``None`` an
    ill-conditioned solve grinds against scipy's own limit instead of reporting failure,
    so the counter stays at zero while the run stalls -- the cap is what converts a hang
    into a countable event.
    """

    PATTERNS = ("MINRES not successful", "CG not successful", "Block CG failed",
                "Failed to build sparse preconditioner")

    def __init__(self, patterns=None):
        self.patterns = tuple(patterns) if patterns else self.PATTERNS
        self.counts = {p: 0 for p in self.patterns}
        self.other = 0

    @property
    def total(self):
        return sum(self.counts.values())

    def __enter__(self):
        import warnings

        self._saved_filters = warnings.filters[:]
        self._saved_show = warnings.showwarning
        warnings.simplefilter("always")          # defeats fvgp's module-level "once"
        warnings.showwarning = self._hook
        return self

    def _hook(self, message, category, filename, lineno, file=None, line=None):
        text = str(message)
        for p in self.patterns:
            if p in text:
                self.counts[p] += 1
                return                            # counted, not printed
        self.other += 1
        self._saved_show(message, category, filename, lineno, file, line)

    def __exit__(self, *exc):
        import warnings

        warnings.showwarning = self._saved_show
        warnings.filters[:] = self._saved_filters
        warnings._filters_mutated()
        return False

    def report(self, n_evals=None, fail_frac=None, label="solve"):
        """Print the tally -- ALWAYS, including at zero, because "no warnings" is only
        informative if you can tell it apart from "never looked". Returns the failure
        fraction when ``n_evals`` is given, and raises past ``fail_frac``."""
        frac = (self.total / n_evals) if n_evals else None
        detail = ", ".join(f"{k.split()[0]}={v}" for k, v in self.counts.items() if v)
        print(f"[{label}] solver non-convergence: {self.total}"
              + (f"/{n_evals} evals ({frac:.1%})" if frac is not None else "")
              + (f"  [{detail}]" if detail else "  (none)")
              + (f"  +{self.other} other warnings" if self.other else ""))
        if fail_frac is not None and frac is not None and frac > fail_frac:
            raise RuntimeError(
                f"{self.total}/{n_evals} solves ({frac:.1%}) did not converge, above the "
                f"{fail_frac:.0%} threshold. The likelihood surface is corrupted; raise "
                f"solve_maxiter, loosen solve_tol, or increase the noise/nugget.")
        return frac


def train_hyperparameters(gp, hp_bounds, max_iter=50, info=True,
                          init_hyperparameters=None, mcmc_prop_distrs="normal",
                          mcmc_prior=None, mcmc_args=None):
    """Optional marginal-likelihood training. `imate` is already required to build
    any gp2Scale GP (see require_imate); training additionally exercises its
    randomised log-determinant heavily. The default flow freezes hyperparameters
    from a validation-scale fit and skips training to keep the 200k run to CG
    solves only.

    THREE THINGS ABOUT TRAINING UNDER gp2Scale that are easy to get wrong:

    * **`max_iter` is not an optimiser iteration count.** gp2Scale forces
      ``method='mcmc'`` (gp.py:791-793) and raises on the gradient path
      (gp_marginal_likelihood.py:216), so `max_iter` becomes `n_updates`: the number of
      Metropolis-Hastings sweeps, i.e. one likelihood evaluation per proposal BLOCK per
      sweep. Each of those is a full sparse covariance rebuild. A value like 30 is a
      30-sample MCMC, not 30 steps of an optimiser, and is not a converged anything.
    * **The return value is not the MAP.** `gp.train` hands back
      ``res["median(x)"]`` -- the componentwise median of the last 1% of the trace
      (gp_training.py:146, gp_mcmc.py:174). At n_updates=400 that is four samples. Read
      ``gp.mcmc_info["max x"]`` for the argmax, and prefer your own median over a
      properly chosen burn-in.
    * **`mcmc_prop_distrs` is where block-MH lives.** Passing a list of
      ``ProposalDistribution`` objects, each owning an ``indices`` subset, gives each
      block its own proposal scale and accept/reject. The default is ONE block over all
      hyperparameters, which couples quantities on wildly different scales (signal
      variances vs support radii) under a single adapted step size.

    ``mcmc_prior`` is also the only per-sample hook available -- `run_in_every_iteration`
    is not forwarded by GPtraining.train -- so it doubles as a logging/checkpoint seam.
    """
    hp_bounds = np.asarray(hp_bounds, float)
    assert hp_bounds.ndim == 2 and hp_bounds.shape[1] == 2, (
        f"hyperparameter_bounds must be (N, 2), got {hp_bounds.shape}")
    n_hps = len(np.asarray(gp.get_hyperparameters()).ravel())
    assert n_hps == len(hp_bounds), (
        f"the GP holds {n_hps} hyperparameters but {len(hp_bounds)} bounds were given. "
        f"fvgp does not validate this (gp.py:1917 out_of_bounds indexes bounds[i] over "
        f"range(len(x))): too many hyperparameters raises a bare IndexError mid-training, "
        f"too few silently resolves via a random uniform draw over the bounds instead."
    )
    gp.train(hyperparameter_bounds=hp_bounds, max_iter=max_iter, info=info,
             init_hyperparameters=init_hyperparameters,
             mcmc_prop_distrs=mcmc_prop_distrs, mcmc_prior=mcmc_prior,
             mcmc_args=(mcmc_args or {}))
    return gp.get_hyperparameters()
