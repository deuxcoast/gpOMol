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

from .cutoff import recalibrate
from .data import stratified_sample_indices
from .kernel import make_wl_block_kernel
from .reduce import SparsePLS
from .wl_features import SparseWLFeaturizer


# ----------------------------- embedding pipeline --------------------------


@dataclass
class WLGPPipeline:
    depth: int = 3
    min_count: int = 5
    pls_components: int = 10
    cutoff_percentile: float = 25.0
    vocab_sample: int = 0  # 0 = fit vocab on ALL train (no OOV); >0 = stratified cap
    cutoff_mult: float = 1.2
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
            depth=self.depth, min_count=self.min_count, cutoff_mult=self.cutoff_mult
        ).fit(fit_atoms)

        X_tr = self.featurizer.transform(atoms, client=client, chunk=chunk)
        self.reducer = SparsePLS(n_components=self.pls_components).fit(X_tr, y)
        Z_tr = self.reducer.transform(X_tr)
        self.dim_ = Z_tr.shape[1]
        self.cutoff_, _ = recalibrate(
            Z_tr, percentile=self.cutoff_percentile, dim=self.dim_
        )
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


def connect_dask(scheduler_file=None, n_workers=16, poll_timeout=1800):
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
        client = Client(scheduler_file=scheduler_file)
        print(f"[dask] connected via {scheduler_file}")
    else:
        client = Client()
        print("[dask] started a local cluster (no scheduler file)")
    if n_workers:
        print(f"[dask] waiting for {n_workers} workers ...")
        client.wait_for_workers(n_workers)
    print(f"[dask] {len(client.scheduler_info()['workers'])} workers ready")
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
    logdet_rtol=0.5,
    args=None,
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
    """

    require_imate()
    from gpcam import GPOptimizer

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

    # The log-determinant is UNAVOIDABLE but, predict-only, it is pure waste. fvgp
    # computes it in the constructor (GPkv.__init__ -> _refresh -> logdet(),
    # gp_kv.py:62,216) no matter what, yet `logdet_KV` is only ever READ by
    # gp_marginal_likelihood.py -- i.e. only if you train. So for a predict-only run
    # we pay imate's stochastic-Lanczos estimate and then throw the answer away.
    #
    # It is not cheap at scale: each SLQ sample costs `lanczos_degree` (20) matvecs
    # against the full sparse KV, and the sample count is driven by error_rtol
    # (default 0.01) up to max_num_samples=5000. Loosening error_rtol makes it stop
    # at min_num_samples=10 -- the floor, ~200 matvecs -- which is all we should pay
    # for a number we discard. Raise it (0.01) if you actually --train.
    _args = dict(args or {})
    _args.setdefault("random_logdet_error_rtol", float(logdet_rtol))

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


def train_hyperparameters(gp, hp_bounds, max_iter=50, info=True):
    """Optional marginal-likelihood training. `imate` is already required to build
    any gp2Scale GP (see require_imate); training additionally exercises its
    randomised log-determinant heavily. The default flow freezes hyperparameters
    from a validation-scale fit and skips training to keep the 200k run to CG
    solves only."""
    gp.train(hyperparameter_bounds=np.asarray(hp_bounds, float), max_iter=max_iter, info=info)
    return gp.get_hyperparameters()
