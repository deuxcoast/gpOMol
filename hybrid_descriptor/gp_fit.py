"""
gp_fit.py
=========

End-to-end orchestration for a hybrid-descriptor GP on the FULL OMol25 train_4M
split (~4M structures), wiring gpCAM's gp2Scale mode. The four component modules
(extensive_mean, features, embedding_kernel, diagnostics) stay stable; this file
holds the OMol25 I/O and the fit/predict flow.

Decisions baked in for full train_4M (see project notes):
  * charges  = Loewdin (NBO is ~33% missing, concentrated in open-shell/metal/
               solvated subsets -> using it would bias the sample to organics).
  * graphs   = geometry-derived (ASE covalent-radius perception); SMILES is only
               ~2% recoverable from provenance, so no SMILES path. Multi-molecule
               records (solvated proteins, electrolyte shells) stay as multi-
               component graphs; WL handles disconnected components.
  * mean     = element counts + NET CHARGE + SPIN. train_4M spans charge/spin
               states whose energy effect is large and must live in the extensive
               mean, not the GP residual.

THE CRUX for train_4M: an EXACT GP at N~4M is feasible only if the covariance
matrix is genuinely sparse. storage ~ 12 * s* * N^2 bytes, so at N=4e6:
  s*=0.06 -> ~11 TB (fits 40 TB) ;  s*=0.2 -> ~38 TB (edge) ;  s*=0.5 -> ~96 TB (no).
Prior evidence had s* flat ~0.5 under broad diversity. So the plan is organised to
MEASURE and, if needed, IMPOSE sparsity (short support radius) before launching the
full run — see diagnostics.sparsity_accuracy_sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from diagnostics import run_falsification
from embedding_kernel import (
    FeatureReducer,
    check_kernel_psd,
    default_hp_bounds,
    make_wendland_mahalanobis,
)
from extensive_mean import ExtensiveEnergyModel
from features import HybridFeatureAssembler

# ----------------------------------------------------------------------------
# Data bundle
# ----------------------------------------------------------------------------


@dataclass
class MolBatch:
    """Everything the pipeline needs about a set of structures. y_total /
    net_charges / spins may be absent at pure-predict time."""

    Z_lists: list
    graphs: list
    positions_list: list
    charges_list: list
    y_total: Optional[np.ndarray] = None
    net_charges: Optional[np.ndarray] = None
    spins: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.Z_lists)

    def extra_context(self):
        """(net_charge, spin) tuples for the extensive-mean extra features."""
        nc = self.net_charges if self.net_charges is not None else np.zeros(len(self))
        sp = self.spins if self.spins is not None else np.ones(len(self))
        return list(zip(nc, sp))


def charge_spin_features(Z_list, ctx) -> np.ndarray:
    """Extensive-mean extra features: [net_charge, spin_multiplicity]. Low-order
    on purpose — this is a prior mean, not a model of the physics."""
    charge, spin = (0.0, 1.0) if ctx is None else (float(ctx[0]), float(ctx[1]))
    return np.array([charge, spin], dtype=float)


# ----------------------------------------------------------------------------
# 0. OMol25 I/O — geometry-derived graphs, Loewdin charges, full diversity
# ----------------------------------------------------------------------------


def build_graph(atoms, cutoff_mult: float = 1.2):
    """
    Geometry-derived connectivity for one record via ASE covalent-radius
    perception. Returns (adjacency, node_labels):
        adjacency[i]   = sorted list of bonded neighbour indices of atom i
        node_labels[i] = atomic number (WL = pure topology; electronics live in
                         the charge channel, not the label)

    cutoff_mult scales the covalent radii. 1.2 is a reasonable default; tighten
    toward ~1.0 to cut spurious bonds in metal/charged cases, loosen to catch
    long bonds. Multi-molecule records naturally yield multiple components; WL
    handles that (the per-atom normalisation keeps it intensive).
    """
    from ase.neighborlist import build_neighbor_list, natural_cutoffs

    cutoffs = natural_cutoffs(atoms, mult=cutoff_mult)
    nl = build_neighbor_list(atoms, cutoffs, self_interaction=False, bothways=True)
    n = len(atoms)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        neigh, _ = nl.get_neighbors(i)
        adjacency[i] = sorted({int(j) for j in neigh if j != i})
    node_labels = atoms.get_atomic_numbers().tolist()
    return adjacency, node_labels


def n_connected_components(adjacency) -> int:
    """Number of connected components (molecules) in a geometry-derived graph."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(adjacency)
    A = lil_matrix((n, n), dtype=np.int8)
    for i, nb in enumerate(adjacency):
        for j in nb:
            A[i, j] = 1
    return connected_components(A.tocsr(), directed=False, return_labels=False)


def load_omol25_subset(
    src: str = "../train_4M",
    n: int = 100_000,
    seed: int = 0,
    charge_key: str = "lowdin_charges",
    cutoff_mult: float = 1.2,
    size_cap: Optional[int] = None,
) -> "MolBatch":
    """
    Random subsample of an OMol25 aselmdb split -> MolBatch. Full diversity: no
    chemistry filtering. Skips only records missing/NaN in `charge_key` (keeps the
    charge channel clean) and, if `size_cap` is set, records larger than it (guards
    against pathological giant solvated systems blowing up memory).

    Accessor is atoms.get_potential_energy() (calc 'energy') and atoms.info[...]
    for charges/charge/spin, matching the train_4M schema.
    """
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    N = len(ds)
    print(f"train_4M: {N:,} structures; sampling {min(n, N):,}")
    idxs = np.random.default_rng(seed).choice(N, size=min(n, N), replace=False)

    Z, G, P, Q, Y, NC, SP = [], [], [], [], [], [], []
    skipped = n_multi = 0
    for i in idxs:
        atoms = ds.get_atoms(int(i))
        if size_cap and len(atoms) > size_cap:
            skipped += 1
            continue
        q = atoms.info.get(charge_key)
        if q is None or np.any(np.isnan(np.asarray(q, dtype=float))):
            skipped += 1
            continue
        adj_lab = build_graph(atoms, cutoff_mult)
        Z.append(atoms.get_atomic_numbers().tolist())
        P.append(atoms.get_positions())
        Q.append(np.asarray(q, dtype=float))
        Y.append(float(atoms.get_potential_energy()))
        NC.append(float(atoms.info.get("charge", 0)))
        SP.append(float(atoms.info.get("spin", 1)))
        G.append(adj_lab)
        if n_connected_components(adj_lab[0]) > 1:
            n_multi += 1

    print(
        f"  kept {len(Y):,}  (skipped {skipped:,}); "
        f"multi-molecule records: {n_multi}/{len(Y)} = {n_multi/max(len(Y),1):.1%}"
    )
    return MolBatch(Z, G, P, Q, np.array(Y), np.array(NC), np.array(SP))


# ----------------------------------------------------------------------------
# 1. Fitted preprocessing bundle (mean + features + reducer), reusable at predict
# ----------------------------------------------------------------------------


@dataclass
class HybridPreprocessor:
    n_components: int = 15
    wl_depth: int = 3
    wl_buckets: int = 256

    mean_model: ExtensiveEnergyModel = None
    assembler: HybridFeatureAssembler = None
    reducer: FeatureReducer = None

    def fit(self, batch: "MolBatch"):
        ctx = batch.extra_context()
        self.mean_model = ExtensiveEnergyModel(
            extra_feature_fn=charge_spin_features
        ).fit(batch.Z_lists, batch.y_total, extra_context=ctx)
        residual = self.mean_model.residual(
            batch.Z_lists, batch.y_total, extra_context=ctx
        )

        self.assembler = HybridFeatureAssembler(
            wl_depth=self.wl_depth, wl_buckets=self.wl_buckets
        )
        X_raw = self.assembler.fit_transform(
            batch.graphs, batch.positions_list, batch.charges_list
        )

        self.reducer = FeatureReducer(n_components=self.n_components).fit(X_raw)
        X = self.reducer.transform(X_raw)
        return X, residual

    def transform(self, batch: "MolBatch"):
        X_raw = self.assembler.transform(
            batch.graphs, batch.positions_list, batch.charges_list
        )
        return self.reducer.transform(X_raw)

    def residual(self, batch: "MolBatch"):
        return self.mean_model.residual(
            batch.Z_lists, batch.y_total, extra_context=batch.extra_context()
        )

    def extensive_energy(self, batch: "MolBatch"):
        return self.mean_model.predict(
            batch.Z_lists, extra_context=batch.extra_context()
        )

    def wl_only_embedding(self, batch: "MolBatch"):
        """Reduced embedding using ONLY the WL channel (zero out the rest), for
        the kNN-skill-vs-WL diagnostic. Reuses the fitted standardiser/reducer so
        the comparison is apples-to-apples in the same reduced space."""
        X_raw = self.assembler.raw_matrix(
            batch.graphs, batch.positions_list, batch.charges_list
        )
        X_std = (X_raw - self.assembler.mean_) / self.assembler.std_
        masked = np.zeros_like(X_std)
        wl = self.assembler.slices_["wl"]
        masked[:, wl] = X_std[:, wl]
        return self.reducer.transform(masked)


# ----------------------------------------------------------------------------
# 2. Gate + fit
# ----------------------------------------------------------------------------


def gate_then_fit(
    batch: "MolBatch",
    target_N: int = 4_000_000,
    n_components: int = 15,
    wendland_k: int = 2,
    dask_client=None,
    gp2Scale_batch_size: int = 10_000,
    run_gate: bool = True,
    mcmc_updates: int = 200,
):
    """
    Full path: preprocess -> falsification gate -> gp2Scale GP -> block-MCMC train.

    Returns (gpo, pre, report). `gpo` is a trained gpCAM GPOptimizer over the
    residual; `pre` is the fitted HybridPreprocessor (needed to predict physical
    energies later); `report` is the FalsificationReport (None if run_gate=False).
    """
    pre = HybridPreprocessor(n_components=n_components)
    X, residual = pre.fit(batch)

    # ---- Falsification gate (cheap; do it before spending cluster time) -------
    report = None
    if run_gate:
        X_wl = pre.wl_only_embedding(batch)
        report = run_falsification(X, X_wl, residual, target_N=target_N)
        print(report.summary())
        if not report.all_passed:
            print(
                "\nGate failed — returning without fitting. "
                "Fix the descriptor / impose shorter support before compute."
            )
            return None, pre, report

    # ---- Kernel + PD guard ----------------------------------------------------
    # explicit backend = dimension-correct Wendland, PD on R^n_components
    kernel = make_wendland_mahalanobis(
        dim=n_components, k=wendland_k, backend="explicit"
    )
    hp_bounds = default_hp_bounds(X, residual)
    init_hps = np.concatenate(
        [[float(np.var(residual))], 0.5 * (hp_bounds[1:, 0] + hp_bounds[1:, 1])]
    )

    # empirical PD check on a subsample — the non-negotiable guard
    sub = X[
        np.random.default_rng(0).choice(len(X), size=min(800, len(X)), replace=False)
    ]
    psd = check_kernel_psd(kernel, sub, init_hps)
    print(
        f"[PD guard] min eigenvalue={psd['min_eigenvalue']:.3e}  "
        f"is_psd={psd['is_psd']}  gram_density={psd['gram_density']:.3f}"
    )
    if not psd["is_psd"]:
        raise RuntimeError(
            "Wendland kernel is NOT PD at dim="
            f"{n_components}. Lower n_components or raise Wendland smoothness "
            "(see embedding_kernel.check_kernel_psd docstring)."
        )

    # ---- gp2Scale GP ----------------------------------------------------------
    from gpcam import GPOptimizer

    gpo = GPOptimizer(
        x_data=X,
        y_data=residual,
        init_hyperparameters=init_hps,
        gp2Scale=True,
        gp2Scale_batch_size=gp2Scale_batch_size,
        kernel_function=kernel,
        dask_client=dask_client,  # pass a distributed.Client at scale
    )

    # ---- Block-MCMC training (recommended over local opt for gp2Scale) --------
    _train_block_mcmc(gpo, hp_bounds, init_hps, n_updates=mcmc_updates)
    return gpo, pre, report


def _train_block_mcmc(gpo, hp_bounds, init_hps, n_updates: int = 200):
    """
    Block Metropolis-Hastings over the hyperparameters, driven against gpCAM's
    exposed log-likelihood. Two natural blocks: signal variance (index 0) and the
    length-scale / support-radius vector (indices 1..D). Sampling the support
    radii together lets the sparsity structure move coherently.
    """
    import numpy as np
    from gpcam import ProposalDistribution, gpMCMC

    D = len(init_hps) - 1

    def in_bounds(v, b):
        return not (np.any(v < b[:, 0]) or np.any(v > b[:, 1]))

    def prior_function(theta, args):
        return 0.0 if in_bounds(theta, args["bounds"]) else -np.inf

    def log_likelihood(hps, args):
        return gpo.log_likelihood(hyperparameters=hps)

    pd_signal = ProposalDistribution([0], init_prop_Sigma=np.array([[0.05]]))
    pd_length = ProposalDistribution(
        list(range(1, D + 1)), init_prop_Sigma=np.eye(D) * 0.01
    )

    mcmc = gpMCMC(
        log_likelihood,
        prior_function,
        [pd_signal, pd_length],
        args={"bounds": hp_bounds},
    )
    result = mcmc.run_mcmc(x0=init_hps, n_updates=n_updates, info=True)
    gpo.set_hyperparameters(result["mean(x)"])
    return result


# ----------------------------------------------------------------------------
# 3. Prediction — restore physical energy and expose calibrated UQ
# ----------------------------------------------------------------------------


def predict_energy(gpo, pre: HybridPreprocessor, batch: "MolBatch"):
    """
    Returns (E_pred, E_std). The GP predicts the intensive residual; we add back
    the extensive mean for the physical energy, and pass the posterior standard
    deviation through unchanged — the extensive mean is deterministic, so all
    predictive uncertainty comes from the GP. That posterior std IS the
    calibrated UQ this whole pipeline exists to deliver.
    """
    X = pre.transform(batch)
    resid_mean = gpo.posterior_mean(X)["m(x)"]
    var = gpo.posterior_covariance(X, variance_only=True)["v(x)"]
    return pre.extensive_energy(batch) + resid_mean, np.sqrt(np.maximum(var, 0.0))


def validate(gpo, pre, batch: "MolBatch"):
    """RMSE and CRPS on held-out molecules, computed on the residual scale via
    gpCAM's built-ins (energy-scale RMSE is identical since the mean is a
    per-molecule deterministic shift)."""
    X = pre.transform(batch)
    residual = pre.residual(batch)
    return {
        "rmse": float(gpo.rmse(X, residual)),
        "crps": float(gpo.crps(X, residual)),
    }
