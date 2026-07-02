"""
gp_fit.py
=========

End-to-end orchestration tying the four steps together and wiring gpCAM's
gp2Scale mode. This is the file you adapt to OMol25 I/O; the four component
modules (extensive_mean, features, embedding_kernel, diagnostics) are meant to
stay stable.

Data flow
---------
    raw molecules ──► ExtensiveEnergyModel ──► intensive residual  (GP y_data)
                 └──► HybridFeatureAssembler ──► standardised features
                                             └──► FeatureReducer (PCA) ──► X (N, D)
    X, residual ──► [FALSIFICATION GATE] ──► gp2Scale GPOptimizer + Wendland
                                          ──► block-MCMC training
    prediction: GP posterior on residual  +  ExtensiveEnergyModel  = physical E,
                and posterior variance = the calibrated UQ that is the whole point.

What you must supply for OMol25 (marked TODO below)
---------------------------------------------------
  * Z_lists        : per-molecule atomic numbers
  * graphs         : per-molecule (adjacency, node_labels) from RDKit/ASE bonds
  * positions_list : per-molecule (n_atoms, 3) coordinates
  * charges_list   : per-molecule Loewdin or NBO partial charges (NOT Mulliken)
  * y_total        : per-molecule DFT total energies
"""

from __future__ import annotations

from dataclasses import dataclass
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
# 0. OMol25 loader — TODO: fill in with fairchem AseDBDataset access
# ----------------------------------------------------------------------------


def load_omol25_subset(n: int, split: str = "train"):
    """
    RETURN: Z_lists, graphs, positions_list, charges_list, y_total

    Sketch (adapt to your working explore_omol25.py / AseDBDataset setup):

        from fairchem.core.datasets import AseDBDataset
        ds = AseDBDataset({"src": "<path>"})
        Z_lists, graphs, pos, charges, y = [], [], [], [], []
        for i in random_indices(len(ds), n):
            atoms = ds.get_atoms(i)
            Z_lists.append(atoms.get_atomic_numbers())
            pos.append(atoms.get_positions())
            charges.append(atoms.info["loewdin_charges"])     # or "nbo_charges"
            y.append(atoms.info["energy"])
            graphs.append(build_graph(atoms))                 # adjacency + labels
    """
    raise NotImplementedError("Wire this to your AseDBDataset / OMol25 access.")


def build_graph(atoms):
    """
    TODO: return (adjacency, node_labels) for one molecule.
      adjacency[i]  = list of bonded neighbour indices of atom i
      node_labels[i] = initial WL label (atomic number is a fine default)

    Build bonds from RDKit (if SMILES/mol available) or from ASE via covalent-
    radius neighbour perception (ase.neighborlist.natural_cutoffs +
    build_neighbor_list). Connectivity-only; geometry lives in the distance
    histogram, not here.
    """
    raise NotImplementedError("Provide molecular-graph construction for OMol25.")


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

    def fit(self, Z_lists, graphs, positions_list, charges_list, y_total):
        self.mean_model = ExtensiveEnergyModel().fit(Z_lists, y_total)
        residual = self.mean_model.residual(Z_lists, y_total)

        self.assembler = HybridFeatureAssembler(
            wl_depth=self.wl_depth, wl_buckets=self.wl_buckets
        )
        X_raw = self.assembler.fit_transform(graphs, positions_list, charges_list)

        self.reducer = FeatureReducer(n_components=self.n_components).fit(X_raw)
        X = self.reducer.transform(X_raw)
        return X, residual

    def transform(self, Z_lists, graphs, positions_list, charges_list):
        X_raw = self.assembler.transform(graphs, positions_list, charges_list)
        return self.reducer.transform(X_raw)

    def wl_only_embedding(self, graphs, positions_list, charges_list):
        """Reduced embedding using ONLY the WL channel (zero out the rest), for
        the kNN-skill-vs-WL diagnostic. Reuses the fitted standardiser/reducer so
        the comparison is apples-to-apples in the same reduced space."""
        X_raw = self.assembler.raw_matrix(graphs, positions_list, charges_list)
        X_std = (X_raw - self.assembler.mean_) / self.assembler.std_
        masked = np.zeros_like(X_std)
        wl = self.assembler.slices_["wl"]
        masked[:, wl] = X_std[:, wl]
        return self.reducer.transform(masked)


# ----------------------------------------------------------------------------
# 2. Gate + fit
# ----------------------------------------------------------------------------


def gate_then_fit(
    Z_lists,
    graphs,
    positions_list,
    charges_list,
    y_total,
    target_N: int = 1_500_000,
    n_components: int = 15,
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
    X, residual = pre.fit(Z_lists, graphs, positions_list, charges_list, y_total)

    # ---- Falsification gate (cheap; do it before spending cluster time) -------
    report = None
    if run_gate:
        X_wl = pre.wl_only_embedding(graphs, positions_list, charges_list)
        report = run_falsification(X, X_wl, residual, target_N=target_N)
        print(report.summary())
        if not report.all_passed:
            print(
                "\nGate failed — returning without fitting. "
                "Fix the descriptor before spending compute."
            )
            return None, pre, report

    # ---- Kernel + PD guard ----------------------------------------------------
    kernel = make_wendland_mahalanobis(dim=n_components)
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


def predict_energy(
    gpo, pre: HybridPreprocessor, Z_lists, graphs, positions_list, charges_list
):
    """
    Returns (E_pred, E_std). The GP predicts the intensive residual; we add back
    the extensive mean for the physical energy, and pass the posterior standard
    deviation through unchanged — the extensive mean is deterministic, so all
    predictive uncertainty comes from the GP. That posterior std IS the
    calibrated UQ this whole pipeline exists to deliver.
    """
    X = pre.transform(Z_lists, graphs, positions_list, charges_list)
    post = gpo.posterior_mean(X)
    resid_mean = post["m(x)"]
    var = gpo.posterior_covariance(X, variance_only=True)["v(x)"]
    E_ext = pre.mean_model.predict(Z_lists)
    return E_ext + resid_mean, np.sqrt(np.maximum(var, 0.0))


def validate(gpo, pre, Z_lists, graphs, positions_list, charges_list, y_total):
    """RMSE and CRPS on held-out molecules, computed on the residual scale via
    gpCAM's built-ins (energy-scale RMSE is identical since the mean is a constant
    shift per molecule)."""
    X = pre.transform(Z_lists, graphs, positions_list, charges_list)
    residual = pre.mean_model.residual(Z_lists, y_total)
    return {
        "rmse": float(gpo.rmse(X, residual)),
        "crps": float(gpo.crps(X, residual)),
    }
