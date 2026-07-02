"""
features.py
===========

Step 2 of the pipeline: the three-channel hybrid descriptor.

    v(M) = [ WL_topology (D_WL) | distance_histogram (n_bins) | charge_scalars (n_c) ]

Every channel is made INTENSIVE so that (a) the descriptor of a big molecule is
comparable to that of a small one and (b) the compact-support kernel is not
tracking an extensive quantity (see extensive_mean.py). The output is a single
fixed-length real vector regardless of molecule size — which is exactly what the
Wendland-Mahalanobis PD construction in embedding_kernel.py requires.

Why NO optimal transport anywhere here
--------------------------------------
The geometry channel is the *representation* that Wasserstein-over-atoms was
reaching for (interatomic distances), but pooled into a fixed histogram and
compared with plain Euclidean distance. We keep the representation and throw away
the OT metric, because a compactly-supported Wendland over an OT distance is not
guaranteed PD (Wendland loses positive-definiteness above a finite dimension;
the infinite-dimensional OT embedding blows past it). Euclidean distance on a
fixed vector sidesteps that entirely.

Dependencies
------------
  * numpy (required)
  * a molecular-graph source for the WL channel. We accept graphs in a minimal
    adjacency form so this file has no hard graph-library dependency; see
    `wl_features`. In practice build these from RDKit or ASE neighbour lists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Channel 1 — Weisfeiler-Lehman topology, hashed to fixed length, intensive
# ----------------------------------------------------------------------------


def _stable_bucket(label: str, n_buckets: int) -> int:
    """Deterministic hash -> bucket. Uses blake2b, NOT Python's builtin hash(),
    which is per-process salted and would make features irreproducible."""
    h = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_buckets


def wl_features(
    adjacency: Sequence[Sequence[int]],
    node_labels: Sequence,
    depth: int = 3,
    n_buckets: int = 256,
) -> np.ndarray:
    """
    Weisfeiler-Lehman subtree-pattern histogram for one molecule, hashed to a
    fixed length and normalised to be intensive.

    Parameters
    ----------
    adjacency : list-of-lists; adjacency[i] = neighbour indices of atom i
    node_labels : per-atom initial labels (e.g. atomic numbers or element symbols)
    depth : number of WL relabelling rounds (h). Larger h = wider neighbourhoods;
        this is the "approximate compact support in graph space" knob.
    n_buckets : fixed feature dimension D_WL for this channel.

    Returns
    -------
    (n_buckets,) float vector, sum-normalised by atom count -> INTENSIVE.

    Notes
    -----
    WL *counts* are extensive (more atoms -> more patterns). Dividing by the atom
    count converts to a per-atom pattern frequency, which is intensive and is the
    right unit given the extensive mean has already removed the size-scaling of E.
    """
    n_atoms = len(node_labels)
    vec = np.zeros(n_buckets, dtype=float)
    labels = [str(l) for l in node_labels]

    # depth 0..h; accumulate hashed pattern counts at every depth
    for d in range(depth + 1):
        for lab in labels:
            vec[_stable_bucket(f"{d}:{lab}", n_buckets)] += 1.0
        if d == depth:
            break
        new_labels = []
        for i in range(n_atoms):
            neigh = sorted(labels[j] for j in adjacency[i])
            new_labels.append(labels[i] + "|" + ",".join(neigh))
        # compress to short digests so strings stay bounded across rounds
        labels = [
            hashlib.blake2b(l.encode(), digest_size=8).hexdigest() for l in new_labels
        ]

    return vec / max(n_atoms, 1)


# ----------------------------------------------------------------------------
# Channel 2 — cutoff-free pairwise-distance histogram, intensive
# ----------------------------------------------------------------------------


def distance_histogram(
    positions: np.ndarray, bin_edges: np.ndarray, overflow: bool = True
) -> np.ndarray:
    """
    Histogram of all intramolecular pairwise distances, normalised to a density.

    Parameters
    ----------
    positions : (n_atoms, 3) Cartesian coordinates (Angstrom).
    bin_edges : (n_bins + 1,) fixed, shared across ALL molecules. Must be fixed
        for the descriptor to be a common fixed-length vector.
    overflow : if True, distances beyond bin_edges[-1] are clipped into the last
        bin so long-range pairs are *recorded* (cutoff-free in spirit) rather
        than dropped. If False they are discarded.

    Returns
    -------
    (n_bins,) float vector summing to 1 -> INTENSIVE (independent of molecule
    size because we normalise by the number of pairs).

    Rotational/translational invariance
    ------------------------------------
    Pairwise distances are invariant to rotation and translation by construction,
    with NO alignment step and NO chemical cutoff radius. This is the property the
    Coulomb-matrix eigenspectrum and SOAP-cutoff routes give up or complicate.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 2:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    if overflow:
        dists = np.minimum(dists, bin_edges[-1] - 1e-9)

    hist, _ = np.histogram(dists, bins=bin_edges)
    total = hist.sum()
    if total == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    return hist.astype(float) / total


def default_distance_bins(r_max: float = 20.0, n_bins: int = 64) -> np.ndarray:
    """Convenience: uniform bins on [0, r_max]. r_max should comfortably exceed
    typical molecular diameters in OMol25 so the overflow bin stays lightly used.
    For heavy-tailed size distributions consider bins on r/(1+r) instead."""
    return np.linspace(0.0, r_max, n_bins + 1)


# ----------------------------------------------------------------------------
# Channel 3 — electronic scalars from Loewdin / NBO charges, intensive
# ----------------------------------------------------------------------------


def charge_features(charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Low-dimensional electronic descriptor from per-atom partial charges.

    USE LOEWDIN OR NBO CHARGES, NOT MULLIKEN. Mulliken populations are notoriously
    basis-set unstable; feeding them here imports that instability as feature
    noise and will show up as an inflated semivariogram nugget. OMol25 ships
    Loewdin and NBO in atoms.info — prefer those.

    Returns [ dipole_per_atom , var(q) , max(q) - min(q) ]  (all intensive).

    Parameters
    ----------
    charges : (n_atoms,) partial charges.
    positions : (n_atoms, 3) coordinates.
    """
    q = np.asarray(charges, dtype=float).ravel()
    positions = np.asarray(positions, dtype=float)
    n = len(q)
    if n == 0:
        return np.zeros(3, dtype=float)

    # translation-invariant dipole: reference to the geometric centroid so the
    # value is well-defined even for net-charged species (common in OMol25).
    centroid = positions.mean(axis=0)
    p_vec = ((positions - centroid) * q[:, None]).sum(axis=0)
    dipole_mag = float(np.linalg.norm(p_vec))

    dipole_per_atom = dipole_mag / max(n, 1)  # keep intensive
    var_q = float(np.var(q))
    range_q = float(q.max() - q.min())
    return np.array([dipole_per_atom, var_q, range_q], dtype=float)


# ----------------------------------------------------------------------------
# Assembler — concatenate the three channels and z-score standardise
# ----------------------------------------------------------------------------


@dataclass
class HybridFeatureAssembler:
    """
    Builds the raw hybrid feature matrix and standardises it.

    Keeps the per-channel column slices so diagnostics.py can construct a
    WL-ONLY distance (to test whether the geometry+charge channels actually lift
    the predictive ceiling above the WL baseline).
    """

    wl_depth: int = 3
    wl_buckets: int = 256
    bin_edges: np.ndarray = field(default_factory=default_distance_bins)

    # learned standardisation
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    slices_: dict = field(default=None, repr=False)

    def _raw_one(self, graph, positions, charges) -> np.ndarray:
        adjacency, node_labels = graph
        wl = wl_features(adjacency, node_labels, self.wl_depth, self.wl_buckets)
        hist = distance_histogram(positions, self.bin_edges)
        chg = charge_features(charges, positions)
        return np.concatenate([wl, hist, chg])

    def raw_matrix(self, graphs, positions_list, charges_list) -> np.ndarray:
        """(N, D_raw) un-standardised features. Records channel slices on first call."""
        rows = [
            self._raw_one(g, p, c)
            for g, p, c in zip(graphs, positions_list, charges_list)
        ]
        X = np.vstack(rows)
        n_wl = self.wl_buckets
        n_hist = len(self.bin_edges) - 1
        self.slices_ = {
            "wl": slice(0, n_wl),
            "hist": slice(n_wl, n_wl + n_hist),
            "charge": slice(n_wl + n_hist, n_wl + n_hist + 3),
        }
        return X

    def fit(self, graphs, positions_list, charges_list) -> "HybridFeatureAssembler":
        X = self.raw_matrix(graphs, positions_list, charges_list)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # dead features -> no scaling, no NaNs
        return self

    def transform(self, graphs, positions_list, charges_list) -> np.ndarray:
        X = self.raw_matrix(graphs, positions_list, charges_list)
        return (X - self.mean_) / self.std_

    def fit_transform(self, graphs, positions_list, charges_list) -> np.ndarray:
        self.fit(graphs, positions_list, charges_list)
        return self.transform(graphs, positions_list, charges_list)
