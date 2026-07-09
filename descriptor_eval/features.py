"""
features.py  (descriptor_eval, Candidate A)
==========================================
Self-contained hybrid descriptor for one ase.Atoms -> fixed-length vector:

    v = [ WL topology (256) | distance histogram (64) | charge scalars (3) ]  = 323

Each channel is intensive. featurize() returns the RAW vector; standardisation is
a population operation (needs the mean/std of the whole set), so it is applied at
the matrix level in standardize(), not per molecule. This file has no dependency
on the hybrid_descriptor package -- the eval framework is independent by design.
"""

from __future__ import annotations

import hashlib

import numpy as np

# ----------------------------- WL topology ---------------------------------


def _bucket(label: str, n_buckets: int) -> int:
    """Deterministic hash -> bucket (blake2b, not Python's salted hash())."""
    return (
        int.from_bytes(hashlib.blake2b(label.encode(), digest_size=8).digest(), "big")
        % n_buckets
    )


def wl_features(
    adjacency, node_labels, depth: int = 3, n_buckets: int = 256
) -> np.ndarray:
    """Weisfeiler-Lehman subtree-pattern histogram, hashed to n_buckets, summed
    over depths 0..depth, normalised by atom count -> intensive frequencies."""
    n = len(node_labels)
    vec = np.zeros(n_buckets, dtype=float)
    labels = [str(l) for l in node_labels]
    for d in range(depth + 1):
        for lab in labels:
            vec[_bucket(f"{d}:{lab}", n_buckets)] += 1.0
        if d == depth:
            break
        labels = [
            hashlib.blake2b(
                (
                    labels[i] + "|" + ",".join(sorted(labels[j] for j in adjacency[i]))
                ).encode(),
                digest_size=8,
            ).hexdigest()
            for i in range(n)
        ]
    return vec / max(n, 1)


# ----------------------------- geometry ------------------------------------


def default_distance_bins(r_max: float = 20.0, n_bins: int = 64) -> np.ndarray:
    return np.linspace(0.0, r_max, n_bins + 1)


def distance_histogram(positions, bin_edges, overflow: bool = True) -> np.ndarray:
    """Histogram of all intramolecular pairwise distances, sum-normalised to 1
    (intensive), rotation/translation invariant, no chemical cutoff."""
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
    tot = hist.sum()
    return (
        hist.astype(float) / tot if tot else np.zeros(len(bin_edges) - 1, dtype=float)
    )


# ----------------------------- charge --------------------------------------


def charge_features(charges, positions) -> np.ndarray:
    """[dipole_per_atom, var(q), max(q)-min(q)] from Loewdin charges. Dipole is
    referenced to the geometric centroid (translation-invariant, well-defined for
    net-charged species) and divided by atom count to stay intensive."""
    q = np.asarray(charges, dtype=float).ravel()
    positions = np.asarray(positions, dtype=float)
    n = len(q)
    if n == 0:
        return np.zeros(3, dtype=float)
    centroid = positions.mean(axis=0)
    dipole = float(np.linalg.norm(((positions - centroid) * q[:, None]).sum(axis=0)))
    return np.array([dipole / max(n, 1), float(np.var(q)), float(q.max() - q.min())])


# ----------------------------- graph + assembly ----------------------------


def build_graph(atoms, cutoff_mult: float = 1.2):
    """Geometry-derived connectivity via ASE covalent-radius perception.
    node label = atomic number; multi-molecule records -> multi-component graphs."""
    from ase.neighborlist import build_neighbor_list, natural_cutoffs

    nl = build_neighbor_list(
        atoms,
        natural_cutoffs(atoms, mult=cutoff_mult),
        self_interaction=False,
        bothways=True,
    )
    n = len(atoms)
    adjacency = [
        sorted({int(j) for j in nl.get_neighbors(i)[0] if j != i}) for i in range(n)
    ]
    return adjacency, atoms.get_atomic_numbers().tolist()


def featurize(
    atoms,
    wl_depth: int = 3,
    wl_buckets: int = 256,
    bin_edges=None,
    charge_key: str = "lowdin_charges",
) -> np.ndarray:
    """One ase.Atoms -> RAW (un-standardised) 323-dim hybrid descriptor."""
    if bin_edges is None:
        bin_edges = default_distance_bins()
    adjacency, labels = build_graph(atoms)
    wl = wl_features(adjacency, labels, wl_depth, wl_buckets)
    hist = distance_histogram(atoms.get_positions(), bin_edges)
    chg = charge_features(atoms.info[charge_key], atoms.get_positions())
    return np.concatenate([wl, hist, chg])


def feature_matrix(atoms_list, **kw) -> np.ndarray:
    """(N, 323) raw feature matrix."""
    return np.vstack([featurize(a, **kw) for a in atoms_list])


def standardize(X: np.ndarray):
    """z-score across the population. Returns (X_std, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std
