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
import time
from dataclasses import dataclass, field

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


# ============================================================================
# WL-only, explicit-vocabulary featurizer (fitted on train)
# ============================================================================
# Replaces the collision-prone blake2b-mod-256 hashing with an EXACT per-depth
# vocabulary learned from the training molecules. Geometry + charge channels are
# dropped (WL-only); an additive kernel can reintroduce them later. Fit builds
# the label->column maps from train; transform emits exact counts (OOV labels
# dropped, rate reported). Depths are separate blocks; depth 0 (bare element
# counts) is dropped by default since the extensive mean already removes
# composition. mode="hashed" reproduces the old behaviour for A/B comparison.


def wl_labels_per_depth(adjacency, node_labels, depth):
    """Per-atom WL labels at each depth 0..depth. Depth 0 = element numbers;
    depth d>=1 = 64-bit blake2b digest of (label | sorted neighbour labels),
    which is the collision-free canonical id of that refined neighbourhood."""
    n = len(node_labels)
    labels = [str(l) for l in node_labels]
    per_depth = [list(labels)]
    for _ in range(depth):
        expanded = [
            labels[i] + "|" + ",".join(sorted(labels[j] for j in adjacency[i]))
            for i in range(n)
        ]
        labels = [
            hashlib.blake2b(e.encode(), digest_size=8).hexdigest() for e in expanded
        ]
        per_depth.append(list(labels))
    return per_depth


@dataclass
class WLFeaturizer:
    """Fitted WL descriptor.

    explicit (default): EXACT per-depth count vocabulary, pruned by min_count
      (drop labels occurring in fewer than min_count TRAIN molecules -- removes
      the huge singleton tail of depth-3 patterns that only bloat D and can't
      generalise). One pass over the molecules (graphs + WL labels computed once
      and reused for both vocab-building and vectorising).
    hashed: legacy 256-bucket hashing (for A/B).

    Counts are per-atom normalised (intensive). Depth 0 dropped unless
    include_depth0=True. Progress + per-phase timing are printed.
    """

    depth: int = 3
    include_depth0: bool = False
    mode: str = "explicit"  # "explicit" | "hashed"
    min_count: int = 2  # explicit: keep labels in >= this many train mols
    n_buckets: int = 256  # hashed mode only
    normalize: bool = True
    cutoff_mult: float = 1.2
    vocab_: dict = field(default=None, repr=False)
    depths_: list = field(default=None, repr=False)
    last_oov_rate_: float = field(default=None, repr=False)

    def __post_init__(self):
        self.depths_ = list(range(0 if self.include_depth0 else 1, self.depth + 1))

    # --- one shared walk: graph + per-depth WL labels for each molecule --------
    def _labels_for(self, atoms_list):
        t0 = time.perf_counter()
        out = []
        for i, atoms in enumerate(atoms_list):
            adj, lab = build_graph(atoms, self.cutoff_mult)
            out.append((len(lab), wl_labels_per_depth(adj, lab, self.depth)))
            if (i + 1) % 2000 == 0:
                print(f"[wl]   ...labels {i + 1}/{len(atoms_list)}")
        print(
            f"[wl] extracted labels for {len(atoms_list)} mols in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return out

    def _build_vocab(self, labels_data):
        df = {d: {} for d in self.depths_}  # label -> #molecules containing it
        for _, pdl in labels_data:
            for d in self.depths_:
                for L in set(pdl[d]):
                    df[d][L] = df[d].get(L, 0) + 1
        raw = sum(len(df[d]) for d in self.depths_)
        vocab = {}
        for d in self.depths_:
            vocab[d] = {}
            for L, c in df[d].items():
                if c >= self.min_count:
                    vocab[d][L] = len(vocab[d])
        self.vocab_ = vocab
        kept = sum(len(vocab[d]) for d in self.depths_)
        print(
            f"[wl] vocab: {raw} raw labels -> {kept} kept "
            f"(min_count={self.min_count}); per-depth "
            f"{ {d: len(vocab[d]) for d in self.depths_} }"
        )

    def _vectorize(self, labels_data):
        offsets, off = {}, 0
        for d in self.depths_:
            offsets[d] = off
            off += len(self.vocab_[d])
        t0 = time.perf_counter()
        X = np.zeros((len(labels_data), off))
        oov = tot = 0
        for i, (n_atoms, pdl) in enumerate(labels_data):
            for d in self.depths_:
                v, base = self.vocab_[d], offsets[d]
                for L in pdl[d]:
                    tot += 1
                    j = v.get(L)
                    if j is None:
                        oov += 1
                        continue
                    X[i, base + j] += 1.0
            if self.normalize and n_atoms > 0:
                X[i] /= n_atoms
        self.last_oov_rate_ = oov / max(tot, 1)
        print(
            f"[wl] vectorized {X.shape} in {time.perf_counter() - t0:.1f}s "
            f"(dropped/OOV {self.last_oov_rate_:.1%} of label occurrences)"
        )
        return X

    def _hashed(self, atoms_list):
        X = np.zeros((len(atoms_list), self.n_buckets))
        for i, atoms in enumerate(atoms_list):
            adj, lab = build_graph(atoms, self.cutoff_mult)
            pdl = wl_labels_per_depth(adj, lab, self.depth)
            for d in self.depths_:
                for L in pdl[d]:
                    X[i, _bucket(f"{d}:{L}", self.n_buckets)] += 1.0
            if self.normalize:
                X[i] /= max(len(lab), 1)
        self.last_oov_rate_ = 0.0
        return X

    @property
    def n_features_(self):
        if self.mode == "hashed":
            return self.n_buckets
        return sum(len(self.vocab_[d]) for d in self.depths_)

    def fit(self, atoms_list):
        if self.mode == "hashed":
            return self
        self._build_vocab(self._labels_for(atoms_list))
        return self

    def fit_transform(self, atoms_list):
        if self.mode == "hashed":
            return self._hashed(atoms_list)
        ld = self._labels_for(atoms_list)  # single walk, reused below
        self._build_vocab(ld)
        return self._vectorize(ld)

    def transform(self, atoms_list):
        if self.mode == "hashed":
            return self._hashed(atoms_list)
        return self._vectorize(self._labels_for(atoms_list))
