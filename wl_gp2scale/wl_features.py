"""
wl_features.py  (wl_gp2scale)
=============================
Explicit-vocabulary Weisfeiler-Lehman descriptor at scale.

Differences from the 10k validation featurizer (which builds a dense
``np.zeros((N, D))``), motivated by the 200k blow-up:

  * Output is a ``scipy.sparse.csr_matrix`` assembled from per-molecule
    (row, col, count) triplets -- a dense (200k, ~10^5) matrix is never formed.
  * ``min_count`` defaults to 5 (tunable to 10) to prune the depth-3 singleton
    tail hard.
  * The vocabulary is fit on a STRATIFIED representative sample (all categories /
    elements) and then FROZEN, so the full transform and any later inference
    share one column space (OOV labels are dropped and the rate reported).
  * The transform is parallelised over molecules (Dask client if given, else a
    multiprocessing Pool, else serial). Each task returns a sparse row-block;
    blocks are stacked with ``scipy.sparse.vstack``.

WL labels use a blake2b hexdigest as a COLLISION-FREE canonical id for a refined
neighbourhood (not a lossy bucket hash).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


# ----------------------------- graph + WL labels ---------------------------


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


def wl_labels_per_depth(adjacency, node_labels, depth):
    """Per-atom WL labels at depths 0..depth. Depth 0 = element numbers; depth
    d>=1 = 64-bit blake2b hexdigest of (label | sorted neighbour labels)."""
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


def _labels_for_one(atoms, depth, cutoff_mult):
    adj, lab = build_graph(atoms, cutoff_mult)
    return len(lab), wl_labels_per_depth(adj, lab, depth)


# ----------------------------- chunk vectoriser ----------------------------
# Module-level (picklable) so Dask / multiprocessing can ship it to workers.


def _vectorize_chunk(atoms_chunk, spec):
    """Return (csr_block, oov, tot) for a chunk of molecules given the frozen
    vocab spec = dict(depths, vocab, offsets, ncols, normalize, depth, cutoff_mult)."""
    depths = spec["depths"]
    vocab = spec["vocab"]
    offsets = spec["offsets"]
    ncols = spec["ncols"]
    normalize = spec["normalize"]
    depth = spec["depth"]
    cutoff_mult = spec["cutoff_mult"]

    rows, cols, vals = [], [], []
    oov = tot = 0
    for i, atoms in enumerate(atoms_chunk):
        n_atoms, pdl = _labels_for_one(atoms, depth, cutoff_mult)
        counts = {}
        for d in depths:
            v, base = vocab[d], offsets[d]
            for L in pdl[d]:
                tot += 1
                j = v.get(L)
                if j is None:
                    oov += 1
                    continue
                col = base + j
                counts[col] = counts.get(col, 0.0) + 1.0
        scale = (1.0 / n_atoms) if (normalize and n_atoms > 0) else 1.0
        for col, c in counts.items():
            rows.append(i)
            cols.append(col)
            vals.append(c * scale)
    block = sp.csr_matrix(
        (np.asarray(vals, float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(atoms_chunk), ncols),
    )
    return block, oov, tot


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ----------------------------- featurizer ----------------------------------


@dataclass
class SparseWLFeaturizer:
    """Fitted sparse explicit-vocab WL featurizer.

    fit(atoms_sample): build the per-depth document-frequency vocab, keep labels
      present in >= min_count sample molecules, freeze the column layout.
    transform(atoms_list, client=None, n_procs=None, chunk=500): emit a
      csr_matrix of per-atom-normalised counts (OOV dropped, rate recorded).
    """

    depth: int = 3
    include_depth0: bool = False
    min_count: int = 5
    normalize: bool = True
    cutoff_mult: float = 1.2
    # frozen state
    vocab_: dict = field(default=None, repr=False)
    offsets_: dict = field(default=None, repr=False)
    depths_: list = field(default=None, repr=False)
    ncols_: int = field(default=None, repr=False)
    last_oov_rate_: float = field(default=None, repr=False)

    def __post_init__(self):
        self.depths_ = list(range(0 if self.include_depth0 else 1, self.depth + 1))

    # -- fit -----------------------------------------------------------------
    def fit(self, atoms_sample):
        t0 = time.perf_counter()
        df = {d: {} for d in self.depths_}
        for k, atoms in enumerate(atoms_sample):
            _, pdl = _labels_for_one(atoms, self.depth, self.cutoff_mult)
            for d in self.depths_:
                for L in set(pdl[d]):
                    df[d][L] = df[d].get(L, 0) + 1
            if (k + 1) % 5000 == 0:
                print(f"[wl-fit]   vocab scan {k + 1}/{len(atoms_sample)}")
        raw = sum(len(df[d]) for d in self.depths_)
        vocab, offsets, off = {}, {}, 0
        for d in self.depths_:
            offsets[d] = off
            vocab[d] = {L: j for j, (L, c) in enumerate(
                (kv for kv in df[d].items() if kv[1] >= self.min_count)
            )}
            off += len(vocab[d])
        self.vocab_, self.offsets_, self.ncols_ = vocab, offsets, off
        print(
            f"[wl-fit] {raw} raw labels -> {off} kept (min_count={self.min_count}) "
            f"from {len(atoms_sample)} sample mols in {time.perf_counter()-t0:.1f}s; "
            f"per-depth { {d: len(vocab[d]) for d in self.depths_} }"
        )
        return self

    @property
    def n_features_(self):
        return self.ncols_

    def _spec(self):
        return {
            "depths": self.depths_,
            "vocab": self.vocab_,
            "offsets": self.offsets_,
            "ncols": self.ncols_,
            "normalize": self.normalize,
            "depth": self.depth,
            "cutoff_mult": self.cutoff_mult,
        }

    # -- transform -----------------------------------------------------------
    def transform(self, atoms_list, client=None, n_procs=None, chunk=500):
        if self.vocab_ is None:
            raise RuntimeError("call fit() before transform().")
        spec = self._spec()
        parts = list(_chunks(list(atoms_list), chunk))
        t0 = time.perf_counter()

        if client is not None:
            # Dask: scatter the (small) frozen spec once, map chunks.
            spec_f = client.scatter(spec, broadcast=True)
            futs = client.map(_vectorize_chunk, parts, [spec_f] * len(parts))
            results = client.gather(futs)
        elif n_procs and n_procs > 1:
            import multiprocessing as mp

            with mp.Pool(n_procs) as pool:
                results = pool.starmap(
                    _vectorize_chunk, [(p, spec) for p in parts]
                )
        else:
            results = [_vectorize_chunk(p, spec) for p in parts]

        blocks = [r[0] for r in results]
        oov = sum(r[1] for r in results)
        tot = sum(r[2] for r in results)
        X = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, self.ncols_))
        self.last_oov_rate_ = oov / max(tot, 1)
        nnz = X.nnz
        dens = nnz / max(X.shape[0] * X.shape[1], 1)
        print(
            f"[wl] transformed {X.shape} (nnz={nnz:,}, density={dens:.2e}) in "
            f"{time.perf_counter()-t0:.1f}s; OOV {self.last_oov_rate_:.1%} of occurrences"
        )
        return X

    def fit_transform(self, atoms_sample, atoms_full=None, **kw):
        """Fit vocab on ``atoms_sample``; transform ``atoms_full`` (defaults to the
        sample). For 200k, pass a stratified sample here and the full list as
        ``atoms_full``."""
        self.fit(atoms_sample)
        return self.transform(atoms_full if atoms_full is not None else atoms_sample, **kw)
