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


def _graph_ase(atoms, cutoff_mult):
    from ase.neighborlist import build_neighbor_list, natural_cutoffs

    nl = build_neighbor_list(
        atoms,
        natural_cutoffs(atoms, mult=cutoff_mult),
        self_interaction=False,
        bothways=True,
    )
    n = len(atoms)
    return [
        sorted({int(j) for j in nl.get_neighbors(i)[0] if j != i}) for i in range(n)
    ]


def _graph_rdkit_ctd(atoms, cutoff_mult):
    """RDKit ``DetermineConnectivity`` -- "connect-the-dots", distance-based with
    valence-aware pruning. ``cutoff_mult`` is ignored (it has no such knob).

    Measured against the ASE threshold on 871 same-molecule conformer families
    (data_exploration/REPORT.md §16): it assigns the same graph to two conformers far
    more often (28.4% of families split, vs 87.7% for the ASE default) and is 2.4x
    faster. It never fails, unlike ``DetermineBonds``, which is 70% failure on this
    dataset (98% on metal complexes) and cannot even be timeout-guarded in-process.
    """
    import os
    from contextlib import contextmanager

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    @contextmanager
    def _quiet():
        # xyz2mol writes "!!! Warning !!!" straight to fd 2 (it fires on every O-H
        # bond here), which the Python logger cannot suppress.
        RDLogger.DisableLog("rdApp.*")
        fd, dn = os.dup(2), os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(dn, 2)
            yield
        finally:
            os.dup2(fd, 2)
            os.close(dn)
            os.close(fd)

    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    m = Chem.RWMol()
    conf = Chem.Conformer(len(Z))
    for i, z in enumerate(Z):
        m.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, Point3D(*[float(x) for x in pos[i]]))
    mol = m.GetMol()
    mol.AddConformer(conf)
    with _quiet():
        rdDetermineBonds.DetermineConnectivity(mol)

    adjacency = [[] for _ in range(len(Z))]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)
    return [sorted(a) for a in adjacency]


def _graph_rdkit_bonds(atoms, cutoff_mult):
    """RDKit ``DetermineBonds`` (xyz2mol) connectivity -- the actual *chemical* graph,
    from a valence-satisfying bond-order assignment. Bond orders are then dropped:
    it is the connectivity that is conformer-invariant (0.0% of same-molecule
    families split, vs 2.3% when the orders are kept, because resonance forms are
    not conformer-stable -- REPORT.md §16).

    **Raises** on the ~70% of `train_4M` it cannot solve (98% of metal complexes),
    and is unusably slow on those same structures -- 1.5 s/mol overall, though only
    1.7 ms/mol on organics, which is what makes the organic-subset experiment in
    `data_exploration/orthogonality.py` affordable. Not a production perceiver.
    """
    from rdkit.Chem import rdDetermineBonds

    from data_exploration.perception_rdkit import _quiet, _rw_mol

    Z = atoms.get_atomic_numbers()
    mol = _rw_mol(Z, atoms.get_positions())
    with _quiet():
        rdDetermineBonds.DetermineBonds(
            mol, charge=int(atoms.info.get("charge", 0)), embedChiral=False)
    adjacency = [[] for _ in range(len(Z))]
    orders = [[] for _ in range(len(Z))]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)
        o = b.GetBondTypeAsDouble()
        orders[i].append(o)
        orders[j].append(o)
    # The bond ORDERS are the only thing here that connect-the-dots cannot produce --
    # DetermineBonds' connectivity step IS DetermineConnectivity, so discarding the
    # orders would make this perceiver byte-identical to `rdkit_ctd`. WL labels nodes,
    # not edges, so the orders are folded into each atom's label as the sorted
    # multiset of its incident bond orders.
    labels = [f"{int(z)}#" + ",".join(f"{o:g}" for o in sorted(od))
              for z, od in zip(Z, orders)]
    return [sorted(a) for a in adjacency], labels


PERCEIVERS = {"ase": _graph_ase, "rdkit_ctd": _graph_rdkit_ctd,
              "rdkit_bonds": _graph_rdkit_bonds}


def build_graph(atoms, cutoff_mult: float = 1.2, perceiver: str = "ase"):
    """Geometry-derived connectivity. node label = atomic number; multi-molecule
    records -> multi-component graphs.

    ``perceiver`` selects how "bonded" is decided:

    ``ase``        (default, unchanged) ASE covalent radii scaled by ``cutoff_mult``.
    ``rdkit_ctd``  RDKit ``DetermineConnectivity`` -- see ``_graph_rdkit_ctd``.

    The default is deliberately the historical behaviour, so every existing result
    stays reproducible; the alternative has to earn its way in on measured R^2.
    """
    try:
        fn = PERCEIVERS[perceiver]
    except KeyError:
        raise ValueError(
            f"unknown perceiver {perceiver!r}; expected one of {sorted(PERCEIVERS)}")
    out = fn(atoms, cutoff_mult)
    # a perceiver may return just the adjacency, or (adjacency, node labels) when it
    # knows something about an atom that its element does not say (e.g. bond orders)
    if isinstance(out, tuple):
        return out
    return out, atoms.get_atomic_numbers().tolist()


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


def _labels_for_one(atoms, depth, cutoff_mult, perceiver="ase"):
    adj, lab = build_graph(atoms, cutoff_mult, perceiver)
    return len(lab), wl_labels_per_depth(adj, lab, depth)


# ----------------------------- chunk vectoriser ----------------------------
# Module-level (picklable) so Dask / multiprocessing can ship it to workers.


def _vectorize_chunk(atoms_chunk, spec):
    """Return (csr_block, oov, tot) for a chunk of molecules given the frozen
    vocab spec = dict(depths, vocab, offsets, ncols, normalize, depth, cutoff_mult,
    perceiver)."""
    depths = spec["depths"]
    vocab = spec["vocab"]
    offsets = spec["offsets"]
    ncols = spec["ncols"]
    normalize = spec["normalize"]
    depth = spec["depth"]
    cutoff_mult = spec["cutoff_mult"]
    # .get so a spec pickled by an older version still deserialises on a worker
    perceiver = spec.get("perceiver", "ase")

    rows, cols, vals = [], [], []
    oov = tot = 0
    for i, atoms in enumerate(atoms_chunk):
        n_atoms, pdl = _labels_for_one(atoms, depth, cutoff_mult, perceiver)
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
    perceiver: str = "ase"          # see build_graph; "ase" keeps historical behaviour
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
            _, pdl = _labels_for_one(atoms, self.depth, self.cutoff_mult,
                                     self.perceiver)
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
            "perceiver": self.perceiver,
        }

    # -- transform -----------------------------------------------------------
    def transform(self, atoms_list, client=None, n_procs=None, chunk=500):
        if self.vocab_ is None:
            raise RuntimeError("call fit() before transform().")
        spec = self._spec()
        parts = list(_chunks(list(atoms_list), chunk))
        t0 = time.perf_counter()

        if client is not None:
            # Scatter the frozen spec once. NOTE the [spec] list-wrap: client.scatter
            # on a bare dict is interpreted as a {key: value} MAPPING, which would
            # publish cluster keys named after our dict's keys ('vocab', 'cutoff_mult',
            # ...). Those get released after the first batch -> workers then fail with
            # "Could not find data: ['cutoff_mult']" and the run hangs.
            spec_f = client.scatter([spec], broadcast=True)[0]
            # Ship molecule chunks as scattered DATA rather than embedding them in the
            # task graph (otherwise the graph is ~120 MiB at 20k, ~1.5 GiB at 200k).
            part_futs = client.scatter(parts)
            futs = client.map(_vectorize_chunk, part_futs, [spec_f] * len(parts))
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
