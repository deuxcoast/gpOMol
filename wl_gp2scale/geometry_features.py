"""
geometry_features.py  (wl_gp2scale)
===================================
3D geometry + electrostatics descriptor at scale -- the second additive-kernel
channel alongside the WL graph (wl_features.py). Ported from
descriptor_eval/geometry.py (self-contained, per the module convention of NOT
importing descriptor_eval), with the WL featurizer's parallel-transform pattern so
it fans out over Dask / multiprocessing at 200k (featurisation is ~22 ms/mol).

The descriptor targets exactly the variance the WL *graph* is blind to: identical
bond graph, different 3D shape (rotamers/pucker/cis-trans) or partial-charge
arrangement -> different intensive energy. Four intensive, rotation/translation/
permutation-invariant channels along a body-order expansion:

  rdf      element-pair partial radial distribution functions (2-body), Gaussian
           broadened, one per unordered element pair among the frozen element set.
  angle    bonded bond-angle histogram (3-body).
  torsion  bonded |dihedral| histogram (4-body, the rotamer coordinate).
  elec     electrostatics from Loewdin partial charges: internal Coulomb sum
           (full + short-range), dipole/quadrupole magnitudes, per-element charge
           moments, global charge spread.

Unlike WL this returns a DENSE (N, ~558) matrix (the histograms are dense). It is
low-dimensional, so the downstream SparsePLS (reduce.py, which sp.csr_matrix-wraps
its input) reduces it to the same 10-D natural-scaled embedding, giving the same
N-invariant cutoff behaviour as the WL channel.

Like the WL vocab, the element set is a population op frozen at fit; the radial /
angular GRIDS are fixed a priori (params), so nothing else needs a train pass.
``channel_slices()`` exposes the four channel spans (parity with
descriptor_eval.geometry) for a future per-channel additive split.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .wl_features import _chunks, build_graph  # reuse graph perception + chunker

# ============================ geometry primitives ===========================


def _gaussian_hist(values, centers, sigma):
    """Gaussian-broadened histogram: each value drops a unit-height Gaussian at its
    location, summed onto the fixed bin centers. Empty input -> zeros."""
    if len(values) == 0:
        return np.zeros(len(centers))
    v = np.asarray(values, dtype=float)[:, None]
    return np.exp(-((centers[None, :] - v) ** 2) / (2.0 * sigma * sigma)).sum(axis=0)


def bond_angles(adjacency, positions):
    """Covalently-bonded bond angles (deg) for every j-i-k with j,k neighbours of i."""
    P = np.asarray(positions, dtype=float)
    out = []
    for i, nbrs in enumerate(adjacency):
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                u = P[nbrs[a]] - P[i]
                v = P[nbrs[b]] - P[i]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 1e-9 and nv > 1e-9:
                    out.append(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))
    return out


def dihedrals(adjacency, positions):
    """|dihedral| (deg, 0..180) over bonded quadruples i-j-k-l about each bond j-k.
    abs() folds enantiomeric conformers together. The rotamer coordinate."""
    P = np.asarray(positions, dtype=float)
    out = []
    n = len(adjacency)
    for j in range(n):
        for k in adjacency[j]:
            if k <= j:  # each bond once
                continue
            for i in adjacency[j]:
                if i == k:
                    continue
                for l in adjacency[k]:
                    if l == j or l == i:
                        continue
                    b1 = P[j] - P[i]
                    b2 = P[k] - P[j]
                    b3 = P[l] - P[k]
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    d = np.linalg.norm(b2)
                    x = n1 @ n2
                    y = np.cross(n1, n2) @ (b2 / d) if d > 1e-9 else 0.0
                    if abs(x) > 1e-12 or abs(y) > 1e-12:
                        out.append(abs(np.degrees(np.arctan2(y, x))))
    return out


# ============================ electrostatics ================================


def coulomb_terms(positions, charges, r_short=3.0):
    """[full_sum, short_range] internal electrostatic energy from partial charges:
    sum_{i<j} q_i q_j / r_ij over the whole molecule / within r_short, each /n_atoms
    (intensive). The full sum is a classical proxy for a genuine additive component
    of the total energy the WL graph cannot represent."""
    P = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float).ravel()
    n = len(q)
    if n < 2:
        return np.zeros(2)
    iu = np.triu_indices(n, k=1)
    diff = P[iu[0]] - P[iu[1]]
    r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    r = np.maximum(r, 1e-6)
    qq = q[iu[0]] * q[iu[1]]
    full = float((qq / r).sum()) / n
    short = float((qq[r < r_short] / r[r < r_short]).sum()) / n
    return np.array([full, short])


def multipole_magnitudes(positions, charges):
    """[|dipole|, |quadrupole|] about the geometric centroid, /n_atoms (intensive,
    translation-invariant). Quadrupole = Frobenius norm of the traceless 2nd moment."""
    P = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float).ravel()
    n = len(q)
    if n == 0:
        return np.zeros(2)
    R = P - P.mean(axis=0)
    dip = np.linalg.norm((R * q[:, None]).sum(axis=0))
    Q = (q[:, None, None] * (R[:, :, None] * R[:, None, :])).sum(axis=0)
    Q = Q - np.trace(Q) / 3.0 * np.eye(3)  # traceless
    quad = np.linalg.norm(Q)
    return np.array([dip, quad]) / n


def element_charge_moments(Z, charges, elements):
    """Per-element [mean, std] of Loewdin charge for each element in the frozen set
    (0 for an absent element). Already intensive."""
    Z = np.asarray(Z)
    q = np.asarray(charges, dtype=float).ravel()
    out = []
    for z in elements:
        m = Z == z
        if m.any():
            out.extend([float(q[m].mean()), float(q[m].std())])
        else:
            out.extend([0.0, 0.0])
    return np.array(out)


# ============================ chunk vectoriser ==============================
# Module-level (picklable) so Dask / multiprocessing can ship it to workers.


def _geom_vector_one(atoms, spec):
    """Raw geometry vector for one ase.Atoms given the frozen ``spec``.
    Returns (vec, n_oov_atoms, n_atoms)."""
    P = np.asarray(atoms.get_positions(), dtype=float)
    Z = np.asarray(atoms.get_atomic_numbers())
    n = len(Z)
    elements = spec["elements"]
    pairs = spec["pairs"]
    channels = spec["channels"]
    parts, n_oov = [], int(np.count_nonzero(~np.isin(Z, elements)))

    if "rdf" in channels:
        rdf = np.zeros((len(pairs), spec["n_rdf"]))
        if n >= 2:
            iu = np.triu_indices(n, k=1)
            diff = P[iu[0]] - P[iu[1]]
            r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            Za, Zb = Z[iu[0]], Z[iu[1]]
            for p, (a, b) in enumerate(pairs):
                m = ((Za == a) & (Zb == b)) | ((Za == b) & (Zb == a))
                if m.any():
                    rdf[p] = _gaussian_hist(r[m], spec["rdf_centers"], spec["sigma_rdf"])
        parts.append((rdf / max(n, 1)).ravel())

    if {"angle", "torsion"} & set(channels):
        adj, _ = build_graph(atoms, spec["cutoff_mult"])
        if "angle" in channels:
            h = _gaussian_hist(bond_angles(adj, P), spec["ang_centers"], spec["sigma_angle"])
            parts.append(h / max(n, 1))
        if "torsion" in channels:
            h = _gaussian_hist(dihedrals(adj, P), spec["tor_centers"], spec["sigma_torsion"])
            parts.append(h / max(n, 1))

    if "elec" in channels:
        q = np.asarray(atoms.info[spec["charge_key"]], dtype=float).ravel()
        parts.append(
            np.concatenate(
                [
                    coulomb_terms(P, q, spec["r_short"]),
                    multipole_magnitudes(P, q),
                    element_charge_moments(Z, q, elements),
                    np.array([float(np.var(q)) if n else 0.0,
                              float(q.max() - q.min()) if n else 0.0]),
                ]
            )
        )
    return np.concatenate(parts), n_oov, n


def _geom_vectorize_chunk(atoms_chunk, spec):
    """Return (dense_block (len(chunk), width), oov_atoms, tot_atoms) for a chunk."""
    rows, oov, tot = [], 0, 0
    for atoms in atoms_chunk:
        v, n_oov, n = _geom_vector_one(atoms, spec)
        rows.append(v)
        oov += n_oov
        tot += n
    return np.vstack(rows) if rows else np.zeros((0, spec["width"])), oov, tot


# ============================ featurizer ====================================


@dataclass
class SparseGeometryFeaturizer:
    """Fitted 3D-geometry + electrostatics featurizer (dense output). Same
    fit/transform(client=,n_procs=,chunk=)/n_features_/last_oov_rate_ interface as
    SparseWLFeaturizer, plus channel_slices(). The element set is frozen at fit; the
    radial/angular grids are fixed a priori.

    ``last_oov_rate_`` = fraction of atoms outside the frozen element set (ignored by
    the rdf/elec channels) -- the geometry analogue of WL's OOV.
    """

    elements: tuple | None = None
    top_k: int = 6
    channels: tuple = ("rdf", "angle", "torsion", "elec")
    r_max: float = 6.0
    n_rdf: int = 24
    sigma_rdf: float = 0.2
    n_angle: int = 18
    sigma_angle: float = 5.0
    n_torsion: int = 18
    sigma_torsion: float = 10.0
    r_short: float = 3.0
    charge_key: str = "lowdin_charges"
    cutoff_mult: float = 1.2
    # frozen state
    elements_: list = field(default=None, repr=False)
    pairs_: list = field(default=None, repr=False)
    slices_: dict = field(default=None, repr=False)
    ncols_: int = field(default=None, repr=False)
    last_oov_rate_: float = field(default=None, repr=False)

    def __post_init__(self):
        bad = set(self.channels) - {"rdf", "angle", "torsion", "elec"}
        if bad:
            raise ValueError(f"unknown channels {bad}")
        self._rdf_centers = np.linspace(0.0, self.r_max, self.n_rdf)
        self._ang_centers = np.linspace(0.0, 180.0, self.n_angle)
        self._tor_centers = np.linspace(0.0, 180.0, self.n_torsion)

    # -- fit: element vocabulary + fixed layout ------------------------------
    def fit(self, atoms_sample):
        t0 = time.perf_counter()
        if self.elements is not None:
            elems = sorted(int(z) for z in self.elements)
        else:
            df = Counter()
            for atoms in atoms_sample:
                df.update(set(int(z) for z in atoms.get_atomic_numbers()))
            elems = sorted(z for z, _ in df.most_common(self.top_k))
        self.elements_ = elems
        self.pairs_ = [
            (elems[i], elems[j]) for i in range(len(elems)) for j in range(i, len(elems))
        ]
        widths = {
            "rdf": len(self.pairs_) * self.n_rdf,
            "angle": self.n_angle,
            "torsion": self.n_torsion,
            "elec": 2 + 2 + 2 * len(elems) + 2,
        }
        self.slices_, off = {}, 0
        for ch in self.channels:
            self.slices_[ch] = (off, off + widths[ch])
            off += widths[ch]
        self.ncols_ = off
        print(
            f"[geom-fit] {len(elems)} elements {elems} -> {len(self.pairs_)} rdf pairs; "
            f"channels={list(self.channels)} -> {off} features "
            f"({time.perf_counter()-t0:.1f}s)"
        )
        return self

    @property
    def n_features_(self):
        return self.ncols_

    def channel_slices(self):
        """{channel -> (start, stop)} column spans for a per-channel additive split."""
        return dict(self.slices_)

    def _spec(self):
        return {
            "elements": self.elements_,
            "pairs": self.pairs_,
            "channels": self.channels,
            "rdf_centers": self._rdf_centers,
            "ang_centers": self._ang_centers,
            "tor_centers": self._tor_centers,
            "n_rdf": self.n_rdf,
            "sigma_rdf": self.sigma_rdf,
            "sigma_angle": self.sigma_angle,
            "sigma_torsion": self.sigma_torsion,
            "r_short": self.r_short,
            "charge_key": self.charge_key,
            "cutoff_mult": self.cutoff_mult,
            "width": self.ncols_,
        }

    # -- transform (parallel; mirrors SparseWLFeaturizer.transform) ----------
    def transform(self, atoms_list, client=None, n_procs=None, chunk=500):
        if self.elements_ is None:
            raise RuntimeError("call fit() before transform().")
        spec = self._spec()
        parts = list(_chunks(list(atoms_list), chunk))
        t0 = time.perf_counter()

        if client is not None:
            spec_f = client.scatter([spec], broadcast=True)[0]
            part_futs = client.scatter(parts)
            futs = client.map(_geom_vectorize_chunk, part_futs, [spec_f] * len(parts))
            results = client.gather(futs)
        elif n_procs and n_procs > 1:
            import multiprocessing as mp

            with mp.Pool(n_procs) as pool:
                results = pool.starmap(_geom_vectorize_chunk, [(p, spec) for p in parts])
        else:
            results = [_geom_vectorize_chunk(p, spec) for p in parts]

        blocks = [r[0] for r in results]
        oov = sum(r[1] for r in results)
        tot = sum(r[2] for r in results)
        X = np.vstack(blocks) if blocks else np.zeros((0, self.ncols_))
        self.last_oov_rate_ = oov / max(tot, 1)
        print(
            f"[geom] transformed {X.shape} in {time.perf_counter()-t0:.1f}s; "
            f"off-vocab atoms {self.last_oov_rate_:.1%}"
        )
        return X

    def fit_transform(self, atoms_sample, atoms_full=None, **kw):
        self.fit(atoms_sample)
        return self.transform(atoms_full if atoms_full is not None else atoms_sample, **kw)
