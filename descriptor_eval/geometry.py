"""
geometry.py  (descriptor_eval, Candidate: 3D geometry + electrostatics)
======================================================================
Self-contained geometry/charge descriptor for one ase.Atoms -> fixed-length
vector, scored against the SAME shared, descriptor-independent residual target
every other candidate uses (data.get_data). This targets exactly the variance
the WL *graph* descriptor is blind to: two molecules with an IDENTICAL bond
graph but different 3D shape (rotamers, ring pucker, cis/trans) or different
partial-charge arrangement are indistinguishable to WL, yet their intensive
energies differ. That gap is the ~0.44 of residual variance WL leaves on the
table (graph-only ceiling R^2 ~0.56 at 200k).

Four physics channels, each already intensive (no blanket /n_atoms -- each
channel carries its own intensive normalisation, so the electrostatic scalars
are not wrongly divided by n):

  rdf      element-pair partial radial distribution functions (Gaussian-broadened
           histograms of interatomic distances, one per unordered element pair).
           The distance histogram in features.py made chemically resolved: "how
           many C-O contacts sit at 2.4 A per atom", etc. Rotation/translation
           invariant. Sees through-space (non-bonded) proximity the graph omits.
  angle    bond-angle histogram over covalently-bonded triples j-i-k (Gaussian
           broadened). Hybridisation / strain the topology can't express.
  torsion  dihedral histogram over bonded quadruples i-j-k-l (|dihedral|, so
           enantiomeric conformers coincide). THE rotamer coordinate -- the most
           direct handle on conformational energy WL cannot see.
  elec     electrostatics from Loewdin partial charges: internal Coulomb sum
           (short-range + full, a literal additive term of the total energy),
           dipole + quadrupole magnitudes about the centroid, per-element charge
           moments (how electron-rich each element is), and global charge spread.

Like ESPH (persistence.py), the element vocabulary is a population op: the set of
elements used for the partial-RDF pairs and the per-element charge moments is
FROZEN at fit time (top_k most common, or an explicit list) so the output width
is fixed across molecules. The radial/angular GRIDS are fixed a priori (params),
so -- unlike the persistence imager -- nothing else needs a train pass.

channel_slices() exposes the four channel spans so a later additive kernel can
give each its own sub-embedding (parity with ElementPHFeaturizer). Requires only
numpy + ase (build_graph, for the covalent connectivity behind angles/torsions).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import features  # build_graph: ASE covalent-radius connectivity (same as WL)

# ============================ geometry primitives ===========================


def _gaussian_hist(values, centers, sigma):
    """Gaussian-broadened histogram: each value drops a unit-height Gaussian at
    its location, summed onto the fixed bin centers. Smoother than hard binning,
    so nearby geometries map to nearby vectors (a friendlier GP embedding). An
    empty value list returns zeros."""
    if len(values) == 0:
        return np.zeros(len(centers))
    v = np.asarray(values, dtype=float)[:, None]
    return np.exp(-((centers[None, :] - v) ** 2) / (2.0 * sigma * sigma)).sum(axis=0)


def bond_angles(adjacency, positions):
    """Covalently-bonded bond angles (degrees) for every j-i-k with j,k neighbours
    of i. Element-blind; the ANGLE value is the 3D info, and WL sees only that a
    triple exists, never its geometry."""
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
    """|dihedral| (degrees, 0..180) over bonded quadruples i-j-k-l about each bond
    j-k. abs() folds mirror-image (enantiomeric) conformers together, matching the
    energy's insensitivity to overall handedness. This is the rotamer coordinate."""
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
    """Internal electrostatic energy from partial charges: [full_sum, short_range]
    each = sum_{i<j} q_i q_j / r_ij over the whole molecule / within r_short, then
    /n_atoms (intensive). The full sum is a genuine additive component of the total
    energy the WL graph cannot represent; the short-range split isolates the
    contact electrostatics (H-bonds, ion pairs) from slow long-range tails."""
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
    translation-invariant, well-defined for net-charged species). Quadrupole is the
    Frobenius norm of the traceless second moment -- charge anisotropy the dipole
    misses."""
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
    (0 for an element absent from a molecule). Already intensive. Captures how
    electron-rich the O's / N's / H's run -- chemically direct polarisation signal
    the graph is blind to."""
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


# ============================ featurizer ====================================


@dataclass
class GeometryFeaturizer:
    """Fitted 3D-geometry + electrostatics descriptor (drop-in for the other
    featurizers: same fit / transform / fit_transform / n_features_ /
    last_oov_rate_ / channel_slices interface).

    elements: explicit atomic numbers for the RDF pairs + per-element charge
      moments; if None, the ``top_k`` most common elements are frozen at fit.
    channels: which of ("rdf","angle","torsion","elec") to emit (ablation).
    r_max/n_rdf/sigma_rdf: radial grid for the partial RDFs (Angstrom).
    n_angle/sigma_angle, n_torsion/sigma_torsion: angular grids (degrees).
    r_short: short-range Coulomb cutoff (Angstrom).
    charge_key: atoms.info key for the per-atom partial charges.
    cutoff_mult: covalent-radius multiplier for the bond graph (matches WL's 1.2).

    last_oov_rate_ = fraction of atoms whose element is outside the frozen set
    (ignored by the rdf / elec channels) -- the geometry analogue of WL's OOV.
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
    pairs_: list = field(default=None, repr=False)  # unordered element pairs (a<=b)
    slices_: dict = field(default=None, repr=False)  # channel -> (start, stop)
    last_oov_rate_: float = field(default=0.0, repr=False)

    def __post_init__(self):
        bad = set(self.channels) - {"rdf", "angle", "torsion", "elec"}
        if bad:
            raise ValueError(f"unknown channels {bad}")
        self._rdf_centers = np.linspace(0.0, self.r_max, self.n_rdf)
        self._ang_centers = np.linspace(0.0, 180.0, self.n_angle)
        self._tor_centers = np.linspace(0.0, 180.0, self.n_torsion)

    # -- element vocabulary + fixed layout -----------------------------------
    def _fit_layout(self, atoms_list):
        if self.elements is not None:
            elems = sorted(int(z) for z in self.elements)
        else:
            df = Counter()
            for atoms in atoms_list:
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
            "elec": 2 + 2 + 2 * len(elems) + 2,  # coulomb, multipole, per-elem, global
        }
        self.slices_, off = {}, 0
        for ch in self.channels:
            self.slices_[ch] = (off, off + widths[ch])
            off += widths[ch]
        self._width = off
        print(
            f"[geom] {len(elems)} elements {elems} -> {len(self.pairs_)} rdf pairs; "
            f"channels={list(self.channels)} -> {off} features"
        )

    # -- per-molecule raw vector ---------------------------------------------
    def _vector(self, atoms):
        P = np.asarray(atoms.get_positions(), dtype=float)
        Z = np.asarray(atoms.get_atomic_numbers())
        n = len(Z)
        parts, n_oov = [], int(np.count_nonzero(~np.isin(Z, self.elements_)))

        if "rdf" in self.channels:
            rdf = np.zeros((len(self.pairs_), self.n_rdf))
            if n >= 2:
                iu = np.triu_indices(n, k=1)
                diff = P[iu[0]] - P[iu[1]]
                r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                Za, Zb = Z[iu[0]], Z[iu[1]]
                for p, (a, b) in enumerate(self.pairs_):
                    m = ((Za == a) & (Zb == b)) | ((Za == b) & (Zb == a))
                    if m.any():
                        rdf[p] = _gaussian_hist(r[m], self._rdf_centers, self.sigma_rdf)
            parts.append((rdf / max(n, 1)).ravel())

        if {"angle", "torsion"} & set(self.channels):
            adj, _ = features.build_graph(atoms, self.cutoff_mult)
            if "angle" in self.channels:
                h = _gaussian_hist(bond_angles(adj, P), self._ang_centers, self.sigma_angle)
                parts.append(h / max(n, 1))
            if "torsion" in self.channels:
                h = _gaussian_hist(dihedrals(adj, P), self._tor_centers, self.sigma_torsion)
                parts.append(h / max(n, 1))

        if "elec" in self.channels:
            q = np.asarray(atoms.info[self.charge_key], dtype=float).ravel()
            parts.append(
                np.concatenate(
                    [
                        coulomb_terms(P, q, self.r_short),
                        multipole_magnitudes(P, q),
                        element_charge_moments(Z, q, self.elements_),
                        np.array([float(np.var(q)) if n else 0.0,
                                  float(q.max() - q.min()) if n else 0.0]),
                    ]
                )
            )
        return np.concatenate(parts), n_oov, n

    def _vectorize(self, atoms_list):
        t0 = time.perf_counter()
        rows, oov, tot = [], 0, 0
        for i, atoms in enumerate(atoms_list):
            v, n_oov, n = self._vector(atoms)
            rows.append(v)
            oov += n_oov
            tot += n
            if (i + 1) % 2000 == 0:
                print(f"[geom]   ...vectorized {i + 1}/{len(atoms_list)}")
        self.last_oov_rate_ = oov / max(tot, 1)
        X = np.vstack(rows)
        print(
            f"[geom] vectorized {X.shape} in {time.perf_counter() - t0:.1f}s "
            f"(off-vocab atoms {self.last_oov_rate_:.1%})"
        )
        return X

    @property
    def n_features_(self):
        return self._width

    def channel_slices(self):
        """Map each physics channel ('rdf','angle','torsion','elec') to its
        (start, stop) column span, so an additive kernel can weight them
        separately (parity with ElementPHFeaturizer.channel_slices)."""
        return dict(self.slices_)

    def fit(self, atoms_list):
        self._fit_layout(atoms_list)
        return self

    def fit_transform(self, atoms_list):
        self._fit_layout(atoms_list)
        return self._vectorize(atoms_list)

    def transform(self, atoms_list):
        if self.elements_ is None:
            raise RuntimeError("call fit() before transform().")
        return self._vectorize(atoms_list)
