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

# Sentinel atomic number for the catch-all "other element" bucket. Z=0 is not a real
# element, so it can never collide with a selected one.
OTHER_Z = 0

# The order channels are emitted in. `_geom_vector_one` appends blocks in THIS order, so
# `channel_slices()` must assign spans in it too -- otherwise a caller passing
# channels=("elec", "rdf") would get slices that mislabel every column.
CANONICAL_CHANNELS = ("rdf", "angle", "torsion", "elec", "strain")

# Strain features. The first block is emitted as an extensive SUM; STRAIN_INTENSIVE
# names are additionally emitted divided by atom count. max/p95 are deliberately NOT
# duplicated: they are already localised (not sums), so an intensive copy would be a
# perfectly collinear column that only hurts the PLS conditioning.
STRAIN_NAMES = ("bond_sq", "bond_abs", "bond_compress", "bond_extend", "bond_max",
                "bond_p95", "angle_sq", "angle_max", "clash", "n_bonds")
STRAIN_INTENSIVE = ("bond_sq", "bond_abs", "bond_compress", "bond_extend",
                    "angle_sq", "clash", "n_bonds")

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


# ============================ strain ========================================
# The census found that max|F| -- a pure off-equilibrium quantity -- explains R^2=0.397
# of the intensive residual while every INPUT-side column reaches < 0.001. Forces are
# labels, so that number names the missing physics rather than providing it. The rdf /
# angle / torsion channels cannot express it: they are population histograms divided by
# atom count, and strain is LOCALISED -- a few over-compressed bonds in a 42-atom
# molecule barely move a normalised histogram.
#
# This channel measures strain directly, as molecular mechanics does (the harmonic
# bonded terms of UFF / MMFF94 / AMBER), with the reference geometry LEARNED from the
# training molecules rather than tabulated. Measured on 20k structures it reaches
# held-out R^2 = 0.345 against y (85% of the max|F| diagnostic) from positions alone.


def _vdw_table():
    """van der Waals radii with a covalent fallback. ase's ``vdw_radii`` is shorter
    than ``covalent_radii`` (104 vs 119) and NaN for many elements."""
    from ase.data import covalent_radii, vdw_radii

    cr = np.asarray(covalent_radii, dtype=float)
    v = np.full(len(cr), np.nan)
    src = np.asarray(vdw_radii, dtype=float)
    v[: len(src)] = src[: len(v)]
    bad = ~np.isfinite(v)
    v[bad] = 2.0 * cr[bad]
    v[~np.isfinite(v)] = 1.7
    return v


_VDW = _vdw_table()


def _mode(x, lo, hi, width):
    """Histogram mode. The MODE, not the mean/median: train_4M is not relaxed minima
    (median max|F| = 3.4 eV/Å), so the strained tail drags the mean and median upward
    while the most common value still tracks the relaxed geometry."""
    if len(x) == 0:
        return float("nan")
    nb = max(int(round((hi - lo) / width)), 1)
    h, edges = np.histogram(x, bins=nb, range=(lo, hi))
    if h.sum() == 0:
        return float(np.median(x))
    k = int(np.argmax(h))
    return float(0.5 * (edges[k] + edges[k + 1]))


def bond_records(adjacency, positions, Z):
    """Every covalent bond as (Za, Zb, deg_a, deg_b, r), the pair ordered by Z.

    Coordination number travels with each end because it proxies BOND ORDER: keying the
    reference on the element pair alone would collapse C-C single / aromatic / double /
    triple (1.53 / 1.39 / 1.33 / 1.20 Å) into one mixture-mode, and deviations from that
    would encode chemistry rather than strain. With coordination the four separate
    cleanly (measured: 1.523 / 1.398 / 1.332 / 1.217)."""
    P = np.asarray(positions, dtype=float)
    out = []
    for i, nbrs in enumerate(adjacency):
        for j in nbrs:
            if j <= i:
                continue
            ki, kj = (i, j) if Z[i] <= Z[j] else (j, i)
            out.append((int(Z[ki]), int(Z[kj]), len(adjacency[ki]), len(adjacency[kj]),
                        float(np.linalg.norm(P[i] - P[j]))))
    return out


@dataclass
class StrainReference:
    """Learned equilibrium geometry: r0 per (element pair, coordination pair) and
    theta0 per (centre element, coordination). Unsupervised -- it never sees y."""

    min_count: int = 30
    r0_: dict = field(default_factory=dict)
    r0_pair_: dict = field(default_factory=dict)
    th0_: dict = field(default_factory=dict)

    def fit(self, records, angles):
        bl, blp, ang = {}, {}, {}
        for a, b, da, db, r in records:
            bl.setdefault((a, b, da, db), []).append(r)
            blp.setdefault((a, b), []).append(r)
        for zc, dg, t in angles:
            ang.setdefault((zc, dg), []).append(t)
        for k, v in bl.items():
            if len(v) >= self.min_count:
                self.r0_[k] = _mode(np.asarray(v), 0.4, 3.5, 0.005)
        for k, v in blp.items():
            if len(v) >= self.min_count:
                self.r0_pair_[k] = _mode(np.asarray(v), 0.4, 3.5, 0.005)
        for k, v in ang.items():
            if len(v) >= self.min_count:
                self.th0_[k] = _mode(np.asarray(v), 0.0, 180.0, 0.5)
        print(f"[geom-fit] strain reference: {len(self.r0_)} (pair, coordination) r0, "
              f"{len(self.r0_pair_)} pair-only r0, {len(self.th0_)} theta0")
        return self

    def r0(self, a, b, da, db):
        v = self.r0_.get((a, b, da, db))
        if v is not None:
            return v
        v = self.r0_pair_.get((a, b))
        if v is not None:
            return v
        from ase.data import covalent_radii
        return float(covalent_radii[a] + covalent_radii[b])

    def theta0(self, zc, dg):
        v = self.th0_.get((zc, dg))
        if v is not None:
            return v
        return {1: 180.0, 2: 180.0, 3: 120.0, 4: 109.5}.get(dg, 109.5)


def strain_features(adjacency, positions, Z, ref, n_atoms):
    """-> the STRAIN_NAMES vector (extensive) followed by the STRAIN_INTENSIVE subset
    divided by atom count."""
    P = np.asarray(positions, dtype=float)
    recs = bond_records(adjacency, P, Z)
    if recs:
        r = np.array([x[4] for x in recs])
        r0 = np.array([ref.r0(x[0], x[1], x[2], x[3]) for x in recs])
        d = r - r0
        ad = np.abs(d)
        f = [float((d ** 2).sum()), float(ad.sum()),
             float((d[d < 0] ** 2).sum()), float((d[d > 0] ** 2).sum()),
             float(ad.max()), float(np.percentile(ad, 95))]
    else:
        f = [0.0] * 6

    th = []
    for i, nbrs in enumerate(adjacency):
        k = len(nbrs)
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                u, v = P[nbrs[a_]] - P[i], P[nbrs[b_]] - P[i]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 1e-9 and nv > 1e-9:
                    ang = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)))
                    # RADIANS, not degrees. theta0 is learned in degrees (the mode of a
                    # degree-valued histogram), but the SQUARED sum inherits the unit:
                    # in degrees a single 30 deg deviation contributes 900 and a large
                    # molecule reaches ~1e6, which is 6 orders of magnitude above every
                    # other feature. Pareto scaling (/sqrt(std)) deliberately preserves
                    # scale, so such a column dominates the PLS embedding and blows the
                    # compact-support cutoff up (measured: cutoff 2.4 -> 180, geom GP
                    # R^2 0.317 -> 0.064). Radians keeps it O(1).
                    th.append(np.radians(ang - ref.theta0(int(Z[i]), k)))
    if th:
        th = np.asarray(th)
        f += [float((th ** 2).sum()), float(np.abs(th).max())]
    else:
        f += [0.0, 0.0]

    # steric clash: non-bonded, non-1-3 pairs inside the vdW contact distance. The
    # bonded and angular terms already own 1-2 and 1-3, so what is left is genuine
    # non-bonded repulsion -- a strain source no bonded term can express.
    clash, n = 0.0, len(Z)
    if n >= 2:
        excl = set()
        for i, nbrs in enumerate(adjacency):
            for j in nbrs:
                excl.add((min(i, j), max(i, j)))
            for a_ in range(len(nbrs)):
                for b_ in range(a_ + 1, len(nbrs)):
                    p, q = nbrs[a_], nbrs[b_]
                    excl.add((min(p, q), max(p, q)))
        iu = np.triu_indices(n, k=1)
        diff = P[iu[0]] - P[iu[1]]
        rr = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        rc = _VDW[Z[iu[0]]] + _VDW[Z[iu[1]]]
        for h in np.where(rr < rc)[0]:
            if (int(iu[0][h]), int(iu[1][h])) not in excl:
                clash += float((rc[h] - rr[h]) ** 2)
    f += [clash, float(len(recs))]

    ext = np.array(f)
    idx = [STRAIN_NAMES.index(nm) for nm in STRAIN_INTENSIVE]
    return np.concatenate([ext, ext[idx] / max(n_atoms, 1)])


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
    real = spec["elements_real"]
    off = ~np.isin(Z, real)
    n_oov = int(np.count_nonzero(off))
    Z_true = Z                      # strain keys its reference by REAL element
    if spec["other_bucket"]:
        # fold every off-vocab atom into the catch-all so it still contributes to the
        # RDF pairs and charge moments instead of being silently dropped
        Z = np.where(off, OTHER_Z, Z)
    # Blocks are keyed by channel and concatenated in CANONICAL_CHANNELS order at the
    # end, so the emitted layout can never drift from the spans channel_slices()
    # reports -- whatever order the caller listed the channels in.
    blocks = {}

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
        blocks["rdf"] = (rdf / max(n, 1)).ravel()

    if {"angle", "torsion", "strain"} & set(channels):
        adj, _ = build_graph(atoms, spec["cutoff_mult"])
        if "angle" in channels:
            h = _gaussian_hist(bond_angles(adj, P), spec["ang_centers"], spec["sigma_angle"])
            blocks["angle"] = h / max(n, 1)
        if "torsion" in channels:
            h = _gaussian_hist(dihedrals(adj, P), spec["tor_centers"], spec["sigma_torsion"])
            blocks["torsion"] = h / max(n, 1)
        if "strain" in channels:
            blocks["strain"] = strain_features(adj, P, Z_true, spec["strain_ref"], n)

    if "elec" in channels:
        q = np.asarray(atoms.info[spec["charge_key"]], dtype=float).ravel()
        blocks["elec"] = (
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
    vec = np.concatenate([blocks[ch] for ch in CANONICAL_CHANNELS if ch in blocks])
    return vec, n_oov, n


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

    ``last_oov_rate_`` = fraction of atoms outside the frozen element set. With
    ``other_bucket`` (the default) those atoms are not dropped -- they are folded into
    a single catch-all pseudo-element -- so the rate is diagnostic, not a loss.

    Element selection (``element_select``) matters more than it looks. The original
    default counted the number of STRUCTURES containing each element, which is
    misaligned with atom abundance: on train_4M that picks H,C,N,O,P,S and EXCLUDES
    fluorine even though F is the 5th most abundant atom in the dataset (1.09% of all
    atoms, vs P at 0.499%). Counting ATOMS (the default) swaps P for F and MEASURABLY
    HELPS: held-out OLS R^2 on the embedding 0.2625 -> 0.2783 at N=2000.

    MEASURED -- element coverage pays at production N and is HARMFUL on small runs.
    The controlling variable is p/n = feature columns / training molecules, and the
    ranking INVERTS between the two regimes (OLS R^2 on the PLS embedding):

        config                        cols   N_tr=1600      N_tr=16000
        top_k=6,  no bucket            558   0.2783 (p/n .35)  0.4263 (.03)
        top_k=10, no bucket           1382  -0.0493 (.86)      0.4366 (.09)
        top_k=10, bucket  (DEFAULT)   1648  -0.1157 (1.03)     0.4484 (.10)

    So the defaults are tuned for N_train >= ~10k, where wider coverage is worth
    +0.022, and ``fit`` warns when p/n exceeds ``max_p_over_n``. For a smoke test at
    N~2k pass ``top_k=6, other_bucket=False``.

    NOT a rare-column problem -- that hypothesis was tested and REJECTED. Pruning rare
    element pairs (``pair_min_frac``) barely moves the small-N failure: at top_k=10,
    dropping 55->38 pairs only lifts R^2 from -0.049 to -0.021, nowhere near the 0.278
    of the narrow configuration. The degradation is monotonic in aggregate width, not
    caused by specific toxic columns, so ``pair_min_frac`` defaults to 0 (off). It is
    kept only so the negative result is not re-derived.
    """

    elements: tuple | None = None
    top_k: int = 10
    element_select: str = "atoms"     # "atoms" (abundance) | "structures" (legacy)
    other_bucket: bool = True         # fold off-vocab atoms into one pseudo-element
    pair_min_frac: float = 0.0        # drop RDF pairs rarer than this fraction of mols
    max_p_over_n: float = 0.25        # warn above this width/(fit molecules) ratio
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
    ref_sample: int = 20_000        # molecules used to learn the strain reference
    strain_min_count: int = 30      # min observations before a key gets its own r0
    # frozen state
    elements_: list = field(default=None, repr=False)      # incl. OTHER_Z if bucketed
    elements_real_: list = field(default=None, repr=False)  # the selected elements only
    pairs_: list = field(default=None, repr=False)
    slices_: dict = field(default=None, repr=False)
    strain_ref_: "StrainReference" = field(default=None, repr=False)
    ncols_: int = field(default=None, repr=False)
    last_oov_rate_: float = field(default=None, repr=False)

    def __post_init__(self):
        bad = set(self.channels) - set(CANONICAL_CHANNELS)
        if bad:
            raise ValueError(f"unknown channels {bad}")
        # normalise to the emission order so slices and layout cannot disagree
        self.channels = tuple(c for c in CANONICAL_CHANNELS if c in set(self.channels))
        self._rdf_centers = np.linspace(0.0, self.r_max, self.n_rdf)
        self._ang_centers = np.linspace(0.0, 180.0, self.n_angle)
        self._tor_centers = np.linspace(0.0, 180.0, self.n_torsion)

    # -- fit: element vocabulary + fixed layout ------------------------------
    def fit(self, atoms_sample):
        t0 = time.perf_counter()
        if self.elements is not None:
            real = sorted(int(z) for z in self.elements)
        else:
            if self.element_select not in ("atoms", "structures"):
                raise ValueError("element_select must be 'atoms' or 'structures'")
            df = Counter()        # ranking counter (depends on element_select)
            atom_df = Counter()   # always per-atom, so coverage is reported honestly
            for atoms in atoms_sample:
                Z = atoms.get_atomic_numbers()
                atom_df.update(int(z) for z in Z)
                # "atoms" ranks by abundance; "structures" counts each element once per
                # molecule (document frequency, the legacy behaviour).
                df.update(int(z) for z in (Z if self.element_select == "atoms" else set(Z)))
            real = sorted(z for z, _ in df.most_common(self.top_k))
            cov = sum(atom_df[z] for z in real) / max(sum(atom_df.values()), 1)
            print(f"[geom-fit] element_select={self.element_select}: top-{self.top_k} "
                  f"covers {cov:.2%} of ATOMS in the fit sample")
        self.elements_real_ = real
        # OTHER_Z is a sentinel pseudo-element every off-vocab atom is mapped to, so no
        # atom is invisible to the descriptor. This matters for metal complexes (20% of
        # train_4M): the metal centre is the defining atom yet is far too rare to earn
        # its own column, and without the bucket it contributed nothing at all.
        elems = ([OTHER_Z] + real) if self.other_bucket else real
        self.elements_ = elems
        pairs = [
            (elems[i], elems[j]) for i in range(len(elems)) for j in range(i, len(elems))
        ]
        # Prune rare element PAIRS -- the direct analogue of SparseWLFeaturizer's
        # min_count pruning of the WL vocabulary's singleton tail, and the reason
        # raising top_k is safe. An I-Br or Se-Se column is zero for ~every molecule
        # with a few large values; pareto scaling turns such a k-of-N feature into a
        # spike ~(N/k)^(1/4) that greedy SIMPLS overfits (under `standard` it is
        # ~sqrt(N/k) and the fit blows up numerically -- measured R^2 ~ -1e50). Dropping
        # the column is strictly better than rescaling it: the signal it carries is
        # unestimable at that frequency anyway.
        if self.pair_min_frac > 0:
            pair_df = Counter()
            for atoms in atoms_sample:
                Z = np.asarray(atoms.get_atomic_numbers())
                if self.other_bucket:
                    Z = np.where(np.isin(Z, real), Z, OTHER_Z)
                zc = Counter(int(z) for z in Z)
                for a, b in pairs:
                    # the RDF column can only be nonzero if the molecule has the atoms
                    if (zc[a] >= 2) if a == b else (zc[a] and zc[b]):
                        pair_df[(a, b)] += 1
            need = max(1, int(round(self.pair_min_frac * len(atoms_sample))))
            kept = [p for p in pairs if pair_df[p] >= need]
            from ase.data import chemical_symbols as _S
            dropped = [p for p in pairs if p not in set(kept)]
            print(f"[geom-fit] rdf-pair prune: {len(pairs)} -> {len(kept)} pairs "
                  f"(min_frac={self.pair_min_frac:g} -> >= {need} of "
                  f"{len(atoms_sample)} mols); dropped e.g. "
                  f"{[f'{_S[a] if a else 'OTH'}-{_S[b] if b else 'OTH'}' for a, b in dropped[:6]]}")
            pairs = kept if kept else pairs
        self.pairs_ = pairs
        if "strain" in self.channels:
            self._fit_strain_reference(atoms_sample)
        widths = {
            "rdf": len(self.pairs_) * self.n_rdf,
            "angle": self.n_angle,
            "torsion": self.n_torsion,
            "elec": 2 + 2 + 2 * len(elems) + 2,
            "strain": len(STRAIN_NAMES) + len(STRAIN_INTENSIVE),
        }
        self.slices_, off = {}, 0
        for ch in self.channels:
            self.slices_[ch] = (off, off + widths[ch])
            off += widths[ch]
        self.ncols_ = off
        # p/n guard. The geometry embedding degrades MONOTONICALLY in width/(training
        # molecules) -- measured OLS R^2 at 1600 train rows: 558 cols (p/n 0.35) 0.278,
        # 728 (0.46) 0.025, 974 (0.61) -0.021, 1382 (0.86) -0.049, 1648 (1.03) -0.116.
        # At 16000 train rows the SAME configurations invert: 558 -> 0.426, 1382 ->
        # 0.437, 1648 -> 0.448. So wide element coverage is right at production N and
        # catastrophic on a small smoke test, and the failure is SILENT -- hence this
        # warning rather than a footgun.
        n_fit = max(len(atoms_sample), 1)
        if off > self.max_p_over_n * n_fit:
            print(f"[geom-fit] WARNING: {off} feature columns from only {n_fit:,} fit "
                  f"molecules (p/n={off/n_fit:.2f} > {self.max_p_over_n}). The PLS "
                  f"embedding degrades badly in this regime -- measured R^2 goes "
                  f"NEGATIVE by p/n~0.6. For a small run pass top_k=6, "
                  f"other_bucket=False (558 cols); the wide default is tuned for "
                  f"N_train >= ~10k.")
        print(
            f"[geom-fit] {len(elems)} elements {elems} -> {len(self.pairs_)} rdf pairs; "
            f"channels={list(self.channels)} -> {off} features "
            f"({time.perf_counter()-t0:.1f}s)"
        )
        return self

    def _fit_strain_reference(self, atoms_sample):
        """Learn r0 / theta0 from the training molecules. This is the only part of fit
        that needs bond perception (~ms per molecule), so it runs on at most
        ``ref_sample`` structures -- mode statistics over element pairs converge on a
        few thousand, long before the full train set. Unsupervised: never sees y."""
        t0 = time.perf_counter()
        sample = atoms_sample
        if self.ref_sample and len(atoms_sample) > self.ref_sample:
            step = max(len(atoms_sample) // self.ref_sample, 1)
            sample = atoms_sample[::step][: self.ref_sample]
        recs, angs = [], []
        for atoms in sample:
            Z = np.asarray(atoms.get_atomic_numbers())
            P = np.asarray(atoms.get_positions(), dtype=float)
            adj, _ = build_graph(atoms, self.cutoff_mult)
            recs.extend(bond_records(adj, P, Z))
            for i, nbrs in enumerate(adj):
                k = len(nbrs)
                for a_ in range(k):
                    for b_ in range(a_ + 1, k):
                        u, v = P[nbrs[a_]] - P[i], P[nbrs[b_]] - P[i]
                        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                        if nu > 1e-9 and nv > 1e-9:
                            angs.append((int(Z[i]), k, float(np.degrees(
                                np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))))
        self.strain_ref_ = StrainReference(min_count=self.strain_min_count).fit(recs, angs)
        print(f"[geom-fit] strain reference from {len(sample):,} molecules "
              f"({len(recs):,} bonds) in {time.perf_counter()-t0:.1f}s")

    @property
    def n_features_(self):
        return self.ncols_

    def channel_slices(self):
        """{channel -> (start, stop)} column spans for a per-channel additive split."""
        return dict(self.slices_)

    def _spec(self):
        return {
            "elements": self.elements_,
            "elements_real": self.elements_real_,
            "other_bucket": self.other_bucket,
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
            "strain_ref": self.strain_ref_,   # plain dicts -> picklable, scatters fine
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
