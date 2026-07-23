"""
persistence.py  (descriptor_eval, Candidate: persistent homology / TDA)
======================================================================
Self-contained persistent-homology descriptor for one ase.Atoms -> fixed-length
vector, scored against the SAME shared, descriptor-independent residual target
every other candidate uses (data.get_data). This tests Marcus's question
directly: do distances between persistence diagrams correlate with the molecular
property?

Pipeline per molecule:
  positions (N,3)  --Vietoris-Rips-->  diagrams {H0, H1, ...}
                   --PersistenceImager (population-fit)-->  fixed-length vector

H0 captures connected-component merge structure, H1 rings/loops, H2 voids/cages
-- exactly the 3D shape the WL *graph* descriptor is blind to. Diagrams are built
from pairwise distances only, so they are rotation/translation invariant. The
image is divided by atom count to stay intensive (comparable to the WL channels
and to the intensive residual target).

Like standardize() and the WL vocabulary, the image GRID (birth/persistence
ranges) is a population operation -- it must be shared across molecules for the
vectors to be comparable -- so a PersistenceImager is FIT on all diagrams first,
then applied per molecule. fit/transform are split so a later train/test harness
(e.g. gp_parity) can fit the grid on TRAIN only.

Requires: ripser (diagrams) + persim (PersistenceImager). Neither is in
requirements.txt -- install into the conda env before running.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

# ----------------------------- diagrams ------------------------------------


def rips_diagrams(positions, maxdim: int = 1, thresh: float | None = None):
    """Vietoris-Rips persistence diagrams of a 3D point cloud.

    Returns a list of (n_k, 2) birth/death arrays, one per homology dimension
    0..maxdim. Infinite and zero-length bars are dropped: the essential H0 class
    (death=inf) and any birth==death points carry no shape the imager can place.
    """
    from ripser import ripser

    positions = np.asarray(positions, dtype=float)
    kw = {"maxdim": maxdim}
    if thresh is not None:
        kw["thresh"] = float(thresh)
    # ripser warns "more columns than rows" / "matrix is square" for small (<=3
    # atom) molecules -- both are benign here: the input is always a point cloud
    # (distance_matrix defaults off), so silence them to keep the log readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dgms = ripser(positions, **kw)["dgms"]

    out = []
    for dgm in dgms:
        dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
        if dgm.size:
            keep = np.isfinite(dgm[:, 1]) & (dgm[:, 1] > dgm[:, 0])
            dgm = dgm[keep]
        out.append(dgm.reshape(-1, 2))
    return out


# ----------------------------- featurizer ----------------------------------

# non-degenerate stand-in used ONLY when a homology dimension is empty across the
# ENTIRE population (e.g. no H1 rings in a tiny local test set) so the imager
# still has a well-defined birth/persistence grid and transform() returns zeros.
_DUMMY_DGM = np.array([[0.0, 1.0], [1.0, 3.0]])


def _pad_range(rng, pixel_size):
    """Widen a degenerate (zero-width) range to one pixel. H0 in a Rips filtration
    has birth==0 for EVERY bar, so skewed births collapse to a zero-width axis ->
    persim rounds it to 0 pixels and the whole H0 image (the merge-distance
    structure) is silently dropped. Padding the birth axis to one pixel keeps H0
    as a (1, n_persistence) column."""
    lo, hi = float(rng[0]), float(rng[1])
    if hi - lo < pixel_size:
        hi = lo + pixel_size
    return (lo, hi)


def _fit_imager(pooled, pixel_size):
    """Fit one PersistenceImager on a pool of diagrams (an empty pool falls back to
    a non-degenerate stand-in so the grid is still defined), padding both axes."""
    from persim import PersistenceImager

    pimgr = PersistenceImager(pixel_size=pixel_size)
    pimgr.fit(pooled if pooled else [_DUMMY_DGM], skew=True)
    pimgr.birth_range = _pad_range(pimgr.birth_range, pixel_size)
    pimgr.pers_range = _pad_range(pimgr.pers_range, pixel_size)
    return pimgr


def _image_or_zeros(imager, dgm):
    """Flattened persistence image of one diagram, or the correctly-sized zero
    vector when the diagram is empty (e.g. a molecule with no atoms of an element)."""
    if len(dgm):
        return np.asarray(imager.transform(dgm, skew=True)).ravel()
    return np.zeros(int(np.prod(imager.resolution)))


@dataclass
class PersistenceFeaturizer:
    """Fitted persistence-image descriptor.

    Computes Rips diagrams per molecule (H0..H_maxdim), fits one persim
    PersistenceImager PER homology dimension on the pooled diagrams (so births
    and persistences share a common grid across the population), then flattens +
    concatenates the per-dimension images into one vector.

    normalize=True divides each molecule's image by its atom count -> intensive.
    """

    maxdim: int = 1
    thresh: float | None = None
    pixel_size: float = 1.0
    normalize: bool = True
    imagers_: list = field(default=None, repr=False)
    # interface parity with WLFeaturizer (gp_parity.evaluate reads .last_oov_rate_);
    # persistence has no vocabulary, so nothing is ever out-of-vocabulary.
    last_oov_rate_: float = field(default=0.0, repr=False)

    # --- one shared walk: diagrams for each molecule ---------------------------
    def _diagrams_for(self, atoms_list):
        t0 = time.perf_counter()
        out = []
        for i, atoms in enumerate(atoms_list):
            dg = rips_diagrams(atoms.get_positions(), self.maxdim, self.thresh)
            out.append((len(atoms), dg))
            if (i + 1) % 2000 == 0:
                print(f"[ph]   ...diagrams {i + 1}/{len(atoms_list)}")
        print(
            f"[ph] computed diagrams for {len(atoms_list)} mols in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return out

    def _fit_imagers(self, diagrams_data):
        self.imagers_ = []
        for d in range(self.maxdim + 1):
            pooled = [dg[d] for _, dg in diagrams_data if dg[d].size]
            self.imagers_.append(_fit_imager(pooled, self.pixel_size))
        res = [tuple(im.resolution) for im in self.imagers_]
        print(
            f"[ph] fit {self.maxdim + 1} imager(s) (pixel_size={self.pixel_size}); "
            f"per-dim resolution {res} -> {self.n_features_} features"
        )

    def _vectorize(self, diagrams_data):
        t0 = time.perf_counter()
        rows = []
        for n_atoms, dg in diagrams_data:
            v = np.concatenate(
                [_image_or_zeros(self.imagers_[d], dg[d]) for d in range(self.maxdim + 1)]
            )
            if self.normalize and n_atoms > 0:
                v = v / n_atoms
            rows.append(v)
        X = np.vstack(rows)
        print(f"[ph] vectorized {X.shape} in {time.perf_counter() - t0:.1f}s")
        return X

    @property
    def n_features_(self):
        return int(sum(np.prod(im.resolution) for im in self.imagers_))

    def fit(self, atoms_list):
        self._fit_imagers(self._diagrams_for(atoms_list))
        return self

    def fit_transform(self, atoms_list):
        dd = self._diagrams_for(atoms_list)  # single walk, reused below
        self._fit_imagers(dd)
        return self._vectorize(dd)

    def transform(self, atoms_list):
        return self._vectorize(self._diagrams_for(atoms_list))


# ============================================================================
# Element-specific persistent homology (ESPH)
# ============================================================================
# The all-atom PersistenceFeaturizer treats every atom as an identical point, so
# its topology is chemistry-blind (a carbon ring and a nitrogen ring look the
# same). ESPH instead runs a SEPARATE Rips filtration on each chemically-defined
# atom subset -- one per single element and, optionally, per element pair -- so
# each channel's topology is tagged with which atoms produced it (the carbon
# skeleton, the oxygen arrangement, the N/O inter-element geometry, ...). This
# reinjects chemical identity the all-atom cloud discards and is the standard way
# TDA descriptors are made competitive on molecular data (Cang & Wei; the
# multiparameter charge variant is the natural follow-on).
#
# A "channel" is a tuple of atomic numbers: (6,) = carbon-only cloud, (7, 8) =
# the combined N+O cloud. The element set and pair list are FROZEN at fit time
# (like a vocabulary) so the output width is fixed across molecules; each
# (channel, homology-dim) gets its own population-fit PersistenceImager.


def _channel_key(channel):
    return "-".join(str(z) for z in channel)


@dataclass
class ElementPHFeaturizer:
    """Element-specific persistence-image descriptor (drop-in for
    PersistenceFeaturizer: same fit / transform / fit_transform / n_features_ /
    last_oov_rate_ interface).

    elements: explicit atomic numbers to use as single-element channels; if None,
      the ``top_k`` most common elements (by how many molecules contain them) are
      chosen at fit time.
    pairs: "none" (single-element channels only) or "all" (also add every
      unordered pair among the chosen elements as a combined two-element cloud).
    min_atoms: a channel whose subset has fewer atoms than this contributes an
      empty (all-zero) image for that molecule.
    normalize: divide each molecule's full vector by its TOTAL atom count, so all
      channels stay intensive and on one scale (matches the all-atom featurizer).
    """

    maxdim: int = 1
    thresh: float | None = 6.0
    pixel_size: float = 0.25
    normalize: bool = True
    elements: tuple | None = None
    top_k: int = 6
    pairs: str = "none"  # "none" | "all"
    min_atoms: int = 1
    # frozen state
    channels_: list = field(default=None, repr=False)   # list of atomic-number tuples
    imagers_: dict = field(default=None, repr=False)     # (channel, dim) -> imager
    last_oov_rate_: float = field(default=0.0, repr=False)

    # -- element vocabulary --------------------------------------------------
    def _fit_channels(self, atoms_list):
        if self.elements is not None:
            elems = sorted(int(z) for z in self.elements)
        else:
            from collections import Counter

            df = Counter()
            for atoms in atoms_list:
                df.update(set(int(z) for z in atoms.get_atomic_numbers()))
            elems = sorted(z for z, _ in df.most_common(self.top_k))
        singles = [(z,) for z in elems]
        pair_ch = []
        if self.pairs == "all":
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    pair_ch.append((elems[i], elems[j]))
        elif self.pairs != "none":
            raise ValueError(f"pairs must be 'none' or 'all', got {self.pairs!r}")
        self.channels_ = singles + pair_ch
        print(f"[esph] {len(elems)} elements {elems} -> {len(self.channels_)} channels "
              f"(pairs={self.pairs})")

    # -- one shared walk: per-channel diagrams for each molecule -------------
    def _diagrams_for(self, atoms_list):
        t0 = time.perf_counter()
        out = []
        for i, atoms in enumerate(atoms_list):
            Z = np.asarray(atoms.get_atomic_numbers())
            P = np.asarray(atoms.get_positions(), dtype=float)
            per_channel = {}
            for ch in self.channels_:
                pts = P[np.isin(Z, ch)]
                if len(pts) >= self.min_atoms:
                    per_channel[ch] = rips_diagrams(pts, self.maxdim, self.thresh)
                else:
                    per_channel[ch] = [np.empty((0, 2)) for _ in range(self.maxdim + 1)]
            out.append((len(atoms), per_channel))
            if (i + 1) % 2000 == 0:
                print(f"[esph]   ...diagrams {i + 1}/{len(atoms_list)}")
        print(f"[esph] computed {len(self.channels_)}-channel diagrams for "
              f"{len(atoms_list)} mols in {time.perf_counter() - t0:.1f}s")
        return out

    def _fit_imagers(self, diagrams_data):
        self.imagers_ = {}
        for ch in self.channels_:
            for d in range(self.maxdim + 1):
                pooled = [pc[ch][d] for _, pc in diagrams_data if pc[ch][d].size]
                self.imagers_[(ch, d)] = _fit_imager(pooled, self.pixel_size)
        print(f"[esph] fit {len(self.imagers_)} imagers "
              f"(pixel_size={self.pixel_size}) -> {self.n_features_} features")

    def _vectorize(self, diagrams_data):
        t0 = time.perf_counter()
        rows = []
        for n_atoms, pc in diagrams_data:
            parts = []
            for ch in self.channels_:
                for d in range(self.maxdim + 1):
                    parts.append(_image_or_zeros(self.imagers_[(ch, d)], pc[ch][d]))
            v = np.concatenate(parts)
            if self.normalize and n_atoms > 0:
                v = v / n_atoms
            rows.append(v)
        X = np.vstack(rows)
        print(f"[esph] vectorized {X.shape} in {time.perf_counter() - t0:.1f}s")
        return X

    @property
    def n_features_(self):
        return int(sum(np.prod(im.resolution) for im in self.imagers_.values()))

    def channel_slices(self):
        """Map each channel to its (start, stop) column span in the output vector,
        so a later additive kernel can give each channel its own sub-embedding."""
        slices, off = {}, 0
        for ch in self.channels_:
            w = int(sum(np.prod(self.imagers_[(ch, d)].resolution)
                        for d in range(self.maxdim + 1)))
            slices[_channel_key(ch)] = (off, off + w)
            off += w
        return slices

    def fit(self, atoms_list):
        self._fit_channels(atoms_list)
        self._fit_imagers(self._diagrams_for(atoms_list))
        return self

    def fit_transform(self, atoms_list):
        self._fit_channels(atoms_list)
        dd = self._diagrams_for(atoms_list)  # single walk, reused below
        self._fit_imagers(dd)
        return self._vectorize(dd)

    def transform(self, atoms_list):
        if self.channels_ is None:
            raise RuntimeError("call fit() before transform().")
        return self._vectorize(self._diagrams_for(atoms_list))
