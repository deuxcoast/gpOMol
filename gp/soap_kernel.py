r"""
soap_kernel.py
==============
Global (molecule-averaged) SOAP kernel as a drop-in `Candidate` for the harness,
to test whether a geometry-carrying descriptor breaks WL's ~2 eV CRPS ceiling.

Why global SOAP
---------------
SOAP encodes each atom's LOCAL 3D environment (a smooth expansion of the
neighbor density in radial x spherical-harmonic basis within a cutoff). WL sees
only connectivity; SOAP sees geometry -- bond lengths, angles, local packing --
which is exactly the signal WL structurally cannot reach.

We AVERAGE the per-atom SOAP power spectra into one vector per molecule
(dscribe average='inner'). Averaging (not summing) makes the molecule descriptor
INTENSIVE, which matches the intensive residual left after the mean function
absorbs the extensive trend -- so it sidesteps the extensivity-vs-stationarity
vise the same way, without a normalize-or-die step.

Kernel
------
Standard SOAP average kernel: cosine-normalize the averaged power spectra, then
k(M,M') = ( p_hat(M) . p_hat(M') )^zeta. PD for integer zeta >= 1. We build this
kernel directly (normalize_kernel=False on the Candidate, since normalization is
done here). The harness converts it to an induced Hilbert distance for the
variogram / support sweep, and can also use it directly as a feature kernel.

Honest caveats
--------------
  * SOAP needs a cutoff r_cut (the property the position paper dislikes). For the
    organic slice ~5-6 A is defensible; it is a parameter here and sweepable.
  * Averaged LOCAL SOAP captures local geometry well but only partially captures
    truly GLOBAL molecular shape (anything larger than r_cut). That is a real
    limitation of the global-average pooling, separate from the cutoff.
  * Dimensionality grows ~ n_species^2 * n_max^2 * (l_max+1); for 5 organic
    species this is a few thousand. High dimension raises distance concentration
    (position paper Sec 7.1), which can HURT the sparsity axis even where it
    helps the predictive axis. Judge prediction and sparsity separately.

Requires: pip install dscribe ase

Run `python soap_kernel.py` for a self-test on a LOCAL-geometry target (energy
set by local neighbor distances) -- SOAP should capture it and connectivity-only
WL should not.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from variogram_screen import Candidate
from wl_kernel import extract_atoms


def _averaged_soap(feats, rcut, nmax, lmax, sigma, species=None, n_jobs=1):
    """One averaged SOAP power-spectrum vector per molecule -> (n_mol, dim)."""
    from ase import Atoms
    from dscribe.descriptors import SOAP

    atoms_list, all_Z = [], set()
    for f in feats:
        numbers, positions = extract_atoms(f)
        atoms_list.append(
            Atoms(
                numbers=list(map(int, numbers)), positions=np.asarray(positions, float)
            )
        )
        all_Z.update(int(z) for z in numbers)
    if species is None:
        species = sorted(all_Z)
    soap = SOAP(
        species=species,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sigma=sigma,
        average="inner",
        periodic=False,
    )
    P = soap.create(atoms_list, n_jobs=n_jobs)
    return np.asarray(P, dtype=float)


def soap_gram(
    feats, rcut=5.0, nmax=6, lmax=4, sigma=0.5, zeta=1, species=None, n_jobs=1
):
    """SOAP average kernel Gram matrix K(M,M') = (p_hat . p_hat)^zeta."""
    P = _averaged_soap(feats, rcut, nmax, lmax, sigma, species=species, n_jobs=n_jobs)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Ph = P / norms  # cosine-normalized
    K = Ph @ Ph.T
    if zeta != 1:
        K = np.clip(K, -1.0, 1.0) ** zeta  # polynomial SOAP kernel (PD)
    return K


def make_soap_candidate(
    name="soap",
    rcut=5.0,
    nmax=6,
    lmax=4,
    sigma=0.5,
    zeta=1,
    species=None,
    n_jobs=1,
    normalize=False,
):
    """Harness Candidate for global-average SOAP. kind='kernel'; the Gram is
    already normalized, so normalize=False by default (the harness won't double-
    normalize)."""

    def fn(fs):
        return soap_gram(
            fs,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            zeta=zeta,
            species=species,
            n_jobs=n_jobs,
        )

    return Candidate(name=name, fn=fn, kind="kernel", normalize_kernel=normalize)


# --------------------------------------------------------------------------- #
# Self-test: a LOCAL-geometry target SOAP should capture and WL should miss
# --------------------------------------------------------------------------- #
def _demo():
    from collections import Counter

    from variogram_screen import empirical_variogram, gram_loo_crps
    from wl_kernel import _grow_molecule, build_adjacency, make_wl_candidate

    rng = np.random.default_rng(5)
    N = 250
    atom_E = {1: -13.6, 6: -1030.0, 7: -1480.0, 8: -2040.0}
    sym = {1: "H", 6: "C", 7: "N", 8: "O"}
    RCUT = 5.0

    feats, energies, symbols = [], [], []
    for _ in range(N):
        na = int(rng.integers(8, 20))
        Z, pos = _grow_molecule(rng, na)
        adj = build_adjacency(Z, pos, 1.2)
        # LOCAL-geometry signal: mean spread of each atom's neighbor distances
        # (a purely geometric, intensive quantity WL cannot see but SOAP can).
        spreads = []
        for i in range(len(Z)):
            if adj[i]:
                d = np.linalg.norm(pos[list(adj[i])] - pos[i], axis=1)
                spreads.append(float(np.std(d)) if len(d) > 1 else 0.0)
        geom = float(np.mean(spreads)) if spreads else 0.0
        e = sum(atom_E[int(z)] for z in Z) + 20.0 * geom + 0.1 * rng.normal()
        feats.append((Z, pos))
        energies.append(e)
        symbols.append([sym[int(z)] for z in Z])

    energies = np.array(energies)
    # element-reference to expose the intensive geometric residual
    elems = sorted({s for ss in symbols for s in ss})
    idx = {e: i for i, e in enumerate(elems)}
    A = np.zeros((N, len(elems)))
    for m, ss in enumerate(symbols):
        for s in ss:
            A[m, idx[s]] += 1
    coef, *_ = np.linalg.lstsq(A, energies, rcond=None)
    y = energies - A @ coef

    print(f"Self-test: {N} molecules; target residual = local neighbor-distance")
    print("spread (pure geometry). SOAP should capture it; WL should be near-blind.\n")

    soap = make_soap_candidate("soap", rcut=RCUT, nmax=6, lmax=4)
    wl = make_wl_candidate("wl_h1", h=1, scale=1.2, normalize=False)

    for cand in (soap, wl):
        K = cand.fn(feats)
        D = (
            np.sqrt(np.clip(np.diag(K)[:, None] + np.diag(K)[None, :] - 2 * K, 0, None))
            if cand.kind == "kernel"
            else K
        )
        vg = empirical_variogram(D, y, n_bins=20)
        feat = gram_loo_crps(K, y)
        print(
            f"  {cand.name:8s}  variogram struct_frac={vg.structured_fraction:.3f}   "
            f"feature-kernel  RMSE={feat['rmse']:.4f}  CRPS={feat['crps']:.4f}"
        )

    print("\nExpect SOAP: higher structured_fraction and lower CRPS than WL, since")
    print("the target is geometric and WL sees only connectivity.")


if __name__ == "__main__":
    _demo()
