"""
explore_omol25.py
=================
First-day exploration of the OMol25 (neutral) validation set, plus the
k-nearest-neighbour distance diagnostic from the gp2Scale-OMol25 position
paper (Section 3 / Remark 1).

What it does
------------
1. Loads an OMol25 *.aselmdb split via fairchem's AseDBDataset.
2. Prints the schema of a single structure (energy, forces, charge, spin,
   elements) so you can see exactly what the data gives you.
3. Computes summary statistics over a random subsample:
     - system sizes (number of atoms)
     - element frequencies
     - total-energy distribution (raw, in eV)
     - charge / spin distribution
4. Runs the k-NN distance diagnostic using a fast, rotation/translation
   invariant first-pass descriptor (sorted Coulomb-matrix eigenspectrum).
   A *bimodal* histogram suggests sparsity will emerge; a *unimodal,
   concentrated* one warns it may not.

Requirements
------------
    pip install fairchem-core ase numpy scipy scikit-learn matplotlib

Usage
-----
    python explore_omol25.py --src /path/to/omol25/val_neutral --n 5000

Notes
-----
- The Coulomb eigenspectrum here is a *placeholder* descriptor for a quick
  go/no-go read. The principled metrics to try next (per the position paper)
  are Wasserstein-on-pairwise-distance-profiles and Weisfeiler-Lehman graph
  distance. Swap `compute_descriptor` to compare them under the same diagnostic.
- We z-score the descriptor features before computing distances so the
  largest eigenvalues don't dominate. Toggle with --no-standardize to see the
  raw-scale behaviour (which is itself informative about concentration).
"""

import argparse
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols  # noqa: F401  (available if needed downstream)

# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_dataset(src):
    """Open an OMol25 aselmdb split. `src` is the directory of .aselmdb files."""
    from fairchem.core.datasets import AseDBDataset

    dataset = AseDBDataset({"src": src})
    print(f"Loaded dataset from {src!r}: {len(dataset):,} structures.")
    return dataset


def inspect_one(dataset, idx=0):
    """Print the full schema of a single structure so you know what's available."""
    atoms = dataset.get_atoms(idx)
    print(f"\n--- single structure schema (idx {idx}) ---")
    print("formula           :", atoms.get_chemical_formula())
    print("n_atoms           :", len(atoms))
    print("elements          :", sorted(set(atoms.get_chemical_symbols())))
    try:
        print("total energy (eV) :", atoms.get_potential_energy())
    except Exception as e:
        print("total energy      : <unavailable>", e)
    try:
        f = atoms.get_forces()
        print("forces shape      :", f.shape, "| max |F| (eV/A):", np.abs(f).max())
    except Exception as e:
        print("forces            : <unavailable>", e)
    # charge & spin live in atoms.info, not as standard ASE properties
    print("atoms.info keys   :", list(atoms.info.keys()))
    print("charge            :", atoms.info.get("charge", "<not found>"))
    print(
        "spin / multiplicity:",
        atoms.info.get("spin", atoms.info.get("spin_multiplicity", "<not found>")),
    )
    return atoms


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------


def summarize(dataset, indices):
    sizes, energies, charges, spins = [], [], [], []
    elem_counter = Counter()

    for idx in indices:
        atoms = dataset.get_atoms(int(idx))
        sizes.append(len(atoms))
        elem_counter.update(atoms.get_chemical_symbols())
        try:
            energies.append(atoms.get_potential_energy())
        except Exception:
            pass
        charges.append(atoms.info.get("charge", np.nan))
        spins.append(
            atoms.info.get("spin", atoms.info.get("spin_multiplicity", np.nan))
        )

    sizes = np.array(sizes)
    energies = np.array(energies, dtype=float)

    print(f"\n--- subsample summary (n = {len(indices)}) ---")
    print(
        f"system size  : min {sizes.min()}, median {int(np.median(sizes))}, "
        f"max {sizes.max()}, mean {sizes.mean():.1f}"
    )
    if energies.size:
        print(
            f"total energy : range [{energies.min():.1f}, {energies.max():.1f}] eV "
            f"(spread {np.ptp(energies):.1f} eV)  <- note the extensivity problem"
        )
        print(
            f"             : per-atom range [{(energies/sizes[:len(energies)]).min():.2f}, "
            f"{(energies/sizes[:len(energies)]).max():.2f}] eV/atom"
        )
    top = elem_counter.most_common(15)
    print("top elements :", ", ".join(f"{el}:{n}" for el, n in top))
    print("n distinct elements seen:", len(elem_counter))

    # Ensure output directories exist
    os.makedirs("./graphs/png", exist_ok=True)
    os.makedirs("./graphs/svg", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(sizes, bins=40, color="#4c72b0")
    ax[0].set(title="System size (atoms)", xlabel="n_atoms", ylabel="count")
    if energies.size:
        ax[1].hist(energies, bins=40, color="#c44e52")
        ax[1].set(title="Total energy (raw, eV)", xlabel="E (eV)")
    syms = [s for s, _ in top]
    ax[2].bar(syms, [elem_counter[s] for s in syms], color="#55a868")
    ax[2].set(title="Top-15 element frequency", xlabel="element")
    fig.tight_layout()
    fig.savefig(f"./graphs/png/omol25_summary_{timestamp}.png", dpi=130)
    fig.savefig(f"./graphs/svg/omol25_summary_{timestamp}.svg", dpi=130)
    print(f"saved -> ./graphs/png/omol25_summary_{timestamp}.png")


# ----------------------------------------------------------------------
# Descriptor + k-NN diagnostic
# ----------------------------------------------------------------------


def coulomb_eigenspectrum(atoms):
    """Sorted Coulomb-matrix eigenvalues: rotation/translation invariant."""
    Z = atoms.get_atomic_numbers().astype(float)
    R = atoms.get_positions()
    n = len(Z)

    diff = R[:, None, :] - R[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.outer(Z, Z) / dist

    # Diagonal: 0.5 * Z_i^2.4  (self-interaction term)
    C[np.diag_indices(n)] = 0.5 * Z**2.4

    eig = np.linalg.eigvalsh(C)  # ascending
    return np.sort(eig)[::-1]  # descending


def build_descriptor_matrix(dataset, indices):
    """Coulomb eigenspectra, zero-padded to the max system size in the sample."""
    spectra = [coulomb_eigenspectrum(dataset.get_atoms(int(i))) for i in indices]
    dim = max(len(s) for s in spectra)
    X = np.zeros((len(spectra), dim))
    for i, s in enumerate(spectra):
        X[i, : len(s)] = s
    return X


def knn_diagnostic(X, k=5, standardize=True):
    """Histogram of k-NN distances. Bimodal => sparsity likely; unimodal => not."""
    from sklearn.neighbors import NearestNeighbors

    if standardize:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1 because self is nearest
    d, _ = nn.kneighbors(X)
    knn_dists = d[:, 1:].ravel()  # drop the self-distance column

    os.makedirs("./graphs/png", exist_ok=True)
    os.makedirs("./graphs/svg", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n--- k-NN distance diagnostic ---")
    print(f"k = {k}, standardized = {standardize}")
    print(f"distance range : [{knn_dists.min():.3f}, {knn_dists.max():.3f}]")
    print(f"mean / median  : {knn_dists.mean():.3f} / {np.median(knn_dists):.3f}")
    cv = knn_dists.std() / knn_dists.mean()
    print(
        f"coeff. of variation (std/mean): {cv:.3f}  "
        f"(<~0.3 hints at concentration / dense covariance risk)"
    )

    plt.figure(figsize=(7, 4))
    plt.hist(knn_dists, bins=80, color="#8172b3")
    plt.title(
        "k-NN distance distribution (Coulomb eigenspectrum)\n"
        "bimodal -> sparsity likely | unimodal+concentrated -> dense risk"
    )
    plt.xlabel("distance to k nearest neighbours")
    plt.ylabel("count")
    plt.yscale("log")
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig(f"./graphs/png/omol25_knn_diagnostic_{timestamp}.png", dpi=130)
    plt.savefig(f"./graphs/svg/omol25_knn_diagnostic_{timestamp}.svg", dpi=130)
    print(f"saved -> ./graphs/svg/omol25_knn_diagnostic_{timestamp}.png")


# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir containing the .aselmdb split")
    p.add_argument("--n", type=int, default=5000, help="subsample size")
    p.add_argument("--k", type=int, default=5, help="neighbours for diagnostic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-standardize", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.src)
    inspect_one(dataset, 0)
    n = min(args.n, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)
    summarize(dataset, indices)
    X = build_descriptor_matrix(dataset, indices)
    knn_diagnostic(X, k=args.k, standardize=not args.no_standardize)


if __name__ == "__main__":
    main()
