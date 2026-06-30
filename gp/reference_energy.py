r"""
reference_energy.py
===================
Per-element energy referencing for OMol25, as a standalone preprocessing step
that runs BEFORE the GP.

Why
---
Total DFT energy is *extensive*: it scales with system size and composition, so
across OMol25 it spans millions of eV dominated by WHICH atoms are present, not
by the geometry/bonding signal we actually want to learn. We model that boring
composition-linear part explicitly and let the GP learn only the residual:

    E_total  ~=  sum_e n_e * eps_e   (+ charge term)   +   E_residual
                 \-------- baseline we fit ---------/       \-- GP target --/

The eps_e are LEARNED parameters, so they are fit on TRAINING DATA ONLY and then
applied (frozen) to both train and test. Fitting on test would leak the target.

What it outputs
---------------
  * Per-element reference energies (and a per-charge coefficient).
  * Referenced residual targets for train and test, keyed by dataset index, so
    the downstream gpCAM run can load them and align with its own descriptor.
  * A before/after spread printout and a histogram, as a sanity check.

Requirements
------------
    pip install ase numpy matplotlib fairchem-core

Usage
-----
    python reference_energy.py --src ../train_4M --n-molecules 3000 \
        --elements organic --size 20 60 --test-frac 0.2 --add-charge

Then in the GP step, load ./referencing/<timestamp>_reference.npz and use
r_train / r_test (aligned to train_idx / test_idx) as the regression target.
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(src):
    """Open an OMol25 aselmdb split. `src` is the directory of .aselmdb files."""
    from fairchem.core.datasets import AseDBDataset

    dataset = AseDBDataset({"src": src})
    print(f"Loaded dataset from {src!r}: {len(dataset):,} structures.")
    return dataset


# ----------------------------------------------------------------------
# Element-set filter (kept consistent with the Wasserstein diagnostic)
# ----------------------------------------------------------------------
def element_filter(mode):
    organic = {"H", "C", "N", "O", "S"}
    organic_ext = organic | {"P", "F", "Cl", "Br", "I"}
    return {
        "organic": lambda s: s <= organic,
        "organic_ext": lambda s: s <= organic_ext,
        "all": lambda s: True,
        "nonorganic": lambda s: not (s <= organic),
    }[mode]


# ----------------------------------------------------------------------
# Collect composition, energy, charge in a single disk pass
# ----------------------------------------------------------------------
def collect_subset(dataset, n_molecules, pool, size_range, mode, seed):
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(dataset), size=min(pool, len(dataset)), replace=False)
    keep = element_filter(mode)
    lo, hi = size_range

    indices, comps, energies, charges = [], [], [], []
    for checked, idx in enumerate(pool_idx, 1):
        if checked % 1000 == 0:
            print(
                f"  scanned {checked}/{len(pool_idx)}, kept {len(indices)}", flush=True
            )
        atoms = dataset.get_atoms(int(idx))
        syms = set(atoms.get_chemical_symbols())
        if not (lo <= len(atoms) <= hi and keep(syms)):
            continue
        try:
            e = atoms.get_potential_energy()
        except Exception:
            continue  # skip structures without an energy label
        indices.append(int(idx))
        comps.append(Counter(atoms.get_chemical_symbols()))
        energies.append(e)
        charges.append(float(atoms.info.get("charge", 0)))
        if len(indices) >= n_molecules:
            break

    print(f"kept {len(indices)} molecules ({mode}) in [{lo},{hi}] atoms")
    return (
        np.array(indices),
        comps,
        np.array(energies, dtype=float),
        np.array(charges, dtype=float),
    )


# ----------------------------------------------------------------------
# Composition matrix + reference fit/apply
# ----------------------------------------------------------------------
def build_composition_matrix(comps, charges, elements, add_charge, add_intercept):
    """Rows = molecules, columns = [per-element counts, (charge), (intercept)]."""
    idx = {e: k for k, e in enumerate(elements)}
    C = np.zeros((len(comps), len(elements)))
    for m, comp in enumerate(comps):
        for e, n in comp.items():
            if e not in idx:
                raise KeyError(f"element {e!r} not in reference set {elements}")
            C[m, idx[e]] += n
    labels = list(elements)
    cols = [C]
    if add_charge:
        cols.append(charges.reshape(-1, 1))
        labels.append("__charge__")
    if add_intercept:
        cols.append(np.ones((len(comps), 1)))
        labels.append("__intercept__")
    return np.hstack(cols), labels


def fit_reference(C_train, E_train):
    eps, *_ = np.linalg.lstsq(C_train, E_train, rcond=None)
    return eps


def apply_reference(C, E, eps):
    return E - C @ eps  # residual target


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir containing the .aselmdb split")
    p.add_argument("--n-molecules", type=int, default=3000)
    p.add_argument(
        "--pool",
        type=int,
        default=80000,
        help="random indices to scan to find the subset",
    )
    p.add_argument(
        "--elements",
        choices=["organic", "organic_ext", "all", "nonorganic"],
        default="organic",
    )
    p.add_argument("--size", type=int, nargs=2, default=[20, 60], metavar=("LO", "HI"))
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument(
        "--add-charge",
        action="store_true",
        help="add a per-charge column (recommended for charged species)",
    )
    p.add_argument(
        "--add-intercept", action="store_true", help="add a constant offset column"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dataset = load_dataset(args.src)

    indices, comps, energies, charges = collect_subset(
        dataset, args.n_molecules, args.pool, tuple(args.size), args.elements, args.seed
    )
    if len(indices) < 10:
        raise SystemExit("Too few molecules kept; widen --size/--pool/--elements.")

    # --- train/test split (positions into the kept arrays) ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(indices))
    n_test = max(1, int(round(args.test_frac * len(indices))))
    test_pos, train_pos = perm[:n_test], perm[n_test:]

    comps_tr = [comps[i] for i in train_pos]
    comps_te = [comps[i] for i in test_pos]

    # --- element set is determined by TRAINING data only ---
    elements = sorted({e for c in comps_tr for e in c})
    train_elem_set = set(elements)

    # drop test molecules containing an element never seen in training
    keep_te = [k for k, c in enumerate(comps_te) if set(c) <= train_elem_set]
    dropped = len(comps_te) - len(keep_te)
    if dropped:
        print(f"WARNING: dropped {dropped} test molecules with unseen elements")
    test_pos = test_pos[keep_te]
    comps_te = [comps_te[k] for k in keep_te]

    # --- build matrices with the FIXED element set, fit on TRAIN only ---
    C_tr, labels = build_composition_matrix(
        comps_tr, charges[train_pos], elements, args.add_charge, args.add_intercept
    )
    C_te, _ = build_composition_matrix(
        comps_te, charges[test_pos], elements, args.add_charge, args.add_intercept
    )

    E_tr, E_te = energies[train_pos], energies[test_pos]
    eps = fit_reference(C_tr, E_tr)
    r_tr = apply_reference(C_tr, E_tr, eps)
    r_te = apply_reference(C_te, E_te, eps)

    # --- verify: the spread should collapse by orders of magnitude ---
    print("\n--- referencing summary ---")
    print(f"train / test molecules     : {len(E_tr)} / {len(E_te)}")
    print(f"raw energy spread (train)  : {np.ptp(E_tr):>14,.1f} eV")
    print(f"referenced spread (train)  : {np.ptp(r_tr):>14,.1f} eV")
    print(f"referenced spread (test)   : {np.ptp(r_te):>14,.1f} eV")
    print(f"referenced RMS (train)     : {np.sqrt(np.mean(r_tr**2)):>14,.3f} eV")
    print(f"referenced RMS (test)      : {np.sqrt(np.mean(r_te**2)):>14,.3f} eV")
    print("\nfitted reference coefficients (eV):")
    for lab, val in zip(labels, eps):
        print(f"    {lab:>14} : {val:12.3f}")

    # --- plot raw vs referenced (train) ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./referencing", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(E_tr, bins=50, color="#c44e52")
    ax[0].set(title="Raw total energy (train)", xlabel="E (eV)", ylabel="count")
    ax[1].hist(r_tr, bins=50, color="#55a868")
    ax[1].set(title="Referenced residual (train)", xlabel="E_residual (eV)")
    fig.tight_layout()
    fig.savefig(f"./referencing/{ts}_referencing.png", dpi=130)
    print(f"\nsaved -> ./referencing/{ts}_referencing.png")

    # --- save artifacts for the GP step (targets keyed by dataset index) ---
    out = f"./referencing/{ts}_reference.npz"
    np.savez(
        out,
        eps=eps,
        labels=np.array(labels),
        train_idx=indices[train_pos],
        test_idx=indices[test_pos],
        E_train=E_tr,
        E_test=E_te,
        r_train=r_tr,
        r_test=r_te,
        elements=np.array(elements),
        add_charge=args.add_charge,
        add_intercept=args.add_intercept,
    )
    with open(f"./referencing/{ts}_reference.json", "w") as f:
        json.dump({lab: float(v) for lab, v in zip(labels, eps)}, f, indent=2)
    print(f"saved -> {out}")
    print(f"saved -> ./referencing/{ts}_reference.json")
    print(
        "\nIn the GP step: load the .npz, use r_train/r_test (aligned to "
        "train_idx/test_idx) as the target."
    )


if __name__ == "__main__":
    main()
