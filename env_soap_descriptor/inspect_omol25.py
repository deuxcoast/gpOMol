"""
inspect_omol25.py
=================
Run ONCE against your OMol25 aselmdb subset. Prints everything needed to finish
make_population() and set pipeline defaults. Paste the output back and I'll wire the
loader, choose the neighbor backend, and set sigma^2 / reference energies.

Usage:
    edit SRC below, then:  python inspect_omol25.py
Only needs fairchem + ase to read; dscribe is optional (used to report SOAP dimension).
"""

import os

import numpy as np

SRC = "../train_4M/"
N_SAMPLE = 3000  # structures to sample for the stats
SEED = 0

# SOAP candidate hypers (only used to report the resulting feature dimension F)
R_CUT, N_MAX, L_MAX = 5.0, 6, 4


def main():
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": SRC})
    n_total = len(ds)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)

    natoms, energies, species_counter = [], [], {}
    comp_rows, all_Z = [], set()
    charge_arrays_seen, info_keys_seen = set(), set()
    example_info = None

    atoms_list = []
    for i in idxs:
        a = ds.get_atoms(int(i))
        Z = a.get_atomic_numbers()
        natoms.append(len(Z))
        all_Z.update(int(z) for z in Z)
        for z in Z:
            species_counter[int(z)] = species_counter.get(int(z), 0) + 1
        # energy: prefer the DFT potential energy in eV
        try:
            e = float(a.get_potential_energy())
        except Exception:
            e = float(a.info.get("energy", np.nan))
        energies.append(e)
        # remember composition for per-element referencing
        comp_rows.append({int(z): int((Z == z).sum()) for z in np.unique(Z)})
        # what per-atom fields exist (charges live here if shipped)
        charge_arrays_seen.update(a.arrays.keys())
        info_keys_seen.update(a.info.keys())
        if example_info is None:
            example_info = dict(a.info)
        if len(atoms_list) < 200:
            atoms_list.append(a)

    natoms = np.array(natoms)
    energies = np.array(energies, float)
    elements = sorted(all_Z)
    # per-element reference energies E ~ sum_Z n_Z * e(Z)  via least squares
    C = np.zeros((len(comp_rows), len(elements)))
    col = {z: j for j, z in enumerate(elements)}
    for r, comp in enumerate(comp_rows):
        for z, c in comp.items():
            C[r, col[z]] = c
    finite = np.isfinite(energies)
    e_ref, *_ = np.linalg.lstsq(C[finite], energies[finite], rcond=None)
    resid = energies[finite] - C[finite] @ e_ref

    print("=" * 70)
    print(f"SOURCE                 {SRC}")
    print(f"total structures       {n_total:,}")
    print(f"sampled                {len(idxs):,}")
    print("-" * 70)
    print("ATOMS PER MOLECULE (sets n_star -> UQ cancellation ~ n_star^2):")
    print(
        f"  min/median/mean/p95/max = {natoms.min()}/{int(np.median(natoms))}/"
        f"{natoms.mean():.1f}/{int(np.percentile(natoms,95))}/{natoms.max()}"
    )
    for tgt in (1e5, 3e5, 1e6):
        print(f"  n_env at {int(tgt):>9,} molecules ~= {int(tgt*natoms.mean()):,}")
    print("-" * 70)
    print(f"ELEMENTS ({len(elements)} distinct) -> SOAP species list:")
    print(f"  {elements}")
    top = sorted(species_counter.items(), key=lambda kv: -kv[1])[:12]
    print(f"  most common Z (by atom count): {[z for z,_ in top]}")
    print("-" * 70)
    print("ENERGY (eV):")
    print(
        f"  raw total        std = {energies[finite].std():.3f}  "
        f"range = [{energies[finite].min():.1f}, {energies[finite].max():.1f}]"
    )
    print(
        f"  after per-element referencing: residual std = {resid.std():.4f} eV  "
        f"(this sets the GP signal scale)"
    )
    print(f"  residual per atom std = {(resid/natoms[finite]).std():.4f} eV/atom")
    print(
        f"  -> suggested noise sigma^2 (start): {max((resid.std()*0.05)**2, 1e-6):.2e}"
    )
    print("-" * 70)
    print("PER-ATOM FIELDS present in atoms.arrays (charges would appear here):")
    print(f"  {sorted(charge_arrays_seen)}")
    print(f"atoms.info keys: {sorted(info_keys_seen)}")
    print(f"example atoms.info: {example_info}")
    print("-" * 70)
    # SOAP dimension (optional)
    try:
        from dscribe.descriptors import SOAP

        soap = SOAP(
            species=elements,
            r_cut=R_CUT,
            n_max=N_MAX,
            l_max=L_MAX,
            periodic=False,
            sparse=False,
        )
        F = soap.get_number_of_features()
        print(
            f"SOAP feature dim F (species={len(elements)}, n_max={N_MAX}, l_max={L_MAX}, "
            f"r_cut={R_CUT}) = {F:,}"
        )
        print(f"  -> PCA target D: keep well below F; suggest D in [10, 20]")
    except Exception as ex:
        print(
            f"dscribe not available here ({ex}); F not computed. Install on the run node."
        )
    print("=" * 70)
    print(
        "Paste this whole block back. I need especially: atoms/mol median+p95,"
        " #elements, residual std, and the atoms.arrays field list."
    )


if __name__ == "__main__":
    main()
