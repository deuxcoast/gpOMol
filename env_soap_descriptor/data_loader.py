"""
data_loader.py
=============
Wires train_4M (ASE-DB aselmdb, read via fairchem AseDBDataset) into the
make_population(n_mol, seed) -> (X_raw, mol_of, y_resid) contract the diagnostic wants.

Three things this gets right for train_4M specifically:
  1. COMPRESSED SOAP. 80 elements -> uncompressed F=577,200 (unusable). compression=
     "mu1nu1" makes F scale linearly (not quadratically) in the element count while
     keeping per-element resolution; "mu2"+species_weighting is the cheaper fallback.
     Only the low-D PCA embedding is ever stored, so the full F exists only transiently.
  2. REFERENCE ON COMPOSITION + CHARGE + SPIN. Raw energy std ~1.5e5 eV collapses to
     ~9 eV after referencing; charge/spin are referenced too because tm_react is charged
     and open-shell. The GP only sees the residual -- the kernel is credited with nothing
     the reference removed. The reference is fit ONCE and applied consistently across N.
  3. NESTED PREFIXES. N grows a single population (perm[:N]) rather than independent
     draws, so nnz/row-vs-N measures density growth, not sampling noise.

Multi-molecule records (solvated_protein, ml_elytes) need no special handling here: SOAP
is per-atom, and y = sum over all atoms in the record is the correct extensive target.
"""

from __future__ import annotations

import os

import numpy as np

# Pauling electronegativity (for mu2 species_weighting fallback); default 1.5 if missing.
_EN = {
    1: 2.20,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    20: 1.00,
    26: 1.83,
    29: 1.90,
    30: 1.65,
    35: 2.96,
    53: 2.66,
}


class OMol25Loader:
    def __init__(
        self,
        src="../train_4M/",
        species=None,
        embed_D=12,
        soap_kwargs=None,
        compression="mu1nu1",
        n_ref_sample=50_000,
        n_embed_mols=2000,
        cache_dir=None,
        seed=0,
    ):
        from fairchem.core.datasets import AseDBDataset

        self.ds = AseDBDataset({"src": src})
        self.N = len(self.ds)
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.N)  # fixed order -> nested prefixes
        self.embed_D = embed_D
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # element set: use provided (from inspection: 80 elements) or scan a sample
        self.species = species or self._scan_species(20_000)
        self._build_soap(compression, soap_kwargs or {})
        self._fit_reference(n_ref_sample)
        self._fit_embedding(n_embed_mols)  # PCA fit on a SOAP sample

    # -------- species + SOAP object -------- #
    def _scan_species(self, n):
        s = set()
        for i in self.perm[:n]:
            s.update(int(z) for z in self.ds.get_atoms(int(i)).get_atomic_numbers())
        return sorted(s)

    def _build_soap(self, compression, soap_kwargs):
        from dscribe.descriptors import SOAP

        kw = dict(r_cut=5.0, n_max=6, l_max=4, periodic=False, sparse=False)
        kw.update(soap_kwargs)
        comp = {"mode": compression}
        if compression == "mu2":
            comp["species_weighting"] = {z: _EN.get(z, 1.5) for z in self.species}
        self.soap = SOAP(species=self.species, compression=comp, **kw)
        self.F = self.soap.get_number_of_features()

    # -------- reference model E ~ composition + charge + spin -------- #
    def _fit_reference(self, n):
        idx = self.perm[: min(n, self.N)]
        col = {z: j for j, z in enumerate(self.species)}
        rows, E = [], []
        for i in idx:
            a = self.ds.get_atoms(int(i))
            v = np.zeros(len(self.species) + 2)
            for z in a.get_atomic_numbers():
                v[col[int(z)]] += 1
            v[-2] = a.info.get("charge", 0)
            v[-1] = a.info.get("spin", 1)
            rows.append(v)
            try:
                E.append(float(a.get_potential_energy()))
            except Exception:
                E.append(float(a.info.get("energy", np.nan)))
        C, E = np.array(rows), np.array(E)
        ok = np.isfinite(E)
        self.ref_coef, *_ = np.linalg.lstsq(C[ok], E[ok], rcond=None)
        self._col = col
        resid = E[ok] - C[ok] @ self.ref_coef
        self.resid_std = float(resid.std())  # sets GP signal scale / sigma^2

    def _reference_energy(self, a):
        v = np.zeros(len(self.species) + 2)
        for z in a.get_atomic_numbers():
            v[self._col[int(z)]] += 1
        v[-2] = a.info.get("charge", 0)
        v[-1] = a.info.get("spin", 1)
        return v @ self.ref_coef

    # -------- embedding: PCA fit on a SOAP SAMPLE (full F held only for the sample) -------- #
    def _fit_embedding(self, n_mols):
        from env_features_kernel import fit_env_embedding

        blocks = [
            self.soap.create(self.ds.get_atoms(int(i))) for i in self.perm[:n_mols]
        ]
        X_sample = np.vstack(blocks)  # (sample_atoms, F)
        self.embedding = fit_env_embedding(X_sample, D=self.embed_D)

    # -------- the population builder -------- #
    def make_population(self, n_mol, seed=0, keep_charges=False):
        """
        Returns (Z, mol_of, y_resid) for the FIRST n_mol molecules of the fixed
        permutation (nested across calls). Z is the D-dim EMBEDDING (SOAP projected on the
        fly, so full-F is never materialized past one molecule). Caches per-N arrays.
        """
        tag = (
            os.path.join(self.cache_dir, f"pop_{n_mol}.npz") if self.cache_dir else None
        )
        if tag and os.path.exists(tag):
            d = np.load(tag)
            return d["Z"], d["mol_of"], d["y"]

        idx = self.perm[:n_mol]
        Z_blocks, mol_of, y, charges = [], [], np.empty(n_mol), []
        for m, i in enumerate(idx):
            a = self.ds.get_atoms(int(i))
            Z_blocks.append(self.embedding.transform(self.soap.create(a)))  # (k, D)
            k = len(a)
            mol_of.extend([m] * k)
            y[m] = float(a.get_potential_energy()) - self._reference_energy(a)
            if keep_charges:
                charges.append(a.info.get("lowdin_charges", np.zeros(k)))
        Z = np.vstack(Z_blocks)
        mol_of = np.asarray(mol_of)
        if tag:
            np.savez(tag, Z=Z, mol_of=mol_of, y=y)
        if keep_charges:
            return Z, mol_of, y, np.concatenate(charges)
        return Z, mol_of, y


if __name__ == "__main__":
    # mock the fairchem/dscribe pieces so the pure-python logic (reference fit, nested
    # prefixes, membership) can be smoke-tested WITHOUT the real deps or data.
    import sys
    import types

    class _MockAtoms:
        def __init__(s, Z, E, charge=0, spin=1):
            s._Z = np.array(Z)
            s._E = E
            s.info = {
                "charge": charge,
                "spin": spin,
                "lowdin_charges": np.zeros(len(Z)),
            }

        def get_atomic_numbers(s):
            return s._Z

        def get_potential_energy(s):
            return s._E

        def __len__(s):
            return len(s._Z)

    rng = np.random.default_rng(0)
    e_true = {1: -13.0, 6: -1030.0, 8: -2040.0}
    mols = []
    for _ in range(500):
        k = rng.integers(4, 40)
        Z = rng.choice([1, 6, 8], size=k)
        E = sum(e_true[int(z)] for z in Z) + rng.normal(0, 2.0)  # extensive + residual
        mols.append(_MockAtoms(Z, E))

    class _MockDS:
        def __init__(s, cfg):
            pass

        def __len__(s):
            return len(mols)

        def get_atoms(s, i):
            return mols[i]

    fc = types.ModuleType("fairchem")
    fcc = types.ModuleType("fairchem.core")
    fcd = types.ModuleType("fairchem.core.datasets")
    fcd.AseDBDataset = _MockDS
    sys.modules.update(
        {"fairchem": fc, "fairchem.core": fcc, "fairchem.core.datasets": fcd}
    )

    # bypass SOAP for the mock (feature step is what needs dscribe)
    L = OMol25Loader.__new__(OMol25Loader)
    L.ds = _MockDS({})
    L.N = len(mols)
    L.rng = rng
    L.perm = rng.permutation(L.N)
    L.species = [1, 6, 8]
    L.cache_dir = None
    L._fit_reference(400)
    print(
        f"reference residual std = {L.resid_std:.3f} eV (truth: injected noise std=2.0)"
    )
    print(
        f"recovered e(Z): H={L.ref_coef[0]:.1f} C={L.ref_coef[1]:.1f} O={L.ref_coef[2]:.1f} "
        f"(truth -13/-1030/-2040)"
    )
    # nested-prefix check
    p1, p2 = L.perm[:100], L.perm[:300]
    print("nested prefixes:", bool(np.array_equal(p1, p2[:100])))
