"""
data.py  (wl_gp2scale)
======================
Load the 200k OMol25 subset and build the descriptor-independent regression
target, plus the per-molecule ``data_id`` category used for block-sparsity.

Target (identical construction to the validation pipeline, reimplemented here so
this module is self-contained):

    y_i = E_total(i) - m(x_i)
    m(x) = intercept + sum_Z n_Z eps_Z + [charge, |charge|, charge^2, spin, n_atoms^2]

``m`` is a ridge-fit extensive (element-referencing) mean; the GP regresses the
intensive residual y. E_total is recoverable as ``y_pred + m(x)``.

Category. ``atoms.info["data_id"]`` labels the OMol25 subset a molecule came from
(proteins, electrolytes, metal complexes, organics, ...). We map the string ids
to contiguous integers so the kernel can skip cross-category blocks. Confirm the
exact key on your shards; override via ``category_key`` if it differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


# ----------------------------- extensive mean ------------------------------


def extra_feats(Z_list, charge: float, spin: float) -> np.ndarray:
    n = float(len(Z_list))
    return np.array([charge, abs(charge), charge**2, spin, n**2], dtype=float)


@dataclass
class ExtensiveMean:
    ridge: float = 1e-6
    elements_: np.ndarray = field(default=None, repr=False)
    index_: dict = field(default=None, repr=False)
    coef_: np.ndarray = field(default=None, repr=False)
    intercept_: float = 0.0

    def _design(self, Z_lists, charges, spins) -> np.ndarray:
        counts = np.zeros((len(Z_lists), len(self.elements_)))
        for i, Zs in enumerate(Z_lists):
            for Z in Zs:
                j = self.index_.get(int(Z))
                if j is not None:
                    counts[i, j] += 1.0
        extra = np.vstack(
            [extra_feats(Zs, c, s) for Zs, c, s in zip(Z_lists, charges, spins)]
        )
        return np.hstack([counts, extra])

    def fit(self, Z_lists, y_total, charges, spins) -> "ExtensiveMean":
        self.elements_ = np.array(sorted({int(z) for Zs in Z_lists for z in Zs}))
        self.index_ = {int(z): j for j, z in enumerate(self.elements_)}
        X = self._design(Z_lists, charges, spins)
        X = np.hstack([np.ones((len(X), 1)), X])
        R = self.ridge * np.eye(X.shape[1])
        R[0, 0] = 0.0
        beta = np.linalg.solve(X.T @ X + R, X.T @ np.asarray(y_total, float))
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, Z_lists, charges, spins) -> np.ndarray:
        return self.intercept_ + self._design(Z_lists, charges, spins) @ self.coef_

    def residual(self, Z_lists, y_total, charges, spins) -> np.ndarray:
        return np.asarray(y_total, float) - self.predict(Z_lists, charges, spins)

    def save(self, path):
        np.savez(path, elements=self.elements_, coef=self.coef_,
                 intercept=self.intercept_)


# ----------------------------- subset + target -----------------------------


@dataclass
class Dataset:
    """One loaded split: molecules, target residual, and integer categories."""
    atoms: list
    y: np.ndarray                    # (N,) intensive residual
    data_id: np.ndarray             # (N,) integer category id
    category_names: list            # index -> original string id
    mean: ExtensiveMean

    def __len__(self):
        return len(self.atoms)


def _encode_categories(raw_ids):
    """Map arbitrary (string) data_id values to contiguous integers, sorted for
    reproducibility. Returns (int_array, names_list)."""
    names = sorted({str(r) for r in raw_ids})
    lut = {name: i for i, name in enumerate(names)}
    return np.array([lut[str(r)] for r in raw_ids], dtype=np.int64), names


def get_data(
    src: str = "train_4M",
    n: int = 200_000,
    seed: int = 0,
    charge_key: str = "lowdin_charges",
    category_key: str = "data_id",
    cache_dir: str = "cache",
) -> Dataset:
    """Draw (or reuse a frozen) n-molecule subset with valid charges, build the
    residual target, and extract integer categories. Indices are frozen to
    ``cache_dir/subset_indices_{n}.npy`` so featurizer/reducer see identical rows.
    """
    from fairchem.core.datasets import AseDBDataset

    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, f"subset_indices_{n}.npy")

    ds = AseDBDataset({"src": src})
    N = len(ds)
    print(f"[data] source has {N:,} structures; requesting n={n:,}")

    if os.path.exists(idx_path):
        idxs = np.load(idx_path)
        print(f"[data] reusing cached subset of {len(idxs):,} indices")
    else:
        rng = np.random.default_rng(seed)
        pool = rng.permutation(N)
        idxs, p = [], 0
        while len(idxs) < n and p < N:
            i = int(pool[p]); p += 1
            q = ds.get_atoms(i).info.get(charge_key)
            if q is not None and not np.any(np.isnan(np.asarray(q, float))):
                idxs.append(i)
            if p % 50_000 == 0:
                print(f"[data]   scanned {p:,}, kept {len(idxs):,}")
        idxs = np.array(idxs)
        if len(idxs) < n:
            raise RuntimeError(f"only {len(idxs)} valid molecules found (< {n})")
        np.save(idx_path, idxs)
        print(f"[data] drew and froze {len(idxs):,} indices (scanned {p:,})")

    atoms, Z, Y, C, S, raw_cat = [], [], [], [], [], []
    for k, i in enumerate(idxs):
        a = ds.get_atoms(int(i))
        atoms.append(a)
        Z.append(a.get_atomic_numbers().tolist())
        Y.append(float(a.get_potential_energy()))
        C.append(float(a.info.get("charge", 0)))
        S.append(float(a.info.get("spin", 1)))
        raw_cat.append(a.info.get(category_key, "unknown"))
        if (k + 1) % 25_000 == 0:
            print(f"[data]   materialised {k + 1:,}/{len(idxs):,} molecules")

    mean = ExtensiveMean().fit(Z, Y, C, S)
    y = mean.residual(Z, Y, C, S)
    cats, names = _encode_categories(raw_cat)
    mean.save(os.path.join(cache_dir, f"mean_model_{n}.npz"))
    np.save(os.path.join(cache_dir, f"y_residual_{n}.npy"), y)
    print(
        f"[data] residual var={np.var(y):.4g} (std {np.std(y):.4g}); "
        f"{len(names)} categories: {names}"
    )
    return Dataset(atoms=atoms, y=y, data_id=cats, category_names=names, mean=mean)


def stratified_sample_indices(data_id: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """Indices of a representative sample covering EVERY category proportionally
    (min 1 per category). Used to fit/freeze the WL vocabulary so it spans all
    categories/elements before the full transform."""
    rng = np.random.default_rng(seed)
    cats = np.unique(data_id)
    per = max(1, size // len(cats))
    picks = []
    for c in cats:
        idx = np.where(data_id == c)[0]
        take = min(len(idx), per)
        picks.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picks)
    if len(out) > size:
        out = rng.choice(out, size=size, replace=False)
    return np.sort(out)
