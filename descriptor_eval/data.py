"""
data.py  (descriptor_eval)
=========================
Builds the shared evaluation data: a reproducible 10k-molecule subset of
train_4M and the DESCRIPTOR-INDEPENDENT target used by every candidate.

Target
------
    y_i = E_total(i) - m(x_i)
    m(x) = intercept + sum_Z n_Z eps_Z + [charge, |charge|, charge^2, spin, n_atoms^2]

`m` is a minimal self-contained extensive (element-referencing) mean fit once by
ridge least squares. Because it uses ONLY energy, element counts, charge and spin
-- nothing descriptor-specific -- the residual y is a fixed property of the 10k
molecules and is reused verbatim for Candidate A (and later B, C). That shared,
descriptor-independent y is what makes the variogram clouds comparable.

The residual is intensive by construction (the per-element terms absorb size), so
it is NOT further divided by n_atoms. Recovery of physical energy is exact and
cheap:  E_total = y_pred + m(x)  -- persist the fitted coefficients (done below).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ----------------------------- extensive mean ------------------------------


def extra_feats(Z_list, charge: float, spin: float) -> np.ndarray:
    """Nonlinear size/charge terms the linear element-count block can't represent."""
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
        X = np.hstack([np.ones((len(X), 1)), X])  # intercept
        R = self.ridge * np.eye(X.shape[1])
        R[0, 0] = 0.0  # don't penalise intercept
        beta = np.linalg.solve(X.T @ X + R, X.T @ np.asarray(y_total, float))
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, Z_lists, charges, spins) -> np.ndarray:
        return self.intercept_ + self._design(Z_lists, charges, spins) @ self.coef_

    def residual(self, Z_lists, y_total, charges, spins) -> np.ndarray:
        return np.asarray(y_total, float) - self.predict(Z_lists, charges, spins)

    def save(self, path):
        np.savez(
            path, elements=self.elements_, coef=self.coef_, intercept=self.intercept_
        )


# ----------------------------- subset + target -----------------------------


def get_data(
    src: str = "../train_4M",
    n: int = 10_000,
    seed: int = 0,
    charge_key: str = "lowdin_charges",
    cache_dir: str = "cache",
):
    """
    Returns (atoms_list, y_residual, mean_model). Reuses a cached frozen index list
    if present (so A/B/C see the IDENTICAL molecules); otherwise draws, admits only
    records with valid `charge_key`, freezes exactly n indices, and caches.
    """
    from fairchem.core.datasets import AseDBDataset

    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, "subset_indices.npy")

    ds = AseDBDataset({"src": src})
    N = len(ds)
    print(f"train_4M: {N:,} structures")

    if os.path.exists(idx_path):
        idxs = np.load(idx_path)
        print(f"  reusing cached subset of {len(idxs):,} indices")
    else:
        rng = np.random.default_rng(seed)
        pool = rng.permutation(N)  # draw order, admit until we hit n
        idxs, p = [], 0
        while len(idxs) < n and p < N:
            i = int(pool[p])
            p += 1
            q = ds.get_atoms(i).info.get(charge_key)
            if q is not None and not np.any(np.isnan(np.asarray(q, float))):
                idxs.append(i)
        idxs = np.array(idxs)
        if len(idxs) < n:
            raise RuntimeError(f"only {len(idxs)} valid molecules found (< {n})")
        np.save(idx_path, idxs)
        print(f"  drew and froze {len(idxs):,} indices (scanned {p:,})")

    atoms_list, Z, Y, C, S = [], [], [], [], []
    for i in idxs:
        a = ds.get_atoms(int(i))
        atoms_list.append(a)
        Z.append(a.get_atomic_numbers().tolist())
        Y.append(float(a.get_potential_energy()))
        C.append(float(a.info.get("charge", 0)))
        S.append(float(a.info.get("spin", 1)))

    mean = ExtensiveMean().fit(Z, Y, C, S)
    y = mean.residual(Z, Y, C, S)
    mean.save(os.path.join(cache_dir, "mean_model.npz"))
    np.save(os.path.join(cache_dir, "y_residual.npy"), y)
    print(
        f"  target: residual var={np.var(y):.4g} (std {np.std(y):.4g}); "
        f"E_total recoverable as y + m(x)"
    )
    return atoms_list, y, mean
