r"""
mean_function.py
================
Promote energy referencing from preprocessing to a GP PRIOR MEAN, so the
compactly-supported kernel models only the residual g = E - m(M).

Why this is the move (and why element-only referencing is not enough)
---------------------------------------------------------------------
The binding conflict at the molecule level is extensivity vs stationarity: a
stationary compactly-supported kernel assumes the target decorrelates past a
finite distance, but an EXTENSIVE target grows with size, so any distance that
respects it never plateaus -> dense. The fix is to let an extensive MEAN FUNCTION
carry the size-scaling, leaving an INTENSIVE residual that a stationary kernel
can actually plateau on.

Per-element referencing (regress E on element counts) is the FIRST such mean
function -- it removes isolated-atom energies (the dominant ~1e4-1e5 eV term).
But the residual still contains cohesive/bonding energy, which is ALSO extensive
(a sum over bonds), so the residual still scales with size and the kernel still
fights stationarity. That is exactly why the WL support sweep on the referenced
residual showed no flat shoulder.

So we go one term further: a BOND-INVENTORY mean (element counts + bond-type
counts) -- a classic extensive bond-energy model. It should absorb the cohesive
trend element counts miss, while leaving intensive chemistry (strain,
conjugation, long-range effects) in the residual for the kernel.

The honesty rail: RESIDUAL-EXTENSIVITY DIAGNOSTIC
-------------------------------------------------
Whether a given mean function actually makes the residual intensive is an
empirical question, not an assumption. `residual_size_diagnostic` measures how
much the residual still tracks molecule size (Spearman, and the fraction of
residual variance a linear size fit explains). If the residual still correlates
with size, the mean function did NOT do its job and the kernel will still be
dense -- use a richer basis. Only once the residual is size-decorrelated is it
worth re-running the support sweep to look for the shoulder.

This module computes the target; the WL screen / support sweep (in the pipeline)
then run on that residual exactly as before. Because the mean is deterministic,
the residual's predictive uncertainty IS the total energy's, so CRPS on g is
directly comparable to CRPS on E from earlier runs.

Run `python mean_function.py` for a self-test on a synthetic energy whose
extensive part is atom + BOND linear: element-only referencing should leave the
residual size-correlated, element+bonds should decorrelate it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


# --------------------------------------------------------------------------- #
# Small stats helper (kept local so this module doesn't import the pipeline)
# --------------------------------------------------------------------------- #
def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


# --------------------------------------------------------------------------- #
# Extensive feature bases for the mean function
# --------------------------------------------------------------------------- #
def element_count_features(symbols_list):
    """Per-molecule element counts. Columns = sorted element symbols."""
    elements = sorted({s for syms in symbols_list for s in syms})
    idx = {e: k for k, e in enumerate(elements)}
    C = np.zeros((len(symbols_list), len(elements)))
    for m, syms in enumerate(symbols_list):
        for s in syms:
            C[m, idx[s]] += 1
    return C, [f"n_{e}" for e in elements], elements


_SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Cu": 29,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}


def bond_type_features(symbols_list, positions_list, scale=1.2):
    """Per-molecule counts of each element-pair bond type, using the same
    geometric covalent-radius bond perception as the WL kernel. Columns = sorted
    'A-B' element pairs. Extensive (sum over bonds), so it carries cohesive
    energy that element counts alone cannot."""
    from wl_kernel import build_adjacency

    per_mol, vocab = [], {}
    for syms, pos in zip(symbols_list, positions_list):
        Z = np.array([_SYMBOL_TO_Z.get(s, 6) for s in syms])
        adj = build_adjacency(Z, np.asarray(pos, float), scale=scale)
        counts = Counter()
        for i in range(len(Z)):
            for j in adj[i]:
                if j > i:  # each undirected bond once
                    a, b = sorted((syms[i], syms[j]))
                    counts[f"{a}-{b}"] += 1
        per_mol.append(counts)
        for k in counts:
            vocab.setdefault(k, len(vocab))
    B = np.zeros((len(per_mol), len(vocab)))
    for m, c in enumerate(per_mol):
        for k, v in c.items():
            B[m, vocab[k]] = v
    return B, [f"b_{k}" for k in vocab]


def build_design(
    data, basis="element", scale=1.2, add_charge=False, add_intercept=False
):
    """Assemble the extensive design matrix for the chosen mean-function basis."""
    C, clabels, elements = element_count_features(data["symbols"])
    cols, labels = [C], list(clabels)
    if basis == "element":
        pass
    elif basis == "element+bonds":
        B, blabels = bond_type_features(data["symbols"], data["positions"], scale)
        cols.append(B)
        labels += blabels
    else:
        raise ValueError(f"unknown basis {basis!r}")
    if add_charge:
        cols.append(np.asarray(data["charges"], float).reshape(-1, 1))
        labels.append("charge")
    if add_intercept:
        cols.append(np.ones((len(data["symbols"]), 1)))
        labels.append("intercept")
    return np.hstack(cols), labels, elements


# --------------------------------------------------------------------------- #
# Fit + residual + diagnostic
# --------------------------------------------------------------------------- #
def fit_and_residual(X, energies):
    """Least-squares extensive mean fit; return (residual, coefs, R^2_of_mean)."""
    energies = np.asarray(energies, float)
    coefs, *_ = np.linalg.lstsq(X, energies, rcond=None)
    fit = X @ coefs
    g = energies - fit
    r2 = 1.0 - np.var(g) / np.var(energies) if np.var(energies) > 0 else 0.0
    return g, coefs, float(r2)


def residual_size_diagnostic(g, sizes):
    """How much does the residual still track molecule SIZE? Returns
    (spearman, frac_var_explained_by_size, slope_eV_per_atom). High values mean
    the residual is still extensive and the kernel will still be dense."""
    g = np.asarray(g, float)
    sizes = np.asarray(sizes, float)
    rho = spearman(g, sizes)
    A = np.vstack([sizes, np.ones_like(sizes)]).T
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    pred = A @ coef
    r2 = 1.0 - np.var(g - pred) / np.var(g) if np.var(g) > 0 else 0.0
    return float(rho), float(r2), float(coef[0])


def compute_target(
    data,
    basis="element",
    scale=1.2,
    add_charge=False,
    add_intercept=False,
    verbose=True,
):
    """Full mean-function decomposition E = m(M) + g(M).

    Returns dict with the residual target `g`, the fitted mean (coefs/labels/R^2),
    the residual-extensivity diagnostic, and a printable verdict. The pipeline
    feeds `g` to the WL screen / support sweep in place of the old `y`.
    """
    energies = np.asarray(data["energies"], float)
    sizes = np.array([len(s) for s in data["symbols"]])
    X, labels, elements = build_design(data, basis, scale, add_charge, add_intercept)
    g, coefs, mean_r2 = fit_and_residual(X, energies)
    rho, size_r2, slope = residual_size_diagnostic(g, sizes)

    # Verdict: is the residual intensive enough for a stationary compact kernel?
    if size_r2 >= 0.10 or abs(rho) >= 0.30:
        verdict = (
            "RESIDUAL STILL EXTENSIVE -> kernel will fight stationarity; "
            "use a richer --mean-function before trusting a flat support sweep."
        )
        intensive = False
    else:
        verdict = (
            "residual looks intensive (size-decorrelated) -> a "
            "compactly-supported kernel can plausibly plateau; run the sweep."
        )
        intensive = True

    info = dict(
        g=g,
        coefs=coefs,
        labels=labels,
        elements=elements,
        mean_r2=mean_r2,
        size_rho=rho,
        size_r2=size_r2,
        size_slope=slope,
        intensive=intensive,
        verdict=verdict,
        basis=basis,
        sizes=sizes,
    )
    if verbose:
        print_mean_report(info, energies)
    return info


def print_mean_report(info, energies):
    g = info["g"]
    print(
        f"\nmean function basis : {info['basis']}  ({len(info['labels'])} extensive features)"
    )
    print(
        f"  mean fit R^2        : {info['mean_r2']:.5f}   "
        f"(fraction of total-energy variance absorbed by the mean)"
    )
    print(
        f"  raw spread          : {np.ptp(energies):,.1f} eV  ->  "
        f"residual spread {np.ptp(g):,.1f} eV  (RMS {np.sqrt(np.mean(g**2)):.3f} eV)"
    )
    print(
        f"  RESIDUAL vs SIZE    : Spearman={info['size_rho']:+.3f}, "
        f"size explains {100*info['size_r2']:.1f}% of residual variance, "
        f"slope={info['size_slope']:+.3f} eV/atom"
    )
    print(f"  -> {info['verdict']}")


# --------------------------------------------------------------------------- #
# Self-test: synthetic energy with an atom+BOND extensive part
# --------------------------------------------------------------------------- #
def _demo():
    from wl_kernel import _grow_molecule, build_adjacency

    rng = np.random.default_rng(2)
    N = 300
    atom_E = {1: -13.6, 6: -1030.0, 7: -1480.0, 8: -2040.0}
    sym = {1: "H", 6: "C", 7: "N", 8: "O"}
    # bond energies by sorted element-pair (the extensive cohesive term)
    bond_E = {
        ("C", "C"): -3.0,
        ("C", "H"): -4.0,
        ("C", "N"): -2.5,
        ("C", "O"): -3.5,
        ("H", "N"): -3.8,
        ("H", "O"): -4.5,
        ("N", "O"): -2.0,
        ("H", "H"): -1.0,
        ("N", "N"): -2.2,
        ("O", "O"): -1.8,
    }

    symbols, positions, comps, energies, charges = [], [], [], [], []
    for _ in range(N):
        na = int(rng.integers(8, 26))
        Z, pos = _grow_molecule(rng, na)
        syms = [sym[int(z)] for z in Z]
        adj = build_adjacency(Z, pos, 1.2)
        e = sum(atom_E[int(z)] for z in Z)  # atom-linear (extensive)
        for i in range(len(Z)):
            for j in adj[i]:
                if j > i:
                    a, b = sorted((syms[i], syms[j]))
                    e += bond_E[(a, b)]  # bond-linear (extensive)
        # INTENSIVE term: a size-INDEPENDENT per-molecule latent (think
        # conformational offset). After the extensive atom+bond mean is removed,
        # only this remains -> the diagnostic should certify it as intensive.
        e += rng.normal(scale=2.0)
        symbols.append(syms)
        positions.append(pos)
        comps.append(Counter(syms))
        energies.append(e)
        charges.append(0.0)

    data = {
        "symbols": symbols,
        "positions": positions,
        "comps": comps,
        "energies": np.array(energies),
        "charges": np.array(charges),
    }

    print("=" * 78)
    print("  SELF-TEST: energy = atom-linear + BOND-linear (extensive) + branching")
    print("  fraction (intensive) + noise. Element-only should leave the residual")
    print("  size-correlated; element+bonds should decorrelate it.")
    print("=" * 78)
    print("\n--- basis = element (per-element referencing only) ---")
    compute_target(data, basis="element")
    print("\n--- basis = element+bonds (bond-inventory mean) ---")
    compute_target(data, basis="element+bonds")
    print("\nIf the bond basis flips the verdict to 'intensive', the decomposition")
    print("works and the residual is ready for the WL support sweep.")


if __name__ == "__main__":
    _demo()
