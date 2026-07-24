"""
stats.py  (data_exploration)
============================
Every census-level aggregate, in one place, written to ``cache/summary.json``.

The figures never compute a headline number themselves -- they read this file (or
the arrays it saves alongside), so poster text, ``REPORT.md`` and the plots cannot
drift apart.

    python -m data_exploration.stats        # -> cache/summary.json, cache/derived.npz

``load_all()`` is the shared entry point for this module and ``figures.py``: it
concatenates the census parts, attaches the ``data_id`` label per row, and (if
``families.py`` has run) the intensive residual target ``y``.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .census import CACHE, load_census

SUMMARY = os.path.join(CACHE, "summary.json")
DERIVED = os.path.join(CACHE, "derived.npz")


def load_all(columns=None, need_y=True):
    """-> ``(c, names, counts, subset, y)``; ``y`` is None if families has not run."""
    c, names, counts, fam_examples = load_census(columns=columns)
    subset = np.array([names.get(int(h), "unknown") for h in c["data_id"]])
    y = None
    fpath = os.path.join(CACHE, "families.npz")
    if need_y and os.path.exists(fpath):
        with np.load(fpath) as d:
            y = d["y"]
        if len(y) != len(subset):
            print("[stats] families.npz row count differs from census; ignoring y")
            y = None
    return c, names, counts, subset, y, fam_examples


def _q(x, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    v = np.percentile(x, qs)
    return {f"p{q}": float(vi) for q, vi in zip(qs, v)} | {
        "mean": float(x.mean()), "std": float(x.std()), "n_finite": int(len(x))
    }


def compute(c, names, counts, subset, y):
    from ase.data import chemical_symbols

    N = len(subset)
    out = {"dataset": {"n_structures": int(N), "n_shards": int(c["shard"].max() + 1)}}

    # ---- subsets and provenance -------------------------------------------
    su, sc = np.unique(subset, return_counts=True)
    order = np.argsort(-sc)
    out["subsets"] = {str(su[i]): int(sc[i]) for i in order}
    top = np.array([names.get(int(h), "?") for h in c["src_top"]])
    tu, tc = np.unique(top, return_counts=True)
    out["src_top"] = {str(tu[i]): int(tc[i]) for i in np.argsort(-tc)}
    grp = np.array([names.get(int(h), "?") for h in c["src_group"]])
    gu, gc = np.unique(grp, return_counts=True)
    out["n_src_groups"] = int(len(gu))
    out["src_group_top20"] = {str(gu[i]): int(gc[i]) for i in np.argsort(-gc)[:20]}

    # ---- elements ----------------------------------------------------------
    B = counts.copy()
    B.data = np.ones_like(B.data)                      # presence, not multiplicity
    n_struct_with = np.asarray(B.sum(axis=0)).ravel()  # structures containing Z
    n_atoms_of = np.asarray(counts.sum(axis=0)).ravel()
    present = np.where(n_struct_with > 0)[0]
    out["elements"] = {
        "n_distinct": int(len(present)),
        "z_present": present.tolist(),
        "by_symbol": {
            chemical_symbols[int(z)]: {
                "n_structures": int(n_struct_with[z]),
                "frac_structures": float(n_struct_with[z] / N),
                "n_atoms": int(n_atoms_of[z]),
            }
            for z in present
        },
    }
    out["atoms_total"] = int(n_atoms_of.sum())

    # ---- size / cost -------------------------------------------------------
    out["size"] = {
        "n_atoms": _q(c["n_atoms"]), "n_heavy": _q(c["n_heavy"]),
        "n_electrons": _q(c["n_elec"]), "n_basis": _q(c["n_basis"]),
        "rg": _q(c["rg"]),
    }
    out["size_per_subset"] = {
        str(s): _q(c["n_atoms"][subset == s]) for s in out["subsets"]
    }

    # ---- charge / spin -----------------------------------------------------
    qu, qc = np.unique(c["charge"], return_counts=True)
    pu, pc = np.unique(c["spin"], return_counts=True)
    out["charge"] = {"hist": {int(k): int(v) for k, v in zip(qu, qc)},
                     "frac_neutral": float((c["charge"] == 0).mean())}
    out["spin"] = {"hist": {int(k): int(v) for k, v in zip(pu, pc)},
                   "frac_singlet": float((c["spin"] == 1).mean()),
                   "frac_unrestricted": float(c["unrestricted"].mean())}

    # ---- composition diversity --------------------------------------------
    fu, fc = np.unique(c["formula"], return_counts=True)
    out["composition"] = {
        "n_unique_formulas": int(len(fu)),
        "frac_rows_in_repeated_formula": float(fc[fc > 1].sum() / N),
        "largest_formula_group": int(fc.max()),
        "zipf_top": sorted(fc.tolist(), reverse=True)[:200],
    }

    # ---- electronic structure ---------------------------------------------
    out["electronic"] = {
        "gap": _q(c["gap"]), "homo": _q(c["homo"]),
        "s_squared_dev": _q(c["s_squared_dev"]), "n_scf": _q(c["n_scf"]),
        "fmax": _q(c["fmax"]), "frms": _q(c["frms"]), "dipole": _q(c["dipole"]),
    }

    # ---- data hygiene ------------------------------------------------------
    e = c["energy"]
    z = np.abs(e - np.median(e)) / (1.4826 * np.median(np.abs(e - np.median(e))))
    # NOTE every structure carries at least one ORCA warning, and all of the
    # high-frequency ones are boilerplate (functional/dispersion notices), so
    # "has a warning" is not a quality signal -- see cache/warnings.json. The
    # panels below use conditions that actually flag a hard calculation.
    def _hyg(m):
        return {
            "frac_scf_ge_100": float((c["n_scf"][m] >= 100).mean()),
            "frac_s2dev_gt_0p1": float(np.nanmean(c["s_squared_dev"][m] > 0.1)),
            "frac_gap_lt_1eV": float(np.nanmean(c["gap"][m] < 1.0)),
            "frac_has_nbo": float(c["has_nbo"][m].mean()),
            "frac_nan_lowdin": float(c["nan_lowdin"][m].mean()),
            "frac_fmax_ge_49p9": float(np.nanmean(c["fmax"][m] >= 49.9)),
        }

    out["hygiene"] = {
        "frac_with_warning": float((c["n_warn"] > 0).mean()),
        "n_nan_lowdin": int(c["nan_lowdin"].sum()),
        **_hyg(np.ones(N, dtype=bool)),
        "per_subset": {str(s): _hyg(subset == s) for s in out["subsets"]},
    }

    # ---- the target, and which free metadata column predicts it ------------
    if y is not None:
        out["target"] = {
            "var": float(np.var(y)), "std": float(np.std(y)), **_q(y),
            "per_subset_var": {str(s): float(np.var(y[subset == s]))
                               for s in out["subsets"]},
        }
        cand = ["n_atoms", "n_heavy", "n_elec", "n_basis", "n_scf", "rg", "gap",
                "homo", "fmax", "frms", "dipole", "q_min", "q_max", "q_rms",
                "s_squared_dev", "charge", "spin", "nl_energy"]
        r2 = {}
        for k in cand:
            x = c[k].astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 1000 and x[m].std() > 0:
                r = np.corrcoef(x[m], y[m])[0, 1]
                r2[k] = float(r * r)
        out["metadata_r2"] = dict(sorted(r2.items(), key=lambda kv: -kv[1]))
        # joint linear fit on the standardised candidates (how much do they add?)
        X = np.column_stack([c[k].astype(float) for k in cand])
        m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        Xm = np.column_stack([np.ones(m.sum()), (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-12)])
        beta, *_ = np.linalg.lstsq(Xm, y[m], rcond=None)
        pred = Xm @ beta
        out["metadata_joint_r2"] = float(1 - np.var(y[m] - pred) / np.var(y[m]))
        out["metadata_joint_n"] = int(m.sum())

    return out


def derived_arrays(c, counts, subset, y):
    """Histograms and matrices the figures need, precomputed once."""
    from ase.data import chemical_symbols  # noqa: F401  (symbols resolved in figures)

    B = counts.copy()
    B.data = np.ones_like(B.data)
    cooc = (B.T @ B).toarray()                    # (119,119) co-occurrence counts
    n_with = np.asarray(B.sum(axis=0)).ravel()
    out = {
        "cooc": cooc.astype(np.int64),
        "n_struct_with_z": n_with.astype(np.int64),
        "n_atoms_of_z": np.asarray(counts.sum(axis=0)).ravel().astype(np.int64),
    }
    return out


def run():
    c, names, counts, subset, y, _ = load_all()
    print("[stats] aggregating ...")
    summary = compute(c, names, counts, subset, y)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(DERIVED, **derived_arrays(c, counts, subset, y))
    print(f"[stats] wrote {SUMMARY} and {DERIVED}")
    return summary


if __name__ == "__main__":
    run()
