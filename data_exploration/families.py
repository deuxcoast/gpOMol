"""
families.py  (data_exploration)
===============================
Redundancy, effective sample size, and the **graph-only ceiling**.

Two questions, both answerable from the census alone:

1. *How many distinct molecules are in "4M structures?"* Conformers, MD frames and
   relaxation steps of the same molecule are separate rows. Grouping rows by their
   provenance family (``sources.parse_source``) turns 4M structures into a much
   smaller number of families, and the inverse-Simpson index of the family-size
   distribution gives a redundancy-corrected effective N.

2. *How much of the regression target can a graph-only descriptor explain, at
   best?* A descriptor that sees only the molecular graph assigns one value to
   every member of a family, so its best possible predictor is the family mean and
   its ceiling is the **between-family variance share**:

       SS_total = SS_between + SS_within,   R^2_max = SS_between / SS_total

   The within share is conformational/geometric energy that no graph can reach.
   This is an a-priori ceiling: no GP is fitted, no descriptor is computed.

Because the family parse is a heuristic (``sources.py``), the decomposition is
reported for **three groupings** that bracket it:

``family``            the parsed family (finest; assumes the parse is right)
``family x formula``  the parse intersected with an exact chemical formula --
                      conservative, a family can never span two compositions
``formula x q x s``   parse-free: same stoichiometry, charge and spin. Coarsest;
                      lumps genuine isomers together, so its within share is an
                      upper bound on what conformation alone contributes.

The regression target is the intensive residual used everywhere else in the repo:
``y = E_total - m(x)`` with ``m`` the ridge-fit extensive (element-referencing)
mean of ``wl_gp2scale/data.py::ExtensiveMean`` -- same design matrix
``[1, element counts, charge, |charge|, charge^2, spin, n_atoms^2]``, refit here on
all 3,986,754 rows by chunked normal equations (the reference implementation
materialises a dense design matrix, which does not fit at 4M).

    python -m data_exploration.families        # -> cache/families.npz + .json
"""

from __future__ import annotations

import json
import os

import numpy as np

from .census import CACHE, load_census

MIX = np.uint64(0x9E3779B97F4A7C15)


# ------------------------------- target ------------------------------------


def extensive_mean(counts, charge, spin, n_atoms, energy, ridge=1e-6, chunk=400_000):
    """Chunked ridge fit of the extensive mean; returns ``(y, info)``.

    Design matrix per row: ``[1, n_Z for each observed element, charge, |charge|,
    charge^2, spin, n_atoms^2]`` -- identical to ``ExtensiveMean`` in
    ``wl_gp2scale/data.py``, accumulated as normal equations so 4M rows never
    materialise densely.
    """
    counts = counts.tocsr()
    used = np.unique(counts.indices)                 # elements actually present
    counts = counts[:, used]
    n = counts.shape[0]
    p = 1 + len(used) + 5
    XtX = np.zeros((p, p))
    Xty = np.zeros(p)
    ymean_acc = 0.0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        C = counts[s:e].toarray().astype(np.float64)
        q = charge[s:e].astype(np.float64)
        X = np.column_stack([
            np.ones(e - s), C, q, np.abs(q), q**2,
            spin[s:e].astype(np.float64), n_atoms[s:e].astype(np.float64) ** 2,
        ])
        yy = energy[s:e].astype(np.float64)
        XtX += X.T @ X
        Xty += X.T @ yy
        ymean_acc += yy.sum()
    R = ridge * np.eye(p)
    R[0, 0] = 0.0
    beta = np.linalg.solve(XtX + R, Xty)

    y = np.empty(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        C = counts[s:e].toarray().astype(np.float64)
        q = charge[s:e].astype(np.float64)
        X = np.column_stack([
            np.ones(e - s), C, q, np.abs(q), q**2,
            spin[s:e].astype(np.float64), n_atoms[s:e].astype(np.float64) ** 2,
        ])
        y[s:e] = energy[s:e].astype(np.float64) - X @ beta

    ybar = ymean_acc / n
    ss_tot = float(np.sum((energy.astype(np.float64) - ybar) ** 2))
    info = {
        "elements": used.tolist(),
        "n_features": int(p),
        "r2_of_mean_model": float(1.0 - np.sum(y**2) / ss_tot),
        "residual_var": float(np.var(y)),
        "residual_std": float(np.std(y)),
        "total_energy_var": float(ss_tot / n),
    }
    return y, beta, info


# ---------------------------- decomposition --------------------------------


def mix(a, b):
    """Combine two 64-bit keys into one (wrapping multiply-xor)."""
    return np.bitwise_xor((a.astype(np.uint64) * MIX), b.astype(np.uint64))


def decompose(y, key, weights_min_size=2):
    """ANOVA split of ``Var(y)`` by group ``key``.

    Two ceilings are returned and they differ a lot; the distinction matters.

    ``r2_max_in_sample`` = ``SS_between/SS_tot`` is what the group means explain
    *on the rows they were computed from*. With 1.2M groups over 4M rows the group
    means have ~one third as many degrees of freedom as data points, so this
    number is inflated -- a singleton group is "explained" perfectly by definition.
    It is a genuine **upper** bound, nothing more.

    ``r2_max_unbiased`` uses the one-way random-effects variance component,
    ``sigma^2_within = SS_within / (n - g)``, and reports
    ``1 - sigma^2_within / Var(y)``: what a descriptor constant within groups could
    achieve on *held-out* rows. This is the number to quote.

    Singleton groups contribute nothing to the within sum of squares, so
    ``frac_rows_in_multi_groups`` is reported alongside -- a grouping that is
    mostly singletons says little about anything.
    """
    y = np.asarray(y, float)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    n, g = len(y), len(cnt)
    gsum = np.bincount(inv, weights=y, minlength=g)
    gmean = gsum / cnt
    ybar = y.mean()
    ss_tot = float(np.sum((y - ybar) ** 2))
    ss_between = float(np.sum(cnt * (gmean - ybar) ** 2))
    ss_within = max(ss_tot - ss_between, 0.0)

    big = cnt >= weights_min_size
    memb_big = big[inv]
    if memb_big.any():
        yb = y[memb_big]
        ss_w_big = float(np.sum((yb - gmean[inv][memb_big]) ** 2))
        var_w_big = ss_w_big / max(memb_big.sum() - big.sum(), 1)
    else:
        ss_w_big, var_w_big = 0.0, 0.0

    var_total = ss_tot / n
    dof = max(n - g, 1)
    var_within_unbiased = ss_within / dof
    return {
        "n_rows": int(n),
        "n_groups": int(g),
        "mean_group_size": float(n / g),
        "max_group_size": int(cnt.max()),
        "frac_rows_in_multi_groups": float(memb_big.mean()),
        "effective_n_inverse_simpson": float(n**2 / np.sum(cnt.astype(float) ** 2)),
        "var_total": var_total,
        "share_between": ss_between / ss_tot if ss_tot else float("nan"),
        "share_within": ss_within / ss_tot if ss_tot else float("nan"),
        "r2_max_in_sample": ss_between / ss_tot if ss_tot else float("nan"),
        "r2_max_unbiased": float(1.0 - var_within_unbiased / var_total)
        if var_total else float("nan"),
        "var_within_unbiased": float(var_within_unbiased),
        "rmse_within_unbiased": float(np.sqrt(var_within_unbiased)),
        "dof_within": int(dof),
        "within_var_multi_groups": var_w_big,
        "within_rmse_multi_groups": float(np.sqrt(var_w_big)),
    }


def size_ccdf(key, max_k=64):
    """``P(family size >= k)`` and the fraction of ROWS living in such families."""
    _, cnt = np.unique(key, return_counts=True)
    ks = np.arange(1, max_k + 1)
    fam = np.array([float((cnt >= k).mean()) for k in ks])
    rows = np.array([float(cnt[cnt >= k].sum() / cnt.sum()) for k in ks])
    return ks, fam, rows, cnt


# --------------------------------- driver ----------------------------------


def run(out_npz=None, out_json=None):
    out_npz = out_npz or os.path.join(CACHE, "families.npz")
    out_json = out_json or os.path.join(CACHE, "families.json")
    cols = ["family", "job", "formula", "charge", "spin", "energy", "n_atoms",
            "data_id", "step"]
    c, names, counts, fam_examples = load_census(columns=cols)

    print("[fam] fitting the extensive mean on all rows ...")
    y, beta, info = extensive_mean(counts, c["charge"], c["spin"], c["n_atoms"],
                                   c["energy"])
    print(f"[fam]   mean-model R2 = {info['r2_of_mean_model']:.6f}; "
          f"residual var = {info['residual_var']:.4g} eV^2 "
          f"(std {info['residual_std']:.3g} eV)")

    qs = mix(c["charge"].astype(np.uint64), c["spin"].astype(np.uint64))
    # ordered coarse -> fine: each level is a strict refinement of the one above,
    # so the ceilings are monotone and bracket the (unmeasured) molecular-graph one
    groupings = {
        "formula": c["formula"],
        "formula_x_charge_spin": mix(c["formula"], qs),
        "family": c["family"],
        "family_x_formula": mix(c["family"], c["formula"]),
        "job": c["job"],
    }

    res = {"target": info, "overall": {}, "per_subset": {}}
    for gname, key in groupings.items():
        d = res["overall"][gname] = decompose(y, key)
        print(f"[fam] {gname:22s} groups={d['n_groups']:>9,} "
              f"ceiling R2 unbiased={d['r2_max_unbiased']:.3f} "
              f"(in-sample {d['r2_max_in_sample']:.3f})  "
              f"rows in multi-groups={d['frac_rows_in_multi_groups']:.2f}")

    subset = np.array([names.get(int(h), "unknown") for h in c["data_id"]])
    for s in np.unique(subset):
        m = subset == s
        res["per_subset"][s] = {
            "n": int(m.sum()),
            "var_total": float(np.var(y[m])),
            **{g: decompose(y[m], k[m]) for g, k in groupings.items()},
        }

    ks, fam_ccdf, row_ccdf, cnt = size_ccdf(c["family"])
    ks2, cmp_ccdf, cmp_row_ccdf, cnt2 = size_ccdf(groupings["formula_x_charge_spin"])
    res["steps"] = {
        "frac_rows_with_step": float((c["step"] >= 0).mean()),
        "mean_frames_per_job": float(res["overall"]["job"]["mean_group_size"]),
    }
    res["family_examples"] = [fam_examples[k] for k in list(fam_examples)[:20]]

    np.savez_compressed(
        out_npz, y=y, beta=beta, subset=subset,
        family=c["family"], job=c["job"], formula=c["formula"],
        ks=ks, fam_ccdf=fam_ccdf, row_ccdf=row_ccdf,
        cmp_ccdf=cmp_ccdf, cmp_row_ccdf=cmp_row_ccdf,
        fam_sizes=np.bincount(np.minimum(cnt, 512)),
        cmp_sizes=np.bincount(np.minimum(cnt2, 512)),
    )
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[fam] wrote {out_npz} and {out_json}")
    return res


if __name__ == "__main__":
    run()
