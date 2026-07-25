"""
descriptors.py  (data_exploration)
==================================
Step 5 of `PLAN.md`: the diagnostics that decide open modelling questions in
`wl_gp2scale/`, run on the Pass B samples from ``geom_sample.py``.

Five analyses, each a separate CLI mode:

``ceiling``      **The number REPORT.md §6 left open.** The molecular-graph ceiling,
                 bracketed there at [0.53, 0.92], measured by grouping structures on
                 the WL multiset hash -- the exact equivalence class of the production
                 descriptor.
``vocab``        WL vocabulary growth vs N and the out-of-vocabulary rate of a fresh
                 draw against a frozen vocabulary. Sizes the 4M run.
``perception``   How much of the "molecular graph" is an artefact of the 1.2x
                 covalent-radius bond perception every descriptor in this repo shares.
``variogram``    gamma(h) and the pairwise-distance distribution per channel and per
                 ``data_id``: is one global length scale defensible?
``cross``        Do nearest neighbours cross ``data_id`` boundaries, and do those
                 pairs carry target correlation? A direct test of the block-sparse
                 kernel's core assumption.

    python -m data_exploration.descriptors ceiling
    python -m data_exploration.descriptors variogram --n 20000

Results are appended to ``cache/descriptors.json`` so the report and figures can
quote them without recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .census import CACHE, load_census
from .geom_sample import load as load_geom

OUT = os.path.join(CACHE, "descriptors.json")


def _save(section, payload):
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    data[section] = payload
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"[desc] wrote cache/descriptors.json :: {section}")


def _pooled_within(y, key):
    """(SS_within, dof, n_groups) for one grouping -- the pooled variance component."""
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    g = len(cnt)
    gsum = np.bincount(inv, weights=y, minlength=g)
    gmean = gsum / cnt
    ss = float(np.sum((y - gmean[inv]) ** 2))
    return ss, int(len(y) - g), g


# ---------------------------------------------------------------------------
# 1. the molecular-graph ceiling
# ---------------------------------------------------------------------------

def ceiling(tag="clusters"):
    """Close the [0.53, 0.92] bracket.

    The estimand is `sigma^2_within(WL)` over the whole 4M, which cannot be measured
    directly without WL-hashing 4M structures. But WL groups are a *refinement* of
    (formula, charge, spin) groups -- same WL labels implies same formula -- so

        sigma^2_w(WL) = sigma^2_w(formula) x rho,
        rho = sigma^2_w(WL) / sigma^2_w(formula)

    and `rho` is a purely *within-composition* quantity: it asks how much of the
    spread inside one composition group the graph resolves. A cluster sample of
    composition groups estimates `rho` directly, while `sigma^2_w(formula)` is
    already known EXACTLY from the 4M census. Multiplying the two transfers the
    sample measurement to the full dataset without ever assuming the sample's
    composition mix matches 4M.
    """
    g = load_geom(tag)
    rows = g["row"]
    c, names, _, _ = load_census(
        columns=["formula", "charge", "spin", "data_id", "family"])
    with np.load(os.path.join(CACHE, "families.npz")) as d:
        y_all = d["y"]
    with open(os.path.join(CACHE, "families.json")) as f:
        fam = json.load(f)

    from .families import mix

    y = y_all[rows]
    qs = mix(c["charge"][rows].astype(np.uint64), c["spin"][rows].astype(np.uint64))
    comp = mix(c["formula"][rows], qs)
    subset = np.array([names.get(int(h), "unknown") for h in c["data_id"][rows]])

    var_y_4m = float(fam["overall"]["formula_x_charge_spin"]["var_total"])
    vw_formula_4m = float(
        fam["overall"]["formula_x_charge_spin"]["var_within_unbiased"])

    ss_c, dof_c, ng_c = _pooled_within(y, comp)
    out = {
        "sample": {"tag": tag, "n_rows": int(len(y)), "n_comp_groups": ng_c},
        "var_y_4M": var_y_4m,
        "var_within_formula_4M": vw_formula_4m,
        "ceiling_formula_4M": 1.0 - vw_formula_4m / var_y_4m,
        "depths": {},
    }
    print(f"[ceiling] cluster sample: {len(y):,} rows in {ng_c:,} composition groups")
    print(f"[ceiling] 4M reference: Var(y) = {var_y_4m:.3f} eV^2, "
          f"within-composition {vw_formula_4m:.3f} eV^2 "
          f"(ceiling {out['ceiling_formula_4M']:.3f})")
    print(f"\n{'grouping':>12}{'groups':>10}{'dof':>9}{'sigma2_w':>11}"
          f"{'rho':>8}{'ceiling':>9}")

    sig_c = ss_c / max(dof_c, 1)
    print(f"{'composition':>12}{ng_c:>10,}{dof_c:>9,}{sig_c:>11.3f}"
          f"{1.0:>8.3f}{out['ceiling_formula_4M']:>9.3f}")

    for depth in (1, 2, 3):
        key = g[f"wl{depth}"]
        ss_w, dof_w, ng_w = _pooled_within(y, key)
        sig_w = ss_w / max(dof_w, 1)
        rho = sig_w / sig_c if sig_c > 0 else float("nan")
        vw_4m = vw_formula_4m * rho
        ceil = 1.0 - vw_4m / var_y_4m
        out["depths"][depth] = {
            "n_wl_groups": ng_w, "dof": dof_w,
            "sigma2_within_sample": sig_w, "rho": rho,
            "var_within_4M": vw_4m, "ceiling_4M": ceil,
            "rmse_within_4M": float(np.sqrt(vw_4m)),
        }
        print(f"{'WL depth ' + str(depth):>12}{ng_w:>10,}{dof_w:>9,}{sig_w:>11.3f}"
              f"{rho:>8.3f}{ceil:>9.3f}")

    # how often does the graph actually split a composition group?
    key3 = g["wl3"]
    _, inv_c = np.unique(comp, return_inverse=True)
    splits = 0
    ncomp = inv_c.max() + 1
    order = np.argsort(inv_c, kind="stable")
    ic = inv_c[order]
    starts = np.flatnonzero(np.r_[True, ic[1:] != ic[:-1]])
    sizes = np.diff(np.r_[starts, len(ic)])
    n_iso, n_dup_rows = 0, 0
    for s, m in zip(starts, sizes):
        members = order[s: s + m]
        k = len(np.unique(key3[members]))
        if k > 1:
            splits += 1
        n_iso += k
        n_dup_rows += m - k
    out["graph_vs_composition"] = {
        "frac_comp_groups_split_by_graph": float(splits / max(ncomp, 1)),
        "mean_graphs_per_composition_group": float(n_iso / max(ncomp, 1)),
        "frac_rows_with_a_wl_twin": float(n_dup_rows / len(y)),
    }
    print(f"\n[ceiling] {100 * splits / max(ncomp, 1):.1f}% of composition groups are "
          f"split by the graph ({n_iso / max(ncomp, 1):.2f} distinct graphs per group)")
    print(f"[ceiling] {100 * n_dup_rows / len(y):.1f}% of sampled rows have a WL twin "
          f"(another row with an identical depth-3 label multiset)")

    # --- CONSISTENCY CHECK: is the WL hash really a *graph* invariant here? ------
    # Same molecule => same bond graph => same WL labels, so the WL grouping must be
    # COARSER than "same molecule (provenance family)" and its within-variance must
    # therefore be LARGER. If it comes out smaller, the hash is separating structures
    # that are the same molecule -- which can only be geometry leaking in through
    # geometry-dependent bond perception, and it means the ceiling above is inflated.
    fam_key = c["family"][rows]
    ss_f, dof_f, ng_f = _pooled_within(y, fam_key)
    sig_f = ss_f / max(dof_f, 1)
    ss_w3, dof_w3, _ = _pooled_within(y, g["wl3"])
    sig_w3 = ss_w3 / max(dof_w3, 1)

    _, inv_f, cnt_f = np.unique(fam_key, return_inverse=True, return_counts=True)
    multi = cnt_f >= 2
    rows_multi = multi[inv_f]
    split = 0
    if rows_multi.any():
        ordf = np.argsort(inv_f, kind="stable")
        iff = inv_f[ordf]
        st_ = np.flatnonzero(np.r_[True, iff[1:] != iff[:-1]])
        sz_ = np.diff(np.r_[st_, len(iff)])
        n_multi = int((sz_ >= 2).sum())
        for s_, m_ in zip(st_, sz_):
            if m_ >= 2 and len(np.unique(g["wl3"][ordf[s_:s_ + m_]])) > 1:
                split += 1
        frac_split = split / max(n_multi, 1)
    else:
        n_multi, frac_split = 0, float("nan")

    out["consistency"] = {
        "sigma2_within_provenance_family_sample": sig_f,
        "sigma2_within_wl3_sample": sig_w3,
        "dof_family": dof_f, "dof_wl3": dof_w3,
        "n_multi_member_families": int(n_multi),
        "frac_families_split_by_wl3": float(frac_split),
        "wl_is_finer_than_molecule": bool(sig_w3 < sig_f),
    }
    print(f"\n[ceiling] CONSISTENCY: within-provenance-family sigma^2 = {sig_f:.3f} "
          f"(dof {dof_f:,}) vs within-WL sigma^2 = {sig_w3:.3f} (dof {dof_w3:,})")
    print(f"[ceiling] {100 * frac_split:.1f}% of the {n_multi:,} multi-member "
          f"provenance families are SPLIT by the depth-3 WL hash")
    if sig_w3 < sig_f:
        print("[ceiling] *** the WL hash is FINER than molecule identity: bond "
              "perception is geometry-dependent, so the ceilings above are "
              "OVERESTIMATES of a true graph descriptor's ceiling ***")

    per = {}
    for s in np.unique(subset):
        m = subset == s
        if m.sum() < 500:
            continue
        ssc, dofc, _ = _pooled_within(y[m], comp[m])
        ssw, dofw, _ = _pooled_within(y[m], key3[m])
        if dofc < 50 or dofw < 50:
            continue
        sc, sw = ssc / dofc, ssw / dofw
        fam_s = fam["per_subset"][s]["formula_x_charge_spin"]
        r = sw / sc if sc > 0 else float("nan")
        vw = fam_s["var_within_unbiased"] * r
        per[s] = {
            "n": int(m.sum()), "rho": r,
            "ceiling_4M": 1.0 - vw / fam["per_subset"][s]["var_total"],
            "ceiling_composition_4M": fam_s["r2_max_unbiased"],
        }
    out["per_subset"] = per
    print(f"\n{'subset':>17}{'n':>8}{'rho':>8}{'comp ceil':>11}{'graph ceil':>12}")
    for s, v in sorted(per.items(), key=lambda kv: -kv[1]["ceiling_4M"]):
        print(f"{s:>17}{v['n']:>8,}{v['rho']:>8.3f}"
              f"{v['ceiling_composition_4M']:>11.3f}{v['ceiling_4M']:>12.3f}")
    _save("ceiling", out)
    return out


# ---------------------------------------------------------------------------
# 2. WL vocabulary growth and OOV
# ---------------------------------------------------------------------------

def vocab(tag="stratified", min_count=2):
    """Vocabulary size vs N, and the OOV rate of a frozen vocabulary.

    ``min_count`` mirrors the production prune (`SparseWLFeaturizer.min_count`): a
    label seen in fewer than that many training molecules never becomes a column, so
    it is not "vocabulary" for any purpose that matters.
    """
    g = load_geom(tag)
    ptr, lab = g["vocab_indptr"], g["vocab_labels"]
    n = len(ptr) - 1
    print(f"[vocab] {n:,} structures, {len(lab):,} (structure, label) incidences")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    sizes = np.unique(np.logspace(2, np.log10(n), 22).astype(int))
    seen_all, seen_twice = set(), set()
    curve_all, curve_kept = [], []
    prev = 0
    for m in sizes:
        for i in perm[prev:m]:
            for L in lab[ptr[i]:ptr[i + 1]]:
                L = int(L)
                if L in seen_all:
                    seen_twice.add(L)
                else:
                    seen_all.add(L)
        curve_all.append(len(seen_all))
        curve_kept.append(len(seen_twice) if min_count == 2 else len(seen_all))
        prev = m

    # power-law fit on the last decade, extrapolated to 4M
    ls, lu = np.log10(sizes[-8:]), np.log10(np.maximum(curve_kept[-8:], 1))
    slope, icept = np.polyfit(ls, lu, 1)
    pred_4m = 10 ** (icept + slope * np.log10(3_986_754))

    # OOV vs the size of the set the vocabulary was frozen on. The single-split
    # number is not actionable -- production freezes on 160k training molecules and
    # will freeze on ~3M at 4M, so what matters is the TREND.
    held = perm[-10_000:] if n > 20_000 else perm[n // 2:]
    pool = perm[: n - len(held)]
    oov_curve = []
    fit_sizes = [s for s in sizes if s <= len(pool)]
    cnt = {}
    prev = 0
    for m in fit_sizes:
        for i in pool[prev:m]:
            for L in lab[ptr[i]:ptr[i + 1]]:
                cnt[int(L)] = cnt.get(int(L), 0) + 1
        prev = m
        frozen = {L for L, k in cnt.items() if k >= min_count}
        tot = miss = 0
        struct = []
        for i in held:
            ls_ = [int(x) for x in lab[ptr[i]:ptr[i + 1]]]
            if not ls_:
                continue
            k_ = sum(1 for L in ls_ if L not in frozen)
            tot += len(ls_); miss += k_
            struct.append(k_ / len(ls_))
        oov_curve.append({
            "n_fit": int(m), "vocab": len(frozen),
            "label_oov_rate": miss / max(tot, 1),
            "mean_structure_oov": float(np.mean(struct)),
            "frac_fully_covered": float(np.mean(np.array(struct) == 0)),
        })

    # per-depth vocabulary shares (only if the scan carried the depth tags)
    per_depth = {}
    if "vocab_depth" in g:
        dep = g["vocab_depth"]
        for dd in (1, 2, 3):
            m = dep == dd
            u = np.unique(lab[m])
            _, c2 = np.unique(lab[m], return_counts=True)
            per_depth[dd] = {
                "unique_labels": int(len(u)),
                "surviving_min_count_2": int(int((c2 >= min_count).sum())),
                "incidences": int(m.sum()),
            }

    out = {
        "n_structures": int(n),
        "sizes": sizes.tolist(),
        "vocab_all": [int(v) for v in curve_all],
        "vocab_min_count_2": [int(v) for v in curve_kept],
        "growth_exponent_last_decade": float(slope),
        "vocab_at_n_min_count_2": int(curve_kept[-1]),
        "vocab_extrapolated_4M": float(pred_4m),
        "per_depth": per_depth,
        "oov_curve": oov_curve,
        "oov": oov_curve[-1] if oov_curve else {},
    }
    print(f"\n{'vocab fit on':>13}{'columns':>11}{'label OOV':>11}"
          f"{'struct OOV':>12}{'fully covered':>15}")
    for r in oov_curve:
        print(f"{r['n_fit']:>13,}{r['vocab']:>11,}{100 * r['label_oov_rate']:>10.1f}%"
              f"{100 * r['mean_structure_oov']:>11.1f}%"
              f"{100 * r['frac_fully_covered']:>14.1f}%")
    if per_depth:
        print(f"\n{'depth':>7}{'unique':>12}{'min_count>=2':>14}{'incidences':>13}")
        for dd, v in per_depth.items():
            print(f"{dd:>7}{v['unique_labels']:>12,}"
                  f"{v['surviving_min_count_2']:>14,}{v['incidences']:>13,}")
    print(f"[vocab] unique labels seen: {curve_all[-1]:,} raw, "
          f"{curve_kept[-1]:,} surviving min_count>={min_count}")
    print(f"[vocab] growth over the last decade: N^{slope:.2f} "
          f"-> extrapolates to {pred_4m:,.0f} columns at 4M")
    _save("vocab", out)
    return out


# ---------------------------------------------------------------------------
# 3. bond perception sensitivity
# ---------------------------------------------------------------------------

def perception(n=6_000, src="train_4M", mults=(1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)):
    """Is the perceived "molecular graph" a graph, or a geometry fingerprint?

    Everything in this repo that says "bond" -- WL labels, the geometry channels, the
    strain reference -- comes from ``build_graph(atoms, cutoff_mult=1.2)``, which
    thresholds interatomic distances. Two conformers of one molecule therefore need
    not receive the same graph, and `ceiling()`'s consistency check says they usually
    do not.

    The metric that matters is **the fraction of provenance families (same molecule,
    different conformer) that the WL hash splits**: it should be 0 for a true graph
    invariant. Everything else here (bonds/atom, ring count) is the mechanism behind
    it. The sweep answers whether any choice of the constant fixes it.
    """
    from fairchem.core.datasets import AseDBDataset

    from .geom_sample import scan_one

    g = load_geom("clusters")
    rows = g["row"]
    c, _, _, _ = load_census(columns=["family"])
    fam = c["family"][rows]
    _, inv, cnt = np.unique(fam, return_inverse=True, return_counts=True)
    # only families with >= 2 members can be split, so only they are informative
    keep = np.flatnonzero(cnt[inv] >= 2)
    rng = np.random.default_rng(0)
    fam_ids = np.unique(inv[keep])
    rng.shuffle(fam_ids)
    sel, take = [], 0
    for f in fam_ids:
        members = keep[inv[keep] == f]
        sel.append(members)
        take += len(members)
        if take >= n:
            break
    sel = np.concatenate(sel)
    famsel = inv[sel]
    print(f"[perception] {len(sel):,} structures from "
          f"{len(np.unique(famsel)):,} multi-member provenance families")

    ds = AseDBDataset({"src": src})
    atoms = [ds.get_atoms(int(g["gidx"][i])) for i in sel]

    # A second, REPRESENTATIVE arm. The family subsample above is drawn from
    # multi-member provenance families, which over-weights the subsets that repeat
    # molecules (metal complexes especially) -- fine for the split metric, wrong for
    # quoting how many bonds or rings a typical structure has.
    gs = load_geom("stratified")
    rep_idx = np.random.default_rng(1).choice(
        len(gs["gidx"]), size=min(2_000, len(gs["gidx"])), replace=False)
    rep_atoms = [ds.get_atoms(int(gs["gidx"][i])) for i in rep_idx]

    out_rows = {}
    print(f"\n{'mult':>6}{'bonds/atom':>12}{'med rings':>11}{'frac acyclic':>13}"
          f"{'families split':>16}{'distinct graphs':>17}")
    for mlt in mults:
        recs = [scan_one(a, mlt)[0] for a in atoms]
        wl3 = np.array([r["wl3"] for r in recs], np.uint64)
        split = tot = 0
        for f in np.unique(famsel):
            m = famsel == f
            if m.sum() >= 2:
                tot += 1
                split += len(np.unique(wl3[m])) > 1
        rep = [scan_one(a, mlt)[0] for a in rep_atoms]
        rings = np.array([r["n_rings"] for r in rep], float)
        bonds = np.array([r["n_bonds"] for r in rep], float)
        na = np.array([r["n_atoms"] for r in rep], float)
        rec = {
            "bonds_per_atom": float((bonds / na).mean()),
            "median_rings": float(np.median(rings)),
            "frac_acyclic": float(np.mean(rings == 0)),
            "frac_families_split": float(split / max(tot, 1)),
            "n_distinct_graphs": int(len(np.unique(wl3))),
            "n_structures": int(len(recs)),
            "n_representative": int(len(rep)),
        }
        out_rows[str(mlt)] = rec
        print(f"{mlt:>6}{rec['bonds_per_atom']:>12.3f}{rec['median_rings']:>11.0f}"
              f"{rec['frac_acyclic']:>13.2f}{100 * rec['frac_families_split']:>15.1f}%"
              f"{rec['n_distinct_graphs']:>17,}")
    best = min(out_rows, key=lambda k: out_rows[k]["frac_families_split"])
    print(f"\n[perception] best multiplier for graph stability: {best} "
          f"({100 * out_rows[best]['frac_families_split']:.1f}% of same-molecule "
          f"families still split)")
    out = {"n_sampled": int(len(sel)),
           "n_families": int(len(np.unique(famsel))),
           "by_mult": out_rows, "best_mult": best}
    _save("perception", out)
    return out


# ---------------------------------------------------------------------------
# 4 + 5. embeddings: variogram, distance concentration, cross-category
# ---------------------------------------------------------------------------

def _embeddings(n, src, seed, dim, geom_channels):
    """Build (and cache) production WL and geometry embeddings on a subsample.

    Cached because `variogram` and `cross` both need them and featurisation is the
    expensive half of step 5.
    """
    from fairchem.core.datasets import AseDBDataset

    cache = os.path.join(CACHE, f"emb_{n}_{seed}_{dim}.npz")
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as d:
            print(f"[emb] reusing {cache}")
            return ({"wl": d["wl"], "geom": d["geom"]}, d["y"], d["subset"])

    from wl_gp2scale.geometry_features import SparseGeometryFeaturizer
    from wl_gp2scale.reduce import SparsePLS
    from wl_gp2scale.wl_features import SparseWLFeaturizer

    g = load_geom("stratified")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(g["gidx"]), size=min(n, len(g["gidx"])),
                             replace=False))
    with np.load(os.path.join(CACHE, "families.npz")) as d:
        y = d["y"][g["row"][sel]]
    subset = g["subset"][sel]

    ds = AseDBDataset({"src": src})
    t0 = time.perf_counter()
    atoms = [ds.get_atoms(int(gi)) for gi in g["gidx"][sel]]
    print(f"[emb] materialised {len(atoms):,} structures in "
          f"{time.perf_counter() - t0:.0f}s")

    emb = {}
    wl = SparseWLFeaturizer(min_count=2)
    Xw = wl.fit_transform(atoms) if hasattr(wl, "fit_transform") else None
    if Xw is None:
        wl.fit(atoms)
        Xw = wl.transform(atoms)
    print(f"[emb] WL: {Xw.shape[1]:,} columns")
    emb["wl"] = SparsePLS(n_components=dim).fit_transform(Xw, y)

    gf = SparseGeometryFeaturizer(channels=tuple(geom_channels))
    gf.fit(atoms)
    Xg = gf.transform(atoms)
    print(f"[emb] geometry ({'+'.join(geom_channels)}): {Xg.shape[1]:,} columns")
    import scipy.sparse as sp

    emb["geom"] = SparsePLS(n_components=dim).fit_transform(sp.csr_matrix(Xg), y)
    np.savez_compressed(cache, wl=emb["wl"], geom=emb["geom"], y=y, subset=subset)
    print(f"[emb] wrote {cache}")
    return emb, y, subset


def _variogram(Z, y, n_pairs=400_000, n_lag=24, seed=0):
    """Empirical gamma(h) from random pairs: 0.5*(y_i - y_j)^2 binned by |z_i - z_j|."""
    rng = np.random.default_rng(seed)
    n = len(y)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    h = np.linalg.norm(Z[i] - Z[j], axis=1)
    gam = 0.5 * (y[i] - y[j]) ** 2
    hi = np.percentile(h, 99)
    edges = np.linspace(0, hi, n_lag + 1)
    b = np.clip(np.digitize(h, edges) - 1, 0, n_lag - 1)
    cnt = np.bincount(b, minlength=n_lag)
    gmean = np.bincount(b, weights=gam, minlength=n_lag) / np.maximum(cnt, 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    sill = float(np.var(y))
    # effective range: first lag reaching 95% of the sill
    rng_idx = np.argmax(gmean >= 0.95 * sill) if np.any(gmean >= 0.95 * sill) else -1
    return {
        "lag": ctr.tolist(), "gamma": gmean.tolist(), "count": cnt.tolist(),
        "sill": sill, "nugget": float(gmean[0]),
        "range": float(ctr[rng_idx]) if rng_idx >= 0 else float("nan"),
        "h_median": float(np.median(h)),
        "h_p01": float(np.percentile(h, 1)),
        "dist_hist": np.histogram(h, bins=60, range=(0, hi))[0].tolist(),
        "dist_edges": np.linspace(0, hi, 61).tolist(),
    }


def variogram(n=20_000, src="train_4M", seed=0, dim=10,
              geom_channels=("rdf", "angle", "torsion", "elec")):
    emb, y, subset = _embeddings(n, src, seed, dim, geom_channels)
    out = {"n": int(len(y)), "dim": dim, "channels": {}}
    for ch, Z in emb.items():
        rec = {"pooled": _variogram(Z, y), "per_subset": {}}
        for s in np.unique(subset):
            m = subset == s
            if m.sum() >= 1500:
                rec["per_subset"][s] = _variogram(Z[m], y[m], n_pairs=200_000)
        out["channels"][ch] = rec
        p = rec["pooled"]
        print(f"\n[variogram] {ch}: nugget {p['nugget']:.2f}  sill {p['sill']:.2f}  "
              f"nugget/sill {p['nugget'] / p['sill']:.2f}  range {p['range']:.3f}  "
              f"median pair distance {p['h_median']:.3f}")
        print(f"{'subset':>17}{'sill':>9}{'nugget':>9}{'n/s':>7}{'range':>8}{'med h':>8}")
        for s, r in sorted(rec["per_subset"].items(),
                           key=lambda kv: -kv[1]["sill"]):
            print(f"{s:>17}{r['sill']:>9.1f}{r['nugget']:>9.1f}"
                  f"{r['nugget'] / r['sill']:>7.2f}{r['range']:>8.3f}"
                  f"{r['h_median']:>8.3f}")
    _save("variogram", out)
    return out


def cross(n=20_000, src="train_4M", seed=0, dim=10, k=10,
          geom_channels=("rdf", "angle", "torsion", "elec")):
    """Does the block-sparse kernel throw away real cross-category information?

    For each structure, find its k nearest neighbours in the embedding and ask (a) how
    often they sit in a different ``data_id``, and (b) whether those cross-category
    neighbours are as informative about `y` as the within-category ones. If they are
    common AND informative, zeroing the cross blocks is discarding signal, not just
    flops.
    """
    from scipy.spatial import cKDTree

    emb, y, subset = _embeddings(n, src, seed, dim, geom_channels)
    names = sorted(set(subset.tolist()))
    idx = {s: i for i, s in enumerate(names)}
    code = np.array([idx[s] for s in subset])
    out = {"n": int(len(y)), "k": k, "subsets": names, "channels": {}}

    for ch, Z in emb.items():
        Z = np.ascontiguousarray(Z, dtype=float)
        tree = cKDTree(Z)
        _, nb = tree.query(Z, k=k + 1)
        nb = nb[:, 1:]                                    # drop self
        same = code[nb] == code[:, None]
        M = np.zeros((len(names), len(names)))
        for a in range(len(names)):
            m = code == a
            if not m.any():
                continue
            cnt = np.bincount(code[nb[m]].ravel(), minlength=len(names))
            M[a] = cnt / cnt.sum()

        # is a cross-category neighbour worth as much as a within-category one?
        yn = y[nb]
        err_same, err_cross = [], []
        for r in range(len(y)):
            s_ = same[r]
            if s_.any():
                err_same.append((y[r] - yn[r][s_].mean()) ** 2)
            if (~s_).any():
                err_cross.append((y[r] - yn[r][~s_].mean()) ** 2)
        vy = float(np.var(y))
        rec = {
            "frac_neighbours_cross_category": float(1.0 - same.mean()),
            "frac_rows_with_any_cross_neighbour": float((~same).any(axis=1).mean()),
            "neighbour_matrix": M.tolist(),
            "r2_from_same_category_neighbours": float(1 - np.mean(err_same) / vy),
            "r2_from_cross_category_neighbours": float(1 - np.mean(err_cross) / vy),
            "n_rows_with_cross": int(len(err_cross)),
        }
        out["channels"][ch] = rec
        print(f"\n[cross] {ch}: {100 * rec['frac_neighbours_cross_category']:.1f}% of "
              f"{k}-NN links cross a data_id boundary; "
              f"{100 * rec['frac_rows_with_any_cross_neighbour']:.1f}% of rows have "
              f"at least one")
        print(f"[cross]   predicting y from the neighbour mean: "
              f"same-category R² {rec['r2_from_same_category_neighbours']:.3f}  "
              f"cross-category R² {rec['r2_from_cross_category_neighbours']:.3f}")
    _save("cross", out)
    return out


MODES = {"ceiling": ceiling, "vocab": vocab, "perception": perception,
         "variogram": variogram, "cross": cross}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=list(MODES) + ["all"])
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--src", default="train_4M")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    if a.mode in ("ceiling", "all"):
        ceiling()
    if a.mode in ("vocab", "all"):
        vocab()
    if a.mode in ("perception", "all"):
        perception()
    if a.mode in ("variogram", "all"):
        variogram(n=a.n, src=a.src, seed=a.seed, dim=a.dim)
    if a.mode in ("cross", "all"):
        cross(n=a.n, src=a.src, seed=a.seed, dim=a.dim, k=a.k)


if __name__ == "__main__":
    main()
