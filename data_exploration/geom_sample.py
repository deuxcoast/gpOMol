"""
geom_sample.py  (data_exploration)
==================================
PASS B -- the structural sample. Everything here needs *positions* and a bond
perception pass, which the 4M census deliberately skipped.

Two samples, drawn for two different jobs, because one sample cannot do both:

``stratified``
    Proportional-with-floor across the 10 ``data_id`` subsets. This is the
    *representative* sample: bond lengths, coordination, fragments, rings,
    compactness, per-element charges (figures F13-F15), and the WL vocabulary
    growth curve (F16).

``clusters``
    A **cluster sample of (formula, charge, spin) groups** with 2 <= size <= cap,
    taking every member of each drawn group. This one exists to measure the
    **molecular-graph ceiling** -- the number `REPORT.md` §6 left bracketed at
    [0.53, 0.92].

Why the second sample has to be a cluster sample: the ceiling is a *within-group*
variance, so it is estimable only from groups with two or more members in hand. A
uniform 200k draw (5% of 4M) shatters the composition groups -- a mean-3.2 group
survives with ~1.16 members -- and would leave almost no within-group pairs. Drawing
whole groups keeps the pairs. Since two structures with the same WL labels
necessarily share a chemical formula, WL groups are a refinement of formula groups,
so **every WL duplicate that exists lives inside some formula group** and this
sample can see all of them.

Per structure we record the bond graph summary, the geometry summary, and the
**WL multiset hash at depths 1-3** -- the hash of the sorted multiset of per-atom WL
labels, which is exactly the object the production descriptor turns into counts. Two
structures sharing a depth-3 hash are *literally indistinguishable* to
`SparseWLFeaturizer`, so grouping on it gives the ceiling for the descriptor we
actually ship, not for an idealised graph invariant.

    python -m data_exploration.geom_sample --sample both --n 200000 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

from .census import CACHE, load_census

# same bond perception as the production descriptor, so "bonded" means one thing
from wl_gp2scale.wl_features import build_graph, wl_labels_per_depth

WL_DEPTH = 3
BOND_BINS = np.linspace(0.6, 3.2, 131)          # Angstrom
ANGLE_BINS = np.linspace(0.0, 180.0, 91)        # degrees
CHARGE_BINS = np.linspace(-2.0, 2.0, 161)       # Lowdin e
MAX_Z = 119
MAX_COORD = 13

COLUMNS = {
    "gidx": "int64",
    "row": "int64",            # index into the census table
    "n_atoms": "int16",
    "n_bonds": "int32",
    "n_frag": "int16",         # disconnected components: 1 molecule or a cluster?
    "n_rings": "int16",        # cyclomatic number  E - V + components
    "n_rot": "int16",          # rotatable single bonds (acyclic, both ends deg>=2)
    "rg": "float32",
    "rg_norm": "float32",      # R_g / n^(1/3): compactness, size-free
    "dmax": "float32",         # longest intramolecular distance
    "min_inter": "float32",    # closest contact BETWEEN fragments (nan if 1 fragment)
    "frag_max": "int16",       # atoms in the largest fragment
    "q_span": "float32",       # max - min Lowdin charge
    "wl1": "uint64",
    "wl2": "uint64",
    "wl3": "uint64",
}


def _blake(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def _components(adj):
    """Connected components of the bond graph -> (labels, n_components)."""
    n = len(adj)
    comp = np.full(n, -1, dtype=np.int32)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if comp[j] < 0:
                    comp[j] = c
                    stack.append(j)
        c += 1
    return comp, c


def _bridges(adj):
    """Bridge edges (edges not on any cycle), iterative Tarjan.

    A rotatable bond is a single acyclic bond with a rotatable environment, so the
    ring membership test is exactly "is this edge a bridge".
    """
    n = len(adj)
    disc = np.full(n, -1, np.int32)
    low = np.zeros(n, np.int32)
    parent = np.full(n, -1, np.int32)
    out = set()
    timer = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        stack = [(root, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if disc[w] < 0:
                    parent[w] = v
                    disc[w] = low[w] = timer
                    timer += 1
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
                if w != parent[v]:
                    low[v] = min(low[v], disc[w])
            if not advanced:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        out.add((min(u, v), max(u, v)))
    return out


def scan_one(atoms, cutoff_mult=1.2):
    """One structure -> (row dict, distribution updates). ONE bond-perception pass."""
    P = np.asarray(atoms.get_positions(), float)
    Z = np.asarray(atoms.get_atomic_numbers())
    n = len(Z)
    adj, labels = build_graph(atoms, cutoff_mult)

    comp, n_frag = _components(adj)
    n_bonds = sum(len(a) for a in adj) // 2
    n_rings = n_bonds - n + n_frag

    bridges = _bridges(adj) if n_bonds else set()
    n_rot = sum(
        1 for (i, j) in bridges
        if len(adj[i]) >= 2 and len(adj[j]) >= 2
    )

    ctr = P - P.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum(ctr * ctr, axis=1))))

    # pairwise distances: n <= 350 so the full matrix is cheap and exact
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    dmax = float(d.max()) if n > 1 else 0.0
    min_inter = float("nan")
    if n_frag > 1:
        cross = comp[:, None] != comp[None, :]
        min_inter = float(d[cross].min())
    frag_max = int(np.bincount(comp).max()) if n else 0

    # ---- distributions (returned as sparse updates, merged by the parent) ----
    bond_upd, coord_upd, ang_upd, q_upd = Counter(), Counter(), Counter(), Counter()
    for i, nbrs in enumerate(adj):
        coord_upd[(int(Z[i]), min(len(nbrs), MAX_COORD - 1))] += 1
        for j in nbrs:
            if j > i:
                a, b = (int(Z[i]), int(Z[j])) if Z[i] <= Z[j] else (int(Z[j]), int(Z[i]))
                bi = int(np.searchsorted(BOND_BINS, d[i, j]) - 1)
                if 0 <= bi < len(BOND_BINS) - 1:
                    bond_upd[(a, b, bi)] += 1
        k = len(nbrs)
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                u, v = P[nbrs[a_]] - P[i], P[nbrs[b_]] - P[i]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 1e-9 and nv > 1e-9:
                    th = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)))
                    ai = int(np.searchsorted(ANGLE_BINS, th) - 1)
                    if 0 <= ai < len(ANGLE_BINS) - 1:
                        ang_upd[(int(Z[i]), min(k, MAX_COORD - 1), ai)] += 1

    q_span = float("nan")
    lq = atoms.info.get("lowdin_charges")
    if lq is not None:
        lq = np.asarray(lq, float)
        if lq.size == n and np.all(np.isfinite(lq)):
            q_span = float(lq.max() - lq.min())
            qi = np.clip(np.searchsorted(CHARGE_BINS, lq) - 1, 0, len(CHARGE_BINS) - 2)
            for z, b in zip(Z, qi):
                q_upd[(int(z), int(b))] += 1

    # ---- WL multiset hashes: what the production descriptor actually sees ----
    per_depth = wl_labels_per_depth(adj, labels, WL_DEPTH)
    wl = [_blake(";".join(sorted(per_depth[dd]))) for dd in (1, 2, 3)]

    row = dict(
        n_atoms=n, n_bonds=n_bonds, n_frag=n_frag, n_rings=n_rings, n_rot=n_rot,
        rg=rg, rg_norm=rg / max(n, 1) ** (1 / 3), dmax=dmax, min_inter=min_inter,
        frag_max=frag_max, q_span=q_span, wl1=wl[0], wl2=wl[1], wl3=wl[2],
    )
    # per-structure unique WL labels (for the vocabulary growth curve)
    uniq = {dd: sorted({_blake(f"{dd}:{L}") for L in per_depth[dd]}) for dd in (1, 2, 3)}
    return row, (bond_upd, coord_upd, ang_upd, q_upd), uniq


# ------------------------------- sampling ----------------------------------


def stratified_rows(subset, n, seed=0, floor=15_000):
    """Proportional across subsets, but never fewer than ``floor`` (or all of it)
    from a small subset -- otherwise SPICE/OrbNet are unresolvable in every panel."""
    rng = np.random.default_rng(seed)
    names, counts = np.unique(subset, return_counts=True)
    take = {}
    # first pass: everyone gets min(floor, their size)
    for nm, cnt in zip(names, counts):
        take[nm] = int(min(floor, cnt))
    left = n - sum(take.values())
    if left > 0:
        room = {nm: int(cnt - take[nm]) for nm, cnt in zip(names, counts)}
        tot = sum(room.values())
        for nm in names:
            take[nm] += int(round(left * room[nm] / tot)) if tot else 0
    picks = []
    for nm, cnt in zip(names, counts):
        idx = np.where(subset == nm)[0]
        picks.append(rng.choice(idx, size=min(int(take[nm]), len(idx)), replace=False))
    out = np.sort(np.concatenate(picks))
    print(f"[geom] stratified sample {len(out):,}: "
          + ", ".join(f"{nm} {int((subset[out] == nm).sum()):,}" for nm in names))
    return out


def cluster_rows(key, n, seed=0, cap=32, min_size=2):
    """Draw whole (formula, charge, spin) groups until ~n rows are collected.

    Groups larger than ``cap`` are subsampled to ``cap`` members: the within-group
    variance stays unbiased (a random subset of a group estimates that group's
    variance) while one 9,835-member group cannot eat the entire budget.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(ks)])
    big = np.flatnonzero(sizes >= min_size)
    perm = rng.permutation(big)
    picks, total = [], 0
    for g in perm:
        members = order[starts[g]: starts[g] + sizes[g]]
        if len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        picks.append(members)
        total += len(members)
        if total >= n:
            break
    out = np.sort(np.concatenate(picks))
    print(f"[geom] cluster sample {len(out):,} rows from {len(picks):,} "
          f"(formula, charge, spin) groups of size >= {min_size} "
          f"(cap {cap}); {len(big):,} such groups exist")
    return out


# ------------------------------- scanning ----------------------------------


def _worker(args):
    chunk_id, gidx, rows, src, cutoff_mult, keep_vocab = args
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    cols = {k: [] for k in COLUMNS}
    bond, coord, ang, q = Counter(), Counter(), Counter(), Counter()
    vocab_ptr, vocab_lab, vocab_dep = [0], [], []
    for gi, ri in zip(gidx, rows):
        row, upd, uniq = scan_one(ds.get_atoms(int(gi)), cutoff_mult)
        row["gidx"], row["row"] = int(gi), int(ri)
        for k in COLUMNS:
            cols[k].append(row[k])
        bond.update(upd[0]); coord.update(upd[1]); ang.update(upd[2]); q.update(upd[3])
        if keep_vocab:
            # depth is carried alongside, not folded into the hash: "is it depth 3
            # that explodes the vocabulary" is the actionable question and it has to
            # stay answerable from the cache.
            for dd in (1, 2, 3):
                vocab_lab.extend(uniq[dd])
                vocab_dep.extend([dd] * len(uniq[dd]))
            vocab_ptr.append(len(vocab_lab))
    out = {k: np.asarray(v, dtype=COLUMNS[k]) for k, v in cols.items()}
    return (chunk_id, out, (dict(bond), dict(coord), dict(ang), dict(q)),
            (np.asarray(vocab_ptr, np.int64), np.asarray(vocab_lab, np.uint64),
             np.asarray(vocab_dep, np.uint8)))


def scan(gidx, rows, src, workers=8, cutoff_mult=1.2, vocab_rows=60_000, chunk=2_000):
    jobs = []
    for c, s in enumerate(range(0, len(gidx), chunk)):
        e = min(len(gidx), s + chunk)
        jobs.append((c, gidx[s:e], rows[s:e], src, cutoff_mult, s < vocab_rows))
    print(f"[geom] scanning {len(gidx):,} structures in {len(jobs)} chunks "
          f"on {workers} workers")

    parts, bond, coord, ang, q = {}, Counter(), Counter(), Counter(), Counter()
    vparts = {}
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for k, (cid, cols, dists, vv) in enumerate(pool.imap_unordered(_worker, jobs)):
            parts[cid] = cols
            bond.update(dists[0]); coord.update(dists[1])
            ang.update(dists[2]); q.update(dists[3])
            if len(vv[1]):
                vparts[cid] = vv
            done = sum(len(v["gidx"]) for v in parts.values())
            el = time.time() - t0
            print(f"[geom]   {done:,}/{len(gidx):,}  {el:6.0f}s  "
                  f"eta {el / max(done, 1) * (len(gidx) - done):6.0f}s", flush=True)

    order = sorted(parts)
    out = {k: np.concatenate([parts[c][k] for c in order]) for k in COLUMNS}
    o = np.argsort(out["row"], kind="stable")
    out = {k: v[o] for k, v in out.items()}

    # vocabulary: ragged per-structure unique label lists, concatenated
    vlab, vdep, vptr = [], [], [0]
    for c in sorted(vparts):
        ptr, lab, dep = vparts[c]
        base = len(vlab)
        vlab.extend(lab.tolist())
        vdep.extend(dep.tolist())
        vptr.extend((ptr[1:] + base).tolist())
    out["vocab_indptr"] = np.asarray(vptr, np.int64)
    out["vocab_labels"] = np.asarray(vlab, np.uint64)
    out["vocab_depth"] = np.asarray(vdep, np.uint8)

    def _pack(counter, ncol):
        if not counter:
            return np.zeros((0, ncol + 1), np.int64)
        return np.array([list(k) + [v] for k, v in counter.items()], np.int64)

    out["bond_hist"] = _pack(bond, 3)      # (Za, Zb, bin, count)
    out["coord_hist"] = _pack(coord, 2)    # (Z, coordination, count)
    out["angle_hist"] = _pack(ang, 3)      # (Z_centre, coordination, bin, count)
    out["charge_hist"] = _pack(q, 2)       # (Z, bin, count)
    print(f"[geom] scan done in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / len(gidx) * 1e3:.1f} ms/mol/worker-set)")
    return out


# --------------------------------- driver ----------------------------------


def out_path(tag, n):
    return os.path.join(CACHE, f"geom_{tag}_{n}.npz")


def run(sample="both", n=200_000, src="train_4M", seed=0, workers=8,
        cutoff_mult=1.2, cap=32, floor=15_000, vocab_rows=60_000, force=False):
    from .families import mix

    cols = ["gidx", "data_id", "formula", "charge", "spin"]
    c, names, _, _ = load_census(columns=cols)
    subset = np.array([names.get(int(h), "unknown") for h in c["data_id"]])

    todo = ["stratified", "clusters"] if sample == "both" else [sample]
    for tag in todo:
        if tag == "stratified":
            rows = stratified_rows(subset, n, seed=seed, floor=floor)
        else:
            qs = mix(c["charge"].astype(np.uint64), c["spin"].astype(np.uint64))
            rows = cluster_rows(mix(c["formula"], qs), n, seed=seed, cap=cap)
        path = out_path(tag, len(rows))
        if os.path.exists(path) and not force:
            print(f"[geom] {path} exists; skipping (use --force)")
            continue
        res = scan(c["gidx"][rows], rows, src, workers=workers,
                   cutoff_mult=cutoff_mult,
                   vocab_rows=vocab_rows if tag == "stratified" else 0)
        res["subset"] = subset[rows]
        np.savez_compressed(path, **res)
        print(f"[geom] wrote {path}")


def load(tag, n=None):
    """Load a scan by tag; with ``n=None`` take the LARGEST cached scan.

    Largest by parsed row count, not by filename -- ``sorted()`` would put the 3k
    smoke-test scan after the 200k one ('3' > '1' lexicographically) and silently
    serve 3,001 rows to every downstream analysis.
    """
    import glob
    import re

    if n:
        files = [out_path(tag, n)] if os.path.exists(out_path(tag, n)) else []
    else:
        files = glob.glob(os.path.join(CACHE, f"geom_{tag}_*.npz"))
        files.sort(key=lambda f: int(re.search(r"_(\d+)\.npz$", f).group(1)))
    if not files:
        raise SystemExit(f"no geom scan for tag {tag!r}; run "
                         f"`python -m data_exploration.geom_sample`")
    with np.load(files[-1], allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", default="both",
                   choices=["both", "stratified", "clusters"])
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--src", default="train_4M")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cutoff-mult", type=float, default=1.2)
    p.add_argument("--cap", type=int, default=32,
                   help="max members drawn from one composition group")
    p.add_argument("--floor", type=int, default=15_000,
                   help="min rows per data_id in the stratified sample")
    p.add_argument("--vocab-rows", type=int, default=60_000,
                   help="rows whose per-structure WL label sets are kept (F16)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(sample=a.sample, n=a.n, src=a.src, seed=a.seed, workers=a.workers,
        cutoff_mult=a.cutoff_mult, cap=a.cap, floor=a.floor,
        vocab_rows=a.vocab_rows, force=a.force)


if __name__ == "__main__":
    main()
