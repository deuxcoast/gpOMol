"""
perception_rdkit.py  (data_exploration)
=======================================
Would RDKit's `rdDetermineBonds` give a better bond graph than ASE's covalent-radius
threshold?

`REPORT.md` §11 showed the ASE-perceived graph is partly a geometry fingerprint: at
the production 1.2x multiplier, **88% of same-molecule conformer families receive
different WL hashes**. The natural question is whether a chemistry-aware perceiver
fixes that. RDKit offers three different things under one name, and they are worth
separating because only one of them is a genuinely different *kind* of information:

``DetermineConnectivity(useVdw=True, covFactor=)``
    the same idea as ASE -- a scaled covalent-radius threshold. Not a new method,
    just a different default constant (1.3 vs our 1.2).
``DetermineConnectivity()``  (default, "connect-the-dots")
    distance-based with valence-aware pruning. Still geometric, slightly smarter.
``DetermineConnectivity(useHueckel=True)``
    extended Hueckel overlap populations -- an electronic-structure criterion, not a
    distance threshold. Genuinely different.
``DetermineBonds()``
    connectivity + **bond orders** via xyz2mol. The bond orders are the real prize:
    they are information a distance threshold can NEVER provide, they make single /
    aromatic / double / triple distinguishable as EDGE LABELS, and they are
    conformer-invariant by construction.

So this module measures, on the same rows as `descriptors.py perception`:

1. **the split rate** -- what fraction of same-molecule provenance families the
   method assigns different graphs to (0 for a true invariant);
2. the **failure rate**, per subset, because xyz2mol assumes organic valences and
   train_4M is 20% metal complexes and 17% open-shell;
3. **cost per molecule**, because whatever wins has to run 4M times.

Bond orders enter the WL hash as edge labels, so ``rdkit_bonds`` and
``rdkit_bonds_topology`` isolate whether any gain comes from the orders or merely
from the connectivity underneath them.

    python -m data_exploration.perception_rdkit --n 2000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import time
import warnings

import numpy as np

from .census import CACHE, load_census
from .geom_sample import load as load_geom

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _quiet():
    """Silence RDKit.

    Two separate channels: the Python-visible RDKit logger, and the C++ ``!!!
    Warning !!!`` stream that xyz2mol writes straight to fd 2 (it fires on every
    O-H bond in the dataset, which would otherwise bury the results). Redirect the
    file descriptor itself, not ``sys.stderr``.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(fd, 2)
        os.close(devnull)
        os.close(fd)


def _blake(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def wl_hash(adj, node_labels, edge_labels=None, depth=3):
    """WL multiset hash, optionally with EDGE labels (bond orders).

    Same construction as `wl_features.wl_labels_per_depth` so the numbers are
    comparable, plus the edge-label slot that ASE perception cannot fill.
    """
    n = len(node_labels)
    labels = [str(l) for l in node_labels]
    for _ in range(depth):
        nxt = []
        for i in range(n):
            if edge_labels is None:
                nbr = sorted(labels[j] for j in adj[i])
            else:
                nbr = sorted(f"{edge_labels[(min(i, j), max(i, j))]}~{labels[j]}"
                             for j in adj[i])
            nxt.append(hashlib.blake2b(
                (labels[i] + "|" + ",".join(nbr)).encode(), digest_size=8).hexdigest())
        labels = nxt
    return _blake(";".join(sorted(labels)))


# ------------------------------- perceivers --------------------------------


def _rw_mol(Z, pos):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    m = Chem.RWMol()
    conf = Chem.Conformer(len(Z))
    for i, z in enumerate(Z):
        m.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, Point3D(*[float(x) for x in pos[i]]))
    mol = m.GetMol()
    mol.AddConformer(conf)
    return mol


def _from_rdkit(mol, with_orders):
    """RDKit mol -> (adjacency, node labels, edge labels or None)."""
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    edges = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
        edges[(min(i, j), max(i, j))] = (
            str(b.GetBondType()) if with_orders else "1")
    return ([sorted(a) for a in adj],
            [a.GetAtomicNum() for a in mol.GetAtoms()],
            edges if with_orders else None)


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    """Abort one perception that runs too long.

    xyz2mol's bond-order search is combinatorial and on some structures (metal
    complexes especially) it does not come back in any useful time. A pipeline that
    has to run 4M times cannot wait, so an over-running molecule is a FAILURE and is
    counted as one.
    """
    import signal

    def _fire(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _pymatgen_graph(method, Z, pos, vacuum=15.0):
    """Coordination from ``pymatgen.analysis.local_env``.

    These algorithms come from materials science, where metal coordination is the
    normal case rather than the awkward one -- which is why they are worth trying on
    `metal_complexes`, the subset every organic-chemistry perceiver refuses (xyz2mol
    fails 98% of it).

    All three are run through a periodic box: the molecule is dropped into a cubic
    lattice padded by ``vacuum`` on every side and any edge to a periodic image is
    discarded. The padding must exceed the algorithm's own search cutoff (CrystalNN
    7 A, VoronoiNN 13 A) or a periodic image can shield a genuine neighbour -- 15 A
    clears both.

    ``CrystalNN``/``VoronoiNN`` are periodic-only by declaration
    (``molecules_allowed = False``). ``EconNN`` advertises ``molecules_allowed =
    True``, but in pymatgen 2026.5.4 its ``get_nn_info`` still calls ``_get_image``,
    which reads ``frac_coords`` off the neighbour -- an attribute a molecule
    neighbour does not have, so the molecule path raises. The box is therefore the
    only route for all three.
    """
    import numpy as np
    from pymatgen.analysis import local_env as le
    from pymatgen.core import Lattice, Structure

    species = [int(z) for z in Z]
    span = pos.max(axis=0) - pos.min(axis=0)
    a = float(span.max() + 2 * vacuum)
    shifted = pos - pos.min(axis=0) + vacuum
    st = Structure(Lattice.cubic(a), species, shifted, coords_are_cartesian=True)
    # VoronoiNN's default tol=0 keeps EVERY tessellation face, which is a
    # space-filling neighbour list rather than a bond graph -- on a 47-atom Pt
    # complex it returns 254 "bonds" and gives Pt a coordination of 12. tol is the
    # minimum solid-angle weight relative to the largest face; 0.5 is the usual
    # "significant face" threshold.
    nn = {"pymatgen_crystalnn": le.CrystalNN,
          "pymatgen_voronoi": lambda: le.VoronoiNN(tol=0.5),
          "pymatgen_econn": le.EconNN}[method]()
    adj = [[] for _ in species]
    for i in range(len(species)):
        for d in nn.get_nn_info(st, i):
            # a nonzero image is a neighbour of a PERIODIC COPY, not of the molecule
            if np.any(np.abs(np.asarray(d["image"], dtype=float)) > 1e-9):
                continue
            adj[i].append(int(d["site_index"]))
    return [sorted(set(x)) for x in adj]


def perceive(method, atoms, charge):
    """-> (adjacency, node labels, edge labels or None). Raises on failure."""
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    if method.startswith("ase_"):
        from wl_gp2scale.wl_features import build_graph

        adj, labels = build_graph(atoms, float(method.split("_")[1]))
        return adj, labels, None

    if method.startswith("pymatgen_"):
        return _pymatgen_graph(method, Z, pos), [int(z) for z in Z], None

    from rdkit.Chem import rdDetermineBonds as rd

    mol = _rw_mol(Z, pos)
    if method == "rdkit_ctd":
        rd.DetermineConnectivity(mol)
    elif method == "rdkit_vdw":
        rd.DetermineConnectivity(mol, useVdw=True, covFactor=1.3)
    elif method == "rdkit_hueckel":
        rd.DetermineConnectivity(mol, useHueckel=True, charge=int(charge))
    elif method in ("rdkit_bonds", "rdkit_bonds_topology"):
        rd.DetermineBonds(mol, charge=int(charge), embedChiral=False)
    else:
        raise ValueError(method)
    return _from_rdkit(mol, with_orders=(method == "rdkit_bonds"))


METHODS = ["ase_1.0", "ase_1.1", "ase_1.2", "rdkit_ctd", "rdkit_vdw",
           "rdkit_hueckel", "rdkit_bonds", "rdkit_bonds_topology",
           "pymatgen_crystalnn", "pymatgen_voronoi", "pymatgen_econn"]


# --------------------------------- driver ----------------------------------


def run(n=2000, src="train_4M", methods=None, depth=3, seed=0, tag="",
        timeout_s=2.0, subsets=None):
    from fairchem.core.datasets import AseDBDataset

    methods = methods or METHODS
    g = load_geom("clusters")
    rows = g["row"]
    c, names, _, _ = load_census(columns=["family", "charge", "data_id"])
    fam = c["family"][rows]
    _, inv, cnt = np.unique(fam, return_inverse=True, return_counts=True)
    eligible = cnt[inv] >= 2
    if subsets:
        want = set(subsets)
        row_sub = np.array([names.get(int(h), "unknown") for h in c["data_id"][rows]])
        eligible &= np.isin(row_sub, list(want))
        print(f"[rdkit] restricted to {sorted(want)}: "
              f"{eligible.sum():,} rows in multi-member families")
    keep = np.flatnonzero(eligible)

    rng = np.random.default_rng(seed)
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
    charges = c["charge"][rows][sel]
    subset = np.array([names.get(int(h), "unknown") for h in c["data_id"][rows][sel]])
    print(f"[rdkit] {len(sel):,} structures from {len(np.unique(famsel)):,} "
          f"multi-member provenance families")

    ds = AseDBDataset({"src": src})
    atoms = [ds.get_atoms(int(g["gidx"][i])) for i in sel]

    out = {}
    print(f"\n{'method':>22}{'ok':>8}{'fail':>7}{'ms/mol':>9}"
          f"{'bonds/atom':>12}{'families split':>16}")
    for meth in methods:
        hashes = np.zeros(len(atoms), dtype=np.uint64)
        ok = np.zeros(len(atoms), dtype=bool)
        bpa = []
        t0 = time.perf_counter()
        with _quiet():
            for k, (a, q) in enumerate(zip(atoms, charges)):
                try:
                    with _deadline(timeout_s):
                        adj, nl, el = perceive(meth, a, q)
                    hashes[k] = wl_hash(adj, nl, el, depth)
                    ok[k] = True
                    bpa.append(sum(len(x) for x in adj) / 2 / max(len(nl), 1))
                except Exception:
                    ok[k] = False
        ms = (time.perf_counter() - t0) / max(len(atoms), 1) * 1e3

        # split rate over families where EVERY member was perceived successfully --
        # a family half of which failed cannot be scored either way
        split = tot = 0
        for f in np.unique(famsel):
            m = famsel == f
            if m.sum() >= 2 and ok[m].all():
                tot += 1
                split += len(np.unique(hashes[m])) > 1
        rec = {
            "n_ok": int(ok.sum()), "n_fail": int((~ok).sum()),
            "fail_rate": float((~ok).mean()), "ms_per_mol": ms,
            "bonds_per_atom": float(np.mean(bpa)) if bpa else float("nan"),
            "families_scored": tot,
            "frac_families_split": float(split / tot) if tot else float("nan"),
            "fail_by_subset": {
                s: float((~ok[subset == s]).mean())
                for s in sorted(set(subset.tolist()))
            },
        }
        out[meth] = rec
        print(f"{meth:>22}{rec['n_ok']:>8,}{rec['n_fail']:>7,}{ms:>9.1f}"
              f"{rec['bonds_per_atom']:>12.3f}"
              f"{100 * rec['frac_families_split']:>15.1f}%")

    print(f"\nfailure rate by subset (blank = 0%):")
    subs = sorted(set(subset.tolist()))
    print(f"{'method':>22}" + "".join(f"{s[:11]:>13}" for s in subs))
    for meth in methods:
        r = out[meth]["fail_by_subset"]
        if all(v == 0 for v in r.values()):
            continue
        print(f"{meth:>22}" + "".join(
            f"{100 * r[s]:>12.0f}%" if r[s] else f"{'':>13}" for s in subs))

    import json

    path = os.path.join(CACHE, f"perception_rdkit{tag}.json")
    with open(path, "w") as f:
        json.dump({"n": int(len(sel)),
                   "n_families": int(len(np.unique(famsel))), "methods": out}, f,
                  indent=1)
    print(f"\n[rdkit] wrote {path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--src", default="train_4M")
    p.add_argument("--methods", nargs="+", default=None, choices=METHODS)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--tag", default="", help="suffix for the output json")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="per-molecule wall-clock budget; over-runs count as failures")
    p.add_argument("--subsets", nargs="+", default=None,
                   help="restrict to these data_id values (e.g. metal_complexes)")
    a = p.parse_args()
    run(n=a.n, src=a.src, methods=a.methods, depth=a.depth, tag=a.tag,
        timeout_s=a.timeout, subsets=a.subsets)


if __name__ == "__main__":
    main()
