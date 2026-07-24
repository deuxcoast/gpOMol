"""
census.py  (data_exploration)
=============================
PASS A -- an **exact** metadata census of every structure in ``train_4M``.

The shards read sequentially at ~3.3k structures/s/process, so all 3,986,754 rows
cost ~4 minutes across 8 workers. There is therefore no reason to sample anything
that can be read straight off ``atoms.info``: this pass is the whole dataset.

Each of the 80 ``.aselmdb`` shards is scanned by one worker into
``cache/census_parts/shard_XX.npz``; ``load_census()`` concatenates them into one
in-memory table (~4M rows x ~30 columns, ~450 MB). Every figure and statistic in
this module reads that table -- the LMDBs are touched exactly once.

    python -m data_exploration.census                 # scan all shards
    python -m data_exploration.census --shards 0 1    # scan a couple (smoke test)
    python -m data_exploration.census --force         # ignore existing parts

Columns are documented in ``COLUMNS`` below. Strings are stored as stable 64-bit
blake2b hashes (``hash_source``) plus per-shard ``{hash: string}`` tables, because
Python's ``hash`` is per-process randomised and full source strings would cost
more than every numeric column combined.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from .sources import _blake, hash_source

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
PARTS = os.path.join(CACHE, "census_parts")

COLUMNS = {
    # identity / provenance
    "sid": "int64",              # OMol25 structure id from atoms.info
    "gidx": "int64",             # global AseDBDataset index (offset + local)
    "shard": "int16",
    "data_id": "uint64",         # hash of the 10-way subset label
    "src_top": "uint64",         # hash of the first source path component
    "src_group": "uint64",       # hash of the production batch directory
    "job": "uint64",             # hash of one ORCA job (family member)
    "family": "uint64",          # hash of the heuristic molecule family
    "step": "int16",             # step index within a job, -1 if none
    # size / cost
    "n_atoms": "int16",
    "n_heavy": "int16",
    "n_elec": "int32",
    "n_ecp_elec": "int16",
    "n_basis": "int32",
    "n_scf": "int16",
    "unrestricted": "bool",
    # state
    "charge": "int16",
    "spin": "int16",
    # energetics
    "energy": "float64",
    "nl_energy": "float32",
    "homo": "float32",           # max over spin channels
    "gap": "float32",            # min over spin channels
    "s_squared": "float32",
    "s_squared_dev": "float32",
    # forces (the off-equilibrium coordinate)
    "fmax": "float32",
    "frms": "float32",
    # geometry (free: positions are already in hand)
    "rg": "float32",             # geometric radius of gyration
    # partial charges
    "has_nbo": "bool",
    "nan_lowdin": "bool",        # the current loader FILTERS these out
    "q_min": "float32",
    "q_max": "float32",
    "q_rms": "float32",
    "dipole": "float32",         # |sum q_i (r_i - com)| from Lowdin charges
    # composition
    "formula": "uint64",         # hash of the sorted element-count vector
    "n_warn": "int16",
}

_NAN = float("nan")


def _shard_paths(src: str):
    import glob

    return sorted(glob.glob(os.path.join(src, "*.aselmdb")))


def _offsets(src: str):
    """Global index offset of each shard, matching ``AseDBDataset``'s own ordering
    so census rows can be joined to the frozen subsets in ``../cache``."""
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    cum = np.asarray(ds._idlen_cumulative, dtype=np.int64)
    return np.concatenate([[0], cum[:-1]]), int(cum[-1])


def scan_shard(args):
    """Worker: one shard -> one .npz part. Returns (shard, n, warn_counter)."""
    shard, path, offset, out_path = args
    from collections import Counter

    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": path})
    n = len(ds)
    cols = {k: np.zeros(n, dtype=v) for k, v in COLUMNS.items()}
    # element counts as COO (avg ~4 distinct elements per molecule -> tiny)
    e_row, e_z, e_n = [], [], []
    names = {}          # hash -> string, for the low-cardinality label columns
    fam_examples = {}   # a few family hash -> string, for spot-checking the parse
    warns = Counter()

    for i in range(n):
        a = ds.get_atoms(i)
        info = a.info
        Z = a.get_atomic_numbers()
        pos = a.get_positions()
        q = int(info.get("charge", 0))
        s = int(info.get("spin", 1))

        top, group, job_h, fam_h, step = hash_source(str(info.get("source", "")), q, s)
        did = str(info.get("data_id", "unknown"))
        for st in (did, top, group):
            names.setdefault(_blake(st), st)
        if len(fam_examples) < 400:
            fam_examples.setdefault(fam_h, str(info.get("source", "")))

        cols["sid"][i] = int(info.get("sid", -1))
        cols["gidx"][i] = offset + i
        cols["shard"][i] = shard
        cols["data_id"][i] = _blake(did)
        cols["src_top"][i] = _blake(top)
        cols["src_group"][i] = _blake(group)
        cols["job"][i] = job_h
        cols["family"][i] = fam_h
        cols["step"][i] = step

        cols["n_atoms"][i] = len(Z)
        cols["n_heavy"][i] = int(np.count_nonzero(Z > 1))
        cols["n_elec"][i] = int(info.get("num_electrons", 0))
        cols["n_ecp_elec"][i] = int(info.get("num_ecp_electrons", 0) or 0)
        cols["n_basis"][i] = int(info.get("n_basis", 0) or 0)
        cols["n_scf"][i] = min(int(info.get("n_scf_steps", 0) or 0), 32767)
        cols["unrestricted"][i] = bool(info.get("unrestricted", False))
        cols["charge"][i] = q
        cols["spin"][i] = s

        cols["energy"][i] = float(a.get_potential_energy())
        cols["nl_energy"][i] = float(info.get("nl_energy", _NAN) or _NAN)
        ho = info.get("homo_energy")
        cols["homo"][i] = float(np.max(ho)) if ho is not None else _NAN
        gp = info.get("homo_lumo_gap")
        cols["gap"][i] = float(np.min(gp)) if gp is not None else _NAN
        cols["s_squared"][i] = float(info.get("s_squared", _NAN))
        cols["s_squared_dev"][i] = float(info.get("s_squared_dev", _NAN))

        try:
            F = a.get_forces()
            fn = np.linalg.norm(F, axis=1)
            cols["fmax"][i] = float(fn.max()) if len(fn) else _NAN
            cols["frms"][i] = float(np.sqrt(np.mean(fn**2))) if len(fn) else _NAN
        except Exception:
            cols["fmax"][i] = cols["frms"][i] = _NAN

        c = pos - pos.mean(axis=0)
        cols["rg"][i] = float(np.sqrt(np.mean(np.sum(c * c, axis=1))))

        cols["has_nbo"][i] = "nbo_charges" in info
        lq = info.get("lowdin_charges")
        if lq is None:
            cols["nan_lowdin"][i] = True
            cols["q_min"][i] = cols["q_max"][i] = cols["q_rms"][i] = _NAN
            cols["dipole"][i] = _NAN
        else:
            lq = np.asarray(lq, dtype=float)
            bad = lq.size != len(Z) or not np.all(np.isfinite(lq))
            cols["nan_lowdin"][i] = bool(bad)
            if bad:
                cols["q_min"][i] = cols["q_max"][i] = cols["q_rms"][i] = _NAN
                cols["dipole"][i] = _NAN
            else:
                cols["q_min"][i] = lq.min()
                cols["q_max"][i] = lq.max()
                cols["q_rms"][i] = float(np.sqrt(np.mean(lq**2)))
                # origin = centre of mass, so the value is well defined for ions too
                # (it is origin-dependent for a charged system -- documented, not hidden)
                com = a.get_center_of_mass()
                cols["dipole"][i] = float(np.linalg.norm((lq[:, None] * (pos - com)).sum(0)))

        w = info.get("warnings") or []
        cols["n_warn"][i] = len(w)
        for msg in w:
            warns[str(msg)[:110]] += 1

        zz, cc = np.unique(Z, return_counts=True)
        e_row.extend([i] * len(zz))
        e_z.extend(zz.tolist())
        e_n.extend(cc.tolist())
        cols["formula"][i] = _blake(
            ";".join(f"{int(z)}:{int(c)}" for z, c in zip(zz, cc))
        )

    np.savez_compressed(
        out_path,
        **cols,
        e_row=np.asarray(e_row, dtype=np.int32),
        e_z=np.asarray(e_z, dtype=np.uint8),
        e_n=np.asarray(e_n, dtype=np.uint16),
        name_hash=np.asarray(list(names.keys()), dtype=np.uint64),
        name_str=np.asarray(list(names.values()), dtype=object),
        fam_hash=np.asarray(list(fam_examples.keys()), dtype=np.uint64),
        fam_str=np.asarray(list(fam_examples.values()), dtype=object),
        allow_pickle=True,
    )
    return shard, n, dict(warns)


def run(src="train_4M", workers=8, shards=None, force=False):
    os.makedirs(PARTS, exist_ok=True)
    paths = _shard_paths(src)
    if not paths:
        raise SystemExit(f"no .aselmdb shards under {src!r}")
    offs, total = _offsets(src)
    sel = range(len(paths)) if shards is None else shards
    jobs = []
    for k in sel:
        out = os.path.join(PARTS, f"shard_{k:03d}.npz")
        if os.path.exists(out) and not force:
            continue
        jobs.append((k, paths[k], int(offs[k]), out))
    print(f"[census] {len(paths)} shards, {total:,} structures; {len(jobs)} to scan")
    if not jobs:
        print("[census] all parts present (use --force to rescan)")
        return

    from collections import Counter

    warns, done, t0 = Counter(), 0, time.time()
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for shard, n, w in pool.imap_unordered(scan_shard, jobs):
            warns.update(w)
            done += 1
            el = time.time() - t0
            print(f"[census] shard {shard:3d} ({n:,} rows)  {done}/{len(jobs)}  "
                  f"{el:6.1f}s  eta {el / done * (len(jobs) - done):6.1f}s", flush=True)
    with open(os.path.join(CACHE, "warnings.json"), "w") as f:
        json.dump(warns.most_common(), f, indent=1)
    print(f"[census] done in {time.time() - t0:.1f}s; "
          f"{len(warns)} distinct warning strings -> cache/warnings.json")


def load_census(columns=None, parts_dir=PARTS):
    """Concatenate the shard parts into one dict of column arrays.

    Also returns ``names`` (hash -> string for data_id / src_top / src_group),
    ``fam_examples``, and the element-count matrix as a scipy CSR ``(N, 119)``.
    """
    import glob

    import scipy.sparse as sp

    files = sorted(glob.glob(os.path.join(parts_dir, "shard_*.npz")))
    if not files:
        raise SystemExit(f"no census parts in {parts_dir}; run `python -m "
                         f"data_exploration.census` first")
    keys = list(columns) if columns else list(COLUMNS)
    chunks = {k: [] for k in keys}
    rows, zs, ns, offset = [], [], [], 0
    names, fam_examples = {}, {}
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            for k in keys:
                chunks[k].append(d[k])
            m = len(d["sid"])
            rows.append(d["e_row"].astype(np.int64) + offset)
            zs.append(d["e_z"])
            ns.append(d["e_n"])
            offset += m
            for h, s in zip(d["name_hash"], d["name_str"]):
                names[int(h)] = str(s)
            for h, s in zip(d["fam_hash"], d["fam_str"]):
                fam_examples.setdefault(int(h), str(s))
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    counts = sp.csr_matrix(
        (np.concatenate(ns).astype(np.int32),
         (np.concatenate(rows), np.concatenate(zs).astype(np.int64))),
        shape=(offset, 119),
    )
    print(f"[census] loaded {offset:,} rows from {len(files)} parts")
    return out, names, counts, fam_examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="train_4M")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shards", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(src=a.src, workers=a.workers, shards=a.shards, force=a.force)


if __name__ == "__main__":
    main()
