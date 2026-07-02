"""
smiles_coverage.py
==================
Estimate how recoverable SMILES / connectivity is across an OMol25 split.

OMol25 records store NO SMILES field (arrays = numbers, positions only), so
"coverage" here means: the fraction of structures whose provenance string
(info['source']) embeds an identifier we can resolve to a SMILES (e.g. a ChEMBL
id), broken down by source dataset (info['data_id']). That breakdown is the real
decision input, because SMILES availability is a per-source-dataset property.

Also prints a companion NBO-coverage tally, since the charge-channel decision
(NBO vs Loewdin) depends on NBO being present and finite.

Usage
-----
    python smiles_coverage.py --src ../train_4M --n 5000
"""

import argparse
import re
from collections import Counter, defaultdict

import numpy as np

# Identifier patterns we know (or suspect) how to resolve to SMILES.
# CHEMBL is confirmed present in OrbNet-Denali-sourced records. The others are
# heuristic starting points — eyeball the example `source` strings this script
# prints per data_id and add patterns for whatever actually dominates train_4M.
ID_PATTERNS = {
    "chembl": re.compile(r"CHEMBL\d+", re.I),
    "pubchem": re.compile(
        r"CID[_-]?\d+", re.I
    ),  # heuristic; verify vs printed examples
}


CHARGE_SCHEMES = ["mulliken_charges", "lowdin_charges", "nbo_charges"]


def _charge_missing(val) -> bool:
    """True if a per-atom charge array is absent, empty, or contains NaN."""
    if val is None:
        return True
    arr = np.asarray(val, dtype=float).ravel()
    return arr.size == 0 or bool(np.any(np.isnan(arr)))


def load_dataset(src):
    from fairchem.core.datasets import AseDBDataset

    ds = AseDBDataset({"src": src})
    print(f"Loaded {len(ds):,} structures from {src!r}")
    return ds


def extract_ids(source: str) -> dict:
    """Return {id_type: matched_string} for every known pattern found in source."""
    hits = {}
    for name, pat in ID_PATTERNS.items():
        m = pat.search(source or "")
        if m:
            hits[name] = m.group(0)
    return hits


def summarize(ds, idxs) -> dict:
    """Walk the sampled structures once, tally provenance + NBO coverage, print."""
    n = len(idxs)
    data_id_counts = Counter()
    id_type_counts = Counter()
    per_dataid = defaultdict(lambda: [0, 0])  # data_id -> [with_resolvable_id, total]
    examples = {}  # data_id -> one sample source string
    any_id = 0
    charge_missing = {s: 0 for s in CHARGE_SCHEMES}
    # data_id -> {scheme: [missing, total]}
    per_dataid_charge = defaultdict(lambda: {s: [0, 0] for s in CHARGE_SCHEMES})

    for i in idxs:
        atoms = ds.get_atoms(int(i))  # OMol25 returns an ASE Atoms
        info = atoms.info
        did = info.get("data_id", "unknown")
        src = info.get("source", "") or ""

        data_id_counts[did] += 1
        examples.setdefault(did, src)
        per_dataid[did][1] += 1

        ids = extract_ids(src)
        if ids:
            any_id += 1
            per_dataid[did][0] += 1
            for t in ids:
                id_type_counts[t] += 1

        for s in CHARGE_SCHEMES:
            miss = _charge_missing(info.get(s, None))
            per_dataid_charge[did][s][1] += 1
            if miss:
                charge_missing[s] += 1
                per_dataid_charge[did][s][0] += 1

    # ---- report --------------------------------------------------------------
    print("\n== SMILES recoverability (via provenance ids) ==")
    print(f"  sampled structures: {n}")
    print(f"  with a resolvable id in source: {any_id}/{n} = {any_id / n:.1%}")
    print(f"  by id type: {dict(id_type_counts)}")

    print("\n== source-dataset (data_id) breakdown ==")
    print(f"  {'data_id':<26}{'count':>8}{'share':>8}{'id-cov':>9}")
    for did, c in data_id_counts.most_common():
        with_id, tot = per_dataid[did]
        print(f"  {str(did):<26}{c:>8}{c / n:>8.1%}{with_id / tot:>9.0%}")
        print(f"      e.g. source={examples[did][:88]!r}")

    print("\n== charge-scheme coverage (missing/NaN fraction) ==")
    for s in CHARGE_SCHEMES:
        print(
            f"  {s:<18} missing {charge_missing[s]:>6}/{n} = {charge_missing[s] / n:.1%}"
        )

    print("\n== per-data_id charge coverage (lowdin-miss / nbo-miss) ==")
    print(f"  {'data_id':<26}{'share':>8}{'lowdin':>10}{'nbo':>8}")
    for did, c in data_id_counts.most_common():
        lo = per_dataid_charge[did]["lowdin_charges"]
        nb = per_dataid_charge[did]["nbo_charges"]
        print(
            f"  {str(did):<26}{c / n:>8.1%}{lo[0] / lo[1]:>10.0%}{nb[0] / nb[1]:>8.0%}"
        )

    return {
        "n": n,
        "any_id_fraction": any_id / n,
        "data_id_counts": dict(data_id_counts),
        "per_dataid_id_coverage": {k: (v[0] / v[1]) for k, v in per_dataid.items()},
        "charge_missing_fraction": {s: charge_missing[s] / n for s in CHARGE_SCHEMES},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="../train_4M", help="directory of .aselmdb files")
    ap.add_argument("--n", type=int, default=5000, help="subsample size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(args.src)
    n = min(args.n, len(ds))
    idxs = np.random.default_rng(args.seed).choice(len(ds), size=n, replace=False)
    summarize(ds, idxs)


if __name__ == "__main__":
    main()
