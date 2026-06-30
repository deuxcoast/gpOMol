r"""
sid_scan.py
===========
Measure the PHYSICAL conformer signal in OMol25 directly, by grouping structures
on their system id (`sid` in atoms.info) instead of on a WL fingerprint.

Why this is the definitive version
----------------------------------
The WL-fingerprint scan (conformer_scan.py) only catches conformers that BOTH
(a) happened to co-sample in a random draw AND (b) map to the same perceived
graph. It gave a lower bound (~1.25 eV RMS on ~12% of a random sample). Grouping
by `sid` removes both limitations: we actively gather every sampled structure of
each system, regardless of whether their graphs collide, so we measure:

  * MAGNITUDE  : among systems with >1 conformer, the DFT energy spread WL (or any
                 geometry-blind model) must incur -- the size of the prize a
                 geometry-carrying descriptor would target.
  * PREVALENCE : what fraction of systems / structures actually have multiple
                 conformers -- i.e. how large a part of OMol25 this affects.

Honesty rails
-------------
  * MAGNITUDE is robust: any system with >=2 sampled conformers gives a valid
    energy-spread estimate, and within a system composition is fixed so the spread
    is identical for raw and per-element-referenced energy.
  * PREVALENCE is sampling-dependent: it reflects how many conformers per system
    landed in the scanned block, not necessarily OMol25's global rate. The scan
    reports its coverage so this is explicit.
  * VALIDATION: we check that every structure in a `sid` group shares the same
    composition. If `sid` were a per-structure id (every group size 1) or grouped
    unlike molecules, that check would fail and the key is wrong -- the report
    says so and suggests alternative keys.

This module holds the analysis (pure numpy, unit-tested below). The collection
that reads `sid`/energy from the dataset lives in the pipeline's run_sid_scan.

Run `python sid_scan.py` for a self-test on synthetic systems with a KNOWN
conformer spread.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _hist_group_sizes(sizes):
    h = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for s in sizes:
        h["5+" if s >= 5 else str(s)] += 1
    return h


def analyze_groups(ids, energies, comp_sig=None, reference_rms=None):
    """Group structure indices by id and measure per-system energy spread.

    ids        : sequence of system ids (one per structure).
    energies   : DFT total energies (eV), one per structure.
    comp_sig   : optional hashable composition signature per structure, used to
                 validate that an id really groups one molecule (same formula).
    reference_rms : optional residual RMS the kernel fights, for a "% of residual"
                 readout.
    """
    ids = list(ids)
    energies = np.asarray(energies, float)
    buckets = defaultdict(list)
    for i, s in enumerate(ids):
        buckets[s].append(i)

    group_sizes = [len(v) for v in buckets.values()]
    multi = [v for v in buckets.values() if len(v) >= 2]

    devs, ranges, stds = [], [], []
    n_struct_multi, consistent = 0, 0
    for g in multi:
        eg = energies[g]
        devs.extend((eg - eg.mean()).tolist())
        ranges.append(float(eg.max() - eg.min()))
        stds.append(float(np.std(eg)))
        n_struct_multi += len(g)
        if comp_sig is not None and len({comp_sig[i] for i in g}) == 1:
            consistent += 1
    devs = np.array(devs)

    return dict(
        n_structures=len(ids),
        n_systems=len(buckets),
        n_multi_systems=len(multi),
        frac_systems_multi=(len(multi) / len(buckets)) if buckets else 0.0,
        n_struct_in_multi=n_struct_multi,
        frac_struct_in_multi=(n_struct_multi / len(ids)) if ids else 0.0,
        rms_within=float(np.sqrt(np.mean(devs**2))) if devs.size else 0.0,
        median_group_std=float(np.median(stds)) if stds else 0.0,
        max_range=float(max(ranges)) if ranges else 0.0,
        mean_group_size=float(np.mean(group_sizes)) if group_sizes else 0.0,
        max_group_size=int(max(group_sizes)) if group_sizes else 0,
        comp_consistent_frac=(consistent / len(multi)) if multi else float("nan"),
        reference_rms=reference_rms,
        group_size_hist=_hist_group_sizes(group_sizes),
    )


def print_report(r, id_key="sid", scanned=None, kept=None):
    print("=" * 84)
    print("  OMol25 CONFORMER SCAN BY SYSTEM ID")
    print(f"  grouping key: atoms.info[{id_key!r}]")
    print("=" * 84)
    if scanned is not None:
        print(f"  scanned {scanned:,} structures, kept {kept:,} in-slice")
    print(f"  distinct systems        : {r['n_systems']:,}")
    print(f"  structures              : {r['n_structures']:,}")
    h = r["group_size_hist"]
    print(
        f"  systems by #conformers  : 1:{h['1']}  2:{h['2']}  3:{h['3']}  "
        f"4:{h['4']}  5+:{h['5+']}   (max {r['max_group_size']})"
    )
    print()
    print(
        f"  PREVALENCE  : {r['n_multi_systems']:,} systems have >1 conformer "
        f"({100*r['frac_systems_multi']:.1f}% of systems, "
        f"{100*r['frac_struct_in_multi']:.1f}% of structures)"
    )
    print()
    if r["n_multi_systems"] == 0:
        print("  No multi-structure groups under this key. Either the scan window")
        print("  contains one structure per system, or this key is a per-structure id.")
        print(
            "  Try a different --sid-key (e.g. 'source', 'data_id') or widen --sid-pool."
        )
        print("=" * 84)
        return
    ref = r["reference_rms"]
    pct = (
        f"  ({100*r['rms_within']/ref:.0f}% of the {ref:.2f} eV residual)"
        if ref
        else ""
    )
    print(f"  MAGNITUDE   : within-system DFT energy spread")
    print(f"     RMS spread (per structure)  : {r['rms_within']:.3f} eV{pct}")
    print(f"     median per-system std        : {r['median_group_std']:.3f} eV")
    print(f"     max within-system range      : {r['max_range']:.3f} eV")
    print()
    cc = r["comp_consistent_frac"]
    flag = (
        "OK (id groups one molecule)"
        if cc >= 0.99
        else "WARNING: id groups differ in composition -- may not be a system id"
    )
    print(
        f"  VALIDATION  : {100*cc:.1f}% of multi-conformer groups have a single "
        f"composition  -> {flag}"
    )
    print("=" * 84)
    print("  RMS spread is the irreducible energy error a geometry-blind model incurs")
    print(
        "  on multi-conformer systems -- the prize a spatial descriptor would target."
    )
    print(
        "  PREVALENCE is sampling-dependent (reflects conformers-per-system that landed"
    )
    print(
        "  in the scan window), so read it as a floor on how widespread the effect is."
    )
    print("=" * 84)


# --------------------------------------------------------------------------- #
# Self-test: synthetic systems with a KNOWN conformer spread
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(11)
    CONF_STD = 1.0  # KNOWN within-system energy std (eV)
    ids, energies, comp = [], [], []
    n_systems = 500
    for s in range(n_systems):
        base = rng.uniform(-2000, -500)
        formula = f"C{rng.integers(3, 12)}H{rng.integers(4, 20)}"  # fixed per system
        k = 1 + (rng.random() < 0.4) * int(rng.integers(1, 6))  # 60% single, 40% multi
        for _ in range(k):
            ids.append(s)
            energies.append(base + (rng.normal(scale=CONF_STD) if k > 1 else 0.0))
            comp.append(formula)
    print(f"Self-test: {n_systems} systems, {len(ids)} structures, multi-conformer")
    print(f"systems given KNOWN energy std = {CONF_STD} eV.\n")
    r = analyze_groups(ids, energies, comp_sig=comp, reference_rms=6.7)
    print_report(r, id_key="sid (synthetic)")
    print("\nRMS spread should land near the injected 1.0 eV, and composition")
    print("consistency should be 100% (each synthetic system has one formula).")


if __name__ == "__main__":
    _demo()
