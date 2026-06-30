"""
conformer_scan.py
=================
Put an eV number on WL's conformer blindness.

The argument
------------
The WL kernel sees only connectivity (the bond graph), not 3D geometry. Two
structures with the SAME WL fingerprint are therefore assigned the SAME predicted
energy. So among any group of structures that share a fingerprint, the spread of
their actual DFT energies is energy the WL kernel *provably cannot explain* -- a
hard floor on its accuracy, independent of kernel/mean-function tuning.

This is a conservative LOWER BOUND on conformer blindness, for two reasons:
  * Geometric bond perception can hand two conformers of the same molecule
    slightly different graphs (a bond near the covalent-radius cutoff flips),
    splitting them into different fingerprint groups -> those conformers are not
    counted here, so the true blindness is at least this large.
  * Within-group energy spread is identical for raw and per-element-referenced
    energy (a shared fingerprint implies identical composition, so referencing
    subtracts the same constant from every group member). We report it in raw eV;
    it equals the referenced-residual spread.

Depth h controls what "same fingerprint" means:
  * h=0  groups by composition only -> spread includes constitutional isomers
         (over-counts; not a conformer measurement).
  * h>=1 groups by local bonding up to h hops -> groups are dominated by genuine
         conformers / stereoisomers of the same molecule. The spread at the depth
         you actually use in the kernel (h=1, 2) is the relevant blindness floor.

We also report a within-group radius-of-gyration spread, to confirm grouped
structures are GEOMETRICALLY distinct (real conformers) rather than duplicates or
pure rotations (which would share a graph trivially).

Run `python conformer_scan.py` for a self-test that injects conformers with a
KNOWN energy spread and checks the diagnostic recovers it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def radius_of_gyration(positions):
    p = np.asarray(positions, float)
    c = p.mean(axis=0)
    return float(np.sqrt(((p - c) ** 2).sum(axis=1).mean()))


def wl_fingerprint_groups(feats, h, scale=1.2):
    """Group molecule indices by identical WL fingerprint at depth h.
    Returns a list of index lists, each with >= 2 members (the multi-member
    fingerprint collisions -- the structures WL cannot tell apart)."""
    from wl_kernel import build_adjacency, wl_feature_matrix

    graphs = []
    for numbers, pos in feats:
        numbers = np.asarray(numbers)
        adj = build_adjacency(numbers, np.asarray(pos, float), scale=scale)
        graphs.append((list(numbers), adj))
    Phi = wl_feature_matrix(graphs, h=h)
    buckets = defaultdict(list)
    for i, row in enumerate(Phi):
        buckets[row.tobytes()].append(i)
    return [idx for idx in buckets.values() if len(idx) >= 2]


def scan(feats, energies, depths=(1, 2, 3), scale=1.2, reference_rms=None):
    """For each depth, group WL-identical structures and measure the within-group
    DFT energy spread (the WL blindness floor) and geometric spread."""
    energies = np.asarray(energies, float)
    N = len(feats)
    rg = np.array([radius_of_gyration(p) for _, p in feats])
    rows = []
    for h in depths:
        groups = wl_fingerprint_groups(feats, h, scale=scale)
        n_grouped = sum(len(g) for g in groups)
        devs, ranges, rg_spreads = [], [], []
        for g in groups:
            eg = energies[g]
            devs.extend((eg - eg.mean()).tolist())  # deviations WL must incur
            ranges.append(float(eg.max() - eg.min()))
            rg_spreads.append(float(np.std(rg[g])))
        devs = np.array(devs)
        rows.append(
            dict(
                h=h,
                n_groups=len(groups),
                n_grouped=n_grouped,
                frac=n_grouped / N if N else 0.0,
                rms_within=float(np.sqrt(np.mean(devs**2))) if devs.size else 0.0,
                max_range=float(max(ranges)) if ranges else 0.0,
                median_rg_spread=float(np.median(rg_spreads)) if rg_spreads else 0.0,
            )
        )
    return dict(rows=rows, N=N, reference_rms=reference_rms)


def print_report(result):
    rows, N = result["rows"], result["N"]
    ref = result.get("reference_rms")
    print("=" * 84)
    print("  WL CONFORMER-BLINDNESS SCAN")
    print("  energy spread among structures WL assigns the SAME fingerprint")
    print("=" * 84)
    print(f"  sample size N = {N}")
    if ref:
        print(f"  reference: residual RMS the kernel fights = {ref:.3f} eV")
    print()
    print(
        f"  {'depth h':>7} {'groups':>7} {'in-groups':>10} {'%data':>7} "
        f"{'RMS spread':>11} {'max range':>10} {'Rg spread':>10}"
    )
    print("  " + "-" * 76)
    for r in rows:
        frac_blind = (
            f"{100*r['rms_within']/ref:.0f}% of resid" if ref and ref > 0 else ""
        )
        print(
            f"  {r['h']:>7d} {r['n_groups']:>7d} {r['n_grouped']:>10d} "
            f"{100*r['frac']:>6.1f}% {r['rms_within']:>10.3f}e {r['max_range']:>9.3f}e "
            f"{r['median_rg_spread']:>9.3f}A   {frac_blind}"
        )
    print("  " + "-" * 76)
    print(
        "  RMS spread = irreducible RMS energy error WL imposes on grouped molecules."
    )
    print("  Rg spread  > 0 confirms grouped structures are geometrically distinct")
    print(
        "             (real conformers), not duplicates. ~0 would mean duplicates/rotations."
    )
    print(
        "  Higher h isolates true conformers; lower h mixes in constitutional isomers."
    )
    if all(r["n_groups"] == 0 for r in rows):
        print(
            "\n  NO multi-member fingerprint groups found. Either this random subsample"
        )
        print(
            "  contains few repeated graphs, or bond perception is splitting conformers."
        )
        print("  To measure the PHYSICAL conformer spread directly, group by OMol25's")
        print(
            "  system/molecule id from atoms.info instead (see pipeline --conformer-id-key)."
        )
    print("=" * 84)


# --------------------------------------------------------------------------- #
# Self-test: inject conformers with a KNOWN energy spread, recover it
# --------------------------------------------------------------------------- #
def _demo():
    from wl_kernel import _grow_molecule, build_adjacency

    rng = np.random.default_rng(7)
    sym_z = {"H": 1, "C": 6, "N": 7, "O": 8}
    base_E = {1: -13.6, 6: -1030.0, 7: -1480.0, 8: -2040.0}
    SCALE = 1.2
    CONF_STD = 0.8  # KNOWN injected conformational energy std (eV)

    feats, energies = [], []
    n_base = 120
    for _ in range(n_base):
        na = int(rng.integers(8, 20))
        Z, pos = _grow_molecule(rng, na)
        adj0 = build_adjacency(Z, pos, SCALE)
        e_base = sum(base_E[int(z)] for z in Z)
        feats.append((Z, pos))
        energies.append(e_base)
        # with prob 0.5, attach 1-3 CONFORMERS: same graph, jittered geometry,
        # energy = base + N(0, CONF_STD). Only keep jitters that preserve the graph.
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                for _try in range(8):
                    jit = pos + rng.normal(scale=0.03, size=pos.shape)
                    if _same_graph(build_adjacency(Z, jit, SCALE), adj0):
                        feats.append((Z, jit))
                        energies.append(e_base + rng.normal(scale=CONF_STD))
                        break
    energies = np.array(energies)

    print(f"Self-test: {len(feats)} structures, conformers injected with KNOWN")
    print(
        f"per-conformer energy std = {CONF_STD} eV (so within-group RMS should ~ {CONF_STD}).\n"
    )
    print_report(
        scan(feats, energies, depths=(1, 2, 3), scale=SCALE, reference_rms=4.7)
    )
    print("\nThe RMS-spread column at h=1..3 should land near the injected 0.8 eV,")
    print("and Rg spread should be > 0 (jittered geometries are distinct).")


def _same_graph(adj_a, adj_b):
    if len(adj_a) != len(adj_b):
        return False
    return all(sorted(a) == sorted(b) for a, b in zip(adj_a, adj_b))


if __name__ == "__main__":
    _demo()
