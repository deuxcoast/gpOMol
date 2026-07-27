"""
bondmult_report.py  (data_exploration)
======================================
Read the `dim_sweep --bond-mult` arms and report them the way the project's own
statistics lesson says to: **paired per-seed differences**, not marginal means.

The marginal std across seeds is large (~0.06 at 20k, and the WL channel is the
worst offender at 0.076) because seeds differ in task difficulty. All arms share
the splits, so differencing arm-by-arm within a seed removes that variance
entirely; judging a 0.01 effect against the marginal std would wrongly dismiss it.

    python -m data_exploration.bondmult_report cache/bondmult_wl_*.npz
"""

from __future__ import annotations

import re
import sys

import numpy as np

CODE_TO_NAME = {0: "wl", 1: "geom", 2: "strain", 3: "elec", 4: "additive"}


def load(paths):
    """-> {(mult, mode): {seed: (ols, gp)}}"""
    out = {}
    for p in paths:
        m = re.search(r"_([0-9]*\.?[0-9]+)\.npz$", p)
        mult = float(m.group(1)) if m else float("nan")
        with np.load(p, allow_pickle=True) as d:
            for seed, code, ols, gp in d["rows"]:
                key = (mult, CODE_TO_NAME[int(code)])
                out.setdefault(key, {})[int(seed)] = (float(ols), float(gp))
    return out


def report(paths, baseline=1.2):
    rec = load(paths)
    mults = sorted({k[0] for k in rec})
    modes = sorted({k[1] for k in rec})
    print(f"arms: bond_mult {mults}  channels {modes}\n")

    for mode in modes:
        seeds = sorted(set().union(*[set(rec[(m, mode)]) for m in mults
                                     if (m, mode) in rec]))
        print(f"=== channel: {mode} ===")
        print(f"{'bond_mult':>10}{'GP mean':>10}{'GP std':>9}"
              + "".join(f"{'s' + str(s):>9}" for s in seeds))
        for m in mults:
            if (m, mode) not in rec:
                continue
            v = np.array([rec[(m, mode)][s][1] for s in seeds])
            print(f"{m:>10.2f}{v.mean():>10.4f}{v.std():>9.4f}"
                  + "".join(f"{x:>9.4f}" for x in v))

        if (baseline, mode) in rec:
            print(f"\n  paired per-seed difference vs bond_mult={baseline} "
                  f"(the historical default):")
            print(f"{'bond_mult':>10}{'mean diff':>11}{'std':>8}{'t-ish':>8}   per-seed")
            for m in mults:
                if m == baseline or (m, mode) not in rec:
                    continue
                d = np.array([rec[(m, mode)][s][1] - rec[(baseline, mode)][s][1]
                              for s in seeds])
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and \
                    d.std(ddof=1) > 0 else float("nan")
                print(f"{m:>10.2f}{d.mean():>+11.4f}{d.std(ddof=1):>8.4f}{t:>8.1f}   "
                      + " ".join(f"{x:+.4f}" for x in d))
        print()

    # the OLS column is the embedding quality without the kernel -- if bond_mult
    # moves OLS and GP together, it is the DESCRIPTOR changing, not the kernel
    print("=== OLS (linear on the embedding) -- same arms, kernel removed ===")
    for mode in modes:
        seeds = sorted(set().union(*[set(rec[(m, mode)]) for m in mults
                                     if (m, mode) in rec]))
        for m in mults:
            if (m, mode) not in rec:
                continue
            v = np.array([rec[(m, mode)][s][0] for s in seeds])
            print(f"  {mode:>8}  bond_mult={m:.2f}  OLS mean={v.mean():.4f} "
                  f"std={v.std():.4f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.endswith(".npz")]
    if not args:
        raise SystemExit(__doc__)
    report(args)
