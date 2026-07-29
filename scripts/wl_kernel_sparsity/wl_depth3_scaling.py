"""Does depth-3 WL overlap get SPARSER with N, and how fast?

The single-N measurement is not decidable on its own. At 4k molecules the depth-3 linear
kernel has density 1.75e-1 -- 7.5x DENSER than the radius kernel we already run at 20k
(2.34e-2 at K=200). Taken at face value that kills it. But density here is not
scale-invariant: the depth-3 pattern vocabulary GROWS with N (~N^0.91 measured
previously), while the number of patterns per molecule stays fixed at ~20, so the chance
that two random molecules share one should FALL roughly like N^-0.91.

If that holds, the ordering reverses somewhere below 200k and the WL kernel proper ends
up sparser than anything a radius gives us. If it does not -- if overlap is driven by a
core of common motifs whose share does not shrink -- then depth-3 WL is dense at every
scale and the idea is dead.

So: measure frac_zero and purity for the depth-3 block at a ladder of N, fit the exponent,
extrapolate to 200k, and compare against the radius kernel's measured density. Everything
is done on the SPARSE product; the Gram is never densified (16k^2 float64 = 2 GB).

The vocabulary is refit at each N, which is the honest thing to do -- a vocabulary frozen
at small N would import the small-N overlap structure into every larger rung and
manufacture the trend being tested.
"""
import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

SCR = os.path.dirname(os.path.abspath(__file__))
LADDER = (1000, 2000, 4000, 8000, 16000)
DEPTH, BOND_MULT = 3, 1.1
# what we actually run today, from sparsity_report at 20k (arm K=200, wl channel)
CURRENT_DENSITY, CURRENT_K = 2.344e-2, 200


def main():
    from wl_gp2scale.data import get_data
    from wl_gp2scale.wl_features import SparseWLFeaturizer

    ds = get_data(src="train_4M", n=20_000, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, _ = train_test_split(idx, test_size=0.2, random_state=42)
    cat_all = ds.data_id[tr]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tr))          # nested subsamples: rung n is a prefix

    print(f"{'N':>7} {'V(depth3)':>11} {'pat/mol':>8} {'frac_zero':>10} {'density':>10} "
          f"{'purity':>8} {'chance':>8} {'lift':>6}")
    print("-" * 78)
    rows = []
    for n in LADDER:
        sub = order[:n]
        atoms = [ds.atoms[tr[i]] for i in sub]
        c = cat_all[sub]
        F = SparseWLFeaturizer(depth=DEPTH, min_count=1, cutoff_mult=BOND_MULT,
                              perceiver="ase", normalize=False).fit(atoms)
        Phi = F.transform(atoms, chunk=500)
        lo = F.offsets_[3]
        B = Phi[:, lo:F.ncols_]                       # depth-3 block only
        B.data[:] = 1.0                               # presence, not multiplicity
        K = (B @ B.T).tocoo()
        keep = K.row != K.col                          # off-diagonal only
        r, cc = K.row[keep], K.col[keep]
        tot = n * (n - 1)
        nz = len(r)
        same = (c[r] == cc.astype(int) * 0 + c[cc])
        chance = float(((c[:, None] == c[None, :]).sum() - n) / tot)
        purity = float(same.mean()) if nz else float("nan")
        dens = nz / tot
        rows.append((n, dens, purity))
        print(f"{n:>7} {B.shape[1]:>11,} {np.median(B.getnnz(axis=1)):>8.0f} "
              f"{1-dens:>10.4f} {dens:>10.3e} {purity:>8.4f} {chance:>8.4f} "
              f"{purity/chance:>6.2f}x")

    # power-law fit density ~ a * N^b, then extrapolate
    N = np.array([r[0] for r in rows], float)
    D = np.array([r[1] for r in rows], float)
    b, loga = np.polyfit(np.log(N), np.log(D), 1)
    print(f"\n  density ~ {np.exp(loga):.3g} * N^{b:.3f}   (R^2 of log-log fit "
          f"{np.corrcoef(np.log(N), np.log(D))[0,1]**2:.4f})")
    for target in (20_000, 200_000, 1_000_000):
        d = np.exp(loga) * target ** b
        print(f"    extrapolated density at N={target:>9,}: {d:.3e}   "
              f"nnz={d*target**2:.2e}   "
              f"({'SPARSER' if d < CURRENT_DENSITY else 'denser'} than our K={CURRENT_K} "
              f"radius kernel at {CURRENT_DENSITY:.2e})")
    P = np.array([r[2] for r in rows], float)
    print(f"\n  purity across the ladder: {np.round(P, 4)}  "
          f"({'rising' if P[-1] > P[0] else 'falling'})")
    print("\n  CAVEAT: extrapolating a power law two decades past the data is a")
    print("  hypothesis, not a measurement. It is cheap to check directly at 200k --")
    print("  featurisation only, no GP, no solve.")


if __name__ == "__main__":
    main()
