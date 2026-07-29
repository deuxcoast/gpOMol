"""Does going DEEPER buy the sparsity that depth 3 does not?

Depth-3 overlap sits at 18% of all pairs and does NOT fall with N (exponent -0.018 over a
16x range). So the overlap is not random collision in a growing vocabulary -- it is a core
of COMMON MOTIFS that recur across every chemical family. A CH3 carbon is a CH3 carbon in
a protein fragment and in an electrolyte.

Deeper subtrees are more specific, so h > 3 must reduce overlap. The question is whether
it reduces it usefully or trivially. The classic WL failure mode is that at large h every
atom's refined label becomes unique to its molecule, phi becomes effectively one-hot, and
k -> 0 for EVERY pair including chemically identical ones. The Gram tends to the identity:
maximally sparse, and useless -- every test point reverts to the prior mean, which is the
mean-reversion failure this project already knows well.

So sparsity alone is not the success criterion. Measured here, per depth:
  * frac_zero          -- the sparsity
  * purity             -- P(same category | connected), against 0.169 chance
  * frac_isolated      -- molecules with NO connection at all. These are the ones the GP
                          cannot say anything about beyond the prior; the analogue of the
                          radius kernel's frac_zero mean-reversion warning.
  * median degree      -- connections per molecule
A usable kernel needs frac_zero HIGH, purity HIGH, and frac_isolated LOW. If the only way
to get sparsity is to isolate everything, WL depth is not the lever.
"""
import numpy as np
from sklearn.model_selection import train_test_split

SUB, BOND_MULT = 4000, 1.1
DEPTHS = (2, 3, 4, 5, 6)
CURRENT_DENSITY = 2.344e-2      # our radius kernel at K=200, 20k, wl channel


def main():
    from wl_gp2scale.data import get_data
    from wl_gp2scale.wl_features import SparseWLFeaturizer

    ds = get_data(src="train_4M", n=20_000, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, _ = train_test_split(idx, test_size=0.2, random_state=42)
    sub = np.random.default_rng(0).choice(len(tr), SUB, replace=False)
    atoms = [ds.atoms[tr[i]] for i in sub]
    c = ds.data_id[tr][sub]
    n = SUB
    tot = n * (n - 1)
    chance = float(((c[:, None] == c[None, :]).sum() - n) / tot)
    print(f"n={SUB}, chance same-category={chance:.4f}, "
          f"our radius kernel density at K=200 is {CURRENT_DENSITY:.2e}\n")
    print(f"{'depth h':>8} {'V(h)':>10} {'pat/mol':>8} {'frac_zero':>10} {'density':>10} "
          f"{'purity':>8} {'lift':>6} {'isolated':>9} {'med deg':>8}")
    print("-" * 92)

    for h in DEPTHS:
        F = SparseWLFeaturizer(depth=h, min_count=1, cutoff_mult=BOND_MULT,
                              perceiver="ase", normalize=False).fit(atoms)
        Phi = F.transform(atoms, chunk=500)
        lo = F.offsets_[h]                       # the DEEPEST block only
        B = Phi[:, lo:F.ncols_]
        B.data[:] = 1.0
        K = (B @ B.T).tocoo()
        keep = K.row != K.col
        r, cl = K.row[keep], K.col[keep]
        nz = len(r)
        dens = nz / tot
        purity = float((c[r] == c[cl]).mean()) if nz else float("nan")
        deg = np.bincount(r, minlength=n)
        print(f"{h:>8} {B.shape[1]:>10,} {np.median(B.getnnz(axis=1)):>8.0f} "
              f"{1-dens:>10.4f} {dens:>10.3e} {purity:>8.4f} "
              f"{purity/chance:>6.2f}x {np.mean(deg == 0):>9.1%} "
              f"{np.median(deg):>8.0f}")

    print(f"\n  A usable kernel needs frac_zero high AND purity high AND isolated low.")
    print(f"  If sparsity only arrives by isolating molecules, every isolated point")
    print(f"  reverts to the prior mean and the kernel has bought nothing.")


if __name__ == "__main__":
    main()
