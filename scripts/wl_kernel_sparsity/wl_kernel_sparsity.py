"""Would the WL kernel PROPER (position paper Eq. 7) be sparse on OMol25?

    k^(h)_WL(G, G') = sum_{d=0..h} <phi^(d)(G), phi^(d)(G')>

phi^(d) is the vector of subtree-pattern COUNTS at depth d. This is a linear kernel in
the raw count space -- PSD by Shervashidze et al. 2011, and it needs no Wendland envelope
and no support radius, because its zeros are COMBINATORIAL: k = 0 exactly when the two
molecules share no subtree pattern.

That is a completely different sparsity mechanism from ours. We take the same phi (39,921
columns, 47 nnz/row at 20k), pareto-scale it, project it onto 10 SUPERVISED PLS
components, and then threshold a Euclidean distance. In a 10-dimensional continuous space
nothing is ever exactly zero, so sparsity cannot emerge and has to be imposed by a radius
-- which is why Remark 1 came back unimodal and why local scaling could not rescue it.
This script asks whether the zeros were there in phi all along.

WHAT IS MEASURED, per depth and cumulatively:
  * frac_zero -- share of off-diagonal molecule pairs with EXACTLY zero overlap. This is
    the sparsity gp2Scale would actually get, with no hyperparameter.
  * the same split within- and across-category, which is what decides whether the
    sparsity is BLOCK sparsity reflecting chemistry (the paper's sec-3 success mode) or
    just uniform thinning.
  * purity = P(same category | nonzero) against the 0.169 chance rate -- of the pairs the
    kernel would actually connect, how many are chemically alike.

TWO CONFIGURATIONS. min_count=2 is production (singleton patterns pruned); min_count=1
keeps every pattern and is closer to Eq. 7 as written. Pruning removes the RAREST
patterns, which are exactly the discriminative ones, so it should reduce sparsity -- worth
knowing by how much.

Depth 0 is included here even though our featurizer excludes it, because Eq. 7 sums from
d=0 and phi^(0) is just the element-count vector: <phi^(0), phi^(0)'> > 0 for any two
molecules sharing carbon. If that term alone is dense, Eq. 7 as literally written cannot
be sparse on a mostly-organic dataset regardless of what the deeper terms do.

Normalisation is irrelevant to every number here -- scaling a row by 1/n_atoms cannot
create or destroy a zero -- so `normalize=False` is used to keep the counts as Eq. 7 has
them, and the zero pattern would be identical either way.
"""
import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

SCR = os.path.dirname(os.path.abspath(__file__))
N, SEED, SUB = 20_000, 42, 4000
DEPTH, BOND_MULT = 3, 1.1


def blocks_by_depth(F, Phi):
    """Split the concatenated feature matrix back into its per-depth column blocks."""
    out, ds = {}, F.depths_
    for i, d in enumerate(ds):
        lo = F.offsets_[d]
        hi = F.offsets_[ds[i + 1]] if i + 1 < len(ds) else F.ncols_
        out[d] = Phi[:, lo:hi]
    return out


def report(K, same, label, n):
    """K is the (sub x sub) Gram of a linear kernel. Everything below is about its
    EXACT zeros, so no tolerance is involved."""
    off = ~np.eye(n, dtype=bool)
    nz = (K != 0) & off
    z = (~nz) & off
    tot = off.sum()
    fz = z.sum() / tot
    fz_w = (z & same).sum() / max((same & off).sum(), 1)
    fz_x = (z & ~same).sum() / max((~same & off).sum(), 1)
    purity = (nz & same).sum() / max(nz.sum(), 1)
    print(f"  {label:<26} {fz:>9.4f} {fz_w:>11.4f} {fz_x:>11.4f} {purity:>9.4f} "
          f"{nz.sum()/tot:>10.2e}")
    return dict(frac_zero=fz, fz_within=fz_w, fz_across=fz_x, purity=purity)


def main():
    from wl_gp2scale.data import get_data
    from wl_gp2scale.wl_features import SparseWLFeaturizer

    ds = get_data(src="train_4M", n=N, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, _ = train_test_split(idx, test_size=0.2, random_state=SEED)
    cat = ds.data_id[tr]
    sub = np.random.default_rng(0).choice(len(tr), SUB, replace=False)
    atoms = [ds.atoms[tr[i]] for i in sub]
    c = cat[sub]
    same = c[:, None] == c[None, :]
    off = ~np.eye(SUB, dtype=bool)
    chance = (same & off).sum() / off.sum()
    print(f"n={SUB} molecules, chance same-category = {chance:.4f}\n")

    # depth 0 = element counts, the term Eq. 7 starts its sum at
    elems = sorted({int(z) for a in atoms for z in a.get_atomic_numbers()})
    ix = {z: j for j, z in enumerate(elems)}
    P0 = np.zeros((SUB, len(elems)))
    for i, a in enumerate(atoms):
        for z in a.get_atomic_numbers():
            P0[i, ix[int(z)]] += 1
    P0 = sp.csr_matrix(P0)

    for mc in (1, 2):
        print("=" * 92)
        print(f"min_count = {mc}   ({'no pruning, closest to Eq. 7' if mc == 1 else 'production'})")
        print("=" * 92)
        F = SparseWLFeaturizer(depth=DEPTH, min_count=mc, cutoff_mult=BOND_MULT,
                              perceiver="ase", normalize=False).fit(atoms)
        Phi = F.transform(atoms, chunk=500)
        blk = blocks_by_depth(F, Phi)
        blk[0] = P0

        print(f"\n  {'kernel':<26} {'frac_zero':>9} {'zero|within':>11} "
              f"{'zero|across':>11} {'purity':>9} {'density':>10}")
        print(f"  {'':<26} {'(sparsity)':>9} {'':>11} {'':>11} "
              f"{'(chance ' + f'{chance:.3f})':>9}")
        print("  " + "-" * 88)

        for d in (0, 1, 2, 3):
            B = blk[d]
            nnz = B.getnnz(axis=1)
            K = (B @ B.T).toarray()
            report(K, same, f"depth {d} alone  (V={B.shape[1]:,})", SUB)
            if d == 0:
                print(f"  {'':<26}   ^ Eq. 7 starts its sum HERE")
        # cumulative: Eq. 7 as written (d=0..3) and as we build it (d=1..3)
        K13 = sum((blk[d] @ blk[d].T).toarray() for d in (1, 2, 3))
        report(K13, same, "SUM d=1..3  (our phi)", SUB)
        K03 = K13 + (blk[0] @ blk[0].T).toarray()
        report(K03, same, "SUM d=0..3  (Eq. 7)", SUB)
        print(f"\n  per-molecule nonzero patterns, by depth: "
              f"{ {d: int(np.median(blk[d].getnnz(axis=1))) for d in (0,1,2,3)} }")
        print()

    print("=" * 92)
    print("READ: frac_zero is the sparsity gp2Scale would get FOR FREE -- no radius, no")
    print("hyperparameter, PSD by construction. Compare with our PLS route, where the")
    print("10-dim continuous embedding admits NO exact zeros and the radius has to")
    print("manufacture them. purity >> chance means the surviving pairs are chemically")
    print("alike, i.e. the sparsity is BLOCK sparsity reflecting chemistry -- the")
    print("position paper's sec-3 success mode.")


if __name__ == "__main__":
    main()
