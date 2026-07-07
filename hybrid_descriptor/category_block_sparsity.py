"""
category_block_sparsity.py
==========================
Quantify Marcus's block-sparsity idea: if pairs from different source categories
(data_id: proteins, electrolytes, metal complexes, organics, ...) are forced to
zero covariance, how much does that raise the feasible N?

The trick only helps to the extent that cross-category pairs would OTHERWISE fall
inside the kernel support radius (i.e. are spuriously close in embedding space).
If different chemistries are already far apart in the embedding, those pairs are
already ~zero and the trick buys little. This script measures exactly that on a
sample, at the support radius that preserves the signal (the variogram range).

Honest scope: this reduces the sparsity CONSTANT (fewer nonzeros), pushing the
feasible-N frontier out. It does NOT change the N^2 growth, so it extends the
affordable range (helps 4M, and beyond) but does not by itself reach 100M.

Usage
-----
    python category_block_sparsity.py --src ../train_4M --n 30000 --n_diag 4000
"""

import argparse

import numpy as np
from analyze_residual import load_for_analysis
from diagnostics import BUDGET_TB, BYTES_PER_NNZ, _pairwise_euclidean, semivariogram
from gp_fit import HybridPreprocessor, MolBatch


def block_sparsity_report(X, data_id, resid, target_N=4_000_000):
    n = len(X)
    cat = np.asarray(data_id)
    D = _pairwise_euclidean(X)
    iu = np.triu_indices(n, k=1)
    dist = D[iu]
    same = cat[iu[0]] == cat[iu[1]]
    p_same = float(same.mean())

    print("\n== category mix (sample) ==")
    cats, counts = np.unique(cat, return_counts=True)
    for c, k in sorted(zip(cats, counts), key=lambda t: -t[1]):
        print(f"  {str(c):<22}{k:>7}  ({k/n:5.1%})")
    print(
        f"  P(random pair is same-category) = {p_same:.3f}  "
        f"-> {1-p_same:.1%} of all pairs are cross-category (candidate free zeros)"
    )

    # signal-preserving radius = variogram range
    r_range = semivariogram(X, resid)["range"]

    def stor(s):
        return BYTES_PER_NNZ * s * target_N**2 / 1e12

    def nbreak(s):
        return float(np.sqrt(BUDGET_TB * 1e12 / (BYTES_PER_NNZ * max(s, 1e-12))))

    radii = sorted(
        set(np.quantile(dist, np.linspace(0.1, 0.95, 12)).tolist() + [r_range])
    )
    print("\n== sparsity with vs without the cross-category zeroing trick ==")
    print(
        f"  target N = {target_N:.0e}, budget {BUDGET_TB:.0f} TB   (marker * = variogram range)"
    )
    print(
        f"  {'radius':>8}{'s*_now':>9}{'s*_block':>10}{'x-cat%ofnz':>12}"
        f"{'TB_now':>9}{'TB_block':>10}"
    )
    for r in radii:
        within = dist <= r
        nz = within.sum()
        s_now = float(within.mean())
        s_block = float((within & same).mean())
        xcat = float((within & ~same).sum() / max(nz, 1))
        mark = " *" if abs(r - r_range) < 1e-9 else ""
        print(
            f"  {r:>8.3g}{s_now:>9.3f}{s_block:>10.3f}{xcat:>11.0%}"
            f"{stor(s_now):>9.1f}{stor(s_block):>10.1f}{mark}"
        )

    # headline at the signal-preserving radius
    within = dist <= r_range
    s_now = float(within.mean())
    s_block = float((within & same).mean())
    print("\n== headline (at variogram range, i.e. keeping the signal) ==")
    print(f"  radius = {r_range:.3g}")
    print(
        f"  without trick: s*={s_now:.3f}  storage={stor(s_now):.1f} TB  "
        f"N_break={nbreak(s_now):.2e}"
    )
    print(
        f"  with trick:    s*={s_block:.3f}  storage={stor(s_block):.1f} TB  "
        f"N_break={nbreak(s_block):.2e}"
    )
    if s_now > 0:
        print(
            f"  => cross-category zeroing removes {1-s_block/s_now:.0%} of nonzeros, "
            f"pushes feasible N by {nbreak(s_block)/nbreak(s_now):.2f}x"
        )
    fits = stor(s_block) <= BUDGET_TB
    print(
        f"  => 4M {'FITS' if fits else 'still does NOT fit'} within budget with the trick"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument("--n", type=int, default=30000, help="load size (embedding fit)")
    ap.add_argument("--n_diag", type=int, default=4000, help="pairwise-analysis size")
    ap.add_argument("--n_components", type=int, default=10)
    ap.add_argument("--reducer", default="pls", choices=["pls", "pca"])
    ap.add_argument("--target_N", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = load_for_analysis(a.src, a.n, a.seed)
    batch = MolBatch(
        Z_lists=data["Z"],
        graphs=data["graphs"],
        positions_list=data["positions"],
        charges_list=data["charges"],
        y_total=data["y"],
        net_charges=data["net_charges"],
        spins=data["spins"],
    )
    pre = HybridPreprocessor(n_components=a.n_components, reducer_method=a.reducer)
    X, resid = pre.fit(batch)

    rng = np.random.default_rng(a.seed)
    m = min(a.n_diag, len(X))
    sel = rng.choice(len(X), size=m, replace=False)
    block_sparsity_report(X[sel], data["data_id"][sel], resid[sel], target_N=a.target_N)


if __name__ == "__main__":
    main()
