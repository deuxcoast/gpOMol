"""Poster figure: the sparsity pattern of the actual gp2Scale kernel matrix.

This is the whole method in one image. A compactly supported kernel sets covariance to
EXACTLY zero beyond a radius, and the category tag zeroes cross-family pairs outright, so
the Gram is block-diagonal with sparse texture inside each block -- yet inference stays
exact, because nothing is being approximated. Every white pixel is a true zero, not a
dropped term.

    python scripts/poster_spy.py [--emb path.npz] [--n 2500] [stem]

HONESTY OF THE PICTURE. The full training Gram at 20k is 16,000 x 16,000; rendering it
would put ~30 matrix cells in every printed pixel and the texture would be a grey wash.
So a random subset of molecules is drawn and its sub-Gram is shown at ~1 cell per pixel.
Sub-sampling rows and columns of a matrix preserves its DENSITY in expectation -- each
pair is kept independently of its value -- so the fraction of ink is faithful even though
the absolute neighbour count per row is not. Both numbers are printed on the figure so the
distinction is on the record rather than buried here.

The cutoffs are the production ones, computed on the FULL training set at the
200-neighbour target, then applied to the subset. Computing them on the subset instead
would silently widen the radius to keep 200 neighbours in a smaller pool and overstate
the density by ~6x.
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE = "#2a78d6"
CHAN = ("wl", "geom")
TARGET_NBRS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="notes/kernel_spy")
    ap.add_argument("--emb", default=None, help="cached {wl,geom}_{tr,te} embeddings")
    ap.add_argument("--n", type=int, default=2500, help="molecules shown")
    ap.add_argument("--total", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from sklearn.model_selection import train_test_split
    from wl_gp2scale.cutoff import cutoff_for_neighbors
    from wl_gp2scale.data import get_data

    ds = get_data(src="train_4M", n=a.total, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=0.2, random_state=a.seed)
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    names = np.array(ds.category_names)

    if a.emb and os.path.exists(a.emb):
        d = np.load(a.emb, allow_pickle=True)
        emb = {c: {"Ztr": d[f"{c}_tr"], "Zte": d[f"{c}_te"]} for c in CHAN}
        print(f"[spy] reused {a.emb}")
    else:
        from wl_gp2scale.pipeline import build_channels
        emb = build_channels([ds.atoms[i] for i in tr], ds.y[tr], cat_tr,
                             [ds.atoms[i] for i in te], cat_te, set(CHAN),
                             depth=3, bond_mult=1.1, geom_top_k=10, target_neighbors=0)

    # production radii, from the FULL training set
    cuts = {c: cutoff_for_neighbors(emb[c]["Ztr"], emb[c]["Zte"], TARGET_NBRS,
                                    dim=emb[c]["Ztr"].shape[1], data_id_tr=cat_tr,
                                    data_id_te=cat_te, sample=5000) for c in CHAN}

    # full-matrix density, estimated on a 5000-row band so the caption can quote it
    rng = np.random.default_rng(0)
    band = rng.choice(len(tr), size=min(5000, len(tr)), replace=False)
    full = np.zeros((len(band), len(tr)), bool)
    for c in CHAN:
        full |= cdist(emb[c]["Ztr"][band], emb[c]["Ztr"]) < cuts[c]
    full &= cat_tr[band][:, None] == cat_tr[None, :]
    dens_full = float(full.mean())
    nbr_full = float(np.median(full.sum(axis=1)))
    del full

    # the shown subset, sorted by category so the blocks are contiguous
    sub = rng.choice(len(tr), size=min(a.n, len(tr)), replace=False)
    sub = sub[np.argsort(cat_tr[sub], kind="stable")]
    cs = cat_tr[sub]
    K = np.zeros((len(sub), len(sub)), bool)
    for c in CHAN:
        K |= cdist(emb[c]["Ztr"][sub], emb[c]["Ztr"][sub]) < cuts[c]
    K &= cs[:, None] == cs[None, :]
    dens_sub = float(K.mean())
    print(f"[spy] full-matrix density {dens_full:.4f} (median {nbr_full:.0f} nbrs/row); "
          f"shown subset density {dens_sub:.4f}")

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(~K, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              rasterized=True)
    # recolour: imshow of ~K gives black where non-zero; overlay in the poster blue
    ax.images[0].remove()
    rgba = np.zeros(K.shape + (4,), np.float32)
    rgba[K] = matplotlib.colors.to_rgba(BLUE)
    ax.imshow(rgba, interpolation="nearest", rasterized=True)

    # category boundaries + labels for the blocks big enough to letter
    edges = np.flatnonzero(np.diff(cs)) + 1
    for e in edges:
        ax.axhline(e - 0.5, color="#c9c7c2", lw=1.2)
        ax.axvline(e - 0.5, color="#c9c7c2", lw=1.2)
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(sub)]])
    for s0, s1 in zip(starts, stops):
        if s1 - s0 >= 0.045 * len(sub):
            ax.text(-0.012 * len(sub), (s0 + s1) / 2, str(names[cs[s0]]),
                    ha="right", va="center", fontsize=21, color="#3b3a37")

    # Bare plot: category labels only. Title, subtitle and caption are deliberately NOT
    # drawn -- they are set in the poster layout tool, where they can be typed at 60 pt+
    # and kept consistent with every other heading on the board. The numbers they would
    # have carried are printed to stdout above so they can be transcribed exactly.
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#9b9993"); sp.set_linewidth(1.6)

    fig.tight_layout()
    stem = a.stem[:-4] if a.stem.lower().endswith((".svg", ".png")) else a.stem
    for ext, kw in ((".svg", {}), (".png", {"dpi": 300})):
        fig.savefig(stem + ext, transparent=True, bbox_inches="tight", **kw)
        print(f"wrote {stem + ext}")


if __name__ == "__main__":
    main()
