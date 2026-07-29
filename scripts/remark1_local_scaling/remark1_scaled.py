"""Remark 1 re-run on LOCALLY SCALED distances -- the gate before any non-stationary
kernel work.

THE HYPOTHESIS. Chemical families overlap in the descriptor (within-family median 0.669
vs across 0.815, a 1.22x separation, ~30% of cross pairs closer than the typical within
pair). One explanation is scale mismatch rather than genuine interpenetration: family A is
tight and family B diffuse (per-family 10-NN distance spans 2.3x), so a cross pair A-B can
be closer than a typical within-B pair purely because B is spread out. If that is the
cause, normalising each point by its own local scale should pull the families apart.

THE NORMALISATION is Zelnik-Manor & Perona local scaling,

    d~(i,j) = d(i,j) / sqrt(sigma_i sigma_j),   sigma_i = distance to i's k-th nearest neighbour

and sigma_i is computed **ignoring the category labels**. That is load-bearing. Deriving
sigma from data_id would encode the family structure we are asking the geometry to reveal,
and would guarantee a positive result that means nothing -- the same imposed-vs-discovered
confusion the block mask already represents. Unsupervised sigma keeps this a discovery test.

THE GATE. The separation ratio must move materially above the stationary 1.22x, and
P(cross pair < within-family median) must fall well below 0.30. If neither moves, the
overlap is genuine interpenetration, local scaling cannot fix it, and no non-stationary
kernel work is justified on sparsity-discovery grounds.

Note what this test can and cannot settle. It asks whether the SPARSITY the position
paper's sec 3 is about becomes discoverable. It does NOT bear on whether an adaptive
bandwidth improves fit or UQ -- our kernel already zeroes cross-category pairs exactly, so
per-category radii are a separate (and PSD-trivial) question that this cannot answer
either way.
"""
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

SCR = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 123)
SUB = 4000
SIGMA_KS = (7, 10, 20, 50)


def local_scale(D, k):
    """sigma_i = distance to the k-th nearest neighbour of i, categories IGNORED."""
    Dk = D.copy()
    np.fill_diagonal(Dk, np.inf)
    return np.sort(Dk, axis=1)[:, k - 1]


def stats(D, same):
    """Separation statistics on whatever distance matrix is handed in."""
    iu = np.triu_indices_from(D, k=1)
    dd, ss = D[iu], same[iu]
    w, x = dd[ss], dd[~ss]
    med_w, med_x = float(np.median(w)), float(np.median(x))
    return dict(within=med_w, across=med_x, ratio=med_x / med_w,
                p_cross_below=float((x < med_w).mean()))


def knn_purity(D, same, ks=(5, 10, 25, 50)):
    Dk = D.copy()
    np.fill_diagonal(Dk, np.inf)
    order = np.argsort(Dk, axis=1)
    r = np.arange(len(D))[:, None]
    return {k: float(np.mean(same[r, order[:, :k]])) for k in ks}


def dip_probe(D, same, k=10):
    """The original Remark 1 modality probe, same-category k-NN distances: fraction of
    mass at < p50/4. A genuine near-zero mode shows up as substantial mass there."""
    Ds = np.where(same, D, np.inf)
    np.fill_diagonal(Ds, np.inf)
    kd = np.sort(Ds, axis=1)[:, k - 1]
    kd = kd[np.isfinite(kd)]
    p50 = np.median(kd)
    return float((kd < p50 / 4).mean())


def main():
    from wl_gp2scale.data import get_data
    ds = get_data(src="train_4M", n=20_000, seed=0)
    idx = np.arange(len(ds.atoms))

    for seed in SEEDS:
        d = np.load(os.path.join(SCR, f"emb_20000_s{seed}.npz"), allow_pickle=True)
        tr, _ = train_test_split(idx, test_size=0.2, random_state=seed)
        cat = ds.data_id[tr]
        rng = np.random.default_rng(0)
        sub = rng.choice(len(cat), size=SUB, replace=False)
        c = cat[sub]
        same = c[:, None] == c[None, :]

        print("=" * 104)
        print(f"SPLIT SEED {seed}   (n={SUB} subsample, chance same-family "
              f"= {np.mean(same[np.triu_indices(SUB, 1)]):.3f})")
        print("=" * 104)

        embs = {"wl": d["wl_tr"][sub], "geom": d["geom_tr"][sub],
                "wl+geom": np.hstack([d["wl_tr"], d["geom_tr"]])[sub]}

        for name, Z in embs.items():
            D = cdist(Z, Z)
            base = stats(D, same)
            pur = knn_purity(D, same)
            dip = dip_probe(D, same)
            print(f"\n  --- {name} ---")
            print(f"  {'scaling':>18} {'within':>8} {'across':>8} {'ratio':>7} "
                  f"{'P(cross<med_w)':>15} {'k5 pure':>8} {'k50 pure':>9} {'dip':>7}")
            print(f"  {'stationary':>18} {base['within']:>8.4f} {base['across']:>8.4f} "
                  f"{base['ratio']:>7.3f} {base['p_cross_below']:>15.3f} "
                  f"{pur[5]:>8.3f} {pur[50]:>9.3f} {dip:>7.3f}")
            for k in SIGMA_KS:
                sig = local_scale(D, k)
                Dt = D / np.sqrt(np.outer(sig, sig))
                s = stats(Dt, same)
                p = knn_purity(Dt, same)
                dp = dip_probe(Dt, same)
                flag = "  <== GATE PASSED" if (s["ratio"] >= 1.5 and
                                               s["p_cross_below"] <= 0.20) else ""
                print(f"  {'local sigma k=' + str(k):>18} {s['within']:>8.4f} "
                      f"{s['across']:>8.4f} {s['ratio']:>7.3f} "
                      f"{s['p_cross_below']:>15.3f} {p[5]:>8.3f} {p[50]:>9.3f} "
                      f"{dp:>7.3f}{flag}")

        if seed == SEEDS[0]:
            # does the per-family SCALE spread -- the thing local scaling is meant to
            # remove -- actually shrink? If not, the normalisation is not doing its job.
            Z = embs["wl+geom"]
            D = cdist(Z, Z)
            print(f"\n  --- does local scaling equalise the per-family scale? "
                  f"(wl+geom, sigma k=10) ---")
            sig = local_scale(D, 10)
            Dt = D / np.sqrt(np.outer(sig, sig))
            print(f"  {'family':<20} {'n':>6} {'med 10-NN stat':>15} {'scaled':>10}")
            rows = []
            for ci in range(len(ds.category_names)):
                m = c == ci
                if m.sum() < 40:
                    continue
                for M, key in ((D, 0), (Dt, 1)):
                    sm = M[np.ix_(m, m)].copy()
                    np.fill_diagonal(sm, np.inf)
                    v = float(np.median(np.sort(sm, axis=1)[:, 9]))
                    if key == 0:
                        a = v
                    else:
                        b = v
                rows.append((ds.category_names[ci], int(m.sum()), a, b))
            rows.sort(key=lambda r: r[2])
            for nm, n, a, b in rows:
                print(f"  {nm:<20} {n:>6} {a:>15.4f} {b:>10.4f}")
            sa = rows[-1][2] / rows[0][2]
            sb = max(r[3] for r in rows) / min(r[3] for r in rows)
            print(f"  --> per-family scale spread: {sa:.1f}x stationary -> {sb:.1f}x scaled")
        print()

    print("=" * 104)
    print("GATE: separation ratio >= 1.5x AND P(cross < within median) <= 0.20")
    print("Stationary baseline is 1.22x / ~0.30. If nothing clears it, the family overlap")
    print("is genuine interpenetration, not scale mismatch, and a non-stationary length")
    print("scale cannot make the sec-3 sparsity discoverable.")


if __name__ == "__main__":
    main()
