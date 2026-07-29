"""Fit a real GP with the WL kernel PROPER and score it against the radius kernel.

    k(G, G') = <phi^(h)(G), phi^(h)(G')>

the deepest-block-only variant of the position paper's Eq. 7. This is the first kernel in
this project whose sparsity is DISCOVERED rather than chosen: k = 0 exactly when two
molecules share no depth-h subtree pattern. No radius, no Wendland envelope, no
compact-support hyperparameter, and PSD by construction (it is a Gram matrix).

WHY THE INPUTS ARE INDICES. phi^(h) has ~10^5 columns, so it cannot be handed to fvgp as
`x_data`. Instead x = [row_id, data_id] and the kernel looks rows up in a sparse matrix it
carries. This works because sort_by_category permutes ROWS -- the ids travel with them --
and because the local dask client runs threads (processes=False), so the ~5 MB matrix is
shared rather than pickled per task.

ARMS, all sharing the M4 prior mean on the SAME cached PLS embeddings, the same 800 test
points, the same jitter, and the same seed as the radius grid -- so the only thing that
changes is the kernel:

  h=4 cosine, category-masked   -- the practical variant, directly comparable to the grid
  h=4 cosine, NO MASK           -- the DISCOVERY test. Our production kernel zeroes
                                   cross-category pairs by fiat (kernel.py:242); dropping
                                   that is the only way to ask whether the WL kernel finds
                                   chemistry on its own, which is what sec 3 actually asks.
  h=5 cosine, category-masked   -- density-matched to our K=200 radius kernel (2.5e-2)
  h=4 raw counts, masked        -- unnormalised, as Eq. 7 writes it. The diagonal then
                                   varies as ||phi||^2 (extensive); cosine fixes it at 1.

The radius-kernel baseline is NOT re-run: arm_s42_k200.npz already holds its mu/err/v_gp
on exactly these test points, so it is reused directly and the comparison is paired.

Traps guarded (see [[fvgp-gotchas]]): krylov mode asserted single; finite solve_maxiter
under SolveWarningCounter; add_noise is NOT applied by predict, so sigma*+noise is formed
here; a fresh Client per GP.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")
N, SEED, NTE, JITTER = 20_000, 42, 800, 4.30
MEAN_CHAN = ("wl", "geom", "strain")
SOLVE = dict(linalg_mode="sparseCG", solve_maxiter=2000, solve_tol=1e-6,
             batch_size=10_000, compute_device="cpu", device="cpu")


class WLSubtreeKernel:
    """k(G,G') = sv * <phi(G), phi(G')>, optionally cosine-normalised and/or
    category-masked. x[:, 0] is a row index into ``Phi``; x[:, 1] is the category tag."""

    def __init__(self, Phi, cats, normalize=True, use_category_tag=True):
        self.Phi = sp.csr_matrix(Phi)
        self.cats = np.asarray(cats)
        self.normalize = normalize
        self.use_category_tag = use_category_tag
        nrm = np.sqrt(self.Phi.multiply(self.Phi).sum(axis=1)).A.ravel()
        # a molecule with no in-vocabulary pattern has ||phi|| = 0. Its row of K is zero
        # whatever we do; the guard only stops a 0/0 in the cosine and keeps that row
        # exactly zero, i.e. the GP reverts to the prior mean there -- which is the honest
        # behaviour, not a patch over it.
        self.n_empty = int((nrm == 0).sum())
        self.nrm = np.where(nrm > 0, nrm, 1.0)

    def __call__(self, x1, x2, hps):
        sv = float(hps[0])
        i1 = np.asarray(x1[:, 0], dtype=int)
        i2 = np.asarray(x2[:, 0], dtype=int)
        K = (self.Phi[i1] @ self.Phi[i2].T).toarray()
        if self.normalize:
            K /= np.outer(self.nrm[i1], self.nrm[i2])
        K *= sv
        if self.use_category_tag:
            K = np.where(self.cats[i1][:, None] == self.cats[i2][None, :], K, 0.0)
        return K


def build_phi(atoms_tr, atoms_te, depth, min_count=1, bond_mult=1.1):
    """Fit the WL vocabulary on TRAIN ONLY, transform both, return the deepest block."""
    from wl_gp2scale.wl_features import SparseWLFeaturizer
    F = SparseWLFeaturizer(depth=depth, min_count=min_count, cutoff_mult=bond_mult,
                           perceiver="ase", normalize=False).fit(atoms_tr)
    lo, hi = F.offsets_[depth], F.ncols_
    A = F.transform(atoms_tr, chunk=500)[:, lo:hi]
    B = F.transform(atoms_te, chunk=500)[:, lo:hi]
    return sp.vstack([A, B]).tocsr(), F


def main():
    from distributed import Client
    from fvgp.gp_lin_alg import _resolve_krylov_mode
    from wl_gp2scale.data import get_data
    from wl_gp2scale.reduce import LinearEmbeddingMean
    from wl_gp2scale.pipeline import (build_gp, predict, release_gp, sort_by_category,
                                      SolveWarningCounter)
    assert _resolve_krylov_mode(None) == "single"

    ds = get_data(src="train_4M", n=N, seed=0)
    idx = np.arange(len(ds.atoms))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    y_tr, y_te = ds.y[tr], ds.y[te]
    cat_tr, cat_te = ds.data_id[tr], ds.data_id[te]
    size_tr = np.array([len(ds.atoms[i]) for i in tr], float)
    size_te = np.array([len(ds.atoms[i]) for i in te], float)
    sel = np.random.default_rng(0).choice(len(y_te), NTE, replace=False)

    # the SAME prior mean the radius grid used, on the SAME cached PLS embeddings
    d = np.load(os.path.join(SCR, f"emb_20000_s{SEED}.npz"), allow_pickle=True)
    Zm_tr = np.hstack([d[f"{n}_tr"] for n in MEAN_CHAN])
    Zm_te = np.hstack([d[f"{n}_te"] for n in MEAN_CHAN])
    mean = LinearEmbeddingMean().fit(Zm_tr, y_tr, size=size_tr)
    r_tr = y_tr - mean.predict(Zm_tr, size=size_tr)
    prior_var = float(np.var(r_tr))
    mu_mean_te = mean.predict(Zm_te, size=size_te)[sel]
    v_mean = mean.predict_var(Zm_te, size=size_te)[sel]

    atoms_tr = [ds.atoms[i] for i in tr]
    atoms_te = [ds.atoms[i] for i in te]
    ntr = len(tr)
    cats_all = np.concatenate([cat_tr, cat_te])

    base = np.load(os.path.join(ARMS, "base_s42.npz"))
    ref = np.load(os.path.join(ARMS, "arm_s42_k200.npz"))
    print(f"[ref] radius kernel K=200: R2={r2_score(ref['y'], ref['mu']):.4f}  "
          f"rho(sigma*,|err|)={spearmanr(np.sqrt(ref['v_gp']), np.abs(ref['err'])).statistic:+.4f}  "
          f"| dist-to-NN rho="
          f"{spearmanr(base['dnn'], np.abs(ref['err'])).statistic:+.4f}\n")

    phi_cache = {}
    arms = [("h=4 cosine, masked", 4, True, True),
            ("h=4 cosine, NO MASK", 4, True, False),
            ("h=5 cosine, masked", 5, True, True),
            ("h=4 raw counts, masked", 4, False, True)]
    rows = []
    for label, h, norm, mask in arms:
        if h not in phi_cache:
            t0 = time.time()
            phi_cache[h] = build_phi(atoms_tr, atoms_te, h)
            print(f"[phi] depth {h} built in {time.time()-t0:.0f}s")
        Phi, F = phi_cache[h]
        kern = WLSubtreeKernel(Phi, cats_all, normalize=norm, use_category_tag=mask)

        # realised structure of THIS kernel on the training set, before any GP
        A = Phi[:ntr]
        A1 = A.copy(); A1.data[:] = 1.0
        Ktr = (A1 @ A1.T).tocoo()
        keep = Ktr.row != Ktr.col
        rr, cc = Ktr.row[keep], Ktr.col[keep]
        if mask:
            m = cat_tr[rr] == cat_tr[cc]
            rr, cc = rr[m], cc[m]
        dens = len(rr) / (ntr * (ntr - 1))
        deg = np.bincount(rr, minlength=ntr)
        purity = float((cat_tr[rr] == cat_tr[cc]).mean()) if len(rr) else float("nan")

        X = np.column_stack([np.arange(len(cats_all), dtype=float), cats_all])
        Xtr = X[:ntr]
        sv = prior_var if norm else prior_var / float(
            np.median(np.asarray(A.multiply(A).sum(axis=1)).ravel()) or 1.0)
        cl = Client(n_workers=4, threads_per_worker=1, processes=False,
                    dashboard_address=None, silence_logs=50)
        t0 = time.time()
        try:
            Xs, ys, _ = sort_by_category(Xtr, r_tr)
            with SolveWarningCounter() as wc:
                gp, _ = build_gp(Xs, ys, None, None, cl, kernel_override=kern,
                                 signal_var=[sv], jitter=JITTER, **SOLVE)
                print(f"[arm] {label}: build={time.time()-t0:.0f}s, {NTE} var solves")
                m_gp, v_gp = predict(gp, X[ntr:][sel], batch=200, variance=True)
            del gp
            release_gp(cl)
        finally:
            cl.close()
        wc.report(n_evals=NTE, label=label)

        mu = mu_mean_te + m_gp
        err = ref["y"] - mu
        sg = np.sqrt(np.maximum(v_gp, 0))
        r2 = r2_score(ref["y"], mu)
        rho = spearmanr(sg, np.abs(err)).statistic
        k5 = max(int(0.05 * NTE), 1)
        rec = len(set(np.argsort(np.abs(err))[-k5:]) & set(np.argsort(sg)[-k5:])) / k5
        rows.append((label, dens, purity, np.mean(deg == 0), np.median(deg),
                     kern.n_empty, r2, rho, rec, time.time() - t0))
        print(f"[arm] {label}: density={dens:.3e} purity={purity:.3f} "
              f"isolated={np.mean(deg==0):.1%} med_deg={np.median(deg):.0f} "
              f"R2={r2:.4f} rho={rho:+.4f} recall={rec:.3f} "
              f"({time.time()-t0:.0f}s)\n")

    print("=" * 108)
    print(f"{'kernel':<26} {'density':>10} {'purity':>8} {'isolated':>9} {'med deg':>8} "
          f"{'R2':>8} {'rho(s*,|e|)':>12} {'top5%':>7}")
    print("-" * 108)
    rr = ref
    print(f"{'radius Wendland K=200':<26} {2.344e-2:>10.3e} {1.000:>8.3f} "
          f"{0.000:>9.1%} {538:>8.0f} {r2_score(rr['y'], rr['mu']):>8.4f} "
          f"{spearmanr(np.sqrt(rr['v_gp']), np.abs(rr['err'])).statistic:>+12.4f} "
          f"{0.158:>7.3f}")
    for (lab, dn, pu, iso, md, ne, r2, rho, rec, t) in rows:
        print(f"{lab:<26} {dn:>10.3e} {pu:>8.3f} {iso:>9.1%} {md:>8.0f} "
              f"{r2:>8.4f} {rho:>+12.4f} {rec:>7.3f}")
    print(f"\n{'dist-to-NN baseline (no GP)':<26} {'':>10} {'':>8} {'':>9} {'':>8} "
          f"{'':>8} {spearmanr(base['dnn'], np.abs(rr['err'])).statistic:>+12.4f}")
    print("\npurity here is P(same category | connected) on the TRAIN Gram; the masked")
    print("arms are 1.000 by construction, so only the NO MASK row is a discovery claim.")


if __name__ == "__main__":
    main()
