"""
channel_variograms.py
=====================
Attribute the global-vs-local behaviour to a specific descriptor channel.

Marcus asked for a variogram on "WL distance" to see whether WL is locally
informative about energy or globally degenerate. But the pipeline has no
standalone WL distance -- WL is one feature block, concatenated with geometry and
charge, PLS-reduced, then compared by Euclidean distance. A WL-only variogram
alone can't tell you which channel carries the long-range structure. So this runs
the SAME Delta-y^2-vs-distance protocol per channel:

    wl        : Euclidean distance on the (standardized) WL feature block
    geometry  : ... on the distance-histogram block
    charge    : ... on the Loewdin-charge block
    hybrid_raw: ... on the full concatenated standardized descriptor
    hybrid_pls: ... on the PLS-reduced embedding the model actually uses

For each channel it reports nugget/sill (how much variance is reachable at all),
the variogram range, and locality = range / max_lag: locality near 1 means the
correlation is GLOBAL (persists across nearly the whole distance range -> fights
compact-support sparsity); small locality means LOCAL structure (sparsity-
friendly). It also times the pairwise distance computation (Marcus's perf check).

Note on WL cost: this uses the HASHED WL feature vectors (already computed), so
distance is a fast Euclidean op -- distinct from a true WL graph-kernel distance,
which would be far more expensive. If Marcus meant the graph-kernel WL, that's a
separate, heavier test.

Usage
-----
    python channel_variograms.py --src ../train_4M --n 1500
"""

import argparse
import time

import numpy as np
from analyze_residual import load_for_analysis
from diagnostics import _pairwise_euclidean, semivariogram
from embedding_kernel import FeatureReducer
from extensive_mean import ExtensiveEnergyModel
from features import HybridFeatureAssembler
from gp_fit import charge_spin_features


def channel_variogram_report(channels: dict, resid: np.ndarray):
    print("\n== per-channel variograms (Delta-y^2 vs distance on that channel) ==")
    print(
        f"  {'channel':<14}{'dims':>6}{'nugget/sill':>13}{'range':>9}"
        f"{'max_lag':>9}{'locality':>10}{'pair_ms':>9}"
    )
    for name, Xc in channels.items():
        Xc = np.ascontiguousarray(Xc, dtype=float)
        t0 = time.perf_counter()
        _ = _pairwise_euclidean(Xc)  # the pairwise cost Marcus asked about
        pair_ms = (time.perf_counter() - t0) * 1e3
        vg = semivariogram(Xc, resid)
        max_lag = float(vg["lags"][-1])
        locality = vg["range"] / max_lag if max_lag > 0 else float("nan")
        print(
            f"  {name:<14}{Xc.shape[1]:>6}{vg['nugget_over_sill']:>13.3f}"
            f"{vg['range']:>9.3g}{max_lag:>9.3g}{locality:>10.2f}{pair_ms:>9.1f}"
        )
    print(
        "\n  locality ~1 => GLOBAL correlation (fights compact-support sparsity);"
        "  small => LOCAL (sparsity-friendly)."
    )
    print(
        "  A high nugget/sill on a channel => that channel's distance barely"
        " tracks the energy residual at all."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default="../train_4M")
    ap.add_argument(
        "--n", type=int, default=1500, help="sample size (pairwise is O(n^2))"
    )
    ap.add_argument("--pls_components", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = load_for_analysis(a.src, a.n, a.seed)
    ctx = list(zip(data["net_charges"], data["spins"]))
    mm = ExtensiveEnergyModel(extra_feature_fn=charge_spin_features).fit(
        data["Z"], data["y"], extra_context=ctx
    )
    resid = mm.residual(data["Z"], data["y"], extra_context=ctx)

    asm = HybridFeatureAssembler()
    Xraw = asm.fit_transform(data["graphs"], data["positions"], data["charges"])
    sl = asm.slices_
    pls = (
        FeatureReducer(n_components=a.pls_components, method="pls")
        .fit(Xraw, resid)
        .transform(Xraw)
    )

    channels = {
        "wl": Xraw[:, sl["wl"]],
        "geometry": Xraw[:, sl["hist"]],
        "charge": Xraw[:, sl["charge"]],
        "hybrid_raw": Xraw,
        "hybrid_pls": pls,
    }
    print(f"[loaded] n={len(resid)}  residual_var={np.var(resid):.4g}")
    channel_variogram_report(channels, resid)


if __name__ == "__main__":
    main()
