"""
diagnostics.py
==============

Step 4: cheap, falsification-first checks on a ~1e5 subsample. Run ALL of these
on the *intensive residual* (extensive_mean.residual), in this order, BEFORE
committing to any full GP fit. Each has an explicit number that kills the
candidate.

Terminology (fixed here to avoid the two-convention trap):
    density  s*  = fraction of molecule pairs that fall WITHIN the support radius
                 = fraction of NON-zero covariance-matrix entries.
    sparsity      = 1 - s* = fraction of zeros.
    Storage grows with density:  bytes ~ 12 * s* * N^2  (CSR, ~12 B / nonzero).
    So you want s* SMALL. The mid-scale sparsity floor is s* < 0.5.

The four checks and their kill numbers
--------------------------------------
  A. kNN skill vs WL-only : hybrid must beat WL by >= REL_GAIN_MIN (relative).
     If it doesn't, the geometry/charge channels added no usable information.
  B. semivariogram nugget : nugget/sill must be < NUGGET_MAX (= WL's ~0.12 ceiling);
     target <= NUGGET_TARGET for the candidate to be worth pursuing.
  C. kNN distance CV       : coefficient of variation >= CV_MIN, else distances
     have concentrated (curse of dimensionality) and no radius yields sparsity.
  D. feasibility           : with measured s*, storage must fit BUDGET_TB at the
     target N; report the N at which it breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ------------------------------ kill thresholds (edit to taste, documented) ---
REL_GAIN_MIN = 0.10  # A: min relative skill gain of hybrid over WL-only
NUGGET_MAX = 0.12  # B: kill if residual nugget/sill >= this (WL ceiling)
NUGGET_TARGET = 0.09  # B: "worth pursuing" if at/below this
CV_MIN = 0.30  # C: kill if kNN distance CV < this
S_STAR_MAX = 0.50  # D: mid-scale sparsity floor (density must be under this)
BUDGET_TB = 40.0  # D: kernel-matrix storage budget
BYTES_PER_NNZ = 12.0  # D: CSR ~ 8B value + 4B column index


# ----------------------------------------------------------------------------
# Small, dependency-light helpers (pairwise on a subsample; O(n^2) is fine here)
# ----------------------------------------------------------------------------


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Full (n, n) Euclidean distance matrix. Intended for n up to a few thousand."""
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _knn_indices(D: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours (excluding self) for each row."""
    order = np.argsort(D, axis=1)
    return order[:, 1 : k + 1]


def density_at_radius(X: np.ndarray, radius: float) -> float:
    """Fraction of molecule pairs within `radius` == matrix density s* == the
    non-zero fraction of the covariance matrix at that support radius. This is the
    quantity that drives storage; feed it the variogram RANGE (correlation length),
    which is the physically meaningful support radius, NOT an arbitrary percentile."""
    X = np.asarray(X, dtype=float)
    D = _pairwise_euclidean(X)
    n = len(X)
    off = D[~np.eye(n, dtype=bool)]
    return float(np.mean(off <= radius))


# ----------------------------------------------------------------------------
# A. kNN predictive skill  (hybrid embedding vs WL-only embedding)
# ----------------------------------------------------------------------------


def knn_skill(X: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """
    Leave-one-out kNN regression skill = 1 - MSE/Var(y)  (an R^2-like score;
    0 == no better than predicting the mean, 1 == perfect). Uses the full
    pairwise matrix, so keep the subsample to a few thousand for this check.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    D = _pairwise_euclidean(X)
    nn = _knn_indices(D, k)
    y_hat = y[nn].mean(axis=1)
    mse = np.mean((y - y_hat) ** 2)
    var = np.var(y)
    return float(1.0 - mse / var) if var > 0 else 0.0


def knn_skill_gain(
    X_hybrid: np.ndarray, X_wl: np.ndarray, y: np.ndarray, k: int = 10
) -> dict:
    """Compare hybrid vs WL-only kNN skill. Kill A if relative gain < REL_GAIN_MIN."""
    s_h = knn_skill(X_hybrid, y, k)
    s_w = knn_skill(X_wl, y, k)
    rel = (s_h - s_w) / abs(s_w) if s_w not in (0.0,) else np.inf
    return {
        "skill_hybrid": s_h,
        "skill_wl": s_w,
        "relative_gain": float(rel),
        "passed": bool((s_h > s_w) and (rel >= REL_GAIN_MIN)),
        "note": (
            "KILL: hybrid adds no usable signal over WL"
            if not ((s_h > s_w) and (rel >= REL_GAIN_MIN))
            else "ok"
        ),
    }


# ----------------------------------------------------------------------------
# B. Semivariogram (nugget / sill / range) on the residual
# ----------------------------------------------------------------------------


def semivariogram(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    max_dist: Optional[float] = None,
    max_pairs: int = 2_000_000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Empirical semivariance gamma(h) = 0.5 * mean[(y_i - y_j)^2] over pairs binned
    by distance h. Reads off:
        nugget  ~ gamma at the shortest lag (unreachable variance / noise floor)
        sill    ~ gamma at large lag (total variance)
        range   ~ lag where gamma reaches ~95% of sill
        nugget/sill = the fraction of variance the descriptor CANNOT reach.

    Kill B if nugget/sill >= NUGGET_MAX. This is the direct test of whether the
    hybrid descriptor breaks WL's ~2 eV / ~12% ceiling.
    """
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(X)

    # sample pairs to bound cost on larger subsamples
    n_all = n * (n - 1) // 2
    if n_all <= max_pairs:
        iu = np.triu_indices(n, k=1)
        ii, jj = iu[0], iu[1]
    else:
        ii = rng.integers(0, n, size=max_pairs)
        jj = rng.integers(0, n, size=max_pairs)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]

    diff = X[ii] - X[jj]
    h = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    semis = 0.5 * (y[ii] - y[jj]) ** 2

    if max_dist is None:
        max_dist = np.quantile(h, 0.95)
    edges = np.linspace(0.0, max_dist, n_bins + 1)
    which = np.clip(np.digitize(h, edges) - 1, 0, n_bins - 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.array(
        [
            semis[which == b].mean() if np.any(which == b) else np.nan
            for b in range(n_bins)
        ]
    )

    valid = ~np.isnan(gamma)
    nugget = float(gamma[valid][0])  # shortest populated lag
    sill = float(np.nanmax(gamma))
    ratio = nugget / sill if sill > 0 else np.inf
    # range: first lag reaching 95% of sill
    reach = np.where(valid & (gamma >= 0.95 * sill))[0]
    rng_val = float(centres[reach[0]]) if len(reach) else float(centres[valid][-1])

    return {
        "nugget": nugget,
        "sill": sill,
        "nugget_over_sill": float(ratio),
        "range": rng_val,
        "lags": centres,
        "gamma": gamma,
        "passed": bool(ratio < NUGGET_MAX),
        "worth_pursuing": bool(ratio <= NUGGET_TARGET),
        "note": (
            "KILL: nugget >= WL ceiling, no reachable variance beyond WL"
            if ratio >= NUGGET_MAX
            else (
                "ok"
                if ratio <= NUGGET_TARGET
                else "marginal: below kill line but above target"
            )
        ),
    }


# ----------------------------------------------------------------------------
# C. kNN distance distribution — CV and sparsity potential
# ----------------------------------------------------------------------------


def knn_distance_stats(
    X: np.ndarray,
    k: int = 10,
    support_radius: Optional[float] = None,
    support_percentiles=(25, 50, 75),
) -> dict:
    """
    Coefficient of variation of pairwise distances (concentration test) plus the
    achievable density s*.

    The MEANINGFUL density is `s_star_at_radius`: the fraction of pairs within
    `support_radius`, where support_radius should be the variogram RANGE (the
    correlation length beyond which covariance is ~0). Pass it in from the
    semivariogram. The per-percentile values are context only — note that s* at
    the p-th percentile is ~p/100 by construction, so a fixed percentile is NOT a
    real sparsity test.

    Kill C if CV < CV_MIN (distances concentrated -> no radius gives sparsity).
    """
    X = np.asarray(X, dtype=float)
    D = _pairwise_euclidean(X)
    n = len(X)
    off = D[~np.eye(n, dtype=bool)]

    mean_d = float(off.mean())
    std_d = float(off.std())
    cv = std_d / mean_d if mean_d > 0 else 0.0

    # context: density at distance percentiles (trivially ~ percentile/100)
    s_star = {}
    for p in support_percentiles:
        r = np.percentile(off, p)
        s_star[p] = float(np.mean(off <= r))

    # the real number: density at the correlation-length support radius
    s_star_at_radius = (
        float(np.mean(off <= support_radius)) if support_radius is not None else None
    )

    # bimodality coefficient (Sarle): >0.555 hints at bimodal (well-separated
    # near/far pairs -> promising for sparsity discovery)
    m = off.mean()
    s = off.std()
    if s > 0:
        g1 = np.mean(((off - m) / s) ** 3)
        g2 = np.mean(((off - m) / s) ** 4) - 3.0
        bc = (g1**2 + 1.0) / (g2 + 3.0)
    else:
        bc = 0.0

    return {
        "cv": float(cv),
        "bimodality_coeff": float(bc),
        "s_star_by_percentile": s_star,
        "s_star_at_radius": s_star_at_radius,
        "support_radius": support_radius,
        "passed_cv": bool(cv >= CV_MIN),
        "note": (
            "KILL: distances concentrated (curse of dimensionality)"
            if cv < CV_MIN
            else "ok"
        ),
    }


# ----------------------------------------------------------------------------
# D. Feasibility — storage vs the 40 TB budget
# ----------------------------------------------------------------------------


def feasibility(s_star: float, target_N: int) -> dict:
    """
    Storage estimate and breaking N for a molecule-level kernel matrix.
        bytes ~ BYTES_PER_NNZ * s_star * N^2
        N_break = sqrt( BUDGET / (BYTES_PER_NNZ * s_star) )
    """
    budget_bytes = BUDGET_TB * 1e12
    storage_bytes = BYTES_PER_NNZ * s_star * (target_N**2)
    n_break = float(np.sqrt(budget_bytes / (BYTES_PER_NNZ * max(s_star, 1e-12))))
    return {
        "s_star": float(s_star),
        "target_N": int(target_N),
        "storage_TB_at_target": storage_bytes / 1e12,
        "N_break": n_break,
        "fits_budget": bool(storage_bytes <= budget_bytes),
        "passed_floor": bool(s_star < S_STAR_MAX),
    }


def sparsity_accuracy_sweep(
    X: np.ndarray, y_resid: np.ndarray, target_N: int = 4_000_000, n_radii: int = 25
) -> dict:
    """
    The decisive tool for the full-4M question: does a support radius exist that is
    BOTH sparse enough to store at target_N AND long enough to capture the
    correlation structure?

    For a sweep of candidate support radii r it reports, at each r:
        s*(r)            = pairwise density = non-zero fraction of the matrix
        storage_TB(r)    = BYTES_PER_NNZ * s*(r) * target_N^2   (vs BUDGET_TB)
        captured(r)      = gamma(r)/sill in [0,1]: how far toward full
                           decorrelation r reaches. captured ~ 1 means pairs beyond
                           r are ~uncorrelated, so truncating support at r is
                           (nearly) free; captured << 1 means truncating at r
                           discards real correlation -> accuracy loss.

    Then it names two radii:
        r_full   = smallest r with captured >= 0.95  (~ the variogram range): the
                   shortest support that keeps essentially all correlation. s* and
                   storage here tell you if EXACT full-accuracy GP fits at target_N.
        r_budget = largest r whose storage <= BUDGET_TB: the most correlation you
                   can afford. captured(r_budget) is the accuracy you'd retain if
                   forced to impose sparsity to fit the budget.

    If r_full fits the budget you are done. If not, the gap between captured(r_full)
    and captured(r_budget) is exactly the accuracy you trade away to run at 4M.
    """
    v = semivariogram(X, y_resid, n_bins=n_radii)
    lags, gamma = v["lags"], v["gamma"]

    D = _pairwise_euclidean(X)
    n = len(X)
    off = D[~np.eye(n, dtype=bool)]

    # captured(r) = how far toward the sill the variogram has climbed by radius r.
    # Robustness: (1) estimate the sill as the MEDIAN of the far-half lags, not a
    # single max bin (a lone noisy far bin shouldn't define it); (2) light moving
    # average; (3) cumulative max, since a variogram is non-decreasing to the sill.
    valid = ~np.isnan(gamma)
    lags_v, gamma_v = lags[valid], gamma[valid]
    half = max(1, len(gamma_v) // 2)
    sill_robust = float(np.median(gamma_v[half:])) or (v["sill"] or 1.0)
    cap_raw = np.clip(gamma_v / sill_robust, 0.0, 1.0)
    smooth = np.convolve(cap_raw, np.ones(3) / 3.0, mode="same")
    cap = np.maximum.accumulate(smooth)
    cap_max = float(cap.max())  # plateau: max *reachable* correlation

    table = []
    for r, c in zip(lags_v, cap):
        s = float(np.mean(off <= r))
        storage_tb = BYTES_PER_NNZ * s * (target_N**2) / 1e12
        table.append(
            {
                "radius": float(r),
                "s_star": s,
                "storage_TB": storage_tb,
                "captured": float(c),
            }
        )

    def _first(pred):
        return next((row for row in table if pred(row)), None)

    def _last(pred):
        return next((row for row in reversed(table) if pred(row)), None)

    # r_full = the KNEE: shortest radius that reaches the plateau (within 2% of
    # cap_max). Using cap_max (not a hardcoded 0.95) means a variogram that
    # saturates BELOW 1.0 -- i.e. has an unstructured floor no radius can reach --
    # is handled correctly: extending support past the knee adds storage but no
    # captured correlation.
    knee = cap_max - 0.02
    r_full = _first(lambda row: row["captured"] >= knee) or table[-1]
    r_budget = _last(lambda row: row["storage_TB"] <= BUDGET_TB)

    full_fits = r_full["storage_TB"] <= BUDGET_TB
    return {
        "target_N": int(target_N),
        "table": table,
        "reachable_correlation": cap_max,  # plateau height (structured frac)
        "unstructured_floor": 1.0 - cap_max,  # variance no support radius reaches
        "r_full": r_full,  # knee: shortest support capturing the plateau
        "r_budget": r_budget,  # most correlation affordable within budget
        "full_accuracy_fits_budget": bool(full_fits),
        "accuracy_traded": (
            None
            if full_fits or r_budget is None
            else r_full["captured"] - r_budget["captured"]
        ),
    }


def format_sweep(sw: dict) -> str:
    L = [
        f"SPARSITY / ACCURACY SWEEP  (target N = {sw['target_N']:.0e}, "
        f"budget {BUDGET_TB:.0f} TB)",
        f"  {'radius':>8}{'s*':>8}{'storage_TB':>12}{'captured':>10}",
    ]
    for row in sw["table"]:
        flag = "" if row["storage_TB"] <= BUDGET_TB else "  <-over budget"
        L.append(
            f"  {row['radius']:>8.3g}{row['s_star']:>8.3f}"
            f"{row['storage_TB']:>12.1f}{row['captured']:>10.2f}{flag}"
        )
    rf, rb = sw["r_full"], sw["r_budget"]
    L.append("-" * 46)
    L.append(
        f"  reachable correlation (plateau): {sw['reachable_correlation']:.2f}"
        f"   unstructured floor: {sw['unstructured_floor']:.2f}"
        f"  (variance no radius can reach)"
    )
    L.append(
        f"  r_full  (knee; captures the plateau): r={rf['radius']:.3g} "
        f"s*={rf['s_star']:.3f} storage={rf['storage_TB']:.1f} TB"
    )
    if sw["full_accuracy_fits_budget"]:
        L.append(
            "  => storable: a support radius captures all REACHABLE "
            "correlation within budget. (Reachable != predictive skill --"
        )
        L.append("     cross-check against kNN skill / retained variance.)")
    elif rb is not None:
        L.append(
            f"  r_budget(max affordable):            r={rb['radius']:.3g} "
            f"s*={rb['s_star']:.3f} captured={rb['captured']:.2f}"
        )
        L.append(
            f"  => full accuracy does NOT fit; imposing sparsity to fit "
            f"budget trades away ~{sw['accuracy_traded']:.0%} of correlation."
        )
    else:
        L.append(
            "  => even the smallest radius exceeds budget at target N; "
            "reduce N or raise the storage budget / node count."
        )
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


@dataclass
class FalsificationReport:
    knn_gain: dict
    variogram: dict
    knn_dist: dict
    feasibility: dict

    @property
    def all_passed(self) -> bool:
        return (
            self.knn_gain["passed"]
            and self.variogram["passed"]
            and self.knn_dist["passed_cv"]
            and self.feasibility["passed_floor"]
        )

    def summary(self) -> str:
        L = []
        g, v, d, f = self.knn_gain, self.variogram, self.knn_dist, self.feasibility
        L.append("FALSIFICATION-FIRST REPORT  (run on the intensive residual)")
        L.append("-" * 62)
        L.append(
            f"A. kNN skill   hybrid={g['skill_hybrid']:.3f} vs WL={g['skill_wl']:.3f}  "
            f"rel_gain={g['relative_gain']:.2f}  "
            f"[{'PASS' if g['passed'] else 'KILL'}]  (need >= {REL_GAIN_MIN:.0%})"
        )
        L.append(
            f"B. semivariogram  nugget/sill={v['nugget_over_sill']:.3f}  "
            f"range={v['range']:.3g}  "
            f"[{'PASS' if v['passed'] else 'KILL'}]  (need < {NUGGET_MAX}, target <= {NUGGET_TARGET})"
        )
        L.append(
            f"C. kNN distance   CV={d['cv']:.3f}  bimodality={d['bimodality_coeff']:.3f}  "
            f"[{'PASS' if d['passed_cv'] else 'KILL'}]  (need CV >= {CV_MIN})"
        )
        L.append(
            f"D. feasibility    s*(@range {d.get('support_radius', float('nan')):.3g})="
            f"{f['s_star']:.3f}  "
            f"storage@N={f['target_N']:.0e}: {f['storage_TB_at_target']:.1f} TB  "
            f"N_break={f['N_break']:.2e}  "
            f"[{'PASS' if f['passed_floor'] else 'KILL'}]  (need s* < {S_STAR_MAX})"
        )
        L.append("-" * 62)
        L.append(
            f"OVERALL: {'PROCEED to GP fit' if self.all_passed else 'DO NOT proceed — a check failed'}"
        )
        return "\n".join(L)


def run_falsification(
    X_hybrid: np.ndarray,
    X_wl: np.ndarray,
    y_resid: np.ndarray,
    target_N: int = 1_500_000,
    k: int = 10,
) -> FalsificationReport:
    """
    One call to gate a candidate. X_hybrid and X_wl should be the reduced
    embeddings of the SAME subsample (X_wl built from the WL channel only), and
    y_resid the intensive residual on that subsample.

    Order matters: the semivariogram is computed first so its RANGE (correlation
    length) can serve as the support radius for the density / feasibility check.
    That is the principled link between "how far correlations reach" and "how
    sparse the covariance matrix is" — not an arbitrary distance percentile.
    """
    g = knn_skill_gain(X_hybrid, X_wl, y_resid, k=k)
    v = semivariogram(X_hybrid, y_resid)
    d = knn_distance_stats(X_hybrid, k=k, support_radius=v["range"])
    f = feasibility(d["s_star_at_radius"], target_N)
    return FalsificationReport(g, v, d, f)
