# Project Summary — Scaling Weisfeiler–Lehman Gaussian Processes for Molecular Energy Prediction on OMol25

*(Prepared as source material for poster generation. Thorough and self-contained; all
statistics are from this project's own experiments unless noted.)*

---

## 1. One-line description

We are building a **sparse, distributed Gaussian Process (GP)** that predicts DFT
molecular energies directly from **graph structure** (Weisfeiler–Lehman descriptors),
engineered to scale to **200,000+ molecules on 16 GPUs** using compact-support kernels
and an iterative solver — and, in the process, we identified and fixed several
methodological problems that determine whether such a model works or transfers across
dataset sizes.

---

## 2. Motivation / problem

- **Goal:** Predict the total energy of a molecule from its structure, as an
  intensive residual on top of an extensive (element-referenced) baseline, using a
  Gaussian Process so we get calibrated uncertainty, not just a point estimate.
- **Dataset:** OMol25 (`train_4M` split), **~3.99 million structures**, spanning **10
  chemically distinct categories** (ani2x, biomolecules, elytes, geom_orca6,
  metal_complexes, orbnet_denali, reactivity, rgd, spice, trans1x). We work on frozen
  subsets of **6k / 20k / 200k** molecules.
- **Core challenge:** Exact GPs are **O(N³)** in time and **O(N²)** in memory —
  infeasible past a few thousand points. Reaching 200k requires (a) a **compactly
  supported kernel** so the covariance matrix is sparse, (b) **distributed kernel
  construction** across GPUs (gpCAM/fvgp's `gp2Scale`), and (c) an **iterative
  conjugate-gradient (CG) solver** instead of a direct factorization.

---

## 3. Approach / pipeline

The end-to-end pipeline (all stages engineered to **never densify** a large matrix):

1. **Molecular graph → Weisfeiler–Lehman (WL) features.** Geometry-derived
   connectivity; per-atom WL relabeling to depth 3; explicit-vocabulary sparse count
   matrix. At depth 3 / `min_count=2` this yields **~54,800 features** (per depth ≈
   7,600 / 20,500 / 26,700), matrix density ~9×10⁻⁴, ~18–19% out-of-vocabulary rate.
2. **Supervised dimensionality reduction (streaming SIMPLS PLS) → 10-D embedding.**
   A custom sparse, streaming Partial Least Squares that folds column
   centering/scaling into implicit matrix–vector products, so the ~55k-column sparse
   feature matrix is **never densified**. Cost is O(nnz) per component.
3. **Compact-support Wendland kernel on the 10-D embedding**, evaluated **on the GPU**
   in ≤10k×10k blocks. Pairs beyond the cutoff radius have exactly zero covariance →
   sparse global matrix. A **category block-sparsity** tag zeroes covariance between
   molecules from different OMol25 subsets.
4. **Distributed GP fit + predict (`gp2Scale`)** on 16 GPUs (4 nodes, NERSC
   Perlmutter), with an iterative sparse-CG solve (optionally preconditioned).

**Regression target:** intensive residual `y = E_total − m(x)`, where `m` is a ridge-fit
extensive mean (element counts + charge/spin terms). Residual variance ≈ **50.3**
(std ≈ 7.1) on the 20k subset.

---

## 4. Key technical contributions and findings

This project's scientific contribution is a sequence of **diagnostic-driven fixes**,
each validated on held-out data. Domain guidance came from **Marcus Noack**, the author
of gpCAM/fvgp/gp2Scale.

### 4.1 Making the embedding scale N-invariant ("natural" PLS scaling)

- **Problem:** The original streaming SIMPLS normalized each component's scores to
  unit norm, so the embedding's absolute scale shrank as **~1/√N**. Pairwise
  distances — and therefore the kernel cutoff radius and *every* length-scale
  hyperparameter — changed with dataset size, so nothing learned at small N transferred
  to large N.
- **Fix:** Store the unit SIMPLS *weight* as the rotation so `transform` returns the
  **natural score** `t = X̃w`. Because `var(t_a)=wₐᵀCov(X̃)wₐ` is a *population*
  quantity, the scale becomes **N-invariant**.
- **Result:** Per-component standard-deviation ratio between a 16k and a 5k fit ≈
  **1.0** (the old normalization gave ≈ **0.56** = √(5/16)). Streaming-vs-batch-vs-
  sklearn PLS R² parity preserved exactly (0.2850).

### 4.2 Feature scaling: standardization is anti-predictive on sparse counts → switch to Pareto scaling (largest predictive win)

- **Diagnostic:** A **scaling × min_count × n_components grid** of held-out OLS R² (3
  seeds, 20k molecules).
- **Root cause identified:** Standardizing (z-scoring) a sparse count matrix inflates a
  feature present in `k` of `N` molecules into a spike of height **~√(N/k)**,
  *independent of the feature's actual count* (e.g. a feature in **2 of 16,000**
  molecules becomes a spike of height **~89**). With ~27,000 such near-singleton
  features, greedy PLS overfits these high-variance flukes.
- **Consequence:** The *production* configuration (standard scaling, `min_count=2`, 10
  components) was **anti-predictive**: held-out OLS R² = **−0.185** (worse than
  predicting the mean).
- **Fix — Pareto scaling** (divide by √std instead of std): the spike shrinks to
  ~(N/k)^¼ ≈ 9.5 for the same feature. Grid result: **best robustness/signal
  trade-off — ~0.345 held-out R² at min_count=2, no toxic tail, and essentially no
  min_count sensitivity.**
- **Key comparison (mean held-out OLS R², 3 seeds, 20k):**
  | scaling | min_count=2, 10 dims | best cell |
  |---|---|---|
  | standard | **−0.185** (anti-predictive) | 0.362 (only at min_count≥10) |
  | **pareto** | **+0.314** (peaks 0.345 @ 6 dims) | 0.361 |
  | center (no scaling) | +0.250 | ~0.25, flat across min_count |
- **Additional insight:** `center` scaling is **completely insensitive to min_count**
  (0.25 → 0.23 across min_count 2→50), *confirming* that standardization — not the
  feature-pruning threshold — was the real cause of the min_count sensitivity we had
  been fighting.
- **Also retracted a false lead:** an earlier finding that "signal peaks at ~4 PLS
  dimensions, so cut to 4" turned out to be an artifact of standard scaling at a single
  seed. The grid showed the optimal dimension *moves* with scaling/min_count, so it is
  not a robust property. We kept **10 dimensions and an isotropic kernel**.

### 4.3 The solve is conditioning-limited, not size-limited

- **Problem:** The kernel (Gram) matrix is **near-singular, condition number ~10⁹**,
  caused by near-duplicate molecules plus a too-dense cutoff. Plain sparse-CG grinds:
  a 20k solve took ~143 s and returned corrupted predictions.
- **Fixes:** (i) **Tighten the cutoff** — percentile 25 → 2 dropped median in-support
  neighbours from **~680 to ~54** and density from **~5% to ~1%**; (ii) a
  **preconditioner** (`sparseCGpre`, per Marcus Noack), which shrinks the *effective*
  condition number CG sees, converging in far fewer iterations to the correct solution
  — the proper fix for a slow solve, versus adding walltime or large jitter.
- **Architectural finding:** `gp2Scale` distributes the *kernel construction* across
  the GPUs, but the **solve is a single multithreaded CPU process on the driver** — the
  4 GPUs sit idle during the solve while it pegs ~118 CPU threads. So the **200k
  bottleneck is the driver-side, conditioning-limited CPU solve**, and conditioning
  (not compute or walltime) is the controlling lever.

### 4.4 Data-driven, N-invariant radius selection (semivariogram + RMSE-vs-distance)

- **Idea enabled by 4.1:** With the embedding scale now N-invariant, the compact-support
  radius can be chosen **once in absolute units** rather than re-derived as a per-N
  percentile.
- **Tools built:** a **semivariogram** (`γ(h)=½⟨(yᵢ−yⱼ)²⟩` vs embedding distance) and an
  **RMSE-vs-nearest-training-neighbour-distance** diagnostic, plus a density/
  conditioning report at any candidate radius.
- **N-invariance verified:** the absolute correlation-range stayed **~0.85–1.0 across
  n = 1.2k → 16k and across different molecule pools**, versus the ~0.55× collapse the
  old 1/√N scaling would have produced — confirming an absolute radius now transfers.
- **Subtle finding:** on a *supervised* PLS embedding, the semivariogram is contaminated
  by the trend PLS builds in (`y` varies linearly along the leading axes → the
  variogram overshoots its sill 5–8× rather than plateauing). The **RMSE-cliff
  radius R_inf ≈ 0.22** (which matches the empirically well-behaved pct=1–2 cutoff of
  ~0.27) is the reliable picker; the tool now auto-detects the trend and reports it.
- **Finding — conditioning binds, not signal:** the *signal* radius (~0.85) would
  imply **~2,200 in-support neighbours at 20k** — far too dense to condition. So there
  is usable predictive signal at distances the kernel cannot currently afford to
  include, which directly motivates the preconditioner work.

---

## 5. Results

- **GP predictive R² (20k, frozen hyperparameters, Wendland gp2Scale):**
  improved from **~0.05** (corrupted: standard scaling + dense ill-conditioned cutoff +
  a category-tag bug) to **0.23** (Pareto embedding + tight cutoff + `sparseCGpre`).
- **Embedding linear ceiling (held-out OLS R²):** **~0.34–0.41** with Pareto scaling.
- **Remaining gap (0.23 GP vs ~0.34 ceiling):** attributed to the compact-support
  kernel being a *local* method fighting a *globally linear* signal (PLS builds a global
  trend). Adding a **linear prior mean** to the GP is the identified next step to close
  this — it would make the OLS ceiling a floor the GP inherits.
- **Scale/transfer:** all hyperparameters and the cutoff radius now **transfer across
  N**, removing the per-N recalibration that previously blocked scaling.

---

## 6. Key statistics (quick reference for figures/callouts)

| Quantity | Value |
|---|---|
| OMol25 source structures (train_4M) | ~3,986,754 |
| Categories | 10 |
| Working subsets | 6k / 20k / 200k |
| WL features (depth 3, min_count 2) | ~54,800 |
| Feature-matrix density / OOV rate | ~9×10⁻⁴ / ~18–19% |
| Embedding dimension | 10-D (PLS, Pareto scaling) |
| Residual target variance (20k) | ~50.3 (std ~7.1) |
| Standardization rare-feature spike (k=2, N=16k) | ~89× |
| Production config OLS R² (standard, mc=2, 10-D) | **−0.185** (anti-predictive) |
| Pareto OLS R² (mc=2) | **+0.31–0.35** |
| Embedding linear ceiling (OLS R²) | ~0.34–0.41 |
| GP predictive R² (20k, frozen hp) | **0.05 → 0.23** |
| Gram condition number | ~10⁹ |
| Cutoff tightening (pct 25 → 2): neighbours | ~680 → ~54 |
| Cutoff tightening: density | ~5% → ~1% |
| N-invariance: per-component std ratio (16k/5k) | ~1.0 (was ~0.56) |
| N-invariance: variogram range across N | ~0.85–1.0 (stable) |
| Compute | 16 GPUs / 4 nodes (NERSC Perlmutter) |

---

## 7. Domain-expert guidance (Marcus Noack, gpCAM/fvgp/gp2Scale author)

Shaped several decisions: (1) predict-only should be **minutes, not hours** — slowness
is conditioning, not size; (2) fix slow solves with a **preconditioner**, not walltime;
(3) **compute is abundant** (16 GPUs is small); (4) **don't normalize PLS** → the
natural-scaling fix; (5) **set the cutoff from the variogram**; (6) start with **frozen
hyperparameters**, train later.

---

## 8. Status and next steps

**Done:** sparse WL featurizer; N-invariant natural-scaled + Pareto PLS; GPU block
Wendland kernel with category block-sparsity; distributed gp2Scale fit/predict on 16
GPUs; conditioning fixes (tight cutoff + `sparseCGpre`); N-invariant radius-selection
diagnostics (semivariogram + RMSE-vs-distance).

**Next:** (1) add a **linear prior mean** to close the GP-vs-OLS gap; (2) run the full
**200k** predict with the Pareto embedding + preconditioner; (3) **train**
hyperparameters (ARD length-scales) once frozen-hyperparameter predict is validated;
(4) optionally detrend the semivariogram (variogram on the OLS residual) for a clean
local correlation length.

---

## 9. Suggested poster framing

- **Headline story:** "Making a graph-kernel Gaussian Process both *predictive* and
  *scalable* on 4M-molecule chemistry — by fixing the statistics of feature scaling and
  the numerics of the solve, not by adding compute."
- **Three-panel narrative:** (1) *Transfer* — natural scaling makes the embedding
  N-invariant; (2) *Predictivity* — Pareto scaling turns an anti-predictive embedding
  (R²=−0.19) into a working one (R²≈0.34); (3) *Scalability* — conditioning, not size,
  is the bottleneck; tight cutoff + preconditioner take GP R² from 0.05 → 0.23.
- **Strong single number for the poster:** **−0.185 → +0.34** (embedding) and
  **0.05 → 0.23** (GP), from methodology fixes alone.
