# Plan: measuring calibration

*Written 2026-07-28. Nothing here has been run yet.*

## Why this matters more than the accuracy work

The OMol25 position paper does not rest its case on accuracy. §7.2 and the conclusion
both put the GP's differentiator elsewhere:

> *"Report calibration of uncertainty estimates, where the GP has a natural advantage. …
> For applications such as active learning for DFT dataset construction, calibrated
> uncertainty is often more valuable than marginal accuracy improvement."*

We have spent this entire effort optimising R² and have **never once measured
calibration**. Given where accuracy currently stands — the kernel contributes ~+0.01 over
a good prior mean — it is entirely possible the project's real result is on an axis we
have not looked at. It is equally possible the uncertainty is badly calibrated, which
would be worth knowing before any claim is made.

`predict(variance=True)` already exists. This is a measurement gap, not a capability gap.

---

## 1. What "calibrated" means here

The GP returns a Gaussian predictive distribution per test point, `N(μ*, σ*²)`. It is
calibrated exactly when the standardised residuals

```
z_i = (y_i − μ*_i) / σ*_i
```

are distributed `N(0, 1)`. Every metric below is a way of testing that one statement.

| metric | what it catches | reading |
|---|---|---|
| `mean(z)` | bias in the mean | ≈ 0 |
| `std(z)` | the headline number | > 1 overconfident (σ* too small); < 1 underconfident |
| coverage at nominal α | shape, not just scale | empirical ≈ nominal at every α |
| miscalibration area | one number for the reliability curve | ∫\|empirical − nominal\|, → 0 |
| NLPD | proper scoring rule | rewards accuracy *and* honest σ |
| CRPS | proper, outlier-robust | use alongside NLPD; NLPD is dominated by tails |
| **sharpness** = `mean(σ*)` | uselessness | **must be reported with coverage** |

Sharpness is not optional. A model predicting σ* = ∞ is perfectly calibrated and worth
nothing. Coverage and sharpness are only meaningful as a pair.

**Reliability diagram**: for each nominal level α, plot the empirical fraction with
`|z| ≤ Φ⁻¹((1+α)/2)` against α. The diagonal is perfect. Curvature tells you *how* it
fails; `std(z)` alone does not.

---

## 2. Three hazards specific to our setup

These are why a naive calibration run would produce a misleading number.

### 2.1 The prior mean's uncertainty is thrown away

We detrend manually: fit `m(z)`, model `y − m(z)` with the GP, add `m(z*)` back. So `σ*`
is the posterior variance **of the residual only**. The linear mean has its own parameter
uncertainty, and under M6 that mean carries ~374 coefficients — not negligible.

**We are therefore guaranteed to be overconfident**, by an amount nobody has quantified.

Fix, and it is exact for a linear mean: add the OLS predictive variance

```
var_mean(x*) = σ̂²  ·  x*ᵀ (HᵀH)⁻¹ x*        σ̂² = RSS / (n − p)
σ*_total²    = σ*_GP²  +  var_mean(x*)
```

Cheap — one triangular solve per test point against an already-factorised `HᵀH`. This
must be in place before any calibration number is quoted, or we are measuring an
artefact of our own architecture.

### 2.2 Compact support means some points get the prior, not a prediction

A test point with no in-support neighbour reverts to the prior mean, and its variance is
the prior variance (`signal_var`). That is not a prediction, it is "no information" — and
it will look *underconfident* because σ* is at its maximum.

Report the covered and uncovered populations **separately**, plus the uncovered fraction.
Pooling them averages a real predictive distribution with a shrug and hides both. At the
radii we currently use `frac_zero` is ~0%, so this may be moot at 20k — but it was ~45%
at the radii we started with, and it will not stay at 0% as N grows with neighbours
pinned.

### 2.3 One global signal variance against 8× category heterogeneity

Measured: residual variance spans **6.3× across molecule-size deciles** and **8.2× across
categories** (`spice` 2.89 → `elytes` 23.59). The kernel has a single signal variance.

This is where the heteroscedasticity finding should finally bite. σ-standardisation did
**not** move R² (+0.0002) — but R² is not the metric a wrong variance breaks. Calibration
is. Expect systematic overconfidence on high-variance groups and underconfidence on low.

**Calibration must be reported per category and per size decile, not pooled.** A pooled
`std(z) ≈ 1` is entirely consistent with being 2× overconfident on half the data and 2×
underconfident on the other half.

---

## 3. The baselines that make this honest

EquiformerV2 provides no uncertainty at all, so "we have uncertainty" is trivially true
and scientifically empty. The comparisons that mean something:

1. **Constant σ**, fitted on train. Beats any model whose σ* carries no information.
2. **The cheap heteroscedastic model we already built** — σ predicted from `log n` +
   category (R² 0.075 on `log r²`). If the GP's σ* is no better than this, the expensive
   machinery is not what is producing the uncertainty.
3. **Error ranking — the one that matters for the stated use case.** For active learning
   you need to know *which* predictions to distrust, which is a ranking question, not a
   coverage question. Report **Spearman(σ*, |y − μ*|)** on held-out data.

That third number is the one I would look at first. A model can be perfectly calibrated
in aggregate and still rank errors at chance, and if it does, it is useless for DFT
dataset construction regardless of what the reliability diagram says.

---

## 4. Cost, and one concrete finding

`posterior_covariance(variance_only=True)` computes `KVsolve(k)` where `k` is the
`(n_train, n_pred)` cross-covariance — **one linear solve per test point**, against one
solve total for the mean.

From the fvgp audit: `_resolve_krylov_mode` is called only from
`calculate_sparse_conj_grad`; `calculate_sparse_minres` has no block path and loops per
column. I predicted from this that MINRES would be the wrong solver for many
right-hand sides.

**Measured (Step 0, n_train=1200, n_test=300), and the prediction was only half right:**

| solver | wall | vs plain CG |
|---|---|---|
| `sparseCG` | 3.1s | — |
| `sparseCG` + `sparse_krylov_mode="block"` | **1.3s** | **2.36×** |
| `sparseMINRES` | 1.4s | 2.21× |

**Block CG is a real 2.4× win over plain CG** — use it. But **MINRES matches block CG**
(1.07×), so the claim that MINRES is the wrong solver for variance is *not* supported at
this scale. Its per-column loop is not the bottleneck here.

Caveat: 300 test points is small, and per-column overhead should scale worse than a block
path. Re-time at n_test = 4000 before committing to a solver for the full run.

Budget by measurement, not by guess: time 500 test points first, then extrapolate. Keep
`--cg-maxiter` finite and the `SolveWarningCounter` armed — an unconverged variance solve
is silent in exactly the same way an unconverged mean solve is, and a corrupted σ* would
be much harder to notice than a corrupted μ*.

---

## 5. Sequence

**Step 0 — verify the variance is right at all. ✅ DONE, PASSES.**
`validate.sparse_vs_dense_parity` now takes `check_variance=True` and compares the sparse
gp2Scale posterior variance against the dense Cholesky reference on the same kernel, same
cutoff, same data — the only difference being the linear algebra.

| solver | mean rel diff | **variance rel diff** | var corr | negatives | over-prior |
|---|---|---|---|---|---|
| `sparseCG` | 2.3e-05 | **1.4e-10** | 1.000000 | 0 | 0 |
| `sparseCG` [block] | 2.3e-05 | **1.4e-10** | 1.000000 | 0 | 0 |
| `sparseMINRES` | 2.1e-04 | **1.0e-09** | 1.000000 | 0 | 0 |

Sparse and dense variance ranges match exactly ([0.1166, 0.2887] against a prior of
0.2896), so the posterior correction is genuinely being applied rather than silently
falling back to the prior.

Worth noting the variance parity is ~5 orders of magnitude *tighter* than the mean
parity. That is expected rather than suspicious: the variance is
`diag(kk) − kᵀ(KV)⁻¹k`, dominated by the exact diagonal term, with the solve entering
only through a correction that is at most ~60% of the prior here. **Solve error affects
the variance far less than the mean** — good news for calibration, and it means a loose
solver tolerance is less dangerous for σ* than for μ*.

The two guards beyond tolerance are the ones that would catch a genuinely broken solve:
zero negative variances, and zero points exceeding the prior variance (the posterior
cannot be more uncertain than the prior).

**Step 1 — add `var_mean(x*)`** (§2.1) to `pipeline.predict`, behind a flag so existing
behaviour is reproducible.

**Step 2 — measure at 20k**, additive kernel, 200 neighbours, M4/M6 mean: z-statistics,
reliability diagram, NLPD, CRPS, sharpness. Split covered/uncovered, and break down by
category and size decile.

**Step 3 — the baselines** (§3), including Spearman error-ranking.

**Step 4 — if per-group calibration is bad**, re-run on the σ-standardised target. This
is the natural home for that work: it did nothing for R² and should do a great deal here.

---

## 6. Pre-registered predictions

Recorded before running, so the results cannot be rationalised afterwards.

1. **Pooled `std(z) > 1`** (overconfident), because §2.1 discards the mean's uncertainty.
   Adding `var_mean` should move it toward 1.
2. **Per-category `std(z)` spanning ~2–3×**, roughly √8 if the single global signal
   variance is the only cause. If the spread is much *larger*, something else is wrong.
3. **Uncovered points underconfident** (σ* pinned at the prior), if any exist.
4. **Spearman(σ*, |error|): genuinely unknown.** This is the one I cannot call, and it is
   the one that decides whether the uncertainty is useful for the paper's stated
   application. If it comes back near zero, that is the headline result of the whole
   calibration exercise and it is a negative one.
