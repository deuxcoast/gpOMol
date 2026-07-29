# Calibration: plan and results

*Plan written 2026-07-28; executed the same day. All numbers: 20k subset, seed 42,
additive kernel (WL + geometry), 200-neighbour radius, M4 prior mean, 800 held-out test
points unless stated.*

## Summary

The OMol25 position paper rests its case for GPs not on accuracy but on **calibrated
uncertainty** (§7.2, conclusion): *"for applications such as active learning for DFT
dataset construction, calibrated uncertainty is often more valuable than marginal
accuracy improvement."* We had optimised R² throughout and never measured calibration.

Headline: **the GP's posterior variance is beaten, on every metric, by the distance to
the nearest training point** — a computation involving no kernel, no solve and no
distributed machinery. The uncertainty is not worthless (it ranks errors at 3× chance),
but the expensive apparatus is not what produces it.

Along the way this found a **silent correctness defect in fvgp's block-CG solver**, and
refuted three of four pre-registered predictions.

---

## Step 0 — is the variance right at all?

`validate.sparse_vs_dense_parity` compared posterior *means* only; the variance had never
been checked against anything. It now takes `check_variance=True`.

| solver | mean rel diff | variance rel diff | var corr | negatives | over-prior |
|---|---|---|---|---|---|
| `sparseCG` | 2.3e-05 | **1.4e-10** | 1.000000 | 0 | 0 |
| `sparseMINRES` | 2.1e-04 | **1.0e-09** | 1.000000 | 0 | 0 |

Passes. Sparse and dense variance ranges match exactly. Note the variance parity is ~5
orders tighter than the mean parity — expected, since the variance is
`diag(kk) − kᵀ(KV)⁻¹k`, dominated by the exact diagonal with the solve entering only
through a correction. **Solve error affects σ\* far less than μ\*.**

### ⚠ `sparse_krylov_mode="block"` is broken — do not use it

Measured at 800 test points, 10 categories, `batch=400`:

| krylov mode | frac at prior variance | corr vs single | wall | **warnings** |
|---|---|---|---|---|
| `single` | 0.005 | — | 464s | 0 |
| `block` | **0.501** | **0.475** | 132s | **0** |

Block CG returns the **prior variance for an entire block** of test points — the first
batch was 100% wrong, the second 0%, consistent with the first block solve returning its
zero initial guess. On those points single CG gives a genuine reduction (mean 5.95 against
a prior of 8.60), so the solve was *discarded*, not approximated. **Zero warnings**;
`SolveWarningCounter` cannot see it because fvgp does not warn on this path.

An earlier draft of this document recommended block CG on the strength of a "2.36×
speedup". That speedup was measuring skipped work. **Retracted.**

Step 0 initially passed *with* block CG because it ran 300 test points in a single batch
against a single dummy category — the failure needs multiple block solves. The parity
test therefore validated the configuration it was run in, not the one production uses.
Worth reporting upstream: a Krylov mode that silently returns wrong results on the
uncertainty path is a library defect.

---

## Step 1 — the prior mean's discarded uncertainty is negligible

We detrend manually, so σ\* describes the *residual* and the mean's parameter uncertainty
is dropped. `LinearEmbeddingMean.predict_var` now returns it.

| | mean | median | p95 |
|---|---|---|---|
| GP posterior var | 7.263 | 8.598 | 8.598 |
| **mean parameter var** | **0.036** | 0.023 | 0.109 |

**0.58% of σ\*²** — a σ inflation of 1.0017× median. Exactly the trace identity
`p/n · σ̂² = 65/16000 × 8.63`. Even M6's 374 coefficients would only reach ~2.3%.

So the exact universal-kriging treatment (which replaces `(HᵀH)⁺` with `(HᵀKV⁻¹H)⁻¹` at a
cost of `p` extra solves) **is not worth building**. `predict_var` documents that it is
the OLS approximation and a lower bound.

---

## Step 2 — coverage is total; we are overconfident; the shape is wrong

**Coverage.** Median **458** in-support same-category train neighbours per test point,
minimum 2, **0.0% uncovered**. The covered/uncovered split this plan originally called for
is a non-issue at these radii. (`sparsity_report`'s `frac_zero` measures *train-vs-train*
and does not describe the prediction task.)

**Overconfidence, and one real bug in our own code.** `posterior_covariance` defaults to
`add_noise=False` and `pipeline.predict` never overrides it, so we compare a **latent
function variance** against an **observed** y — the observation noise is simply absent.

| jitter | σ includes | std(z) | cov68 *(0.683)* | cov95 *(0.950)* | NLPD |
|---|---|---|---|---|---|
| 0.78 | latent only *(as shipped)* | **1.242** | 0.685 | 0.875 | 2.563 |
| 0.78 | + noise | 1.161 | 0.700 | 0.890 | 2.536 |
| 2.00 | + noise | 1.052 | 0.743 | 0.917 | **2.518** |
| 4.30 | + noise | 0.924 | 0.780 | 0.938 | 2.532 |
| 8.60 | + noise | 0.783 | 0.825 | 0.975 | 2.591 |

**No jitter fixes this, because the scale is not what is wrong.** At every jitter `cov68`
is *above* target while `cov95` is *below* it. A mis-scaled Gaussian moves both the same
way; these move in opposite directions. The three criteria also disagree about the best
jitter (NLPD picks 2.0, `std(z)` picks 4.3, `cov95` picks 8.6) — the same symptom.

---

## Steps 4 and 4b — where the heavy tails come from

**σ-standardisation makes calibration worse.** Fitting the GP on `r/σ̂` (σ̂ from `log n` +
category, both known at test time) and rescaling back:

| arm | std(z) | cov68 | cov95 | **excess kurtosis** | R² |
|---|---|---|---|---|---|
| raw residual, jitter 4.30 | 0.924 | 0.780 | 0.938 | **1.84** | 0.7205 |
| σ-standardised, jitter 0.50 | 1.281 | 0.660 | 0.885 | **3.41** | 0.7209 |

The per-category `std(z)` spread *did* close (2.21× → 1.43×), so the mechanism works —
but σ̂ explains only **7.5% of log-variance**, and dividing by a noisy estimate
manufactures heavier tails than it removes. Net negative. R² unchanged throughout.

**The tails are a small outlier population.**

| trim top | std(z) | excess kurtosis | share of Σz² |
|---|---|---|---|
| 0% | 0.887 | **5.70** | — |
| 0.5% *(4 points)* | 0.820 | 2.41 | 14.8% |
| 2% *(16 points)* | 0.737 | 0.73 | 31.9% |
| 5% *(40 points)* | 0.654 | **0.05** | **47.8%** |

**5% of test points carry half the squared error**; removing them leaves an essentially
Gaussian z. The QQ plot is straight through the core and departs only at the extremes.

**And they are extrapolation, not descriptor aliasing.**

| | top 2% by \|z\| | rest |
|---|---|---|
| distance to nearest train NN | **0.945** | 0.724 |
| \|Δy\| to that neighbour | **8.79** | 1.91 |
| molecule size | 59 | 42 |

`Spearman(|z|, distance) = +0.247`, `Spearman(|z|, |Δy|) = +0.312`. Outliers sit *farther*
from the training data, are larger, and are energetically distinct. Enrichment is real but
secondary: `geom_orca6` 3.95×, `elytes` 2.50×, `reactivity` and `trans1x` 0.00×.

**This should have been good news.** Aliased points would be undetectable in principle —
close in the descriptor, so they look well covered. Sparse-region points are exactly what a
GP posterior variance is built to flag. The information is available; we are not using it.

---

## Step 3 — is σ\* useful? No: a distance beats it

Each candidate gets its own optimal global scale fitted on half the test set and evaluated
on the other, so NLPD/CRPS compare methods **at their best scale** and Spearman measures
ranking independent of scale.

| σ candidate | Spearman | top-5% recall | NLPD | CRPS | miscal area |
|---|---|---|---|---|---|
| constant | — | 0.050 *(chance)* | 2.478 | 1.5234 | 0.1054 |
| size + category | +0.2977 | 0.100 | 2.396 | 1.4931 | 0.1007 |
| **dist-to-NN** *(no GP at all)* | **+0.2999** | **0.200** | **2.379** | **1.4893** | **0.0821** |
| GP σ\* (latent) | +0.2093 | 0.150 | 2.428 | 1.5035 | 0.0933 |
| GP σ\* + noise | +0.2135 | 0.150 | 2.444 | 1.5086 | 0.0978 |

The GP's uncertainty is **not worthless** — it ranks at 3× chance on top-5% recall. But it
is beaten on *every* metric by the distance to the nearest training point, and on most by a
trivial `log n` + category model.

**Why.** Sharpness is nearly constant across categories (2.44–2.89). With a 200-neighbour
radius and a median of 458 in-support neighbours, even the "far" points have ample
coverage, so σ\* barely varies — while `dist-to-NN` measures local sparsity directly, which
is precisely what the outliers are made of.

**Caveat on precision.** The evaluation half is 400 points, so top-5% recall is 20 points
and a 0.05 difference is one molecule. Spearman on 400 has SE ≈ 0.05, making 0.213 vs
0.300 about 1.7 SE. No single metric is decisive; the consistent direction across five is.

---

## Pre-registered predictions: scorecard

Recorded before running, and three of four were wrong.

1. **`std(z) > 1`, and adding the mean's variance moves it toward 1.** Direction
   ✅ (1.242), cause ❌ — the mean's variance is 0.58% and moves nothing. I attributed the
   overconfidence to the wrong thing twice: first the mean, then the jitter value.
2. **Per-category `std(z)` spanning ~2–3×.** ✅ Measured 2.67×.
3. **Uncovered points underconfident.** ❌ Void — there are none.
4. **`Spearman(σ*, |error|)`: genuinely unknown.** Answered: **+0.21, and beaten by a
   baseline with no GP in it.**

---

## What this means, and what to do next

**The finding that changes the plan: there is an accuracy/calibration tradeoff in the
support radius, and we have only ever tuned one side of it.** A wide radius improves R² and
flattens σ\*; a tight one makes σ\* responsive to local density, which is exactly the signal
the outliers carry. Every radius decision in this project was made on R² alone.

That makes the radius work — which the prior-mean result had seemingly superseded — live
again, on a different axis. **The next experiment is the 60/200/500 neighbour grid scored
on `Spearman(σ*, |error|)` and top-5% recall rather than R².** If σ\* overtakes `dist-to-NN`
at a tighter radius, the GP earns its uncertainty claim and the tradeoff has to be managed
explicitly. If it does not, the honest conclusion is that on this dataset the GP's
uncertainty is not worth its cost, and that is a result worth reporting.

Two smaller items, both cheap:

- **`predict` should expose `add_noise`.** There is currently no way to obtain the
  predictive variance of an *observation*, and the default silently gives the wrong
  quantity for calibration.
- **Harden the variance parity test** — multiple categories, enough test points to force
  several block solves. As written it passed with a solver discarding half its work.
