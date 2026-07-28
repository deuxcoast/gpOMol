# The prior mean

*Last updated 2026-07-28.*

This document covers what the GP's prior mean is, why each piece of it exists, and the
measurements behind each decision. It matters more than its length suggests: **the prior
mean currently does about 97% of the predictive work, and the last change to it was worth
roughly four times the entire kernel.**

---

## 1. There are two means, at two different layers

Easy to confuse, and they do different jobs.

### Layer 1 — `ExtensiveMean`, applied in `data.get_data`

Before any GP exists, `get_data` fits an `ExtensiveMean` to the raw total energies and
**stores the residual as the target** ([data.py:210](data.py)). Its design matrix is

```
[ per-element atom counts ]  +  [ charge, |charge|, charge², spin, n² ]
```

This is the extensivity model: total energy is, to first order, a sum of per-atom
contributions. Removing it converts an extensive target that spans orders of magnitude
into a residual the GP can actually work with.

**Consequence that trips people up:** composition, `n_atoms` and `n²` are already gone —
*linearly, and independently of any descriptor*. Adding them to the GP's prior mean is a
no-op. We tested it; see §3.

### Layer 2 — `LinearEmbeddingMean`, the GP's prior mean

`reduce.LinearEmbeddingMean` is the prior mean in gp2Scale's sense (Noack et al. 2025,
Eq. 2). The GP models `y − m(z)` and predictions add `m(z*)` back.

```
m(z, n) = b₀ + bᵀz + cᵀs(n) + dᵀ(z · log n)
                     └─ size ─┘  └─ interaction ─┘
where s(n) = [n, log n, √n, n³]
```

The first two terms are the original linear mean; the last two are the 2026-07-28
addition.

---

## 2. Why a prior mean at all

The Wendland kernel has **compact support**: beyond its radius, covariance is exactly
zero. A test point with no in-support neighbour therefore gets the prior mean and nothing
else. With the default zero mean that is *literally zero* — and it happened to about 45%
of test points at the radii we started with.

A linear mean turns that failure into a floor: uncovered points fall back to the OLS
prediction rather than to zero, so the GP inherits linear accuracy everywhere and adds
local corrections where it has coverage. **This roughly doubled R² when it was
introduced** and remains one of the two biggest wins in the project.

There is a second reason. PLS bakes a global linear trend into the embedding by
construction, so removing it leaves a much closer to stationary residual — which is
precisely what a compact-support kernel is good at, and what gp2Scale is designed for.

### Why detrend manually instead of using fvgp's `prior_mean_function`

Both are supported. For frozen hyperparameters the point predictions are identical, and
the manual route cannot perturb the sparse solve. The fvgp route is only needed if the
mean is to be trained *jointly* with the kernel, which we do not do.

---

## 3. The size interaction — the 2026-07-28 change

### What was measured

20k molecules, seed 42, held-out test:

| prior mean | var(residual) | **test R²** |
|---|---|---|
| intercept only | 50.92 | −0.000 |
| linear on embedding *(previous production)* | 10.74 | 0.6262 |
| + size terms `[n, log n, √n, n³]` | 10.67 | 0.6273 |
| + per-element counts | 10.29 | 0.6355 |
| **+ size × embedding** ← adopted | **8.60** | **0.7292** |

**+0.103 held-out R².** For scale, the gp2Scale kernel has never contributed more than
+0.028 over the linear mean, and the best full GP result at the time was 0.6543 — so this
one term, *with no kernel at all*, beat the best GP by 0.075.

### Why the interaction and not the main effects

Size on its own buys +0.001, and element counts +0.009. Nearly the whole effect is the
`z · log n` block. That follows directly from §1: `ExtensiveMean` already removed
composition and size **additively**. What it cannot express is that *the embedding's
relationship to the residual depends on molecule size* — a per-atom descriptor average
means something different for a 5-atom molecule than for a 300-atom one. That is a
product, not a sum, so no additive composition model can reach it.

### Why `log n` for the interaction

Tested against raw `n` and `√n`; `log n` was the best of the three and is the most stable
numerically over a 2–341 atom range. The `s(n)` block keeps `[n, √n, n³]` available as
main effects so the interaction is not carrying curvature that belongs in the trend.

### Design decisions

**Opt-out, not opt-in** (`--no-mean-size-interaction`). A term worth +0.10 should be the
default; the flag exists to reproduce ladder rungs measured before 2026-07-28, all of
which used the plain linear mean.

**`size=None` reproduces the old model exactly.** Verified against a hand-built OLS
design. Old results stay reproducible without a code checkout.

**`predict()` raises if the mean was fitted with sizes and called without them.** Silently
dropping the term would cost more than the entire kernel while looking like a normal run.

**Size features are standardised internally.** `n³` reaches ~4×10⁷ while the embedding is
O(1). OLS is invariant to an invertible linear reparameterisation of the design, so this
changes the arithmetic and not the fit — verified to match an unstandardised design.

**The OLS baseline uses the same mean model.** `dim_sweep`'s `OLS_R2` column must track
whatever the GP is given, or `GP_R2 − OLS_R2` stops measuring the kernel's contribution
and silently becomes "kernel plus the difference between two mean models".

---

## 4. Which channels feed the mean

`--mean-channels` is deliberately independent of `--channels` (the kernel blocks).
Production is:

```
--channels wl geom additive  --mean-channels wl geom strain
```

So **strain is in the mean only** — no Gram block. The reasoning is the project's
channel-placement criterion: a prior-mean term costs no Gram block, adds no neighbours,
cannot worsen conditioning, and reaches every test point *including uncovered ones*. A
channel earns a kernel block only if its effect is genuinely local and non-additive.
Strain is a sum of harmonic bond terms — an energy estimate, additive and global — so the
mean is its correct home. Measured directly on the organic subset: moving the WL channel
from kernel to mean cost only +0.0077, i.e. its Gram block was buying almost nothing.

One consequence worth knowing when reading logs: `mean_chan` is computed once per seed and
used for **every** channel mode, so `OLS_R2` is identical across the `wl`, `geom` and
`additive` rows of a given seed. That is a useful self-consistency check — and it also
means the `wl` row is not comparable to a WL-only reference from an older run.

---

## 5. What the mean does *not* fix

The prior mean removes trend. It cannot remove **heteroscedasticity**, and we have a lot
of it.

The semivariogram of the residual never plateaus: γ/sill climbs to 3.38 (WL) and 3.91
(geometry) instead of flattening at 1. Adopting the size interaction — which cut residual
variance by 20% and added 0.10 R² — moved that to 3.28 and 3.74. **Essentially unchanged.**
The drift is not a missing mean term.

What it is:

| grouping | variance range | ratio |
|---|---|---|
| molecule size (deciles) | 5.49 → 29.56, monotone | **6.3×** |
| category (`data_id`) | `spice` 2.89 → `elytes` 23.59 | **8.2×** |

Standardising the residual within size deciles moves the overshoot to 2.76 / 3.36 — real
but partial, with category likely accounting for much of the rest.

**Implications.** A stationary kernel with a single signal variance must compromise across
groups differing 6–8× in variance: too smooth for `elytes`, too rough for `spice`. That is
a better candidate for the stubborn +0.02 kernel ceiling than the support radius ever was.
The two candidate fixes — a heteroscedastic `noise_variances` vector (fvgp accepts one
per point; we currently pass `jitter * ones(n)`), and the paper's §4.3 non-stationary bump
term, which *is* a spatially varying signal variance — both live outside the mean.

Also worth recording: γ/sill ≈ 0.62 (WL) and 0.46 (geometry) in the *shortest* distance
bin, so half to two-thirds of the residual variance is already present between nearest
neighbours and is unavailable to any kernel at any radius.

---

## 6. Reproducing the measurements

The mean ladder, the heteroscedasticity tables and the variogram diagnostics come from a
single script pattern: build the channels via `pipeline.build_channels`, fit each candidate
mean with `np.linalg.lstsq`, score held-out with `r2_score`, and feed the residual to
`radius.semivariogram` / `radius.range_from_variogram`. `range_from_variogram` returns
`None` here by design — it refuses to report a range for a variogram that never saturates.

Sanity checks worth re-running after any change to this file:

- `size=None` reproduces plain OLS exactly.
- Standardised and unstandardised designs give identical predictions.
- `predict()` without sizes raises when the mean was fitted with them.
- `dim_sweep`'s `OLS_R2` is identical across channel modes within a seed.
