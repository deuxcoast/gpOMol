# The support radius, scored on calibration instead of R²

*Executed 2026-07-28. 20k subset, data seed 0, split seeds 42/7/123, additive kernel
(WL + geometry), M4 prior mean on WL+geometry+strain with the size interaction, jitter
4.30, 800 held-out points per arm carrying a posterior variance. Scripts:
`scripts/radius_calibration/`.*

## What this was testing

`CALIBRATION.md` ended on a live lead: **there is an accuracy/calibration tradeoff in the
support radius, and every radius decision in this project was made on R² alone.** The
argument was mechanical. At a 200-neighbour radius the median test point has 458
in-support neighbours, so every point is amply covered, σ\* barely varies, and a σ that
does not vary cannot rank anything — whereas `dist-to-NN` measures local sparsity
directly, and local sparsity is what the outliers are made of. Tighten the radius, the
reasoning went, and σ\* becomes responsive to density; the price is R².

So: re-run the neighbour grid scored on `Spearman(σ*, |err|)` and top-5% recall rather
than R², and see whether σ\* overtakes `dist-to-NN` at a tighter radius.

**Pre-registered prediction: tightening the radius raises the variability of σ\* and
therefore its ranking skill. It is exactly backwards.**

---

## Design

For each split seed the embeddings are built **once**; each arm then recomputes only the
per-channel cutoff at its target neighbour count `K` and rebuilds the GP. Within a seed
the embedding, the prior mean, the test subset and both GP-free baselines are therefore
*identical* across `K`, and the only thing moving between arms is the radius. `dist-to-NN`
does not depend on `K` at all, which makes `Spearman(σ*) − Spearman(dist-to-NN)` a
**paired per-seed difference** — the statistic `channel_placement` argued for once WL
turned out to have a seed-to-seed std of 0.076 at 20k.

The grid the plan asked for was 60/200/500. **15 and 30 were added** because the
hypothesis under test is "tighter is better" and at `K`=60 the median test point still
keeps ~190 in-support neighbours: had the trend run the predicted way, the answer would
have sat outside the grid. The tight arms are also the cheap ones.

Two departures from `CALIBRATION.md` Step 3's scoring, both buying precision where that
run named its own main weakness ("the evaluation half is 400 points, so a 0.05 difference
in top-5% recall is one molecule"):

1. **Spearman and top-5% recall are scale-free** — a global scale cannot reorder
   anything — so they are computed on all 800 points rather than a held-out half.
2. NLPD/CRPS/miscalibration do need a fitted scale, so they keep the split, but
   **2-fold cross-fitted** (fit on A score B, fit on B score A, average).

Run at `linalg=sparseCG` with a finite `solve_maxiter=2000` under `SolveWarningCounter`.
**Zero solver non-convergence in all 15 arms, 12,000 variance solves.** fvgp's Krylov mode
is asserted `single` at startup — the `block` mode returns the prior variance for entire
blocks of test points with no warning (`CALIBRATION.md` §Step 0).

**Reproduction check.** Re-scored under Step 3's exact protocol (seed 42 alone, single
split, metrics on the eval half), this pipeline returns the published table to four
decimal places on all five candidates — `dist-to-NN` +0.2999 / 0.200 / 2.379 / 1.4893,
GP σ\* +0.2093 / 0.150 / 2.428 / 1.5035. Same pipeline; the numbers below differ only
because there are now three seeds and twice the points.

---

## Result 1 — the tradeoff does not exist. Both objectives want a WIDER radius.

Mean over three seeds. `CV(σ*)` is `sd(σ*)/mean(σ*)`: a σ that does not vary cannot rank
anything, whatever its average level.

| K | cutoff wl | cutoff geom | nbrs/test pt | frac 0-nbr | **R²** | **CV(σ\*)** | cost/arm |
|---|---|---|---|---|---|---|---|
| 15 | 0.230 | 0.988 | 54 | 1.6% | 0.7181 | 0.0719 | 39 s |
| 30 | 0.264 | 1.108 | 102 | 0.7% | 0.7209 | 0.0861 | 64 s |
| 60 | 0.303 | 1.249 | 190 | 0.3% | 0.7240 | 0.1039 | 111 s |
| 200 | 0.393 | 1.565 | 538 | 0.0% | 0.7293 | 0.1424 | 307 s |
| 500 | 0.488 | 1.948 | 1226 | 0.0% | **0.7337** | **0.1808** | 672 s |

Both columns rise monotonically with the radius, in all three seeds, and the paired
per-seed increments track each other almost exactly (see the figure's middle panel). **The
radius does not trade accuracy against calibration. It is aligned with both**, and the
configuration that is best on R² is also the one where σ\* carries the most information.

There is no cheap corner either: the winning arm is also **17× the cost** of the tightest
one, because the posterior variance is one linear solve per test point against a matrix
that a wide radius makes dense.

## Result 2 — why the predicted mechanism was backwards

| K | ρ(σ\*, dist-to-NN) | ρ(σ\*, #nbrs) | median σ\*/√prior |
|---|---|---|---|
| 15 | +0.658 | −0.762 | **0.991** |
| 30 | +0.661 | −0.743 | 0.979 |
| 60 | +0.671 | −0.721 | 0.957 |
| 200 | +0.707 | −0.670 | 0.885 |
| 500 | **+0.750** | −0.596 | **0.794** |

`CALIBRATION.md` had the failure mode of a *wide* radius right — total coverage flattens
σ\* — but assumed the tight end was therefore better. It is worse, for a reason the
original argument missed: **σ\* varies because points differ in how much the data reduces
the prior, and that requires neighbours to exist.** At `K`=15 the median point retains
99.1% of its prior σ; virtually nothing is being reduced, so σ\* is pinned near the prior
for *everyone*. A σ pinned at the prior is flatter than a σ pinned at a well-informed
value. Compact support has two ways to destroy the variability of the posterior variance,
and this project had only anticipated one of them.

The first column is the uncomfortable one. σ\* is already substantially a reconstruction
of `dist-to-NN` (ρ ≈ 0.66–0.75) — and it becomes **more** like the free baseline as the
radius widens. The GP's best configuration is the one where its posterior variance most
closely imitates a quantity available without it.

## Result 3 — σ\* does not overtake the baseline at any radius

`Spearman(σ, |err|)` and top-5% recall, all 800 points, mean over three seeds:

| K | size+category *(no GP)* | dist-to-NN *(no GP)* | GP σ\* (latent) | GP σ\* + noise |
|---|---|---|---|---|
| | ρ / recall | ρ / recall | ρ / recall | ρ / recall |
| 15 | +0.2975 / 0.167 | +0.2581 / 0.192 | +0.1461 / 0.150 | +0.1684 / 0.267 |
| 30 | +0.3001 / 0.158 | +0.2676 / 0.183 | +0.1476 / 0.150 | +0.1613 / 0.225 |
| 60 | +0.3010 / 0.158 | +0.2767 / 0.200 | +0.1514 / 0.158 | +0.1589 / 0.217 |
| 200 | +0.3083 / 0.142 | +0.2897 / 0.192 | +0.1703 / 0.158 | +0.1742 / 0.175 |
| 500 | **+0.3166** / 0.158 | +0.2939 / 0.192 | +0.1924 / 0.175 | +0.1953 / 0.175 |

The paired statistic, `Spearman(σ*) − Spearman(dist-to-NN)`, one difference per seed:

| K | per-seed differences | mean | SE | verdict |
|---|---|---|---|---|
| 15 | −0.143, −0.078, −0.115 | −0.1120 | 0.0187 | baseline wins |
| 30 | −0.156, −0.083, −0.122 | −0.1200 | 0.0211 | baseline wins |
| 60 | −0.163, −0.090, −0.123 | −0.1254 | 0.0213 | baseline wins |
| 200 | −0.156, −0.091, −0.111 | −0.1194 | 0.0193 | baseline wins |
| 500 | −0.125, −0.086, −0.093 | −0.1014 | 0.0120 | baseline wins |

**Negative at every radius, in every seed, at 5–8 SE.** Widening the radius *narrows* the
gap (−0.125 at `K`=60 → −0.101 at `K`=500) but comes nowhere near closing it, and the
matrix is near-dense by then — the direction that would help is the direction in which
gp2Scale stops being gp2Scale.

Top-5% recall does not discriminate: every arm is a tie within ±1 SE (SE 0.04–0.07 on
3 seeds × 40 points). It is too coarse a statistic at this test-set size to carry a
conclusion, and should not be quoted as one.

**And the strongest ranker is the cheapest thing on the list.** `size+category` — `log n`
plus category dummies, no geometry, no descriptor, no GP — beats `dist-to-NN` by
+0.019…+0.039 (paired, 3/3 seeds positive at K ≤ 60) and beats the GP by 0.124…0.153
(paired, SE ≈ 0.027–0.032, ~5 SE). Step 3 had these two effectively tied at +0.2977 vs
+0.2999 on one seed; with three seeds and twice the points the ordering is
**size+category > dist-to-NN > GP σ\***.

---

## Correction to CALIBRATION.md: the NLPD/CRPS row is not reproducible

Step 3 reported `dist-to-NN` as the winner on NLPD (2.379), CRPS (1.4893) and
miscalibration (0.0821) as well as on ranking. **That column does not survive more seeds,
and the reason is a defect in the candidate rather than noise.**

A σ proportional to the distance to the nearest training point asserts σ → 0 for a
molecule that has a near-duplicate in the training set. The Gaussian MLE scale
`c = √mean((err/s)²)` is then dominated by the single smallest `s`, so one molecule sets
the scale for all 800. Seed 7 contains such a point (`dist-to-NN` = 3.7e−5, |err| = 0.10);
seeds 42 and 123 do not. **Step 3 ran seed 42 only.**

| seed | NLPD, dist-to-NN raw | min(dist-to-NN) | ratio of the two cross-fitted scales |
|---|---|---|---|
| 42 | 2.430 | 7.4e−02 | 1.1 |
| 123 | 2.533 | 4.0e−02 | 1.2 |
| **7** | **294.831** | **3.7e−05** | **34.1** |

Floored at its own 1st percentile — `s = √(d² + d₀²)`, the minimum defensible
regularisation of a σ model that can otherwise claim infinite confidence — the candidate
is well behaved again and **still wins**, at every radius:

| K = 200 | NLPD | CRPS | miscal | std(z) |
|---|---|---|---|---|
| size+category | 2.510 | 1.5704 | 0.0816 | 1.0347 |
| dist-to-NN raw | 99.932 | 5.3227 | 0.1498 | 8.7175 |
| **dist-to-NN floored** | **2.469** | **1.5684** | **0.0766** | 1.0125 |
| GP σ\* (latent) | 2.554 | 1.5995 | 0.0957 | 1.0231 |
| GP σ\* + noise | 2.576 | 1.6042 | 0.1000 | 1.0297 |

So the *conclusion* of Step 3 stands on all five metrics; its NLPD/CRPS/miscal **numbers**
for `dist-to-NN` should be replaced by the floored ones, and the raw candidate should be
described as what it is — unusable without a floor, and the least robust of the five.

---

## The small-category caveat, and the one live lead it opens

`cutoff_for_neighbors` masks cross-category distances to `inf` and then drops the
non-finite k-th distances, so a category with fewer than `K` training rows contributes
nothing to the median that sets the radius:

| K | categories excluded | share of test set |
|---|---|---|
| ≤ 60 | none | 0.0% |
| 200 | `orbnet_denali`, `spice` | 2.4% |
| 500 | `orbnet_denali`, `spice`, `rgd` | 4.8% |

This is not just bookkeeping. Per-category `Spearman(σ*, |err|)`, pooled over seeds:

| category | n_test | K=15 | K=30 | K=60 | K=200 | K=500 | dist-to-NN |
|---|---|---|---|---|---|---|---|
| ani2x | 141 | +0.171 | +0.187 | +0.191 | +0.194 | **+0.246** | +0.186 |
| biomolecules | 490 | **+0.131** | +0.111 | +0.087 | +0.037 | +0.014 | +0.183 |
| elytes | 472 | +0.098 | +0.096 | +0.086 | +0.078 | +0.087 | +0.088 |
| geom_orca6 | 113 | +0.309 | +0.340 | +0.393 | **+0.418** | +0.374 | +0.311 |
| metal_complexes | 493 | +0.049 | +0.053 | +0.064 | +0.105 | **+0.153** | +0.145 |
| reactivity | 470 | +0.098 | +0.097 | +0.106 | +0.119 | **+0.121** | +0.120 |
| rgd | 47 | **+0.119** | +0.083 | +0.072 | +0.015 | +0.007 | +0.257 |
| trans1x | 113 | +0.310 | +0.349 | **+0.400** | +0.384 | +0.374 | +0.260 |

**The pooled "wider is better" hides a genuine split.** `metal_complexes` and `ani2x` want
the widest radius on offer; `biomolecules` and `rgd` are *ruined* by it (rgd +0.119 →
+0.007, biomolecules +0.131 → +0.014) and are precisely where `dist-to-NN` wins by the
largest margin (rgd +0.257 against the GP's +0.007). A single global radius is being
asked to serve categories whose optimum differs by an order of magnitude in `K`.

That is the one place this result points forward rather than back, and it is the same
lever the notes already flagged as untested: the gp2Scale methods paper's §4.2
*length-scale* non-stationarity ρ(x), of which **a per-category radius is the cheap
discrete version**. Everything needed to try it exists — `build_gp` already takes
per-channel `(start, stop, cutoff)` specs, and because the Gram is *already* exactly
block-diagonal by category (`kernel.py:242` zeroes cross-category pairs unconditionally),
per-category radii are automatically PSD: a block-diagonal matrix whose blocks are each
PSD is PSD. No Paciorek–Schervish machinery is needed for the discrete version.

### But NOT as a route to discovering sparsity — that was tested and failed

*Scripts: `scripts/remark1_local_scaling/`.* It is tempting to read the per-category scale
spread as evidence that a non-stationary length scale could make the position paper's §3
block sparsity *emerge* — that families overlap only because family A is tight and B
diffuse (per-family 10-NN distance spans 2.3×), so a cross pair can be closer than a
typical within-B pair purely from scale mismatch. That is a testable claim and it is
**false**.

Re-running Remark 1 on locally-scaled distances — `d̃ᵢⱼ = dᵢⱼ/√(σᵢσⱼ)` with σᵢ the distance
to i's k-th nearest neighbour, computed **ignoring `data_id`** so the test measures
discovery rather than imposition — on `wl+geom`, mean over the three seeds:

| scaling | separation ratio | P(cross < within median) | k=5 purity | k=50 purity | modality probe |
|---|---|---|---|---|---|
| stationary | 1.225 | 0.291 | 0.710 | 0.616 | 0.000 |
| local σ, k=10 | 1.279 | 0.231 | 0.714 | 0.618 | 0.000 |
| local σ, k=50 | 1.263 | 0.224 | 0.717 | 0.625 | 0.000 |
| **pre-registered gate** | **≥ 1.50** | **≤ 0.20** | | | |

Everything moves the right way in all three seeds and both geometry channels, so the
effect is real — and far too small, covering ~20% of the distance to the gate.

**The internal control is what makes it decisive: the normalisation worked.** The
per-family scale spread it targets collapsed from **2.3× to 1.7×**. The confound was
removed and the families still did not separate, so the overlap is **genuine
interpenetration, not scale mismatch**.

Three details harden it: the decision-relevant number barely moves (k=5 neighbour purity
0.710 → 0.714 — a compact-support kernel's neighbour *sets* are unchanged, and the ratio
gains are happening in tails the kernel never reaches); Remark 1's own modality probe
stays at 0.000 under every σ, so no near-zero mode appears; and on the WL channel alone
local scaling makes separation *worse* (1.220 → 1.161).

**Conclusion.** ρ(x) cannot make §3's block structure emerge from this descriptor — and
since cross-category covariance is already exactly zero, there is nothing left for it to
discover at the family level anyway. A per-category radius remains worth trying as
**tuning**, and should be reported as such; the paper's discovery claim is not available
to it.

*(Incidental, and consistent with the historical CG conditioning trouble: the WL channel
contains exact duplicates — 78 of 4000 sampled points have a co-located neighbour and one
cluster has 14 identical embeddings — which is why σᵢ is undefined there for k ≤ 10. The
geometry channel has none.)*

---

## Verdict against the position paper

The plan set the go/no-go in advance: *"If σ\* overtakes `dist-to-NN` at a tighter radius,
the GP earns the position paper's UQ claim and the tradeoff must be managed explicitly. If
it does not, the honest conclusion is that on this dataset the GP's uncertainty is not
worth its cost."*

**It does not, at any radius, in any seed.** The lead is closed:

* There is no accuracy/calibration tradeoff to manage — the radius is aligned with both.
* σ\* is beaten on ranking by two GP-free baselines, one of which (`log n` + category)
  requires no descriptor at all, and on proper scoring rules by the floored distance.
* σ\* is itself ρ ≈ 0.75 correlated with the baseline it loses to, and the correlation
  *increases* in the configuration where the GP does best.

This is the third independent finding pointing the same way — the prior mean does ~97% of
the accuracy (`PRIOR_MEAN.md`), the descriptor does not discover sparsity (the paper's own
Remark 1 go/no-go), and the posterior variance does not earn its solve. **None of the three
is a verdict on gp2Scale**; §7.1/§7.2 of the position paper describes this regime
(high-dimensional → sparsely distributed points → "challenging to discover naturally
occurring sparsity") and prescribes a different descriptor — Wasserstein distance on
rotationally-invariant pairwise distance profiles, or the WL kernel proper — which we have
never built. The honest statement is that **the UQ claim fails for our descriptor**, and
the descriptor is the untested variable.
