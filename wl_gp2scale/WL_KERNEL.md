# The WL kernel proper: sparsity IS discoverable, and discovering it does not help

*Executed 2026-07-29. 20k subset, data seed 0, split seeds 42/7/123, M4 prior mean on the
same cached PLS embeddings as the radius grid, jitter 4.30, 800 held-out points per arm.
Scripts: `scripts/wl_kernel_sparsity/` (diagnostics) and `scripts/wl_gp/` (the GP fits).*

## Summary

The position paper's §3 gating question is *"will the MCMC with a sparsity-preferring
prior naturally discover a sparse covariance structure on this dataset?"* Every earlier
result in this project answered **no** — but always for our descriptor, a supervised
10-dimensional PLS compression in which no exact zero can ever occur, so sparsity had to
be imposed by a radius. This tests the paper's own alternative, the WL kernel on subtree
counts (Eq. 7), whose zeros are combinatorial.

Three findings, in increasing order of how much they cost the thesis:

1. **Sparsity is discoverable.** At depth ≥ 4 the WL kernel, with *no category mask at
   all*, connects molecules at **2.8× chance purity**. No radius, no hyperparameter, no
   Wendland envelope, PSD by construction. That is §3's success mode, achieved.
2. **Discovering it buys nothing over imposing it.** The unmasked arm's held-out R² is
   0.7973 against the masked arm's 0.7977.
3. **It is the best kernel we have for accuracy and the worst for uncertainty** — and
   uncertainty is the paper's entire claim.

---

## Eq. 7 as written is dense; the sparsity lives in the deepest block only

Measured on 4000 molecules (`scripts/wl_kernel_sparsity/`), chance purity 0.169:

| kernel | frac_zero | purity | density |
|---|---|---|---|
| depth 0 (element counts) | 0.0029 | 0.169 | 9.97e−1 |
| depth 1 | 0.0594 | 0.169 | 9.41e−1 |
| depth 2 | 0.4125 | 0.190 | 5.88e−1 |
| depth 3 | 0.8250 | 0.320 | 1.75e−1 |
| **Σ d=1..3** (our φ) | 0.0594 | 0.169 | 9.41e−1 |
| **Σ d=0..3** (Eq. 7) | **0.0029** | 0.169 | 9.97e−1 |

φ⁽⁰⁾ is the element-count vector, so it connects any two molecules sharing carbon. The
deeper problem is **the summation**: a sum is nonzero if *any* term is, so `Σ d=1..3` has
exactly the density of depth 1 alone. **Eq. 7 discards the sparsity of its own deep
terms.** Everything below therefore uses the **deepest block only**, `k = ⟨φ⁽ʰ⁾, φ⁽ʰ⁾′⟩`.

`min_count` is exactly irrelevant to the off-diagonal zero pattern — pruning singletons
removes only patterns present in one molecule, which contribute to no off-diagonal inner
product. Production's `min_count=2` costs nothing in kernel sparsity.

**Depth-3 density does not fall with N**: exponent −0.018 (R² 0.34, no trend) over
1k→16k while the vocabulary grows 11× (16,023 → 180,275 patterns). Overlap is not random
collision in a growing pattern space; it is a core of common motifs every molecule shares.
Depth, not scale, is the lever — and it does not collapse the way the classic WL failure
mode predicts. Sparsity and purity improve *together*:

| depth h | density | purity | isolated | median degree |
|---|---|---|---|---|
| 3 | 1.75e−1 | 1.89× | 6.2% | 519 |
| 4 | 5.76e−2 | 2.83× | 20.1% | 60 |
| 5 | 2.53e−2 | 3.90× | 32.4% | 13 |
| 6 | 1.77e−2 | 4.50× | 42.1% | 2 |

The cost is isolation, and it is survivable *here specifically* because isolated points
revert to the prior mean and `PRIOR_MEAN.md` establishes the mean already carries ~97% of
the accuracy. In a zero-mean GP this would be fatal.

---

## The GP fits

Seed 42, paired against the radius kernel on identical test points and prior mean.
`purity` is `P(same category | connected)` on the train Gram — 1.000 by fiat in the masked
arms, so **only the NO MASK row is a discovery claim**.

| kernel | density | purity | isolated | med deg | R² | ρ(σ\*,\|err\|) | top-5% |
|---|---|---|---|---|---|---|---|
| radius Wendland K=200 | 2.34e−2 | 1.000 | 0.0% | 538 | 0.8079 | **+0.2088** | 0.158 |
| h=4 cosine, masked | 2.77e−2 | 1.000 | 18.9% | 79 | 0.7977 | +0.0570 | 0.025 |
| **h=4 cosine, NO MASK** | 5.89e−2 | **0.470 = 2.8×** | 15.5% | 249 | 0.7973 | +0.0715 | 0.000 |
| h=5 cosine, masked | 1.65e−2 | 1.000 | 30.2% | 23 | 0.7981 | +0.0320 | 0.025 |
| **h=4 raw counts, masked** | 2.77e−2 | 1.000 | 18.9% | 79 | **0.8102** | +0.0067 | 0.000 |
| `dist-to-NN` *(no GP)* | — | — | — | — | — | **+0.2995** | — |

Zero solver non-convergence in every arm; fvgp's Krylov mode asserted `single` at startup.

### The accuracy result replicates and is the first kernel-side win in this project

`h=4 raw counts` against the radius kernel, paired within seed:

| seed | WL R² | radius R² | ΔR² | WL ρ | radius ρ | Δρ |
|---|---|---|---|---|---|---|
| 42 | 0.8102 | 0.8079 | +0.0023 | +0.0067 | +0.2088 | −0.2021 |
| 7 | 0.7370 | 0.7149 | +0.0221 | +0.0309 | +0.1289 | −0.0981 |
| 123 | 0.6781 | 0.6650 | +0.0131 | +0.1051 | +0.1733 | −0.0682 |
| | | | **+0.0125 ± 0.0057** | | | **−0.1228 ± 0.0406** |

Both significant at 2 SE, both consistent in sign across all three seeds. (The radius
column means 0.7293, matching `RADIUS_CALIBRATION.md`'s three-seed figure — the two
experiments are consistent.)

**+0.0125 is large in the only context that matters.** `PRIOR_MEAN.md` measured the
radius kernel's *entire* contribution over the prior mean at **+0.007**. The WL kernel
roughly **triples what the kernel contributes**, and does it with a median degree of 79
against 538 — comparable predictions from a seventh of the neighbours.

Note it is the **raw-count** arm that wins. Cosine normalisation costs ~0.012 R². Counts
are extensive and energy is extensive; this is the same lever `strain_feature` found.

### The uncertainty result is a rout, and that is the axis the paper rests on

Every WL configuration ranks errors at +0.007…+0.105, against +0.13…+0.21 for the radius
kernel and +0.30 for a distance with no GP in it. Top-5% recall is at or below the 0.05
chance rate in all four arms. **The kernel whose sparsity is principled produces a worse
posterior variance than the kernel whose sparsity we made up.**

The mechanism is visible in the table and differs by variant:

* **Cosine** fixes every molecule's prior variance at 1, so σ\* is driven almost entirely
  by *whether* a molecule is isolated (19–30% are) rather than by how densely its
  neighbourhood is sampled. It goes crudely bimodal instead of tracking local data density.
* **Raw counts** makes the diagonal scale as `‖φ‖² ∝ size`, and size is already in the
  prior mean — so σ\* ranks errors at essentially zero (+0.0067) while giving the best
  point predictions.

Neither tracks local data density, which is what error ranking needs and what
`dist-to-NN` measures directly.

---

## What this settles

§3's gating question has a **more interesting answer than "no"**: sparsity *is*
discoverable on OMol25 at WL depth ≥ 4, it *is* chemically meaningful (2.8× chance), it
costs no hyperparameter, and the resulting kernel *predicts better* than the compact-support
Wendland we have been tuning for months. Every premise of the paper's success mode holds.

**And the conclusion the paper draws from that premise does not follow.** The benefit it
promises — calibrated uncertainty for active learning, *"often more valuable than marginal
accuracy improvement"* (§7.2) — is precisely what the discovered-sparsity kernel is worst
at. It improves the thing the paper says it does not need and degrades the thing it does.

Combined with `RADIUS_CALIBRATION.md` (σ\* loses to a GP-free distance at every radius) and
the local-scaling result (non-stationarity cannot make family structure emerge), the
position is now: **on this dataset, every route we have tested to a useful GP posterior
variance fails, including the one the paper prescribes.** The remaining untested
prescription is §6.4 — Wasserstein distance on rotationally-invariant pairwise distance
profiles — which is a distance, and so would feed the same compact-support machinery whose
σ\* we have now measured three ways.

Honest limits: 20k, one dataset subset, energy only. The accuracy win is +0.0125 ± 0.0057
and deserves a larger-N confirmation before it is leaned on. None of this touches forces.
