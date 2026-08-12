# Handoff prompt

Paste the block below to start a fresh session.

---

Continuing the gpOMol project (`wl_gp2scale`: exact gp2Scale GP on OMol25).

Read the memory index, then these in order:

  1. `omol25-position-paper-goals`        — why this project exists
  2. `crps-skill-and-percat-sv`           — THE METRIC THE PAPER REPORTS, and the one lever that moved it
  3. `percat-radius-and-likelihood-gate`  — the training gate, CORRECTED 2026-08-06
  4. `sparsity-not-discovered`            — the evidence behind the block mask

FRAMING, AND IT HAS CHANGED. The paper is **not** a GP-vs-GNN comparison. It is about
**scaling an exact GP to a 4M-structure molecular dataset with meaningful
uncertainties**, reporting accuracy, MSE and CRPS. Judge proposals against that, not
against beating a GNN. Several sessions were spent optimising a σ*-vs-dist-to-NN gap
that is a model-selection diagnostic, not a paper result — don't reopen it.

WHERE WE ARE. 200k runs clean on Perlmutter: GP R² 0.7469 on 40k test points, build
247 s, 800 variance solves at 7.45 s/pt, zero solver failures. Calibration is good
(cov95 0.955 at 20k). The open weakness is **sharpness**: the GP's per-point σ is worth
only a few percent of CRPS over a single constant variance, and a plain dist-to-NN
baseline gets more. Per-family signal variance is the one lever that moved it — CRPS
skill 1.42% → 2.99%, paired +1.57 at t=12.3, free on accuracy, PSD-free.

## TASK 1: start training the GP

Marcus's plan, phases 1–3 already built and committed:

* **Phase 1 PASSED** — Gibbs-Wendland is numerically PSD (`scripts/gibbs_psd_gate.py`).
  The prefactor exponent MUST be the ambient embedding dimension, not the Wendland's
  design dimension; d=3 gives a relative min-eigenvalue of −4.7e-2 that our jitter does
  not rescue.
* **Phase 2 DONE** — `GibbsWendlandKernel` in `wl_gp2scale/kernel.py`, `--gibbs` in
  `scripts/product_kernel.py`. Reduces bit-exactly to the stationary kernel when ℓ is
  constant (`scripts/test_gibbs_kernel.py`, max diff 0.00e+00).
* **Phase 3 DONE** — `scripts/percat_likelihood.py --kernel gibbs` profiles ℓ per
  family. Run it at 40k first; it takes ~5 min and needs no cluster.

Read the gate memo before designing the MCMC. Its headline was CORRECTED: "the
likelihood has no interior optimum" was a grid artifact. Widened to m=2048, every
informative family turns over — but at **81–100% block density**, so the objection is
computational, not identifiability. MCMC *will* converge; the sparsity-preferring prior's
job is to trade likelihood against cost. Specify its strength in **neighbours per row**
so it reads as a stated budget. The saturated optima are flat (reactivity gives up only
~190 nats across an 8× range in ℓ), so the prior can pull ℓ down a long way cheaply —
that number is the argument to put to Marcus.

Also unfinished and cheap: `--sv-fit minus-noise` exists but is untested at scale. At
200k the model asserts ~58% more variance than the errors have (std(z) 0.795), because
`sv = var(residual)` has the jitter added *on top*. And `score_calibration.py` now
reports a `scale-free` skill column that fits each candidate its own σ scale on half the
points — use it, because raw CRPS skill conflates σ's scale with its shape and went
NEGATIVE at 200k for that reason alone.

## TASK 2: is the cross-family block mask justified?

`kernel.py` zeroes cross-family covariance unconditionally. **This makes the model
mathematically identical to ten independent per-family GPs** — a block-diagonal
covariance factorises exactly, so a point in family c has a posterior depending only on
family c's data. So "could we just run several smaller GPs?" is not an alternative
design; it is a restatement of what we already have.

That matters for the paper: "an exact GP on 4M structures" is currently ten exact GPs on
~400k each. Still a real result, a different claim, and a referee will notice.

What the data says so far, and it does NOT support the mask:

* Families overlap heavily — within-family median distance 0.669 vs across-family 0.815,
  a **1.22× separation**, with ~30% of cross-family pairs closer than the typical
  within-family pair (`sparsity-not-discovered`). k-NN distance distributions are
  unimodal in both channels.
* Nearest neighbours are 71% same-family at k=5 against 16.9% chance — chemistry is a
  GRADIENT here, not modes. The mask imposes the position paper's success structure
  rather than discovering it.
* Leave-one-family-out (`scripts/lofo_uq.py`): with the mask, a held-out family gets
  σ* = the prior EXACTLY for every point — constant, hence useless for ranking, which is
  precisely the acquisition case.

Things to measure:

1. **Does cross-family covariance help?** Run the same config with and without the mask
   at 20k and compare R², CRPS skill and coverage. `lofo_uq.py` already runs maskless and
   is the closest existing harness. Watch the confound: dropping the mask changes the
   realised density, so re-match it before comparing.
2. **What does the mask cost at 4M?** Per-family GPs of ~400k are far cheaper than one
   4M GP, so if the mask is right, the scaling story should be told that way honestly —
   and if it is wrong, cross-family covariance is information being thrown away.
3. **Is there a middle option?** A Gibbs kernel with ℓ varying continuously WITHIN
   families but the mask retained is what Marcus specified (his option B, already built).
   Dropping the mask entirely costs the free PSD guarantee, the exact block-separable
   likelihood profiling (~1 s/eval instead of ~20 s), and the sparsity the 4M target
   needs — the driver-side CSR is already ~2.5 GB at 200k and gp2Scale assembles it on
   the CLIENT (`gp_prior.py:294-306`), so ~50 GB at 4M is the binding constraint. Ask
   Marcus whether that assembly can be distributed; it is the concrete risk to the
   headline claim.

## Gated out — do NOT reopen

* **GNN descriptors.** Closed with Marcus after a full screen (PR #62). A learned
  embedding gives 2.2× better k-NN prediction but no better near-neighbour resolution
  (g(1) 0.6195 vs our 0.6158), worse drift, and its σ* advantage is present on random
  splits and gone on held-out chemistry.
* **More PLS rank / different metrics.** The nugget floor is ~0.61 at g(1) and rank
  flattens by ~28 dims/channel. L1, L0.5 and L∞ all lose on the axis that matters.
* **The WL subtree kernel at scale** (density is scale-invariant), **M0 at scale**
  (−0.36 R²), **PCA/unsupervised compression for the kernel**, **the §4.3 bump kernel**.

## Process rules that have cost time

* Smoke-run every batch on ONE seed at n=2000 before launching a sweep.
* Check the scorer's `loaded N arms` line — a missing arm produces a plausible wrong
  number, not an error.
* **Never quote a metric computed with different parameters as a baseline.** g(k) is not
  scale-free; a hardcoded baseline from a different sample size made a GNN look 0.12
  better than it was, and the number was reported three times at three magnitudes before
  being caught.
* 20k is for SCREENING, scale is for CONFIRMING. wl×strain looked like a win at 20k over
  6 seeds and the advantage was gone by 80k.
