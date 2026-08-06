# Handoff prompt

Paste the block below to start a fresh session.

---

Continuing the gpOMol project (`wl_gp2scale`: exact gp2Scale GP on OMol25).

Read the memory index, then these in order:

  1. `omol25-position-paper-goals`   — why this project exists
  2. `wl-gp2scale-next-steps`        — current state and the next experiment
  3. `product-kernel-uq`             — the breakthrough, read with the goals memo

FRAMING. The goal comes from the OMol25 position paper
(`~/Documents/textbooks/Mathematics/Gaussian Processes/gp2scale_OMol25.pdf`): show a large
EXACT GP can compete with GNNs like EquiformerV2, where the GP's advantage is CALIBRATED
UNCERTAINTY for active learning in DFT dataset construction, not accuracy. Judge proposals
against that goal, not against R².

WHERE WE ARE. For months the GP's posterior variance lost to a plain nearest-neighbour
distance on every UQ metric, and five separate levers failed to move it (support radius,
non-stationary local scaling, the paper's own WL subtree kernel, prior-mean strength,
per-category radii). Then a MULTIPLICATIVE kernel worked:

  k = k_wl * k_geom   (intersection support)   instead of   k_wl + k_geom   (union)

Density-matched, 6 seeds, 20k: +0.068 paired improvement in
`Spearman(sigma*,|err|) - Spearman(dist-to-NN,|err|)` at t = 7.5, mean-independent, and
free on accuracy. At the M1 mean rung the gap reaches -0.012 +- 0.007, which is 1.8 SE
from parity and the first time sigma* has beaten the `size+category` baseline.

Mechanism, predicted before the run: union support calls a molecule covered if it has
neighbours close in graph topology OR geometry; intersection needs BOTH. Coverage is what
sigma* measures, so a stricter coverage notion makes sigma* respond to genuine novelty.

THE TASK. Run the product kernel at 200k on Perlmutter and see whether the result holds at
scale. The command shape, the baseline arm to pair it against, and the cost warnings
(featurisation ~10 min; variance is ONE SOLVE PER TEST POINT; `--diag-sample 1000` because
the density band is a dense driver-side block) are all in `wl-gp2scale-next-steps`.

Before running, check `fvgp-gotchas` — several traps there silently corrupted results
(block-CG returns the PRIOR variance with no warning; `add_noise` defaults False;
unconverged solves warn once per process). Also read the "Gated out" section of
`wl-gp2scale-next-steps` before proposing anything: the MCMC campaign over rho(x) and the
WL subtree kernel at scale are both closed by measurement, not by opinion.

Two process rules that cost time last session: smoke-run every batch on ONE seed before
launching the sweep, and check the scorer's `loaded N arms` line — a missing arm produces
a plausible wrong number rather than an error.
