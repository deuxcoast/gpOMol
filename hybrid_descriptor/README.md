# Hybrid-descriptor GP for OMol25 (gpCAM / gp2Scale)

A modular pipeline for **Candidate #1**: a three-channel hybrid descriptor fed to
an exact Gaussian process via gpCAM's `gp2Scale` mode, targeting the **1–2M
molecule** regime where calibrated uncertainty is the deliverable (not chasing
100M, which the storage arithmetic rules out at the molecule level).

The design is _falsification-first_: a cheap gate on a 10⁵ subsample must pass
before any cluster time is spent, and every diagnostic runs on the **intensive
residual**, never on total energy.

```
raw molecules ─► ExtensiveEnergyModel ─► intensive residual ────────────┐ (GP y_data)
             └─► HybridFeatureAssembler ─► standardised features        │
                                        └─► FeatureReducer (PCA) ─► X ───┤
X, residual ─► [FALSIFICATION GATE] ─► gp2Scale GPOptimizer + Wendland ──┘
                                    └─► block-MCMC training
predict: GP posterior on residual + extensive mean = physical E,  +  posterior σ (UQ)
```

## Files

| File                  | Role                                                                       |
| --------------------- | -------------------------------------------------------------------------- |
| `extensive_mean.py`   | Element-referencing linear model → intensive residual (Step 1)             |
| `features.py`         | WL, distance-histogram, charge channels + assembler (Step 2)               |
| `embedding_kernel.py` | PCA reducer + Wendland-Mahalanobis kernel + PD guard (Step 3)              |
| `diagnostics.py`      | Falsification gate with explicit kill numbers (Step 4)                     |
| `gp_fit.py`           | Orchestration + gp2Scale wiring + prediction; **edit the OMol25 I/O here** |

## Step 1 — extensive prior mean

Total energy is extensive; the GP should never try to model that. We fit
`E_ext(M) = intercept + Σ_Z n_Z ε_Z (+ optional low-order composition terms)` by
ridge least squares over the whole training set and **subtract it**, handing the
GP the residual as `y_data`.

Why pre-subtract instead of a gpCAM `mean_function`: element references fit on
~1M molecules are pinned to numerical precision, so folding them into the trained
hyperparameter vector buys no meaningful uncertainty and just enlarges the MCMC.
More importantly, pre-subtraction makes the residual an explicit array — so every
number in the gate is computed on it, and the sparse kernel can never be silently
credited with variance the mean removed.

Sanity check first: `ExtensiveEnergyModel.reference_energies()` should return
per-element values close to known atomic energies. If residual variance is still
size-correlated (visible as a size-scaling nugget), add columns via
`extra_feature_fn` (bond/ring counts, `n_atoms²`) — do **not** reach for a fancier
kernel.

## Step 2 — the hybrid descriptor (all channels intensive)

`v(M) = [ WL (D_WL) | distance-histogram (n_bins) | charge scalars (3) ]`

- **Topology — WL.** Weisfeiler–Lehman subtree patterns, hashed to a fixed length
  with `blake2b` (not Python's salted `hash()`, which would be irreproducible),
  summed over depths 0..h, then divided by atom count → intensive per-atom pattern
  frequencies.
- **Geometry — distance histogram.** All intramolecular pairwise distances,
  histogrammed on fixed shared bins, normalised to sum 1. Rotation/translation
  invariant with no alignment and no chemical cutoff (long pairs overflow into the
  last bin). **This is the representation Wasserstein-over-atoms was reaching for,
  kept as a fixed vector and compared with plain Euclidean distance so the kernel
  stays PD** (a compact-support Wendland over an OT metric is not PD — see Step 3).
- **Electronics — charge scalars.** `[dipole_per_atom, var(q), max(q)−min(q)]` from
  **Löwdin or NBO** charges. **Do not use Mulliken** — its basis-set instability
  enters here as feature noise and shows up as an inflated nugget. OMol25 ships
  Löwdin/NBO in `atoms.info`.

Concatenate and z-score standardise (`HybridFeatureAssembler`). The assembler
records per-channel column slices so the gate can build a **WL-only** embedding
for the skill comparison.

## Step 3 — reduction + kernel (this is where PD lives)

PCA to `D` dimensions, then an **ARD Wendland** on the PCA coordinates. PCA
decorrelates the axes; ARD length scales rescale them → a diagonal Mahalanobis in
PCA space = a full Mahalanobis in feature space, with the metric _learned_. (No
whitening: ARD already learns the per-axis scale.)

### The dimension trap — read before setting `n_components`

A Wendland ψ\_{d₀,k} is positive-definite **only on R^{d₀} and below**. The
Wendland in the gp2Scale paper (Eq. 3) is a **d₀ = 3** construction — it was built
for 3-D spatial data. Applying it at D = 15–25 is **not** PD-guaranteed, and a
passing empirical check on a small subsample does **not** prove PD-ness (a finite
sample can miss the offending configuration; at full scale the Cholesky then
fails). The rule:

> **embedding dimension `D` ≤ Wendland design dimension `d₀`.**

Two clean ways to satisfy it, both provided in `embedding_kernel.make_wendland_mahalanobis`:

- **`backend="explicit"` (default, use for D = 15–25):** a genuine ψ\_{D,k} with
  `d₀ = D` (closed forms for k∈{0,1,2,3}, validated against the textbook ψ\_{3,1..3};
  k=2 gives C⁴, the analog of the Matérn-5/2 the position paper argues for). PD on
  R^D by Wendland's theorem.
- **`backend="gpcam"` (fast/sparse, use only for D ≤ 3):** gpCAM's native
  `wendland_anisotropic` (d₀ = 3). Warns if you hand it D > 3.

There is a genuine tension — gp2Scale's native kernel _wants_ D = 3, the chemical
descriptor _wants_ D = 15–25. Don't assume; **measure** it: compare kNN skill at
D = 3 (native) against D = 15–25 (explicit) to see what compressing to 3 actually
costs. `FeatureReducer.retained_variance()` is the first cheap read on that.

Whichever backend, run `check_kernel_psd` on a subsample before fitting. Treat it
as a **necessary** guard (a materially negative eigenvalue ⇒ not PD, stop), not a
sufficiency proof — the sufficiency comes from `D ≤ d₀`.

### Hyperparameter layout

```
hps[0]      = signal variance                     bounds ≈ [1e-3, 10·var(resid)]
hps[1:D+1]  = per-PCA-axis length scales/support  bounds ≈ [1e-2·range, 2·range]
```

The length scales _are_ the anisotropic support radii: pairs farther apart than
the support get exactly zero covariance, which creates the sparse matrix
`gp2Scale` exploits. Keep upper bounds tight — length scales that grow too large
destroy sparsity and defeat the framework. `default_hp_bounds` sets these from the
reduced embedding.

## Step 4 — the falsification gate (`run_falsification`)

All four run on the intensive residual over a ~10⁵ subsample (the pairwise checks
use a few thousand at a time). Kill numbers are module-level constants in
`diagnostics.py`.

| Check                       | Passes if                                        | Meaning of failure                                |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| **A. kNN skill vs WL-only** | hybrid beats WL by ≥ `REL_GAIN_MIN` (10%)        | geometry+charge added no usable signal            |
| **B. semivariogram nugget** | `nugget/sill < NUGGET_MAX` (0.12); target ≤ 0.09 | can't break WL's ~2 eV / ~12% ceiling             |
| **C. kNN distance CV**      | `CV ≥ CV_MIN` (0.30)                             | distances concentrated → no radius gives sparsity |
| **D. feasibility**          | `s* < S_STAR_MAX` (0.50) at the target N         | matrix too dense to store within 40 TB            |

**On s\* (fixed convention to avoid the two-definition trap):** `s*` is the
**density** = fraction of pairs within the support radius = non-zero fraction of
the covariance matrix. Sparsity = 1 − s\*. Storage ≈ `12·s*·N²` bytes, so you want
s\* small. Crucially, s\* is computed at the **variogram range** (the correlation
length), _not_ an arbitrary distance percentile — density at the p-th percentile
is trivially ~p/100 and tests nothing. The range is where covariance actually
decays to ~0, so it is the physically meaningful support radius linking "how far
correlations reach" to "how sparse the matrix is."

Storage arithmetic (descriptor-independent): at s\* ≈ 0.06 (your ~94%-zeros
organic figure) the matrix breaks the 40 TB budget near N ≈ 7.5M; at s\* ≈ 0.5,
near N ≈ 2.6M. Either way molecule-level 100M is out — which is why the target is
1–2M and the contribution is calibrated UQ, whose advantage window coincides with
that feasibility window.

## Running it

```python
from gp_fit import load_omol25_subset, gate_then_fit, predict_energy, validate
from distributed import Client

# 1. supply OMol25 data (fill in load_omol25_subset / build_graph for AseDBDataset)
Z_lists, graphs, positions, charges, y_total = load_omol25_subset(n=100_000)

# 2. gate on the subsample; only fits if the gate passes
client = Client(); client.wait_for_workers(4)
gpo, pre, report = gate_then_fit(
    Z_lists, graphs, positions, charges, y_total,
    target_N=1_500_000, n_components=15, dask_client=client,
)

# 3. predict physical energy + calibrated uncertainty
E_pred, E_std = predict_energy(gpo, pre, Z_lists, graphs, positions, charges)
print(validate(gpo, pre, Z_lists, graphs, positions, charges, y_total))  # rmse, crps
```

To sanity-check the gate logic and the from-scratch numpy paths without gpCAM,
run `_smoke_test.py` (synthetic data; not part of the pipeline).

## What you still owe the pipeline (OMol25-specific, marked `TODO`)

- `load_omol25_subset` — `AseDBDataset` access; pull `energy`, positions, and
  `loewdin_charges`/`nbo_charges` from `atoms.info`.
- `build_graph` — connectivity (RDKit if you have SMILES, else ASE
  `natural_cutoffs` + `build_neighbor_list`); return `(adjacency, node_labels)`.

## Honest bottom line

If the gate fails at B (nugget) or A (no gain over WL), that is the real answer:
the hybrid descriptor doesn't beat WL and the contribution is the calibrated-UQ
story in the 1–2M regime, not a new accuracy record. The gate is built to tell you
that _before_ you spend the cluster, not after.
