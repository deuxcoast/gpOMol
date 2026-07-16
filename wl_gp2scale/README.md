# wl_gp2scale

Explicit-vocabulary Weisfeiler–Lehman descriptor + a distributed, **block-sparse
gp2Scale** Gaussian-process kernel, for scaling energy regression toward **200k
molecules on 16 GPUs** (4 Perlmutter GPU nodes).

Self-contained: imports nothing from `descriptor_eval/` or `hybrid_descriptor/`. It
reuses their *ideas* (explicit-vocab WL, the `ψ(r)=(1−r)⁴(4r+1)` Wendland, PLS(10),
`data_id` category blocks) and scales them.

## Status (measured, not aspirational)

| check | result |
|---|---|
| sparse gp2Scale == dense `scipy.cdist` reference | **R² 0.1188 vs 0.1188** at 20k, `corr=1.000000` |
| matches `descriptor_eval/gp_parity.py` (~0.09–0.12) | **yes**, at `--min-count 2` |
| streaming SIMPLS == sklearn batch PLS | identical to 4 dp (1e-16 on synthetic) |
| CUDA path (fp64 `cdist`, gp2Scale, sparseCG, GPU binding) | **works** (50k, 4 GPUs) |
| does more training data improve R²? | **UNANSWERED — see "What is not yet known"** |

## Files
| file | role |
|------|------|
| `data.py` | OMol25 loader, `ExtensiveMean` residual target, `data_id` → int categories |
| `wl_features.py` | `SparseWLFeaturizer` → `csr_matrix` (frozen vocab, Dask-parallel transform) |
| `reduce.py` | `SparsePLS` streaming SIMPLS (sparse, supervised) + batch-PLS parity |
| `kernel.py` | `make_wl_block_kernel`; `check_kernel_psd`, `check_kernel_diagonal` |
| `cutoff.py` | `recalibrate`, `sparsity_report`, `suggest_percentile` |
| `pipeline.py` | embedding pipeline, category tag/sort, Dask connect, `build_gp`/`predict` |
| `validate.py` | pre-run checklist (parity, sparsity, PSD, diagonal, PLS parity) |
| `diagnose.py` | `--mode bisect` (attribute a failure) / `--mode sweep` (size the cutoff) |
| `run_200k.py` | full-run CLI; also usable as a learning-curve tool |

## Environment

See `../README.txt`. Short version: conda env `gpomol`, **`pip install imate`**
(mandatory — gp2Scale cannot even be constructed without it), and launch Dask with
`../launch-dask-conda.sh` (backgrounded, from the repo root), **not**
`launch-dask-moduleGPU.sh`.

## Usage

```bash
# 1. Validate (local CPU, no allocation needed)
python -m wl_gp2scale.validate --n 20000 --device cpu --workers 4 --min-count 2

# 2. Validate on GPU
./allocate_GPUs.sh 1 4
./launch-dask-conda.sh 4 > dask_launch.log 2>&1 &     # wait for "scheduler up"
python -m wl_gp2scale.validate --n 50000 --device cuda --workers 4 --min-count 2 \
    --cutoff-pct 10 --scheduler-file $SCRATCH/scheduler_file_gpOmol.json

# 3. Size the 200k cutoff (CPU-only; needs no GPU and no Dask)
python -m wl_gp2scale.diagnose --mode sweep --n 20000 --ntr 3000 --min-count 2 \
    --pcts 25,10,5,2,1 --target-n 196000

# 4. Full run
./allocate_GPUs.sh 4 16
./launch-dask-conda.sh 16 > dask_launch.log 2>&1 &
python -m wl_gp2scale.run_200k --n 200000 --workers 16 --device cuda \
    --min-count 2 --cutoff-pct 10 --no-variance
```

**Pass `--min-count 2`.** The default is 5 (a scale-motivated guess that was never
validated); every result above used 2, which is what `gp_parity.py` uses.

Outputs of `run_200k`: `cache/preds_200k.npz` (`y_true`, `y_pred` residual, `var`,
`r2`, `rmse`, the frozen knobs, `category_order`), `cache/preds_200k_parity.png`
(parity plot), and `cache/mean_model_*.npz` + `cache/y_residual_*.npy` to recover
`E_total = y_pred + mean.predict(...)`.

---

## Design notes — the non-obvious parts

### Nothing is trained. There is no MCMC.
The GP is given a **one-element** hyperparameter vector:

| quantity | how it's set | learned? |
|---|---|---|
| `signal_var` = `hps[0]` | `var(y_train)`, closed form | no |
| `cutoff` | a percentile of the embedding's pairwise distances | **no — a fixed constant in the kernel** |
| length scale | folded into `cutoff`; no separate parameter | n/a |

`--train` is off by default and `cutoff_is_hp=False`. This mirrors `gp_parity.py`,
which is why it reproduces that baseline. It is *not* the usual gpCAM idiom (ARD
Wendland with trained per-axis length scales that double as support radii — that's
`hybrid_descriptor/embedding_kernel.py` and the `gpOmol.ipynb` notebook).

The cutoff *is* the maximum distance at which two molecules have nonzero covariance
(ψ(r)=0 for r ≥ cutoff), and ψ tapers smoothly within it — but the radius and the
taper shape are both fixed. Making the radius a trained hyperparameter is a one-line
flag, but then **sparsity — and therefore driver memory — varies during
optimisation**, and you must size memory for the bound anyway.

### torch.cdist's default mode silently corrupts the kernel
`torch.cdist` defaults to `compute_mode="use_mm_for_euclid_dist_if_necessary"`, the
Gram expansion `‖a‖²+‖b‖²−2a·b`. It suffers catastrophic cancellation for identical
points and returns a **nonzero self-distance** (3.05e-05 float32, 9.31e-10 float64),
so the Wendland diagonal lands below `signal_var`. `scipy`'s `cdist` returns exactly 0.

This Gram is near-singular (compact support + duplicate molecules; measured
**cond ≈ 9.4e9**), so a ~2e-4 relative perturbation amplified into R² 0.049 → 0.027
while raising no error. Hence: `compute_mode="donot_use_mm_for_euclid_dist"` and
**`dtype="float64"`** (both load-bearing), plus `check_kernel_diagonal` as a guard —
`K[i,i]` must equal `signal_var` *exactly*. Do not "optimise" either back.

### The cutoff percentile is a density knob and does NOT transfer across N
In-support pairs ≈ `pct/100 × P(same category)`, so **neighbours/point scales with N**:

| N_train | pct | nbrs/pt | driver peak |
|---|---|---|---|
| 16,000 | 25 | 818 | 0.5 GB |
| 196,000 | **25** | **10,023** | **~70 GB** |
| 196,000 | 2.04 | 818 | ~5.8 GB |

Use `cutoff.suggest_percentile(N, target_neighbors, frac_same_category)`, or the
sweep's `nbr@N` column. What transfers is the **neighbour count**, not the percentile.

Memory is **driver-side**: gp2Scale distributes kernel *evaluation*, but fvgp gathers
the COO components and assembles one scipy CSR on the client (`gp_prior.py:294-306`),
where the solve runs. Budget the driver's RAM (~256 GB), not `n_workers × 30 GiB`.

### Prediction cost is wildly asymmetric
`posterior_mean` uses the precomputed `KVinvY` → **one solve total**, cheap.
`posterior_covariance` does `KVsolve(k)` → **one solve per test point** on the full
N×N system; at 196k that is hours-to-days for a few thousand test points. The
cross-covariance `k` is also built **dense** (196k × 4k ≈ 6.3 GB).
So: `--predict-batch` bounds the memory (not the solve count), and `--no-variance`
gives mean-only. Want variance at 200k? Keep the test set in the *hundreds*.

### Other gotchas
- **`imate` is required to construct any gp2Scale GP**, not just to train one.
  `build_gp` calls `require_imate()` and fails early.
- **One live gp2Scale GP per dask client** (fvgp `WeakValueDictionary` guard). `del`
  yours, then `pipeline.release_gp(client)`; or use a fresh client.
- **The kernel returns a DENSE block.** fvgp's worker-side wrapper does
  `sparse.coo_matrix(k)` itself and gathers only non-zeros, so the global matrix is
  still sparse and the dense N×N never exists. Returning sparse breaks
  `posterior_covariance` (it needs `np.diag(kk)` and a dense CG right-hand side).
- **PSD:** `ψ(r)=(1−r)⁴(4r+1)` is the d0=3 Wendland, not guaranteed PD on a 10-D
  embedding. In practice `min_eig ≈ −2e-15` (roundoff — PSD but *singular*), and
  `--backend wendland_d0` is the dimension-correct fallback. A near-zero `min_eig` is
  not itself a bug; it is why float64 matters.
- **Vocabulary is fit on ALL train molecules** by default (`vocab_sample=0`).
  Subsampling leaves train labels out-of-vocabulary and silently discards signal
  (it cost 20.6% of label occurrences once).
- `atoms.info["data_id"]` supplies the category (override via `get_data(category_key=)`).

## What is not yet known

**Whether more training data improves R².** Every R² above comes from a GP trained on
3000 points: `validate`'s parity harness pins the GP at `--parity-n` regardless of
`--n`, because it needs a dense Cholesky reference to compare against. It is a
*parity* tool, not a learning-curve tool. Likewise `diagnose --mode sweep` measures R²
at `--ntr` (1–102 neighbours) while production would see 757–9,789 — so its **R²
column cannot pick the 200k cutoff**; only its density/memory columns are reliable.

The deciding measurement is the full-train learning curve, with the neighbour count
held roughly fixed:

```bash
python -m wl_gp2scale.run_200k --n 20000 --cutoff-pct 25 --test-size 0.2 \
    --min-count 2 --no-variance --out cache/lc_20k.npz   # 16k train, ~818 nbrs
python -m wl_gp2scale.run_200k --n 50000 --cutoff-pct 10 --test-size 0.2 \
    --min-count 2 --no-variance --out cache/lc_50k.npz   # 40k train, ~1022 nbrs
```

If R² climbs, 200k is justified. Encouraging sign: the PLS linear probe went
**−0.2660 (16k) → +0.1212 (40k)**, i.e. the representation is still improving with
data. Caveat: those two runs draw different test sets, so they are comparable in
expectation but not matched — `descriptor_eval/learning_curve.py` does this properly
with a fixed test set and nested train prefixes.

Also open: `sparseLU` vs `sparseCG` at 200k (LU was ~100× more accurate in the
bisect — `max|Δ|` 2.3e-06 vs 2.2e-04 — but its fill-in at 200k is the concern).
