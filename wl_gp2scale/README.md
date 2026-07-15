# wl_gp2scale

Explicit-vocabulary Weisfeiler–Lehman descriptor + a distributed, **block-sparse
gp2Scale** Gaussian-process kernel, for scaling energy regression to **200k
molecules on 16 GPUs** (4 Perlmutter GPU nodes).

This is a new, self-contained module: it imports nothing from `descriptor_eval/`
or `hybrid_descriptor/`. It reuses their *ideas* (explicit-vocab WL, the dense
`ψ(r)=(1−r)⁴(4r+1)` Wendland, PLS(10), `data_id` category blocks) but scales them:

- **Never materialises the dense N×N covariance.** Each gp2Scale block is built on
  the GPU; fvgp's worker-side wrapper extracts its non-zeros (`sparse.coo_matrix`)
  in global coordinates and gathers only those, so the global matrix is sparse and
  the solve uses the iterative sparse path (`sparseCG`). (The kernel returns the
  dense block — required by fvgp 4.8.3's prediction path, which uses the kernel
  output as `np.diag(kk)` and the CG right-hand side; returning a sparse block
  breaks `posterior_covariance`. The dense block is a transient local to each
  worker.)
- **Neighbor search = `torch.cdist` within blocks.** No FAISS/cuML (neither is
  installed). gp2Scale already tiles into ≤10k×10k blocks; each worker does one
  on-GPU `cdist`, applies the compact-support Wendland, and thresholds to sparse.
- **Category block-sparsity.** The embedding carries an integer `data_id` tag in
  its last column; disjoint-category blocks return a pre-built empty block with no
  distance computed, and straddling blocks get a per-pair mask.
- **WL at scale.** Sparse CSR output, `min_count=5`, vocabulary frozen on a
  stratified sample of all categories, transform parallelised over molecules.
- **Supervised reduction that stays sparse.** Streaming SIMPLS via sparse matvecs
  (centering folded in, never densified) → 10-D embedding.
- **Recalibrated cutoff.** Percentile of the 200k embedding's pairwise distances.

## Files
| file | role |
|------|------|
| `data.py` | 200k OMol25 loader, `ExtensiveMean` residual target, `data_id` → int categories |
| `wl_features.py` | `SparseWLFeaturizer` → `csr_matrix` (frozen vocab, parallel transform) |
| `reduce.py` | `SparsePLS` streaming SIMPLS (sparse, supervised) + batch-PLS parity |
| `kernel.py` | `make_wl_block_kernel` gp2Scale GPU block kernel; `check_kernel_psd` |
| `cutoff.py` | `recalibrate`, `sparsity_report` |
| `pipeline.py` | embedding pipeline, category tag/sort, Dask connect, `build_gp`/`predict` |
| `validate.py` | pre-run checklist |
| `run_200k.py` | full-run CLI |

## Environment (Perlmutter)
```bash
module load python/3.11-24.1.0
source ./gpomol/bin/activate      # venv with gpcam 8.4.1, torch 2.8, dask 2025.5.1
pip install imate                 # REQUIRED — see below (not in requirements.txt)
```

> **`imate` is mandatory for gp2Scale.** gpcam 8.4.1 / fvgp 4.8.3 import `imate`
> inside the gp2Scale constructor (randomised log-determinant), so it is needed to
> *even instantiate* a gp2Scale `GPOptimizer` — not only for training. It is not in
> the repo's `requirements.txt`. `build_gp` calls `require_imate()` and fails early
> with this guidance if it's missing. Install and validate it on Perlmutter first;
> if the manylinux wheel is unavailable, it may need a source build against the
> module's BLAS/LAPACK.

## 1. Validate first (20k–50k, 1–2 GPUs or local CPU)
Confirms: sparse≈dense parity, realized nnz fits memory, sparseCG converges at
jitter 1e-6, streaming-PLS R² == batch-PLS R², kernel PSD.
```bash
# local CPU smoke test (a few dask workers, no allocation needed)
python -m wl_gp2scale.validate --n 20000 --device cpu --workers 4
# to compare R² apples-to-apples with descriptor_eval/gp_parity.py add: --min-count 2

# single GPU node (uses the srun-launched workers for proper per-task GPU binding)
./allocate_GPUs.sh 1 4
./launch-dask-moduleGPU.sh 4
python -m wl_gp2scale.validate --n 50000 --device cuda --workers 4 \
    --scheduler-file $SCRATCH/scheduler_file_gpOmol.json
```
All five checks print `PASS`/values, including **predictive R² vs truth** (compare
to gp_parity.py ~0.09–0.12). If parity fails or CG stalls: **tighten the cutoff**
(`--cutoff-pct` lower) rather than adding jitter, or switch `--backend wendland_d0`
(PD on R¹⁰ by construction).

## 2. Full run (200k, 16 GPUs)
```bash
./allocate_GPUs.sh 4 16               # 4 nodes × 4 GPUs, account m4055_g
./launch-dask-moduleGPU.sh 16         # writes $SCRATCH/scheduler_file_gpOmol.json
python -m wl_gp2scale.run_200k --n 200000 --workers 16 --device cuda
```
Predict-only by default (hyperparameters frozen from validation, so the run is CG
solves only). `imate` is still required just to build the gp2Scale GP (above);
`--train` additionally exercises its log-determinant.

Outputs:
- `cache/preds_200k.npz` — `y_true`, `y_pred` (residual), `var`, `r2`, `rmse`,
  frozen `signal_var`/`cutoff`/`dim`/`min_count`/`depth`/`pls`/`cutoff_pct`, and
  `category_order` (the sort permutation).
- `cache/preds_200k_parity.png` — parity plot (pred vs true), points coloured by
  posterior std.
- `cache/mean_model_200000.npz`, `cache/y_residual_200000.npy` — recover physical
  energy as `E_total = y_pred + mean.predict(...)`.
- stdout — R²/RMSE vs baseline, realized sparsity/cutoff diagnostics, WL OOV rate
  and per-depth vocab sizes, and timings.

## Notes
- **One gp2Scale GP per dask client.** fvgp 4.8.3 forbids two live gp2Scale GPs on
  the same client (scatter refcount race). To build another sequentially, `del`
  your GP then call `pipeline.release_gp(client)`; or use a fresh client.
- Confirm the category key: this module reads `atoms.info["data_id"]`
  (override via `data.get_data(category_key=...)`).
- `ψ(r)=(1−r)⁴(4r+1)` is the d0=3 Wendland; on the 10-D embedding PD rests on
  compact support + a tight cutoff (diagonal dominance) + jitter 1e-6 + CG.
  `--backend wendland_d0` is the dimension-correct fallback.
