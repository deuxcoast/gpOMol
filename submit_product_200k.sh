#!/bin/bash
# The product-kernel UQ comparison at 200k. Perlmutter batch job.
#
#     sbatch submit_product_200k.sh                  # PROBE: seed 42, nte=50
#     PROBE=0 SEEDS="42" sbatch submit_product_200k.sh        # real, one seed
#     PROBE=0 SEEDS="7 123" sbatch submit_product_200k.sh     # the rest
#
# WHAT THIS TESTS. At 20k, k = k_wl * k_geom (intersection support) beat
# k_wl + k_geom (union) by +0.068 +- 0.009 on Spearman(sigma*,|err|) - Spearman(dNN,|err|),
# density-matched over 6 seeds. This asks whether that survives 10x the data. Both arms
# must run at the SAME global radius and the SAME realised density or the comparison is
# meaningless, which is why both scripts are given the SAME --diag-sample.
#
# RUN THE PROBE FIRST. The dominant cost is posterior variance: ONE LINEAR SOLVE PER TEST
# POINT against the full 160k system (fvgp gp_posterior; measured 0.58 s/point at 20k).
# 800 points x 2 arms x 3 seeds is somewhere between 2.5 and 12 hours depending on how CG
# iteration counts grow with N, and nobody has measured that here. The probe runs seed 42
# at nte=50 into a SEPARATE --out so the real run does not skip it (both scripts treat an
# existing output file as "already done"), and prints the per-point cost to extrapolate
# from before committing an 8h allocation.

#SBATCH -N 4
#SBATCH -n 16                      # 16 tasks = 16 GPUs = 16 dask workers
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -G 16
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -A m4055_g
#SBATCH -J product_200k
#SBATCH -o product_200k_%j.out
#SBATCH -e product_200k_%j.err

set -u
N_WORKERS=16                       # must equal -n above
N_MOL="${RUN_N:-200000}"
SEEDS="${SEEDS:-42}"
PROBE="${PROBE:-1}"

# --- driver-side memory, the thing that actually breaks at this N -------------------
# Every density/cutoff diagnostic materialises a dense (diag_sample x n_train) float64
# block ON THE DRIVER -- the dask workers are not involved. At n_train=160k that is
# 6.4 GB per channel at the 5000 default and 1.28 GB at 1000. Both scripts must get the
# SAME value: it sets the global radius each arm runs at, so a mismatch silently
# un-matches the two arms' densities while each script's own numbers stay consistent.
DIAG_SAMPLE=1000

if [ "$PROBE" = "1" ]; then
    NTE=50
    OUT="cache/product_200k_probe"
    SEEDS="42"
    echo "=== PROBE: seed 42, nte=$NTE -> $OUT (measure s/point, then set PROBE=0) ==="
else
    NTE=800
    OUT="cache/product_200k"
fi

ENV_BIN="${ENV_BIN:-$HOME/.conda/envs/gpomol/bin}"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
cd "$SLURM_SUBMIT_DIR"

export MALLOC_TRIM_THRESHOLD_=0
export PYTHONUNBUFFERED=1          # else print() block-buffers and an 8h job looks stalled
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1

# fail fast if the env is wrong, rather than after queueing for hours. imate is a HARD
# requirement of gpcam 8.4.1's gp2Scale constructor and is NOT in requirements.txt.
( cd / && python -c "import torch, gpcam, imate, dask, wl_gp2scale" ) || {
    echo "ERROR: $ENV_BIN/python cannot import torch/gpcam/imate/dask/wl_gp2scale" >&2
    exit 1
}
echo "python: $(which python)   PYTHONPATH=$PYTHONPATH"
echo "n=$N_MOL seeds=[$SEEDS] nte=$NTE diag_sample=$DIAG_SAMPLE out=$OUT"

# per-JOB scheduler file: concurrent jobs would otherwise rm+write the same $SCRATCH
# path and connect to each other's scheduler
sched="$SCRATCH/scheduler_file_gpOmol_${SLURM_JOB_ID}.json"
rm -f "$sched"

dask scheduler --interface hsn0 --scheduler-file "$sched" &
sched_pid=$!
until [ -f "$sched" ]; do sleep 2; done
echo "scheduler up -> $sched"

srun -n "$N_WORKERS" -o "dask_worker_${SLURM_JOB_ID}.txt" dask worker \
    --memory-limit="30 GiB" --scheduler-file "$sched" \
    --interface hsn0 --nworkers 1 --nthreads 1 &
workers_pid=$!

common=(--n "$N_MOL" --seeds $SEEDS --mean M1 --nte "$NTE"
        --workers "$N_WORKERS" --device cuda --diag-sample "$DIAG_SAMPLE"
        --scheduler-file "$sched" --out "$OUT")

# BASELINE FIRST, deliberately. It builds and caches cache/emb_${N_MOL}_s{seed}.npz
# (~10 min/seed: 316 s vocab scan + 222 s transform at 200k), which the product arm then
# reuses -- so the two arms are compared on byte-identical embeddings rather than two
# featurisation runs that could drift. M1 not M4: M4 buys +0.08 R2 but its residual is
# more size-structured, and a size+category sigma model then beats the GP. UQ is the goal.
echo "=== arm 1/2: additive baseline (percat_radius.py --arm global) ==="
python -u scripts/percat_radius.py --arm global "${common[@]}"
rc=$?
if [ $rc -ne 0 ]; then
    echo "baseline arm failed (rc=$rc); NOT running the product arm -- it would have no"
    echo "baseline to pair against and score_percat would report it against nothing." >&2
    kill "$workers_pid" "$sched_pid" 2>/dev/null; rm -f "$sched"; exit $rc
fi

echo "=== arm 2/2: product kernel (product_kernel.py) ==="
python -u scripts/product_kernel.py "${common[@]}"
rc=$?

echo "=== scoring ==="
# CHECK THE 'loaded N arms' LINE. A missing arm file makes this print a plausible wrong
# number rather than an error; that has happened once already (product-M4 was silently
# scored against the M1 baseline).
python -u scripts/score_percat.py "$OUT"

echo "driver exited with code $rc; tearing down cluster"
kill "$workers_pid" "$sched_pid" 2>/dev/null
rm -f "$sched"
exit $rc
