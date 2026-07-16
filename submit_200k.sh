#!/bin/bash
# Batch job: the full 200k gp2Scale GP run, self-contained.
#
# Unlike the interactive flow (salloc -> launch-dask-conda.sh in one terminal ->
# python in another), a batch job runs ONE script start to finish with no human in
# the loop, so every step that was easy to forget is baked in here. Submit with:
#
#     sbatch submit_200k.sh
#
# Why batch at all: pct=25 (the signal-optimal cutoff) is projected at ~5.4h of
# driver-side solve, past the 4h `interactive` QOS cap. `regular` QOS allows the
# longer walltime; -t 08:00:00 gives comfortable margin.

#SBATCH -N 4                       # 4 nodes
#SBATCH -n 16                      # 16 tasks = 16 GPUs = 16 dask workers
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -G 16
#SBATCH -C gpu
#SBATCH -q regular                 # NOT interactive (that caps at 4h)
#SBATCH -t 08:00:00
#SBATCH -A m4055_g
#SBATCH -J wl_gp2scale_200k
#SBATCH -o run_200k_%j.out         # stdout  (%j = job id)
#SBATCH -e run_200k_%j.err         # stderr

set -u
N_WORKERS=16                       # must equal -n above
N_MOL="${RUN_N:-200000}"           # override for a cheap test: RUN_N=20000 sbatch ...

# --- environment (same lessons as launch-dask-conda.sh: PATH, not module load) ---
ENV_BIN="${ENV_BIN:-$HOME/.conda/envs/gpomol/bin}"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"   # so workers import wl_gp2scale
cd "$SLURM_SUBMIT_DIR"

export MALLOC_TRIM_THRESHOLD_=0
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1

# fail fast if the env is wrong, rather than after queueing for hours
( cd / && python -c "import torch, gpcam, imate, dask, wl_gp2scale" ) || {
    echo "ERROR: $ENV_BIN/python cannot import torch/gpcam/imate/dask/wl_gp2scale" >&2
    exit 1
}
echo "python: $(which python)   PYTHONPATH=$PYTHONPATH"

# --- launch the Dask cluster (both backgrounded, unlike the interactive script) ---
sched="$SCRATCH/scheduler_file_gpOmol.json"
rm -f "$sched"

dask scheduler --interface hsn0 --scheduler-file "$sched" &
sched_pid=$!
until [ -f "$sched" ]; do sleep 2; done
echo "scheduler up -> $sched"

# one worker per task, backgrounded so the driver below can run in the foreground
srun -n "$N_WORKERS" -o "dask_worker_${SLURM_JOB_ID}.txt" dask worker \
    --memory-limit="30 GiB" --scheduler-file "$sched" \
    --interface hsn0 --nworkers 1 --nthreads 1 &
workers_pid=$!

# --- the driver runs in the FOREGROUND; when it returns, the job ends ------------
# pct=25 = signal-optimal cutoff (the reason for the 8h job). --no-variance because
# posterior variance is one solve PER test point. connect_dask waits for all 16.
python -m wl_gp2scale.run_200k \
    --n "$N_MOL" --min-count 2 --cutoff-pct 25 --test-size 0.02 \
    --no-variance --workers "$N_WORKERS" --device cuda \
    --scheduler-file "$sched" --out "cache/preds_${N_MOL}.npz"
rc=$?

echo "driver exited with code $rc; tearing down cluster"
kill "$workers_pid" "$sched_pid" 2>/dev/null
exit $rc
