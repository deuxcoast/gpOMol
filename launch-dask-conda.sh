#!/bin/bash
# Launch a Dask scheduler + one worker per SLURM task for the `gpomol` CONDA env.
#
# Why this exists (vs launch-dask-moduleGPU.sh):
#   * That script does `source ./gpomol/bin/activate`, i.e. a VENV in the repo dir,
#     as documented in README.txt. The actual environment on this system is a CONDA
#     env (~/.conda/envs/gpomol). The source fails, and because there is no `set -e`
#     the script sails on.
#   * It then relies on `module load python/3.11-24.1.0`, which HAS dask but NOT
#     torch/gpcam/wl_gp2scale. The workers therefore register and look healthy, then
#     die as soon as they are asked to unpickle the kernel function.
#   * The module load also PREPENDS its python to PATH, so activating conda before
#     running the script would not help either.
#
# Instead of activating anything, this prepends the env's bin/ to PATH directly --
# no conda shell-function machinery needed in a non-interactive script -- and srun
# inherits it (--export=ALL is the default), so the workers get the right python.
#
# USAGE (the srun below stays in the FOREGROUND by design, to keep the workers
# alive, so background this script or use a second shell):
#
#     ./allocate_GPUs.sh 1 4
#     ./launch-dask-conda.sh 4 &
#     python -m wl_gp2scale.validate --n 50000 --device cuda --workers 4 \
#         --min-count 2 --cutoff-pct 10 \
#         --scheduler-file $SCRATCH/scheduler_file_gpOmol.json
#
# Override the env location with ENV_BIN=/path/to/env/bin ./launch-dask-conda.sh 4

set -u

number_of_workers=${1:?usage: launch-dask-conda.sh <number_of_workers>  (match salloc -n)}
ENV_BIN="${ENV_BIN:-$HOME/.conda/envs/gpomol/bin}"
scheduler_file=$SCRATCH/scheduler_file_gpOmol.json

# Refuse a second concurrent launch, BEFORE the slow import check so it fails fast.
# Running this twice is self-destroying and the symptoms are baffling: the second
# run's `rm -f` deletes the live scheduler file, its scheduler fights the first for
# the port, and its srun blocks forever because the first srun already holds every
# task in the allocation.
if pgrep -u "$USER" -f "dask scheduler" > /dev/null 2>&1; then
    echo "ERROR: a 'dask scheduler' is already running for $USER" >&2
    echo "       pids: $(pgrep -u "$USER" -f 'dask scheduler' | tr '\n' ' ')" >&2
    echo "       Refusing to start a second cluster. To reset:" >&2
    echo "         pkill -u $USER -f 'dask scheduler'; pkill -u $USER -f 'dask worker'" >&2
    echo "         pkill -u $USER -f 'srun.*dask'; rm -f $scheduler_file" >&2
    exit 1
fi

if [ ! -x "$ENV_BIN/python" ]; then
    echo "ERROR: no python at $ENV_BIN. Set ENV_BIN=/path/to/your/env/bin" >&2
    exit 1
fi
# Deliberately NO `module load python/...` here: it would prepend its own python and
# shadow the env, which is the trap in launch-dask-moduleGPU.sh.
export PATH="$ENV_BIN:$PATH"

# Fail fast rather than let workers register and then die on unpickling the kernel.
python -c "import torch, gpcam, imate, dask, wl_gp2scale" || {
    echo "ERROR: $ENV_BIN/python cannot import torch/gpcam/imate/dask/wl_gp2scale." >&2
    echo "       (wl_gp2scale is imported from the CWD -- run this from the repo root.)" >&2
    exit 1
}
echo "python : $(which python)"
echo "dask   : $(which dask)"

export slurm_cpu_bind="cores"
export MALLOC_TRIM_THRESHOLD_=0
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1

rm -f "$scheduler_file"

echo "starting scheduler -> $scheduler_file"
dask scheduler --interface hsn0 --scheduler-file "$scheduler_file" &

sleep 5
until [ -f "$scheduler_file" ]; do sleep 5; done
echo "scheduler up; starting $number_of_workers workers"

# Foreground on purpose: this srun is what keeps the workers alive. Worker stdout
# goes to dask_worker_info.txt. srun inherits PATH (and the salloc GPU binding, so
# each task gets its own GPU via --gpus-per-task=1).
srun -n "$number_of_workers" -o dask_worker_info.txt dask worker \
    --memory-limit="30 GiB" \
    --scheduler-file "$scheduler_file" \
    --interface hsn0 \
    --nworkers 1 \
    --nthreads 1
