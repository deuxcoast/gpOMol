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
# A PREVIOUS instance of this script can still be waiting or still holding an srun
# even when its scheduler is dead, so 'dask scheduler' alone is not enough to detect
# it. Use a PID LOCKFILE rather than pgrep on our own name: `$(pgrep ...)` forks a
# subshell that inherits this script's command line, so a name match counts itself
# and the guard fires on every run.
lockfile="${scheduler_file%.json}.lock"
if [ -f "$lockfile" ]; then
    _prev=$(cat "$lockfile" 2>/dev/null || true)
    if [ -n "${_prev:-}" ] && kill -0 "$_prev" 2>/dev/null; then
        echo "ERROR: another launch-dask-conda.sh is already running (pid $_prev)" >&2
        echo "       Refusing to start a second cluster. To reset:" >&2
        echo "         kill $_prev; rm -f $lockfile" >&2
        echo "         pkill -u $USER -f 'dask scheduler'; pkill -u $USER -f 'dask worker'" >&2
        echo "         pkill -u $USER -f 'srun.*dask'; rm -f $scheduler_file" >&2
        exit 1
    fi
    rm -f "$lockfile"          # stale: the writer is gone
fi

if pgrep -u "$USER" -f "dask scheduler" > /dev/null 2>&1 \
   || pgrep -u "$USER" -f "srun.*dask worker" > /dev/null 2>&1; then
    echo "ERROR: a dask scheduler or a dask srun is already running for $USER" >&2
    echo "       pids: $(pgrep -u "$USER" -f 'dask scheduler|srun.*dask worker' | tr '\n' ' ')" >&2
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

# Claim the lock now that the guards have passed, and release it on ANY exit so a
# crashed run never blocks the next one.
echo $$ > "$lockfile"
trap 'rm -f "$lockfile"' EXIT

# The repo is not pip-installed, so `wl_gp2scale` is only importable via the CWD --
# and that does NOT reach the cluster. `dask scheduler`/`dask worker` are installed
# console scripts: their sys.path[0] is the env's bin/, not this directory. The
# scheduler must import wl_gp2scale to deserialize the task graph, so without this
# it dies with "ModuleNotFoundError: No module named 'wl_gp2scale'". PYTHONPATH is
# inherited by the backgrounded scheduler and by srun (--export=ALL), so it fixes
# both. (A local Client() never hit this: its workers inherit the parent's sys.path.)
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Check imports the way the SCHEDULER and WORKERS will see them: from a neutral cwd,
# so only PYTHONPATH can supply wl_gp2scale. Running `python -c` from the repo root
# would pass via the implicit cwd entry on sys.path and hide a broken PYTHONPATH --
# a false pass that is exactly how the scheduler's ModuleNotFoundError got through.
( cd / && python -c "import torch, gpcam, imate, dask, wl_gp2scale" ) || {
    echo "ERROR: $ENV_BIN/python cannot import torch/gpcam/imate/dask/wl_gp2scale" >&2
    echo "       with PYTHONPATH=$PYTHONPATH" >&2
    echo "       (run this from the repo root so wl_gp2scale is on PYTHONPATH)" >&2
    exit 1
}
echo "python     : $(which python)"
echo "dask       : $(which dask)"
echo "PYTHONPATH : $PYTHONPATH"

export slurm_cpu_bind="cores"
export MALLOC_TRIM_THRESHOLD_=0
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1

# NETWORK INTERFACE. hsn0 is Perlmutter's HPE Slingshot and does not exist on every
# cluster -- on Lawrencium the high-speed fabric is InfiniBand (ib0), and dask fails
# with "'hsn0' is not a valid network interface". Detect instead of hardcoding, in
# descending order of speed, and let DASK_INTERFACE override.
if [ -z "${DASK_INTERFACE:-}" ]; then
    for _if in hsn0 ib0 eth0; do
        if [ -e "/sys/class/net/$_if" ]; then DASK_INTERFACE=$_if; break; fi
    done
fi
if [ -z "${DASK_INTERFACE:-}" ]; then
    echo "ERROR: no usable network interface found. Available:" >&2
    ls /sys/class/net >&2
    echo "       set DASK_INTERFACE=<name> and re-run." >&2
    exit 1
fi
echo "interface  : $DASK_INTERFACE  (available: $(ls /sys/class/net | tr '\n' ' '))"

rm -f "$scheduler_file"

echo "starting scheduler -> $scheduler_file"
dask scheduler --interface "$DASK_INTERFACE" --scheduler-file "$scheduler_file" &
sched_pid=$!

# BOUNDED wait, and check the scheduler is still alive while waiting.
#
# This used to be `until [ -f "$scheduler_file" ]; do sleep 5; done`, which is a trap
# with a genuinely baffling symptom: if the scheduler dies during startup (a bad
# --interface, a port clash) the file never appears, so the script spins here
# forever instead of exiting. It then HIJACKS the next launch -- as soon as that run
# writes the scheduler file, this stale loop satisfies and fires its OWN srun, so two
# sruns fight over the same tasks and both die with
# "tasks N-M: Exited with exit code 1" / "Force Terminated". The top-of-script guard
# does not catch it either, because by then this instance's scheduler is dead.
for _ in $(seq 120); do
    [ -f "$scheduler_file" ] && break
    if ! kill -0 "$sched_pid" 2>/dev/null; then
        echo "ERROR: the scheduler exited during startup (see its traceback above)." >&2
        echo "       Nothing to clean up; fix the cause and re-run." >&2
        exit 1
    fi
    sleep 1
done
if [ ! -f "$scheduler_file" ]; then
    echo "ERROR: no scheduler file after 120s; killing the scheduler." >&2
    kill "$sched_pid" 2>/dev/null
    exit 1
fi
echo "scheduler up (pid $sched_pid); starting $number_of_workers workers"

# Foreground on purpose: this srun is what keeps the workers alive. Worker stdout
# goes to dask_worker_info.txt. srun inherits PATH (and the salloc GPU binding, so
# each task gets its own GPU via --gpus-per-task=1).
srun -n "$number_of_workers" -o dask_worker_info.txt dask worker \
    --memory-limit="30 GiB" \
    --scheduler-file "$scheduler_file" \
    --interface "$DASK_INTERFACE" \
    --nworkers 1 \
    --nthreads 1
