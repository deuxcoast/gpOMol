#!/bin/bash
# Tear down a dask cluster started by launch-dask-conda.sh, and clear its state.
#
# launch-dask-conda.sh now cleans up after itself on exit, so this should rarely be
# needed. It still matters in the cases where a trap cannot run: a SIGKILL, a login
# node reboot, or a shell closed out from under a backgrounded launch. It is also the
# safe thing to reach for when the launcher refuses to start and you want the reset in
# one command rather than five pkills copied out of an error message.
#
# Safe to run when nothing is up: every step is idempotent and exits 0.
#
#     ./stop-dask.sh          # tear down and report
#     ./stop-dask.sh --check  # report only, change nothing

set -u

scheduler_file="${SCHEDULER_FILE:-$SCRATCH/scheduler_file_gpOmol.json}"
lockfile="${scheduler_file%.json}.lock"
check_only=""
[ "${1:-}" = "--check" ] && check_only=1

pids() { pgrep -u "$USER" -f 'dask scheduler|dask worker|srun.*dask' 2>/dev/null | tr '\n' ' '; }

echo "scheduler file : $scheduler_file $([ -f "$scheduler_file" ] && echo '(present)' || echo '(absent)')"
echo "lock file      : $lockfile $([ -f "$lockfile" ] && echo "(present, pid $(cat "$lockfile" 2>/dev/null))" || echo '(absent)')"

# A lockfile whose writer is still alive means a launch script is genuinely running.
# Killing its cluster out from under it is how you get two half-dead clusters, so say
# so and stop rather than guessing.
if [ -f "$lockfile" ]; then
    owner=$(cat "$lockfile" 2>/dev/null || true)
    if [ -n "${owner:-}" ] && kill -0 "$owner" 2>/dev/null; then
        echo
        echo "NOTE: launch-dask-conda.sh is still running (pid $owner)."
        echo "      Stop it first -- 'kill $owner' -- and its own EXIT trap will clean"
        echo "      up the scheduler and both files. Then re-run this if anything is left."
        [ -n "$check_only" ] || exit 1
    fi
fi

before=$(pids)
echo "dask processes : ${before:-none}"

if [ -n "$check_only" ]; then
    echo
    echo "(--check: nothing changed)"
    exit 0
fi

if [ -n "${before// /}" ]; then
    # Order matters: kill the srun first so slurm does not restart tasks underneath us,
    # then the workers it was holding, then the scheduler they were reporting to.
    pkill -u "$USER" -f 'srun.*dask'     2>/dev/null || true
    pkill -u "$USER" -f 'dask worker'    2>/dev/null || true
    pkill -u "$USER" -f 'dask scheduler' 2>/dev/null || true
    sleep 2
    left=$(pids)
    if [ -n "${left// /}" ]; then
        echo "escalating to SIGKILL for: $left"
        pkill -9 -u "$USER" -f 'dask scheduler|dask worker|srun.*dask' 2>/dev/null || true
        sleep 1
    fi
fi

rm -f "$scheduler_file" "$lockfile"

after=$(pids)
if [ -n "${after// /}" ]; then
    echo
    echo "ERROR: processes survived: $after" >&2
    echo "       Inspect with: ps -fp $after" >&2
    exit 1
fi
echo
echo "clean: no dask processes, no scheduler file, no lock."
echo "Note this does NOT release your slurm allocation -- 'scancel <jobid>' or 'exit'"
echo "the salloc shell for that."
