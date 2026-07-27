#!/bin/bash
# Interactive GPU allocation on LAWRENCIUM.
#
# allocate_GPUs.sh is Perlmutter-only and cannot work here: it passes
# `--account m4055_g` (a NERSC repo), `--qos interactive` (a NERSC QoS),
# `--constraint gpu` (how Perlmutter picks GPU nodes -- Lawrencium picks them by
# PARTITION), and no --partition at all, which is what produces
# "Invalid partition name specified".
#
# Nothing here is guessed. Set these once per shell (or in ~/.bashrc) from the
# values SLURM itself reports:
#
#     sacctmgr -n show assoc user=$USER format=Account%20,Partition%15,QOS%40
#     sinfo -o "%18P %8a %12l %6D %10G %N" | grep -i -E "gpu|es1"
#
#     export LRC_ACCOUNT=...      # e.g. pc_something / lr_something
#     export LRC_PARTITION=...    # the GPU partition (es1 on Lawrencium)
#     export LRC_QOS=...          # e.g. es_normal / es_debug
#     export LRC_GPU_TYPE=...     # optional; Lawrencium GPU nodes are heterogeneous
#                                 # so --gres often needs a type, e.g. V100 / A40
#
# USAGE:  ./allocate_GPUs_lrc.sh <nodes> <total_tasks> [gpus_per_node]
#
# The THREE-PLACES RULE still applies and is the single most common way to burn an
# allocation here: <total_tasks> must equal the argument to launch-dask-conda.sh AND
# the --workers passed to the python driver. A mismatch does not error -- the run
# blocks forever waiting for workers that never arrive.
set -euo pipefail

nodes=${1:?usage: $0 <nodes> <total_tasks> [gpus_per_node]}
tasks=${2:?usage: $0 <nodes> <total_tasks> [gpus_per_node]}
gpus_per_node=${3:-$(( tasks / nodes ))}

missing=()
[[ -z "${LRC_ACCOUNT:-}"   ]] && missing+=(LRC_ACCOUNT)
[[ -z "${LRC_PARTITION:-}" ]] && missing+=(LRC_PARTITION)
[[ -z "${LRC_QOS:-}"       ]] && missing+=(LRC_QOS)
if (( ${#missing[@]} )); then
    echo "ERROR: unset: ${missing[*]}" >&2
    echo "Discover them with:" >&2
    echo "  sacctmgr -n show assoc user=\$USER format=Account%20,Partition%15,QOS%40" >&2
    echo "  sinfo -o \"%18P %8a %12l %6D %10G %N\" | grep -i -E 'gpu|es1'" >&2
    exit 2
fi

# Lawrencium requests GPUs with --gres, not --gpus-per-task/--constraint. The node
# types are mixed, so a bare `gpu:N` can land you on hardware you did not intend.
if [[ -n "${LRC_GPU_TYPE:-}" ]]; then
    gres="gpu:${LRC_GPU_TYPE}:${gpus_per_node}"
else
    gres="gpu:${gpus_per_node}"
    echo "NOTE: LRC_GPU_TYPE unset -> requesting '${gres}'. Lawrencium GPU nodes are" >&2
    echo "      heterogeneous; set LRC_GPU_TYPE to pin the hardware." >&2
fi

# $SCRATCH is a NERSC convention and is usually unset here, which would silently make
# launch-dask-conda.sh write its scheduler file to /scheduler_file_gpOmol.json.
if [[ -z "${SCRATCH:-}" ]]; then
    export SCRATCH="/global/scratch/users/$USER"
    echo "NOTE: SCRATCH was unset; using $SCRATCH (export it in ~/.bashrc to persist)." >&2
fi

# es_debug caps well below es_normal, and salloc rejects the whole request if --time
# exceeds the QoS limit, so this has to be settable rather than hardcoded.
walltime=${LRC_TIME:-04:00:00}

set -x
salloc --nodes "$nodes" -n "$tasks" \
       --ntasks-per-node="$gpus_per_node" \
       --gres="$gres" \
       --partition "$LRC_PARTITION" \
       --qos "$LRC_QOS" \
       --account "$LRC_ACCOUNT" \
       --time "$walltime"
