#!/bin/bash

export slurm_cpu_bind="cores"
number_of_workers=$1  # MUST BE SAME AS SBATCH -n

scheduler_file=$SCRATCH/scheduler_file_gpOmol.json
rm -f $scheduler_file

malloc_trim_threshold_=0
module load python/3.11-24.1.0
source ./gpomol/bin/activate

echo we have nodes: ${slurm_job_nodelist}

echo "$sdn_ip_addr"


hn=$(hostname -s)
port="8789"
echo ${port}
echo "starting scheduler"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1

dask scheduler \
    --interface hsn0 \
    --scheduler-file $scheduler_file &

dask_pid=$!

# Wait for the scheduler to start
sleep 5
until [ -f $scheduler_file ]
do
     sleep 5
done

echo "starting workers"
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \


srun -o dask_worker_info.txt dask worker  --memory-limit="30 GiB" \
    --scheduler-file $scheduler_file \
    --interface hsn0 \
    --nworkers 1 \
    --nthreads 1 \


