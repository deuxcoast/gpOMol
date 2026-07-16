# gpOMol — environment and run notes (Perlmutter)

## Environment

The environment in use is a **conda** env named `gpomol` (python 3.11), at
`~/.conda/envs/gpomol`.

An earlier version of this file documented a **venv** flow:

```bash
python -m venv gpomol
gpomol source/bin/activate        # also malformed; should be `source gpomol/bin/activate`
```

No such venv exists on this system, and that mismatch has cost real debugging time:
`launch-dask-moduleGPU.sh` still sources `./gpomol/bin/activate` and silently
continues when it fails (see [Running](#running-gpu)). If you ever do standardise on
a venv, fix the launch scripts at the same time.

```bash
conda activate gpomol
pip install -r requirements.txt
pip install imate                 # REQUIRED for gp2Scale; NOT in requirements.txt
python -m ipykernel install --user --name gpomol --display-name gpomol
```

> **`imate` is mandatory, not optional.** gpcam 8.4.1 / fvgp 4.8.3 import it inside
> the gp2Scale constructor (for the randomised log-determinant), so it is needed to
> even _instantiate_ a gp2Scale `GPOptimizer` — not just to train one. Without it:
> `Exception: You have activated 'gp2Scale'. You need to install imate manually.`

A fresh interactive node does **not** have the env active. If `python` resolves to
`/usr/lib64/python2.7` you will get a `SyntaxError` on f-strings — that means you
forgot `conda activate gpomol` (your prompt should show `(gpomol)`).

## Running (GPU)

The **same number** must appear in all three places — `salloc -n`, the launch
script's argument, and `--workers`. A mismatch (e.g. launching 4 workers and asking
for 16) makes the run block waiting for workers that will never arrive, silently
burning the allocation. `connect_dask` now fails after `worker_timeout` naming both
counts instead of hanging.

```bash
N=4                                           # nodes*4 == tasks == GPUs == workers
./allocate_GPUs.sh 1 $N                       # <nodes> <tasks=GPUs>, account m4055_g
./launch-dask-conda.sh $N > dask_launch.log 2>&1 &
grep -c "Register worker" dask_launch.log     # wait until this reaches $N
python -m wl_gp2scale.validate --n 50000 --device cuda --workers $N \
    --min-count 2 --scheduler-file $SCRATCH/scheduler_file_gpOmol.json
```

For the full run it is `N=16` with `./allocate_GPUs.sh 4 16` (4 nodes x 4 GPUs).

Use `launch-dask-conda.sh`, **not** `launch-dask-moduleGPU.sh`. The latter sources
the non-existent venv, does not `set -e`, and so continues with
`module load python/3.11-24.1.0`. That python **has `dask` but not
`torch`/`gpcam`/`wl_gp2scale`**, so its workers register and look perfectly healthy,
then die the moment they are handed the kernel function to unpickle. The module load
also prepends its own python to `PATH`, so `conda activate gpomol` beforehand does
not rescue it either.

Three things about `launch-dask-conda.sh`:

- **Background it** (`&`). Its `srun` stays in the foreground on purpose — that is
  what keeps the workers alive. It is not hanging.
- **Redirect to a log.** It prints nothing for ~10s while torch imports; without a
  log it looks dead, and launching a second copy is self-destroying (the second
  `rm -f` deletes the live scheduler file, the schedulers fight for the port, and the
  second `srun` blocks forever on the allocation's tasks). The script now refuses a
  second launch, but redirect anyway.
- **Run it from the repo root.** It puts the repo on `PYTHONPATH` so the scheduler
  and workers can import `wl_gp2scale`. The repo is not pip-installed, and
  `dask scheduler` / `dask worker` are console scripts whose `sys.path` does **not**
  include your CWD — without `PYTHONPATH` the scheduler dies deserialising the task
  graph with `ModuleNotFoundError: No module named 'wl_gp2scale'`.

To reset a wedged cluster:

```bash
pkill -u $USER -f 'dask scheduler'; pkill -u $USER -f 'dask worker'
pkill -u $USER -f 'srun.*dask'; rm -f $SCRATCH/scheduler_file_gpOmol.json
```

`squeue -u $USER --steps` showing only `<jobid>.extern` is **normal** — that step
exists in every allocation and is not a leftover. The scheduler is not a SLURM step,
so check it with `pgrep -u $USER -f dask`.

## Layout

| path                 | role                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `wl_gp2scale/`       | distributed WL + gp2Scale GP for 200k molecules (see its `README.md`)                                                              |
| `descriptor_eval/`   | 10k-scale WL descriptor evaluation; `gp_parity.py` is the dense CPU reference kernel (R² ~0.09–0.12) that `wl_gp2scale` reproduces |
| `hybrid_descriptor/` | 4M-scale gp2Scale pipeline, ARD Wendland–Mahalanobis kernel                                                                        |
| `train_4M/`          | OMol25 ASE-LMDB shards (gitignored)                                                                                                |
