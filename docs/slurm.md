# Slurm

Yet, we use `conda` tool to handle virtual environments and packages on the cluster.
The name of the conda environment (`ot`) is hardcoded into scripts.

Inside each script we enable jax to use:
- 64-bit float numbers,
- gpu,
- platform allocator of the memory with the preallocation switched off.

# Slurm Job Submission Examples

```bash
sbatch -J task-name --time 14-00:00:00 --export=RESULT_DIR=results/synthetic scripts/run_benchmark_online.sh \
    config/generators/2D/problem_set.yaml configs/runners/all_solvers.yaml
```

```bash
sbatch -J task-name --time 14-00:00:00 --export=DATA_FILE=datasets/synthetic.h5,RESULT_DIR=results/synthetic \
    scripts/run_benchmark_hdf5.sh config/generators/2D/problem_set.yaml configs/runners/all_solvers.yaml
```

```bash
sbatch -J task-name --time 14-00:00:00 --export=DATA_DIR=datasets/synthetic,RESULT_DIR=results/synthetic \
    scripts/run_benchmark.sh config/generators/2D/problem_set.yaml configs/runners/all_solvers.yaml
```
