# Utils for OT Methods Benchmark

## Slurm

### On *Compute Canada* clusters

The basic script to run the computations is `scripts/basic_slurm_task.sh`.

One can monitor the GPU usage on the node with the following command, which runs `nvidia-smi` every 30 seconds
```
$ srun --jobid 123456 --pty watch -n 30 nvidia-smi
```

