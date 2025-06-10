# Utils for OT Methods Benchmark

## Slurm

### On *Compute Canada* clusters

The basic script to run the computations is `scripts/basic_slurm_task.sh`.

One can monitor the GPU usage on the node with the following command, which runs `nvidia-smi` every 30 seconds
```
$ srun --jobid 123456 --pty watch -n 30 nvidia-smi
```

## Synthetic datasets

To create synthetic dataset first need to create config file for generation:

```yaml
generators:
  1D-gaussians-64:
    generator: uot.problems.generators.GaussianMixtureGenerator
    dim: 1
    num_components: 1
    n_points: 64
    num_datasets: 30
    borders: (-6, 6)
    cost_fn: uot.utils.costs.cost_euclid_squared
    use_jax: true
    seed: 42
```

Class specified in `generator` field will be used and all other fields will be passed as init arguments to it. Section name (in this case `1D-gaussians-64`) will be used as generator name. Multiple generators in one config are allowed.

```
$ pixi run serialize --config configs/generators/gaussians.yaml --export-dir datasets/synthetic
```

In `export-dir` folders with serialized problems for each generators will be created. In the same folder `meta.yaml` will be created with copy of generator config. 