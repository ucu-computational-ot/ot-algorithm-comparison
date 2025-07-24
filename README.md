# Utils for OT Methods Benchmark

- See [docs/index.md](docs/index.md) for full documentation on main pipeline.
- See [docs/slurm.md](docs/slurm.md) for examples related to slurm.
- See [docs/color_transfer.md](docs/color_transfer.md) for detailed explanation of the color transfer experiment.
- See [docs/mnist.md](docs/mnist.md) for detailed explanation of the MNIST Classification experiment.

## Installing Pixi

This project uses [Pixi](https://prefix.dev/docs/pixi/) to manage dependencies. Follow the official installation instructions for your platform:

- [Linux](https://prefix.dev/docs/pixi/install#linux)
- [macOS](https://prefix.dev/docs/pixi/install#macos)
- [Windows](https://prefix.dev/docs/pixi/install#windows)

After installation run `pixi install` to set up the environment. Available tasks can be invoked with `pixi run <task>`.

### Common commands

- `pixi run serialize --config <config.yaml> --export-dir <directory>` - create problem datasets from `config.yaml` in the target directory.
- `pixi run benchmark --config <config.yaml> --folds <n> --export <file>` - run experiments using the configuration for `n` folds and write results to `file`.
- `pixi run lint` or `ruff check .` to lint the code.


## Slurm

### On *Compute Canada* clusters

The generic script for both SLURM and local runs is `scripts/run_benchmark.sh`.
For example:
`sbatch scripts/run_benchmark.sh configs/generators/gaussians.yaml configs/runners/gaussians.yaml`

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

## Running experiments

To run experiments, first create config file like:

```yaml
param-grids:
  regularizations:
    - reg: 1
    - reg: 0.001

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    jit: true
    param-grid: regularizations

problems:
  dir: datasets/synthetic
  names:
    - 1D-gaussians-64
  
experiment: 
  name: Measure on Gaussians
  function: uot.experiments.measurement.measure_time
```

Here you can define solvers and their param-grids (solver will be run for each set of params). Also in `problems` section with `dir` the `export-dir` of serialization is specified (see previous section) and with names specific folders with problems in that directory

```
$ pixi run benchmark --config configs/runners/gaussians.yaml --folds 1 --export results/raw/gaussians.csv
```

With `export` one can secify where to put csv-report of experiment

## Color Transfer

To run a Color Transfer experiment, first create config file like:

```yaml
param-grids:
  epsilons:
    - reg: 1
    - reg: 0.01

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    param-grid: epsilons
    jit: true

bin-number: 16
batch-size: 100000
pair-number: 3
images-dir: ./datasets/images
output-dir: ./outputs/color_transfer
rng-seed: 42

drop-columns:
  - transport_plan
  - u_final
  - v_final

experiment: 
  name: Time and test
  function: uot.experiments.measurement.measure_time_and_output
```
For detailed explanation of the parameters, please refer to [docs/color_transfer.md](docs/color_transfer.md).

The corresponding pixi command example:
```
pixi run color-transfer --config ./configs/color_transfer/example.yaml
```

There is also a feature to create a dashboard for visual comparison of the input images and results - the corresponding command is:
```
pixi run color-transfer-visualization --origin_folder <path_to_input_images> --results_folder <path_to_resulting_images>
```

## MNIST Classification

The MNIST classification experiment is performed in two steps.

- Distance matrix calculation.
- Classification itself.

For detailed config examples for each of them, please refer to [docs/mnist.md](docs/mnist.md).

The corresponding pixi commands:

- Step 1:
```
pixi run mnist_distances --config ./configs/mnist_dist_example.yaml
```

- Step 2:
```
pixi run mnist_classification --config ./configs/mnist_classification_example.yaml
```

## Linting

This project uses [Black](https://black.readthedocs.io/) and [Ruff](https://docs.astral.sh/ruff/) for code style. Run `pixi run lint` or `ruff check .` to lint the code.
