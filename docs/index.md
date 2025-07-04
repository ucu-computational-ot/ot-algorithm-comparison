# OT Algorithm Comparison Framework

This documentation describes the structure of the repository and how to use it
for benchmarking optimal transport (OT) algorithms. It is aimed at both users
who want to run experiments with the provided tools and developers who plan to
contribute new solvers or problem generators.

## Repository Overview

The project is organised as a normal Python package with the main source code
under the `uot/` directory. Important subpackages are:

- **`uot.algorithms`** – standalone algorithm implementations used in the
  experiments.
- **`uot.data`** – utilities for representing measures and loading datasets.
- **`uot.problems`** – classes that describe OT problems as well as generators
  that create synthetic datasets.
- **`uot.solvers`** – interfaces and concrete solvers. Each solver implements
  the `BaseSolver` API.
- **`uot.experiments`** – functions and helpers for running benchmarks.

Configuration files for dataset generation and experiment runs live in the
`configs/` directory while convenience scripts for SLURM jobs are placed under
`scripts/`.

## Problem Structure

All experiments revolve around *problems* that specify the input measures and
the associated cost matrices.  The base class `MarginalProblem` lives in
`uot.problems.base_problem`.  At the moment only `TwoMarginalProblem` is fully
implemented.  It represents a pair of measures $(\mu,\nu)$ together with a cost
function.  Problems are serialised and later consumed by the benchmark runner so
that dataset generation and solver execution are decoupled.

## Installation

The recommended way to set up the environment is via [pixi](https://pixi.sh/):

```bash
pixi install
```

Alternatively you can rely on standard `pip` using the `requirements.txt`
file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generating Synthetic Problems

Synthetic datasets are created from YAML configuration files. Each file can
define one or more generators.  A minimal example is shown below
(`configs/generators/gaussians.yaml`):

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

The key `generator` specifies the fully qualified class name of a subclass of
`ProblemGenerator`. All remaining fields are passed as arguments to the class
constructor.

### Available Generators

- **`GaussianMixtureGenerator`** – samples Gaussian mixture models on a fixed
  grid.  Parameters include `dim`, `num_components`, `n_points` and optional
  Wishart hyper-parameters.
- **`CauchyGenerator`** – creates 1‑D Cauchy-distributed marginals.  It accepts
  `dim`, `n_points`, `borders` and the usual common options.
- **`ExponentialGenerator`** – produces 1‑D exponential distributions with a
  random scale parameter.
- **`PairedGenerator`** – composes two other generators by drawing `mu` from the
  first and `nu` from the second.  The nested generator configurations are given
  in `gen_a_cfg` and `gen_b_cfg`.

YAML anchor syntax is used extensively in the provided configs to avoid
repetition.  Anchors are introduced with `&name` and later referenced using
`*name`:

```yaml
defaults: &g
  dim: 1
  n_points: 32

my_dataset:
  <<: *g
  generator: uot.problems.generators.GaussianMixtureGenerator
```

Multiple generators may be listed in a single file and will be serialised one by
one.

Datasets are serialised with:

```bash
pixi run serialize --config <path/to/generator.yaml> --export-dir <output/dir>
```

A folder for each generator will be created under `export-dir` along with a
`meta.yaml` file storing the configuration.

Serialising problems allows expensive dataset generation to happen once.  The
runner later loads them via `ProblemStore` and `ProblemIterator`.  When using the
YAML configs, loading happens automatically so you rarely need to interact with
these classes directly.

To visually inspect a dataset you can run `python -m uot.problems.inspect_store
--dataset <path> --outdir plots` which saves plots of the stored distributions.

## Running Experiments

Benchmark runs are configured in YAML as well. An example runner configuration
is `configs/runners/gaussians.yaml`:

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

`param-grids` define sets of solver parameters. Each entry under `solvers`
references a `BaseSolver` subclass and optionally one of the parameter grids.
The `problems` section points to the previously serialised datasets. Finally the
`experiment` section chooses one of the measurement functions.

A runner file may use YAML anchors in the same spirit as generator configs.  For
example common solver parameters can be defined once and reused:

```yaml
defaults: &run
  jit: true

solvers:
  sinkhorn:
    <<: *run
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    param-grid: regularizations
```

The available measurement functions are defined in
`uot.experiments.measurement`:

- **`measure_time`** – record execution time.
- **`measure_solution_precision`** – compare solver output with the ground truth
  optimal cost when available.
- **`measure_with_gpu_tracker`** – additionally track GPU and CPU resource
  usage.

Use the function that best matches the metrics you want to collect.

Run the benchmark with:

```bash
pixi run benchmark --config <path/to/runner.yaml> --folds 1 --export results.csv
```

The command produces a CSV table with the collected metrics. The `folds`
argument controls how many times the experiment is repeated.

Internally the CLI calls `run_pipeline` from `uot.experiments.runner`.  The
function loads the problems, repeats them according to `folds`, and iterates
over every solver/parameter combination.  Its output is a single pandas data
frame that you can post-process or save to CSV.

## Implementing New Solvers

To add a custom solver create a subclass of `BaseSolver` in `uot/solvers`:

```python
from uot.solvers.base_solver import BaseSolver

class MySolver(BaseSolver):
    def solve(self, marginals, costs, **kwargs):
        # your optimisation code
        return {"transport_plan": ..., "cost": ...}
```

Solvers are expected to accept a sequence of `BaseMeasure` objects (`marginals`)
and a sequence of cost arrays. They return a dictionary of results and metrics.
Any additional keyword arguments are taken from the corresponding entry in the
runner configuration.

A solver becomes available in configs via its import path, e.g.
`uot.solvers.my_solver.MySolver`.

### Built‑in Solvers

- `LinearProgrammingTwoMarginalSolver` – wraps the exact solver from the
  `ot` package.
- `SinkhornTwoMarginalSolver` – JAX implementation of entropic Sinkhorn.
- `GradientAscentTwoMarginalSolver` – first‑order solver using gradient ascent.
- `LBFGSTwoMarginalSolver` – quasi‑Newton optimisation via `jaxopt.LBFGS`.

When adding your own solver make sure it returns at least the transport plan and
its cost.  Additional metrics can be included freely.  Once implemented, list
the solver in a runner YAML and provide a parameter grid if needed.

## Writing Problem Generators

Generators produce lists of OT problems. Implement a subclass of
`ProblemGenerator` in `uot/problems/generators` and override the `generate`
method:

```python
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem

class MyGenerator(ProblemGenerator):
    def generate(self, n_points: int, **kwargs) -> list[TwoMarginalProblem]:
        # create measures and costs
        return [TwoMarginalProblem(...)]
```

The generator can then be referenced in a YAML configuration as shown earlier.

Generators should yield `TwoMarginalProblem` instances.  They are typically used
only during the serialisation stage, after which the pickled problems are loaded
by the runner.  When designing a new generator aim to make all parameters
configurable via the constructor so that they can be specified from YAML.

## Using Existing Tools Programmatically

While YAML configuration files are the main interface, all functionality can be
used directly from Python. The following snippet illustrates how to create a
generator, serialise problems and run a solver without reading any config:

```python
from uot.problems.generators import GaussianMixtureGenerator
from uot.problems.problem_serializer import save_problems
from uot.solvers.sinkhorn import SinkhornTwoMarginalSolver
from uot.experiments.measurement import measure_time
from uot.experiments.experiment import Experiment

# generate dataset
gen = GaussianMixtureGenerator()
problems = gen.generate(n_points=64, num_datasets=10)

# run experiment
exp = Experiment("timing", measure_time)
solver = SinkhornTwoMarginalSolver
results = exp.run_on_problems(problems, solver, reg=1e-3)
results.to_csv("results.csv", index=False)
```

## Contributing

1. Fork the repository and create a new branch for your feature.
2. Ensure `pre-commit` checks and unit tests pass before opening a pull request.
3. Document new modules and add tests when appropriate.

See the existing tests under the `tests/` directory for examples.

---

For detailed usage examples consult the configuration files inside
`configs/`. Additional SLURM job templates are available under `scripts/`.
