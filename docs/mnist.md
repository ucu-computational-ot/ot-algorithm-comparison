## MNIST Classification

The MNIST classification experiment is performed in two steps.

- **Distance matrix calculation.**

Corresponding config example:

```yaml
param-grids:
  epsilons:
    - reg: 1
    - reg: 0.01

solvers:
  sinkhorn:
    solver: uot.solvers.sinkhorn.SinkhornTwoMarginalSolver
    jit: true
    param-grid: epsilons
  
batch-size: 5000
output-dir: ./outputs/mnist/costs
```

    - "batch-size" represents the number of operations done simultaneously when working with JAX;
    - "output-dir" is the path to a folder where the resulting distance matrices will be stored.

- **Classification itself.**

Corresponding config example:

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

sample-sizes:
  - 100
  - 250

costs-dir: ./outputs/mnist/costs
output-dir: ./outputs/mnist/classification
```

    - "sample-sizes" specify how many numbers will be sampled for training and accuracy analysis;
    - "costs-dir" corresponds to the directory with distance matrices per solver configuration, geterated on the previous step;
    - "output-dir" is the path to a folder where the resulting metrics will be stored.


On the first step, distance matrices for the whole MNIST dataset are calculated per solver configuration specified.
On the second, for every solver configuration and sample size, a SVM taken from scikit-learn will be trained, with kernel matrices based on the abovementioned OT distance matrices. The output consists of accuracy per each of these trained SVM.


The corresponding pixi commands:

- Step 1:
```
pixi run mnist_distances --config ./configs/mnist_dist_example.yaml
```

- Step 2:
```
pixi run mnist_classification --config ./configs/mnist_classification_example.yaml
```
