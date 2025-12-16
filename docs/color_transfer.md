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

bin-number:
  - 16
  - 32
soft-extension:
  - no
  - yes
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
- "bin-number" can be a single integer or a list of integers, and the benchmark will iterate over each value; it affects the detailedness of color grids created for the images;
- "batch-size" represents the number of operations done simultaneously when working with JAX;
- "pair-number" represents the number of individual experiments performed per solver configuration (not including the warm-up runs);
- "images-dir" is the path to the directory with original images;
- "output-dir" is the path to a folder where the resulting images and output dataframe will be stored;
- "rng-seed" is specified for reproducibility;
- "drop-columns" allows the user to drop certain columns from the resulting dataframe;
- "soft-extension" accepts either a single boolean/yes-no value or a list of them. Each value triggers a full pass of the benchmark, saving images and metrics for the requested soft-extension mode (e.g., `["no", "yes"]` runs both variants back-to-back);
- "experiment" section works similarly to the one used in the main pipeline. **Important:** the measurement function chosen must return a transport_plan among other metrics. If the user doesn't require it in the final dataframe, it can be added to "drop-columns".

The experiment is carried out as such: for each solver, **pair-number** number of source-target image pairs will be generated, then the optimal transport plan between each of them is calculated and the source is transported to the target based on that transport plan.

The corresponding pixi command example:
```
pixi run color-transfer --config ./configs/color_transfer/example.yaml
```

There is also a feature to create a dashboard for visual comparison of the input images and results - the corresponding command is:
```
pixi run color-transfer-visualization --origin_folder <path_to_input_images> --results_folder <path_to_resulting_images>
```
