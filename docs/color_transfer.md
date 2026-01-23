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
displacement-interpolation:
  - 0.0
  - 1.0
color-space: rgb
# active-channels: [r, g]
batch-size: 100000
pair-number: 3
images-dir: ./datasets/images
rng-seed: 42

drop-columns:
  - transport_plan
  - monge_map
  - u_final
  - v_final

experiment: 
  name: Time and test
  output-dir: ./outputs/color_transfer
```
- "bin-number" can be a single integer or a list of integers, and the benchmark will iterate over each value; it affects the detailedness of color grids created for the images.
- "batch-size" represents the number of operations done simultaneously when working with JAX.
- "pair-number" represents the number of individual experiments performed per solver configuration (not including the warm-up runs).
- "images-dir" is the path to the directory with original images.
- "experiment.output-dir" is the path to a folder where the resulting images and output dataframe will be stored.
- "rng-seed" is specified for reproducibility.
- "drop-columns" allows the user to drop certain columns from the resulting dataframe. Use it to remove large artifacts like `transport_plan` or `monge_map`.
- "soft-extension" accepts either a single boolean/yes-no value or a list of them. Each value triggers a full pass of the benchmark, saving images and metrics for the requested soft-extension mode (e.g., `["no", "yes"]` runs both variants back-to-back).
- "displacement-interpolation" accepts a single alpha or list of alphas in [0, 1] for post-processing the map. Each alpha runs as a separate mode.
- "color-space" selects the space used to build OT marginals (supported: `rgb`, `lab`/`cielab`).
- "active-channels" optionally selects a subset of channels by name or index (e.g., `[l, a]` or `[0, 1]`), reducing the OT dimension.
- "experiment" section works similarly to the one used in the main pipeline.

The experiment is carried out as such: for each solver, **pair-number** number of source-target image pairs will be generated, then the optimal transport plan between each of them is calculated and the source is transported to the target based on that transport plan.

Outputs are written to a timestamped subfolder under `experiment.output-dir` and include `color_transfer_results.csv` plus any saved images.

### Back-and-Forth solver notes

- For `BackNForthSqEuclideanSolver`, the returned Monge map is in index coordinates.
- The CIC map construction mirrors the CIC pushforward; pushforward-from-map and pushforward-from-potential should be close (up to interpolation error).
- The adaptive map is a representative average of adaptive samples, so a map-based pushforward can differ from the adaptive pushforward. This is expected; use CIC for tighter agreement if needed.

The corresponding pixi command example:
```
pixi run color-transfer --config ./configs/color_transfer/example.yaml
```

There is also a feature to create a dashboard for visual comparison of the input images and results - the corresponding command is:
```
pixi run color-transfer-visualization --origin_folder <path_to_input_images> --results_folder <path_to_resulting_images>
```
