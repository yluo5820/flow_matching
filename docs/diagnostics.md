# Reading Outputs And Diagnostics

This guide explains how to read the plots, CSVs, and arrays produced by the `fm_lab`
commands. Update this file whenever a diagnostic output is added, renamed, or changes
meaning.

## Run Directory Layout

Most commands write a directory with this shape:

```text
run_dir/
  config.yaml
  metadata.json
  metrics.json                 # training runs only; includes trained_steps and early_stopping
  checkpoint.pt                # training runs only
  samples/                     # only when sample arrays are written
  trajectories/                # only when trajectory arrays are written
  diagnostics/                 # only when diagnostic tables/arrays are written
  plots/                       # only when figures are written
```

`config.yaml` is the effective config for that run. `metadata.json` records library
versions and git metadata. Prefer comparing runs by their saved configs and CSVs rather
than by memory of the command that produced them.

Artifact subdirectories are created lazily. For example, a dry run may only contain
`config.yaml` and `metadata.json`, while field diagnostics may contain `diagnostics/`
without `samples/` or `trajectories/`.

Training sample artifacts are drawn with `sampling.seed` when provided, otherwise
`experiment.seed`. The same source batch is used for every solver in
`plots/generated_samples_nfe*.png`, and the same trajectory initial positions are used
for every solver in `plots/trajectories_*_nfe*.png`. These references are saved as
`samples/source_reference.npy` and `trajectories/source_reference_nfe*.npy`.

The `.npy` files are the raw tensors behind plots and diagnostics:

- `samples/source_reference.npy`: source initial positions used for generated samples.
- `samples/target_reference.npy`: target samples shown in sample plots.
- `samples/<solver>_nfe*.npy`: final generated samples for a solver.
- `trajectories/source_reference_nfe*.npy`: trajectory initial positions.
- `trajectories/<solver>_nfe*.npy`: full ODE paths with shape `(time, trajectory, dim)`.

Keep them when you want to replot, inspect coordinates, compare solvers, compute new
metrics, or reproduce exactly which points were shown. They are not needed to resume
training; `checkpoint.pt`, `config.yaml`, and `metrics.json` are the important files for
that.

## 3D Toy Runs

The current training, sampling, solver, and kNN path diagnostics support 3D tensors.
Sample and trajectory PNGs render 3D axes when the saved tensors have at least three
coordinates. The saved `.npy` arrays contain the full coordinate data.

Grid ambiguity heatmaps are only produced for 2D runs. For 3D configs, read
`knn_ambiguity` and `bayes_gap` in `diagnostics/ambiguity_time.csv` instead.

Available 3D toy targets:

| Target | Config | What it stresses |
|---|---|---|
| Spherical shell | `configs/toy/gaussian_to_spherical_shell_linear_3d.yaml` | Thin shell geometry. |
| Swiss roll | `configs/toy/gaussian_to_swiss_roll_linear_3d.yaml` | Curved connected manifold. |
| Swiss roll with straightness regularization | `configs/toy/gaussian_to_swiss_roll_linear_3d_straight.yaml` | A/B test for learned-flow straightening. |
| 3D Gaussian mixture | `configs/toy/gaussian_to_gaussian_mixture_linear_3d.yaml` | Mode coverage and separated clusters. |
| 3D Gaussian mixture with direction-only flow | `configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only.yaml` | Label-conditioned straight-line direction stress test. |
| Multi Swiss roll | `configs/toy/gaussian_to_multi_swiss_roll_linear_3d.yaml` | Multimodal curved manifolds. |
| Torus | `configs/toy/gaussian_to_torus_linear_3d.yaml` | Hole/topology and tube geometry. |
| Multi torus | `configs/toy/gaussian_to_multi_torus_linear_3d.yaml` | Multiple disconnected topological components. |
| Helix mixture | `configs/toy/gaussian_to_helix_mixture_linear_3d.yaml` | Thin curved 1D structures in 3D. |
| Nested spherical shells | `configs/toy/gaussian_to_nested_spherical_shells_linear_3d.yaml` | Multiple radial shells and radial ambiguity. |

## Image Runs

MNIST is represented as flattened 784D image vectors, with `image_shape` metadata used
only for plotting. The first image-space config is:

| Target | Config | What it stresses |
|---|---|---|
| MNIST | `configs/mnist/mnist_linear_baseline.yaml` | High-dimensional image-space FM baseline with image-grid plots. |

## Training Outputs

### `plots/generated_samples_nfe*.png`

This plot shows target samples and generated samples from each configured solver.
For 3D runs, each panel is a 3D scatter plot over coordinates `x0`, `x1`, and `x2`.
For image runs such as MNIST, each panel is an image grid: target images first, followed
by generated images for each solver.
All solver panels start from the same saved `samples/source_reference.npy` source batch.
Toy configs default to a single plotting solver, `rk4`, so this plot is primarily a
sample-quality view rather than a solver-comparison view.

Read it as:

- Target panel: reference samples from the data distribution.
- Solver panels: samples generated by integrating the learned vector field from source noise.
- Good toy fit: generated cloud has the same coarse support, modes, and shape as target.
- Bad toy fit: generated cloud remains Gaussian-like, collapses to one region, leaks far away,
  or misses modes.
- Good MNIST fit: generated grids show digit-like strokes with diverse classes.
- Bad MNIST fit: images look like noise, gray mush, saturated pixels, or repeated templates.

Do not over-interpret fine details from a small smoke run. Use enough steps and enough
generated samples before judging shape quality.

By default, the plot shows all generated samples requested by `sampling.n_samples`.
Use `sampling.plot_max_points` or CLI `--plot-max-points` only when plotting becomes too
slow or visually too dense.

### `plots/trajectories_*_nfe*.png`

This plot shows generated ODE trajectories from source to final sample.
For 3D runs, each line is rendered in 3D over coordinates `x0`, `x1`, and `x2`.
All solver trajectory plots start from the same saved
`trajectories/source_reference_nfe*.npy` positions.

Read it as:

- Neutral gray lines: individual trajectories; color does not encode a metric.
- Smooth, coherent paths usually indicate a better behaved vector field.
- Sharp turns or tangled paths can indicate high curvature or solver stress.
- This is qualitative; pair it with field and solver diagnostics.

### `diagnostics/training_history.csv`

Columns:

| Column | Meaning |
|---|---|
| `step` | Training step. |
| `loss` | Total optimized training loss. |
| `flow_matching_loss` | Conditional flow matching velocity loss. |
| `straightness_loss` | Unweighted learned-flow straightness penalty, when enabled. |
| `straightness_weighted` | Weighted contribution added to total loss, when enabled. |
| `direction_loss` | Direction-only objective angle loss, when enabled. |
| `speed_loss` | Direction-only scalar speed prediction loss, when enabled. |
| `direction_weighted` | Weighted direction loss contribution. |
| `speed_weighted` | Weighted speed loss contribution. |
| `direction_speed_vector_mse` | Raw vector MSE diagnostic for `s(t,x,a)n(a)` vs target velocity. |
| `direction_alignment_cos2_mean` | Mean squared cosine alignment between target velocity and learned direction. |
| `direction_alignment_cos2_p10/p50/p90` | Distribution summaries for squared direction alignment. |
| `perpendicular_residual_mean` | Mean target velocity energy orthogonal to learned direction. |
| `speed_abs_mean`, `speed_abs_p90` | Signed-speed magnitude summaries. |
| `direction_pairwise_abs_mean` | Mean absolute pairwise direction similarity over the logged batch. |
| `loss_ema` | EMA-smoothed loss when early stopping is enabled. |

Loss is useful for optimization sanity, but it is not sufficient evidence of good generation.
When early stopping is enabled, `training.steps` is the maximum step count. Check
`metrics.json` for `trained_steps`, `requested_steps`, and the `early_stopping` block
to see whether the run stopped because the logged loss plateaued.
`metrics.json` also records the effective objective block.

### `plots/training_loss.png`

This is the plotted version of `diagnostics/training_history.csv`.

Read it as:

- `loss`: total optimized loss.
- `flow_matching_loss`: ordinary conditional flow matching loss.
- `straightness_weighted`: weighted straightness term added to the total loss, when enabled.
- `straightness_loss`: raw unweighted straightness estimate, when enabled.
- Downward trend: optimization is working.
- Flat trend: training may have converged, learning rate may be too small, or the model may be underpowered.
- Spikes: expected with stochastic minibatches, but large persistent spikes can indicate instability.

For judging sample quality, pair the loss curve with `plots/generated_samples_nfe*.png`.
A low loss does not guarantee the generated distribution has the right geometry or mode coverage.

For `direction_only_straight` runs, read direction and speed metrics separately:

- Low `direction_loss` / high `direction_alignment_cos2_mean`: directions align with sampled target velocities.
- High `perpendicular_residual_mean`: the coupling is asking one source label to point in
  incompatible directions. This is expected to be worst under independent coupling and
  should usually improve with more coherent pairings such as minibatch OT.
- High `speed_abs_mean` or `speed_abs_p90`: the speed network may be compensating for poor directions.
- High `direction_pairwise_abs_mean`: directions may be collapsing to similar lines.
- Compare `direction_weighted` and `speed_weighted` to decide whether `--direction-weight`
  needs to be increased for the next run.

For `model_generated` / teacher-coupled distillation runs, these metrics are measured
against the frozen teacher endpoint paired with each source, not an independently sampled
data endpoint. Low direction and speed losses mean the straight student is imitating the
teacher transport; sample plots are still needed to judge whether the teacher and student
match the real target distribution.

`metrics.json` also includes `sampling.line_containment` for label-conditioned runs. The
off-line values should be close to numerical zero because trajectories are constrained to
`a + R n(a)` by construction.

## Path-Law Ambiguity Diagnostics

Produced by `fm-lab-diagnostics` and by the `path` stage of `fm-lab-run-comparison`.
These diagnostics do not require a trained model. They analyze the chosen source,
coupling, and path.

### Concept

For sampled hidden path variables, the code computes:

```text
X_t = path(x0, x1, t)
U_t = target velocity at t
```

The key object is the conditional velocity law:

```text
L(U_t | X_t = x)
```

If many incompatible velocities are plausible near the same spacetime point, a
deterministic MSE-trained vector field has to average them. That is velocity ambiguity.

### `plots/ambiguity_heatmap_t*.png`

This is a 2D grid estimate of local velocity ambiguity at one time `t`.

Each cell contains:

```text
Tr Cov(U_t | X_t falls in this spatial bin)
```

Read it as:

- Bright/high value: local target velocities disagree; deterministic FM is harder there.
- Dark/low value: local target velocities mostly agree.
- Blank/NaN: not enough samples landed in that cell.
- Sparse or blocky heatmap: increase `--n-samples` or reduce `--bins`.

Important caveats:

- The plot currently shows grid indices, not physical coordinate tick labels.
- Color scale is per plot, so compare numeric CSV values before making cross-run claims.
- A high value means velocity ambiguity, not necessarily literal path crossing.
- For minibatch OT couplings, large diagnostic sample requests are estimated by concatenating
  multiple OT minibatches capped by `coupling.max_exact_size`.

The raw grid is saved as:

```text
diagnostics/grid_ambiguity_t*.npz
```

Arrays inside:

| Array | Meaning |
|---|---|
| `heatmap` | Per-cell trace covariance. |
| `counts` | Number of samples in each cell. |
| `x_edges` | Spatial bin edges for x-coordinate. |
| `y_edges` | Spatial bin edges for y-coordinate. |

### `diagnostics/ambiguity_time.csv`

Columns:

| Column | Meaning | How to read |
|---|---|---|
| `t` | Path time. | Compare profiles over time. |
| `grid_ambiguity` | Mass-weighted grid/bin estimate. | Higher means more local disagreement. |
| `grid_valid_bins` | Number of bins with enough samples. | Low values mean grid estimate is weak. |
| `knn_ambiguity` | kNN local covariance estimate. | More robust in sparse settings. |
| `bayes_gap` | kNN estimate of irreducible deterministic MSE. | Higher means harder deterministic regression. |

Use `knn_ambiguity` and `bayes_gap` as the primary scalar signals when the grid is sparse.

## Learned-Field Diagnostics

Produced by `fm-lab-field-diagnostics` and by the `field` stage of
`fm-lab-run-comparison`.

### `diagnostics/field_stats.csv`

Columns:

| Column | Meaning | How to read |
|---|---|---|
| `t` | Path time. | Profiles over time. |
| `acceleration_mean` | Mean material acceleration norm. | Curvature proxy; higher means less straight/stable flow. |
| `acceleration_max` | Max material acceleration norm. | Highlights worst sampled regions. |
| `acceleration_sq_mean` | Mean squared acceleration norm. | Penalizes spikes more strongly. |
| `jacobian_frobenius_mean` | Mean Frobenius norm of `J_x v`. | Overall local sensitivity. |
| `jacobian_spectral_mean` | Mean spectral norm of `J_x v`. | Lipschitz/stiffness proxy. |
| `divergence_mean` | Mean divergence. | Expansion/contraction proxy. |
| `divergence_std` | Divergence variability. | Higher means less uniform local volume change. |

Interpretation:

- High ambiguity with high acceleration suggests ambiguous path laws may be forcing curved fields.
- High Jacobian spectral values suggest the field may be sensitive to solver step size.
- These are sampled estimates, not formal guarantees.

## Solver Sensitivity Diagnostics

Produced by `fm-lab-solver-sensitivity` and by the `solver` stage of
`fm-lab-run-comparison`.

### `diagnostics/solver_sensitivity_nfe*.csv`

Each row compares two solvers at fixed NFE.

Columns:

| Column | Meaning | How to read |
|---|---|---|
| `nfe` | Neural function evaluations / solver steps. | Lower NFE is harder. |
| `schedule` | Time grid schedule. | `uniform`, `cosine`, etc. |
| `solver_i`, `solver_j` | Solver pair. | Pairwise comparison. |
| `mmd` | Kernel MMD between generated samples. | Higher means solvers generate different distributions. |
| `sliced_wasserstein` | Sliced Wasserstein distance. | Higher means larger distributional spread. |

### `plots/solver_sensitivity_*_nfe*.png`

This is a pairwise matrix over solvers.

Read it as:

- Dark/low values: solvers agree.
- Bright/high values: the learned field is solver-sensitive.
- If sensitivity drops as NFE increases, the issue is likely numerical discretization.
- If sensitivity remains high at large NFE, inspect field diagnostics and sample quality.

## Geometry Diagnostics

Produced by `fm-lab-geometry` and by the optional `geometry` comparison stage.

### `diagnostics/geometry_time.csv`

Columns:

| Column | Meaning | How to read |
|---|---|---|
| `t` | Path time. | Geometry mismatch over the path. |
| `radial_deviation_mean` | Mean distance to nearest configured radius. | Higher means more off-shell/off-manifold excursion. |
| `radial_deviation_max` | Worst sampled radial deviation. | Useful for spikes/outliers. |
| `radial_velocity_abs_mean` | Mean radial/normal velocity magnitude in 2D. | Higher means more normal movement. |
| `tangent_velocity_abs_mean` | Mean tangent velocity magnitude in 2D. | Higher means more along-manifold movement. |
| `normal_tangent_ratio` | Radial magnitude divided by tangent magnitude. | Higher means normal motion dominates. |

Interpretation:

- For circular/shell-like data, a geometry-aware path should reduce radial deviation.
- If radial deviation is high, the path is spending time away from the data geometry.
- If normal/tangent ratio is high, the target velocity is dominated by off-manifold movement.

## Completed-Run Comparison Outputs

Produced by `fm-lab-compare-runs`.

### `plots/generated_samples_nfe*.png`

This is a side-by-side sample-quality comparison across completed training runs. The
target panel uses `samples/target_reference.npy` from the first run; each run panel uses
`samples/<solver>_nfe*.npy` from that run. The command rejects runs whose `source` or
`data` config blocks differ.

Read it as:

- Same support/modes as target: better sample quality.
- Mode collapse or off-manifold points in one panel: that run is weaker for generation.
- Equal axes across panels: visual scale is comparable.

### `plots/training_loss_comparison.png`

This overlays one training-history column, `loss` by default, across completed runs.
Use `--loss-key flow_matching_loss` or another column when you want to compare a specific
objective component.

Read it as optimization behavior only. Lower loss does not automatically mean better
samples, especially when objectives differ or a regularization term changes the total loss.

### `summary.json`

Records the compared run directories, labels, selected solver/NFE, input `.npy` paths, and
plot paths. Use it to recover exactly which runs were compared.

## Matrix Comparison Outputs

Produced by `fm-lab-run-comparison`.

### `summary.csv`

One row per variant. It contains aggregate means/maxes from training, path diagnostics,
field diagnostics, and solver sensitivity.

For the first research experiment, compare:

| Question | Columns to inspect |
|---|---|
| Which path/coupling has lower ambiguity? | `path_knn_ambiguity_mean`, `path_bayes_gap_mean` |
| Which learned field is more curved? | `field_acceleration_mean_mean` |
| Which learned field is stiffer? | `field_jacobian_spectral_mean_mean` |
| Which model is more solver-sensitive? | `solver_sliced_wasserstein_max_max` |
| Did training optimize? | `final_loss` |

### `report.md`

A compact human-readable summary of key metrics. Treat it as a starting point, not a full
analysis. For paper-style claims, inspect the per-variant CSVs and plots.

## Practical Reading Order

For a trained toy experiment:

1. Open `plots/generated_samples_nfe*.png` to confirm the model roughly learned the target.
2. Read `diagnostics/ambiguity_time.csv` to see whether the path/coupling was ambiguous.
3. Read `diagnostics/field_stats.csv` to see whether the learned field became curved/stiff.
4. Read `diagnostics/solver_sensitivity_nfe*.csv` to see solver robustness.
5. Use `summary.csv` from comparison runs to compare variants.

The central hypothesis is supported only if lower ambiguity tends to align with easier
learning, lower curvature/stiffness, and lower solver sensitivity. If those quantities do
not move together, that negative result is scientifically useful.
