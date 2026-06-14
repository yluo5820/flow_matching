# CLI Lookup

This file is the maintained lookup table for `fm_lab` command-line tools. When a new
console script is added to `pyproject.toml`, update this file in the same commit.

Run commands from the project environment:

```bash
conda activate /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.conda/fm_lab
```

Without activation, prefix commands with `.conda/fm_lab/bin/`.

When `--output-dir` is omitted, commands use the config/default output directory but do
not overwrite an existing run. If `runs/example` already exists, the next default run is
created as `runs/example_1`, then `runs/example_2`, and so on. Passing `--output-dir`
uses that exact path.

## Maintenance Checklist

When adding or changing a CLI:

1. Update the `[project.scripts]` section in `pyproject.toml`.
2. Update the matching section in this file.
3. Update `docs/diagnostics.md` if outputs or metric semantics changed.
4. Add or update tests that cover the new command or artifact shape.
5. Run:

```bash
.conda/fm_lab/bin/python -m ruff check .
.conda/fm_lab/bin/python -m pytest
```

## Command Summary

| Command | Purpose | Primary input | Main outputs |
|---|---|---|---|
| `fm-lab-train` | Train one flow model and sample it. | Toy YAML config | checkpoint, metrics, samples, loss/sample/trajectory plots |
| `fm-lab-diagnostics` | Estimate path-law ambiguity before training. | Toy YAML config | ambiguity CSV/JSON, heatmaps, raw grids |
| `fm-lab-field-diagnostics` | Measure learned-field curvature/Jacobian stats. | Checkpoint | field stats CSV |
| `fm-lab-solver-sensitivity` | Compare generated samples across solvers/NFEs. | Checkpoint | pairwise distance CSVs, matrices |
| `fm-lab-geometry` | Measure path geometry mismatch. | Geometry-capable YAML config | geometry CSV/JSON, time profile |
| `fm-lab-compare-runs` | Compare completed runs with same source/target. | Training run directories | side-by-side samples, overlaid loss curves |
| `fm-lab-mnist-eval` | Evaluate a completed MNIST image-generation run. | MNIST run directory | pixel/classifier/nearest-neighbor metrics, nearest-neighbor plot |
| `fm-lab-run-comparison` | Run a controlled multi-variant experiment. | Comparison matrix YAML | summary CSV/JSON, Markdown report |

## `fm-lab-train`

Train one model from a config and write a normal run directory.

```bash
fm-lab-train \
  --config configs/toy/two_moons_baseline.yaml \
  --steps 10000 \
  --n-samples 8192 \
  --n-trajectories 128 \
  --nfe 64 \
  --output-dir runs/demo_two_moons \
  --device cpu
```

Key options:

| Option | Meaning |
|---|---|
| `--config` | YAML experiment config. Required. |
| `--output-dir` | Override `experiment.output_dir`. |
| `--dry-run` | Create run directory and metadata without training. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--steps` | Override `training.steps`. When early stopping is enabled, this is the maximum step count. |
| `--batch-size` | Override `training.batch_size`. |
| `--n-samples` | Override `sampling.n_samples`. |
| `--n-trajectories` | Override `sampling.n_trajectories`. |
| `--nfe` | Override `sampling.nfe`. |
| `--plot-max-points` | Override `sampling.plot_max_points`; defaults to all generated samples. |
| `--trajectory-target-max-points` | Override `sampling.trajectory_target_max_points`. |
| `--objective` | Override `objective.name`, e.g. `flow_matching`. |
| `--objective-loss` | Override `objective.loss`, currently `mse`. |
| `--straightness-weight` | Override `objective.straightness.weight`; `0` disables it. |
| `--straightness-sample-size` | Override `objective.straightness.sample_size`. |
| `--direction-weight` | Override `objective.direction_weight` for `direction_only_straight`. |
| `--speed-weight` | Override `objective.speed_weight` for `direction_only_straight`. |

Sampling artifacts use `sampling.seed` when set, otherwise `experiment.seed`. The same
source batch is reused for every solver in generated-sample plots, and the same trajectory
initial positions are reused for every solver trajectory plot.

Training objectives are configured in YAML under `objective`. The default is ordinary
conditional flow matching MSE:

```yaml
objective:
  name: flow_matching
  loss: mse
```

To test direction-only straight flow, use the dedicated YAML config:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only.yaml \
  --steps 50000 \
  --batch-size 256 \
  --n-samples 4096 \
  --n-trajectories 128 \
  --nfe 64 \
  --device auto
```

When splitting commands across lines, keep `\` as the final character on the line. A
trailing space after `\` makes the shell stop the command early.

The direction-only config uses a label-conditioned model. The objective does not restrict
the coupling, but the current toy config uses minibatch OT pairing:

```yaml
coupling:
  name: minibatch_ot
  max_exact_size: 1024

model:
  name: direction_speed_mlp

objective:
  name: direction_only_straight
  direction_weight: 2.0
  speed_weight: 1.0
```

This is label-conditioned/Lagrangian flow matching. The model carries the source label
through sampling and predicts `s(t,x,a) n(a)`, so learned-field diagnostics that assume an
Eulerian `v(x,t)` are intentionally unsupported for this objective.

Use `coupling.name: independent` to measure the cheap independent-coupling stress-test
ceiling. Use `coupling.name: minibatch_ot` when you want more coherent source-target
pairs for the same direction-only parameterization. Other coupling implementations can
also be used as long as their own `pair(x0, x1)` logic supports the training batch.
For `minibatch_ot`, `training.batch_size` or `--batch-size` must be less than or equal
to `coupling.max_exact_size`.

To use a trained baseline model as the coupling for direction-only distillation, point a
`model_generated` coupling at the baseline checkpoint:

```yaml
coupling:
  name: model_generated
  checkpoint_path: runs/gaussian_to_gaussian_mixture_linear_3d_base/checkpoint.pt
  solver: rk4
  nfe: 32
  schedule: uniform
```

This ignores the sampled target batch during training. Each source batch is integrated
through the frozen teacher model, and the generated endpoint becomes the target paired
with that source. The included config is:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only_distill.yaml \
  --steps 50000 \
  --n-samples 4096 \
  --n-trajectories 128 \
  --nfe 64 \
  --device auto
```

If `speed_loss` dominates the total loss, sweep `--direction-weight` without editing YAML:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only.yaml \
  --direction-weight 10.0 \
  --speed-weight 1.0
```

To train the recommended local MNIST image-space check:

```bash
fm-lab-train \
  --config configs/mnist/mnist_image_unet_ot.yaml \
  --steps 100000 \
  --n-samples 256 \
  --n-trajectories 16 \
  --nfe 64 \
  --device auto
```

The recommended config uses centered, dequantized pixels, a small time-conditioned
image U-Net, and minibatch OT with batch size 128. The older
`configs/mnist/mnist_linear_baseline.yaml` keeps the flattened MLP baseline for showing
why image inductive bias matters. MNIST configs use the standard IDX gzip files under
`data/mnist` and have `data.download: true`, so the first run attempts to download them.
MNIST sample and trajectory PNGs are image grids instead of coordinate scatter plots.

To test the learned-flow straightness regularizer:

```yaml
objective:
  name: flow_matching
  loss: mse
  straightness:
    weight: 1.0e-2
    sample_size: 256
```

This adds `weight * ||d_t v_theta(x_t,t) + J_x v_theta(x_t,t) v_theta(x_t,t)||^2`
on sampled training path points. It uses the learned field `v_theta` for the advective
velocity and requires second-order autograd, so it is more expensive than plain FM.

The same settings can be applied from the CLI:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_swiss_roll_linear_3d.yaml \
  --straightness-weight 1.0e-2 \
  --straightness-sample-size 256
```

To test the low-order learned acceleration interpolant, use the dedicated YAML config:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_learned_acceleration_3d.yaml \
  --steps 50000 \
  --n-samples 8192 \
  --n-trajectories 128 \
  --nfe 64 \
  --device auto
```

This keeps the velocity model Eulerian, but replaces the linear training path with:

```text
I(t, x0, x1) = x0 + t * (x1 - x0) + h(t) * A_psi(x0, x1)
```

The default basis is `h(t)=t(1-t)`. The endpoint-velocity-preserving ablation uses:

```yaml
path:
  name: learned_acceleration
  basis: endpoint_bump
```

The path is trained with a staged two-optimizer schedule:

```yaml
objective:
  name: flow_matching
  straightness:
    weight: 1.0e-2
    sample_size: 256
  interpolant_acceleration:
    weight: 1.0e-3

training:
  learned_acceleration:
    warmup_steps: 5000
    theta_steps: 1
    psi_steps: 1
    psi_lr: 1.0e-4
```

During warmup only the velocity model is updated. After warmup, the trainer alternates
velocity-model updates against FM plus Burgers residual and interpolant updates against
Burgers residual plus `interpolant_acceleration.weight * ||A_psi||^2`. There are no
extra CLI flags for this v1 path; edit YAML to sweep the basis, weights, warmup, or
`psi_lr`.

Compare it against the linear baseline with:

```bash
fm-lab-compare-runs \
  --run-a runs/gaussian_to_gaussian_mixture_linear_3d \
  --label-a linear \
  --run-b runs/gaussian_to_gaussian_mixture_learned_acceleration_3d \
  --label-b learned_acceleration \
  --nfe 64
```

Early stopping is configured in YAML under `training.early_stopping`:

```yaml
training:
  steps: 50000
  log_every: 500
  early_stopping:
    enabled: true
    warmup_steps: 10000
    patience_steps: 5000
    min_delta: 1.0e-3
    ema_alpha: 0.3
```

The stopper monitors the EMA-smoothed logged training loss after `warmup_steps`.
Training stops when that monitor has not improved by at least `min_delta` for
`patience_steps`.

Main outputs:

```text
run_dir/
  checkpoint.pt
  metrics.json
  samples/
  samples/source_reference.npy
  trajectories/
  trajectories/source_reference_nfe*.npy
  diagnostics/training_history.csv
  plots/training_loss.png
  plots/generated_samples_nfe*.png
  plots/trajectories_*_nfe*.png
```

## `fm-lab-mnist-eval`

Evaluate a completed MNIST image-generation run. This is the local gatekeeper for image
experiments: it checks pixel range/statistics, generated-sample diversity, nearest
training images, and optional classifier recognizability/diversity.

```bash
fm-lab-mnist-eval \
  --run-dir runs/mnist_image_unet_ot \
  --solver auto \
  --nfe 64 \
  --max-samples 256 \
  --reference-samples 2048 \
  --device auto
```

Main outputs:

```text
run_dir/
  diagnostics/mnist_eval_<solver>_nfe*.json
  diagnostics/mnist_eval_<solver>_nfe*.csv
  plots/mnist_nearest_neighbors_<solver>_nfe*.png
```

The first classifier run trains a small cached MNIST CNN under `artifacts/`; later runs
reuse it. Use `--skip-classifier` for quick range/diversity/nearest-neighbor checks.

## `fm-lab-diagnostics`

Estimate ambiguity from the chosen source/coupling/path without training a model.

```bash
fm-lab-diagnostics \
  --config configs/toy/two_moons_baseline.yaml \
  --n-samples 8192 \
  --bins 64 \
  --knn-k 32 \
  --output-dir runs/demo_two_moons_path_diag \
  --device cpu
```

Key options:

| Option | Meaning |
|---|---|
| `--config` | YAML experiment config. Required. |
| `--output-dir` | Diagnostics run directory. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--n-samples` | Samples per configured `t` value. |
| `--bins` | Grid cells per axis for 2D ambiguity heatmaps. |
| `--knn-k` | Neighbor count for kNN ambiguity/Bayes gap. |
| `--save-raw` | Save raw sampled `x_t` and target velocities. |

Main outputs:

```text
diagnostics/ambiguity_time.csv
diagnostics/ambiguity_time.json
diagnostics/grid_ambiguity_t*.npz
plots/ambiguity_time.png
plots/ambiguity_heatmap_t*.png
```

## `fm-lab-field-diagnostics`

Measure curvature/material acceleration and Jacobian statistics for a trained checkpoint.

```bash
fm-lab-field-diagnostics \
  --checkpoint runs/demo_two_moons/checkpoint.pt \
  --n-samples 512 \
  --output-dir runs/demo_two_moons_field_diag \
  --device cpu
```

Key options:

| Option | Meaning |
|---|---|
| `--checkpoint` | Training checkpoint. Required. |
| `--config` | Optional config override. Defaults to checkpoint config. |
| `--output-dir` | Diagnostics run directory. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--n-samples` | Samples per configured `t` value. |

Main output:

```text
diagnostics/field_stats.csv
```

## `fm-lab-solver-sensitivity`

Sample one trained model with several solvers and compare generated distributions.

```bash
fm-lab-solver-sensitivity \
  --checkpoint runs/demo_two_moons/checkpoint.pt \
  --n-samples 2048 \
  --max-metric-samples 2048 \
  --output-dir runs/demo_two_moons_solver_diag \
  --device cpu
```

Most toy configs now default to one plotting solver, `rk4`. Use this command only when
you explicitly want pairwise solver comparisons; otherwise training plots focus on sample
quality and trajectories for the configured solver.

Key options:

| Option | Meaning |
|---|---|
| `--checkpoint` | Training checkpoint. Required. |
| `--config` | Optional config override. Defaults to checkpoint config. |
| `--output-dir` | Diagnostics run directory. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--n-samples` | Generated samples per solver. |
| `--max-metric-samples` | Maximum samples used by MMD/SW metrics. |
| `--schedule` | Override solver schedule, e.g. `uniform`, `cosine`. |

Main outputs:

```text
samples/*_nfe*.npy
diagnostics/solver_sensitivity.json
diagnostics/solver_sensitivity_nfe*.csv
plots/solver_sensitivity_*_nfe*.png
```

## `fm-lab-geometry`

Measure path geometry mismatch, such as radial deviation and radial/tangent velocity.

```bash
fm-lab-geometry \
  --config configs/toy/annulus_linear.yaml \
  --n-samples 4096 \
  --output-dir runs/annulus_geometry \
  --device cpu
```

Key options:

| Option | Meaning |
|---|---|
| `--config` | YAML experiment config. Required. |
| `--output-dir` | Diagnostics run directory. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--n-samples` | Samples per configured `t` value. |
| `--save-raw` | Save sampled `x_t`, velocity, and radial deviation arrays. |

Main outputs:

```text
diagnostics/geometry_time.csv
diagnostics/geometry_time.json
plots/geometry_time.png
```

## `fm-lab-compare-runs`

Compare completed training run directories. This does not retrain; it reads saved
`samples/<solver>_nfe*.npy` arrays and `diagnostics/training_history.csv` from each run.
All runs must have the same `source` and `data` config blocks.

```bash
fm-lab-compare-runs \
  --runs runs/swiss_roll_plain runs/swiss_roll_straight \
  --labels plain straight \
  --nfe 64 \
  --solver rk4 \
  --output-dir runs/comparisons/swiss_roll_plain_vs_straight
```

Key options:

| Option | Meaning |
|---|---|
| `--runs` | Completed training run directories. Requires at least two. |
| `--labels` | Optional labels, one per run. Defaults to each run's `experiment.name`. |
| `--output-dir` | Comparison output directory. |
| `--nfe` | NFE suffix to compare. Default: `64`. |
| `--solver` | Solver sample file prefix. Default: `rk4`; use `auto` if each run has one matching sample file. |
| `--loss-key` | Training-history column to overlay. Default: `loss`. |
| `--max-points` | Maximum points shown in the generated-sample comparison plot. |

Main outputs:

```text
comparison_dir/
  config.yaml
  metadata.json
  summary.json
  plots/generated_samples_nfe64.png
  plots/training_loss_comparison.png
```

## `fm-lab-run-comparison`

Run a multi-variant matrix. This is the preferred command for the first research
experiment in the brief: two moons, independent coupling vs minibatch OT coupling.

```bash
fm-lab-run-comparison \
  --matrix configs/experiments/two_moons_indep_vs_ot.yaml
```

Fast smoke version:

```bash
fm-lab-run-comparison \
  --matrix configs/experiments/two_moons_indep_vs_ot.yaml \
  --steps 1000 \
  --n-samples 512 \
  --diagnostic-samples 1024 \
  --field-samples 128 \
  --nfe 16 \
  --device cpu
```

Key options:

| Option | Meaning |
|---|---|
| `--matrix` | Comparison matrix YAML. Required. |
| `--output-dir` | Override matrix `experiment.output_dir`. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--stages` | Comma-separated stages. Default: `train,path,field`; add `solver` when needed. |
| `--steps` | Override `training.steps` for every variant. |
| `--n-samples` | Override sampling and solver metric sample counts. |
| `--diagnostic-samples` | Override path-law diagnostic samples. |
| `--field-samples` | Override learned-field diagnostic samples. |
| `--nfe` | Override sampling NFE and solver-sensitivity NFE list. |

Main outputs:

```text
comparison_dir/
  matrix.yaml
  summary.csv
  summary.json
  report.md
  variants/
    independent_linear/
    minibatch_ot_linear/
```
