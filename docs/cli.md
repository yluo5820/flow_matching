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
| `--n-samples` | Override `sampling.n_samples`. |
| `--n-trajectories` | Override `sampling.n_trajectories`. |
| `--nfe` | Override `sampling.nfe`. |
| `--plot-max-points` | Override `sampling.plot_max_points`; defaults to all generated samples. |
| `--trajectory-target-max-points` | Override `sampling.trajectory_target_max_points`. |
| `--objective` | Override `objective.name`, e.g. `flow_matching`. |
| `--objective-loss` | Override `objective.loss`, currently `mse`. |
| `--straightness-weight` | Override `objective.straightness.weight`; `0` disables it. |
| `--straightness-sample-size` | Override `objective.straightness.sample_size`. |

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
