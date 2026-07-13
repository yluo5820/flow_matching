# Flow Matching Research Playground

`fm_lab` is a small PyTorch research playground for testing whether path/coupling ambiguity, latent geometry mismatch, learned-field curvature, and solver sensitivity are linked in flow matching.

The project follows the implementation brief in `flow_matching_research_playground_brief.md`. The first target is a fully inspectable 2D conditional flow matching lab before moving to image-scale or latent experiments.

## Development Stages

1. **Project setup**: package metadata, config loading, deterministic seeding, run directories, and reproducibility metadata.
2. **Minimal 2D FM baseline**: toy distributions, Gaussian source, independent coupling, linear path, MLP velocity model, FM training loop, Euler/Heun/RK4 sampling, and plots.
3. **Paths and couplings**: minibatch OT coupling, rectified/reflow placeholder, spherical path, and tangent-normal synthetic path.
4. **Diagnostics**: grid/kNN ambiguity, Bayes regression gap, velocity clouds, curvature, Jacobian/stiffness, and solver sensitivity.
5. **Solver suite**: midpoint, Dormand-Prince or `torchdiffeq` wrapper, schedules, and NFE sweeps.
6. **Controlled experiments**: independent vs OT on two moons/checkerboard, geometry-aware paths on shell data, and high-dimensional non-crossing sanity checks.

## Quick Start

Create the project-local conda environment with the package dependencies:

```bash
conda create -p .conda/fm_lab python=3.11 pip -y
.conda/fm_lab/bin/python -m pip install -e ".[dev]"
```

Create a dry-run experiment directory from the default toy config:

```bash
.conda/fm_lab/bin/fm-lab-train --config configs/toy/two_moons_baseline.yaml --dry-run
```

Run a very small smoke training job:

```bash
.conda/fm_lab/bin/fm-lab-train --config configs/toy/two_moons_baseline.yaml --steps 100 --n-samples 256 --n-trajectories 32 --nfe 16 --output-dir runs/smoke --device cpu
```

The full default config uses a longer toy training budget and saves checkpoint, metrics, generated samples, trajectories, and plots.

### Class-conditional generation

Class conditioning uses a learned class embedding and a learned null token for
classifier-free guidance. During training, `dropout_probability` replaces class labels
with the null token. During sampling, requested classes repeat to fill
`sampling.n_samples`, and `generated_labels.npy` is saved with the generated samples.

```yaml
conditioning:
  enabled: true
  num_classes: 10
  embedding_dim: 128
  dropout_probability: 0.15

sampling:
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  classifier_free_guidance:
    scale: 2.0
```

See `configs/mnist/mnist_class_conditional_cfg.yaml` for a complete image U-Net
example. Conditional training requires a target implementing `sample_with_labels()`
and a coupling that preserves target indices. Independent and minibatch-OT couplings
support this interface.

## Documentation

- [CLI lookup](docs/cli.md): maintained reference for every `fm-lab-*` command.
- [Reading outputs and diagnostics](docs/diagnostics.md): how to interpret plots, CSVs, and summary metrics.

Run path-law ambiguity diagnostics without training:

```bash
.conda/fm_lab/bin/fm-lab-diagnostics --config configs/toy/two_moons_baseline.yaml --n-samples 1024 --bins 32 --output-dir runs/moons_ambiguity
```

Run learned-field diagnostics from a checkpoint:

```bash
.conda/fm_lab/bin/fm-lab-field-diagnostics --checkpoint runs/smoke/checkpoint.pt --n-samples 128 --device cpu
.conda/fm_lab/bin/fm-lab-solver-sensitivity --checkpoint runs/smoke/checkpoint.pt --n-samples 256 --device cpu
```

Supported fixed-step time schedules are `uniform`, `quadratic`, `reverse_quadratic`, and `cosine`. Add `dopri5` or `rk45` to `solvers.names` to use the SciPy Dormand-Prince/RK45 wrapper for small CPU-oriented checks.

Run geometry diagnostics for shell or annulus paths:

```bash
.conda/fm_lab/bin/fm-lab-geometry --config configs/toy/annulus_linear.yaml --n-samples 1024 --output-dir runs/annulus_geometry
```

Run the first controlled comparison from the research brief:

```bash
.conda/fm_lab/bin/fm-lab-run-comparison --matrix configs/experiments/two_moons_indep_vs_ot.yaml
```

Evaluate a completed local MNIST image run:

```bash
.conda/fm_lab/bin/fm-lab-mnist-eval --run-dir runs/mnist_image_unet_ot --solver auto --nfe 64 --device auto
```

Useful toy configs:

```text
configs/toy/two_moons_baseline.yaml       # independent coupling + linear path
configs/toy/two_moons_ot.yaml             # minibatch OT coupling + linear path
configs/toy/circles_tangent_normal.yaml   # polar tangent-normal path
configs/geometry_explorer/datasets/mnist/original/models/image_unet_ot.yaml  # MNIST original image U-Net + minibatch OT
configs/mnist/mnist_linear_baseline.yaml   # deliberately naive flattened MNIST baseline
```

## Repository Layout

```text
configs/
  toy/
fm_lab/
  data/
  sources/
  couplings/
  paths/
  models/
  training/
  solvers/
  diagnostics/
  experiments/
  plotting/
  utils/
scripts/
notebooks/
tests/
```

Every experiment should save its config, metadata, raw arrays, plots, checkpoints, and metrics under `runs/`.
