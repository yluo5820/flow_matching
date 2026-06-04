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

Create an environment with the project dependencies, then install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Create a dry-run experiment directory from the default toy config:

```bash
fm-lab-train --config configs/toy/two_moons_baseline.yaml --dry-run
```

Training and diagnostics are implemented incrementally by stage.

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
