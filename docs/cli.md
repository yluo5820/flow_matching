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
| `fm-lab-explorer` | Build and launch the unified geometry explorer. | Dataset variant/projection/run configs | SQLite registry, dataset variants, projection views, trajectory views |
| `fm-lab-diagnostics` | Estimate path-law ambiguity before training. | Toy YAML config | ambiguity CSV/JSON, heatmaps, raw grids |
| `fm-lab-field-diagnostics` | Measure learned-field curvature/Jacobian stats. | Checkpoint | field stats CSV |
| `fm-lab-solver-sensitivity` | Compare generated samples across solvers/NFEs. | Checkpoint | pairwise distance CSVs, matrices |
| `fm-lab-geometry` | Measure path geometry mismatch. | Geometry-capable YAML config | geometry CSV/JSON, time profile |
| `fm-lab-compare-runs` | Compare completed runs with same source/target. | Training run directories | side-by-side samples, overlaid loss curves |
| `fm-lab-mnist-eval` | Evaluate a completed MNIST image-generation run. | MNIST run directory | pixel/classifier/nearest-neighbor metrics, nearest-neighbor plot |
| `fm-lab-run-comparison` | Run a controlled multi-variant experiment. | Comparison matrix YAML | summary CSV/JSON, Markdown report |
| `fm-lab-sample-checkpoint` | Resample a trained checkpoint without retraining. | Completed run/checkpoint | samples, trajectories, sample/trajectory plots |
| `fm-lab-sampling-timesteps` | Register sampler timesteps as Geometry Explorer classes. | Completed run/checkpoint | timestep-labeled dataset variant and optional view |
| `fm-lab-imbdiff-eval` | Evaluate class-imbalanced CIFAR generation. | Generated/real Inception caches or generated arrays | FID, KID, Recall, IS, classwise and frequency-group reports |
| `fm-lab-imbdiff-cm-probe` | Probe learned CM capacity allocation. | Official CM run directories and checkpoints | Paired functional, Fourier, and gradient-routing reports |
| `fm-lab-imbdiff-cm-dropout-probe` | Separate expert and dropout contributions to the CM branch distance. | One official CM checkpoint | Independent/paired/disabled-dropout distances and gradient comparisons |
| `fm-lab-fashion-mnist-lt-eval` | Evaluate balanced conditional Fashion-MNIST generation. | Generated/real classifier caches or generated arrays | Fashion-FID, KID, Recall, IS, classwise and head/middle/tail reports |
| `fm-lab-fashion-geometry-frequency` | Run the gated Fashion-MNIST geometry-by-frequency bridge. | Frozen Stage-0 or Stage-1 YAML | geometry gate, all-class cyclic configs, budget gate, evaluations, response analysis |
| `fm-lab-synthetic-long-tail` | Run the gated synthetic long-tail geometry experiment. | Frozen experiment YAML | pools, gates, run ledger, evaluations, effect summary, living report |
| `fm-lab-long-tail-geometry-stage0` | Validate the counterfactual long-tail observation pipeline before measurement runs. | Ordinary-FM checkpoint and Stage-0 YAML config | fail-closed report, paired probe manifests, validated gradient rows |
| `fm-lab-long-tail-geometry-observation0` | Establish the reproducible-gradient noise ceiling before any mapping experiment. | Locked Observation-0 preregistration and three raw-checkpoint runs | immutable manifests, checkpoint sketches, reliability table, noise-ceiling decision |

## `fm-lab-fashion-geometry-frequency`

Inspect or run the Fashion-MNIST geometry-selection gate. Stage 0 reserves disjoint
500-image probes, leaves 5,000 training candidates per class, and does not train a
generative model.

```bash
fm-lab-fashion-geometry-frequency \
  --config configs/fashion_mnist_geometry_frequency/stage0.yaml \
  stage0 --device auto --dry-run
```

Remove `--dry-run` to extract the raw-PCA and DINOv2-PCA features and apply the frozen
split-half/subsample stability rule. A failed selection gate is a valid scientific
outcome and does not enable downstream training.

The failed trio gate enables only the separately frozen all-class fallback. Inspect its
eleven-condition primary design—one balanced reference and ten equal-exposure cyclic
support rotations—without training:

```bash
fm-lab-fashion-geometry-frequency \
  --config configs/fashion_mnist_geometry_frequency/stage1_all_classes.yaml \
  stage1-plan --dry-run
```

Calibrate the balanced model sequentially at 2,000, 5,000, and, only if needed, 10,000
steps. A later budget resumes exactly from the preceding checkpoint. The frequency
commands fail closed until `stage1-calibration-status` selects a budget.

```bash
fm-lab-fashion-geometry-frequency \
  --config configs/fashion_mnist_geometry_frequency/stage1_all_classes.yaml \
  stage1-calibrate --training-steps 2000 --device auto
```

After the budget passes, `stage1-run --condition class_balanced_offset_00` runs one
rotation, `stage1-run-all` runs all missing rotations, and `stage1-analyze` writes the
classwise support responses and all ten preregistered geometry correlations. The full
rotation command is intended for a user terminal when its projected runtime exceeds 30
minutes.

## `fm-lab-synthetic-long-tail`

Execute the preregistered synthetic study one gated stage at a time. Use `plan` or
`matrix --dry-run` to inspect the frozen 36-run matrix without starting a process.

```bash
fm-lab-synthetic-long-tail \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  matrix --dry-run
```

The stage commands are `plan`, `build-pools`, `calibrate-renderer`, `train-oracle`,
`pilot`, `smoke`, `matrix`, `evaluate`, `aggregate`, and `report`. Downstream training
fails closed when renderer, oracle, metric-control, or pilot gates are absent or failed.

## `fm-lab-long-tail-geometry-stage0`

Run this gate before scheduling any counterfactual frequency-mapping experiment. It
rejects CM, objective modifiers, capacity adapters, early stopping, unpaired probes,
inexact checkpoint replay, inaccurate gradient sketches, and failed synthetic controls.

```bash
fm-lab-long-tail-geometry-stage0 \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_geometry_stage0.yaml \
  --checkpoint runs/fashion_mnist_lt_geometry_stage0/checkpoints/step_000500.pt \
  --output-dir runs/fashion_mnist_lt_geometry_stage0 \
  --device auto
```

The command exits nonzero on the first failed gate. Its report is still written, and
gradient-row artifacts are never produced when a pre-gradient gate fails.

## `fm-lab-long-tail-geometry-observation0`

Prepare the locked offset-0 pilot and its three ordinary-FM seed configs:

```bash
fm-lab-long-tail-geometry-observation0 prepare \
  --preregistration configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0
```

Train each generated config into its printed exact run directory. Then collect one
registered seed at a time and analyze only after all three registry rows are measured:

```bash
fm-lab-long-tail-geometry-observation0 collect \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0 \
  --run-dir runs/long_tail_geometry/fashion_mnist/observation0/mapping_0/seed_0 \
  --device auto
fm-lab-long-tail-geometry-observation0 analyze \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0
```

The collector is resumable only when checkpoint, manifest, layer, and preregistration
digests still match. If the primary decision is `escalate_probe_rows`, repeat `collect`
for all seeds and `analyze` with `--escalated`. This is the single locked increase from
16 to 32 microbatches per cell; it is not itself evidence for Outcome D. This CLI has
no Stage-1 command and cannot schedule a nonzero frequency mapping.

## `fm-lab-fashion-mnist-lt-eval`

The fast long-tail benchmark trains on exponentially subsampled Fashion-MNIST
but evaluates against the untouched official Fashion-MNIST test split. Its
canonical protocol requests 1,000 generated samples per class, so all ten
conditional distributions contribute equally to the global metrics.

Refresh the editable install, then train and sample the four controlled
continuous IR100 variants and the full balanced-data baseline. All five predict
the clean target while optimizing velocity loss; three IR100 variants add CBDM,
OC, or CM respectively. CM is independent of OC and compares its capacity-on
and capacity-off branches in clean-target space. Its default low-rank branch
covers the decoder, while other U-Net sections remain selectable. They use
JiT-style logit-normal time sampling `(-0.8, 0.8)` and apply the same `0.05`
denominator floor to prediction and supervision. Evaluation retains the
controlled Euler/NFE-64 generation protocol. These short runs intentionally
disable EMA and use the selected raw checkpoint with CFG scale 1.0.

```bash
.conda/fm_lab/bin/python -m pip install -e .

.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss.yaml \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss \
  --device auto
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cbdm.yaml \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_cbdm \
  --device auto
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_oc.yaml \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_oc \
  --device auto
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_cm \
  --device auto
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_balanced_x_vloss.yaml \
  --output-dir runs/fashion_mnist_balanced/x_vloss \
  --device auto
```

An explicit output directory is never auto-suffixed: each directory above must
be empty (or absent) before a fresh controlled run. Then extract frozen
Fashion-MNIST classifier features and report FID, KID, Inception Score,
generative recall, per-class scores, and frequency-group scores for each run:

```bash
.conda/fm_lab/bin/fm-lab-fashion-mnist-lt-eval \
  --generated-samples runs/fashion_mnist_lt_ir100/x_vloss/samples/euler_nfe64.npy \
  --generated-labels runs/fashion_mnist_lt_ir100/x_vloss/samples/generated_labels.npy \
  --generative-checkpoint runs/fashion_mnist_lt_ir100/x_vloss/checkpoint.pt \
  --generation-method x_vloss --sampler euler --nfe 64 \
  --guidance-scale 1.0 --generative-weights raw --generation-seed 0 \
  --data-root data/fashion_mnist --download \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss/evaluation
.conda/fm_lab/bin/fm-lab-fashion-mnist-lt-eval \
  --generated-samples runs/fashion_mnist_lt_ir100/x_vloss_cbdm/samples/euler_nfe64.npy \
  --generated-labels runs/fashion_mnist_lt_ir100/x_vloss_cbdm/samples/generated_labels.npy \
  --generative-checkpoint runs/fashion_mnist_lt_ir100/x_vloss_cbdm/checkpoint.pt \
  --generation-method x_vloss_cbdm --sampler euler --nfe 64 \
  --guidance-scale 1.0 --generative-weights raw --generation-seed 0 \
  --data-root data/fashion_mnist --download \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_cbdm/evaluation
.conda/fm_lab/bin/fm-lab-fashion-mnist-lt-eval \
  --generated-samples runs/fashion_mnist_lt_ir100/x_vloss_oc/samples/euler_nfe64.npy \
  --generated-labels runs/fashion_mnist_lt_ir100/x_vloss_oc/samples/generated_labels.npy \
  --generative-checkpoint runs/fashion_mnist_lt_ir100/x_vloss_oc/checkpoint.pt \
  --generation-method x_vloss_oc --sampler euler --nfe 64 \
  --guidance-scale 1.0 --generative-weights raw --generation-seed 0 \
  --data-root data/fashion_mnist --download \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_oc/evaluation
.conda/fm_lab/bin/fm-lab-fashion-mnist-lt-eval \
  --generated-samples runs/fashion_mnist_lt_ir100/x_vloss_cm/samples/euler_nfe64.npy \
  --generated-labels runs/fashion_mnist_lt_ir100/x_vloss_cm/samples/generated_labels.npy \
  --generative-checkpoint runs/fashion_mnist_lt_ir100/x_vloss_cm/checkpoint.pt \
  --generation-method x_vloss_cm --sampler euler --nfe 64 \
  --guidance-scale 1.0 --generative-weights raw --generation-seed 0 \
  --data-root data/fashion_mnist --download \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss_cm/evaluation
.conda/fm_lab/bin/fm-lab-fashion-mnist-lt-eval \
  --generated-samples runs/fashion_mnist_balanced/x_vloss/samples/euler_nfe64.npy \
  --generated-labels runs/fashion_mnist_balanced/x_vloss/samples/generated_labels.npy \
  --generative-checkpoint runs/fashion_mnist_balanced/x_vloss/checkpoint.pt \
  --generation-method balanced_x_vloss --sampler euler --nfe 64 \
  --guidance-scale 1.0 --generative-weights raw --generation-seed 0 \
  --imbalance-factor 0.01 \
  --data-root data/fashion_mnist --download \
  --output-dir runs/fashion_mnist_balanced/x_vloss/evaluation
```

The report includes Many/Medium/Few FID. The balanced baseline intentionally
keeps `--imbalance-factor 0.01`, so group membership follows the IR100 training
frequencies rather than treating the balanced training counts as frequency
groups.

If the evaluator checkpoint is absent, the command trains it once on the
balanced training split and validates its held-out accuracy before use. Supply
`--generated-cache` and `--real-cache` together to recompute reports without
feature extraction. Cache provenance must match exactly.
Generated-array evaluation hashes the arrays and generative checkpoint, and
records the method, sampler, NFE, guidance scale, and seed so modified files or
incompatible sampling protocols cannot silently reuse a cache.

## `fm-lab-imbdiff-eval`

Evaluate generated CIFAR samples using the reference TensorFlow-FID Inception
features. FID and KID are reference-compatible; Recall, Inception Score,
classwise FID, and rank-third Many/Medium/Few FID are documented extensions.

```bash
fm-lab-imbdiff-eval \
  --generated-samples runs/imbdiff/cifar10/samples/ddim.npy \
  --generated-labels runs/imbdiff/cifar10/samples/generated_labels.npy \
  --dataset cifar10 --data-root data/cifar10 \
  --class-counts runs/imbdiff/cifar10/class_counts.json \
  --output-dir runs/imbdiff/cifar10/evaluation
```

Use `--generated-cache` with `--real-cache` to recompute reports without
rerunning Inception feature extraction. The exact TensorFlow-FID weight file is
required; the command never substitutes torchvision ImageNet weights.

For a fast multi-method screen, `--skip-recall --skip-classwise-fid` omits the
two expensive diagnostics while retaining overall FID/KID, Inception Score,
and pooled Many/Medium/Few FID. The default remains the complete report.

## `fm-lab-imbdiff-cm-probe`

Probe existing official ImbDiff-CM checkpoints without retraining. Held-out
CIFAR rows, Gaussian noise, discrete timesteps, and OC transfer draws are fixed
in a reusable manifest. The command compares capacity-on and capacity-off
predictions and separately measures gradients from the denoising, consistency,
and diversity terms into general and LoRA parameters.

```bash
fm-lab-imbdiff-cm-probe \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/oc_capacity_only \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/released_cm \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/pure_cm \
  --checkpoint-steps 20000,40000,60000 \
  --timesteps 50,250,500,750,950 \
  --samples-per-class 1 \
  --weights ema \
  --mixed-precision auto \
  --channels-last on \
  --device cuda \
  --output-dir /root/autodl-tmp/runs/imbdiff_matrix60k/cm_mechanism_probe
```

Use `--functional-only --checkpoint-steps 60000 --timesteps 500` for a quick
end-to-end smoke. The full probe writes `manifest.json`, `summary.json`,
`functional_rows.csv`, `gradient_summary.csv`, per-checkpoint JSON reports,
and a compact `report.md`. Existing output manifests are reused and checked by
SHA-256 so checkpoint comparisons remain paired.

## `fm-lab-imbdiff-cm-dropout-probe`

Run a fixed-checkpoint training-mode diagnostic of the two CM forward passes.
The released condition uses independent dropout masks; the controls replay one
mask across both passes, disable dropout, or disable expert capacity in both
independent-mask passes. This command does not retrain or modify the checkpoint.

```bash
fm-lab-imbdiff-cm-dropout-probe \
  --checkpoint /root/autodl-tmp/runs/imbdiff_matrix60k/released_cm/checkpoint.pt \
  --timesteps 100,500,900 \
  --classes auto \
  --classes-per-group 2 \
  --samples-per-class 1 \
  --dropout-repeats 10 \
  --weights ema \
  --channels-last on \
  --device cuda \
  --output-dir /root/autodl-tmp/runs/imbdiff_matrix60k/cm_dropout_probe
```

`--classes auto` chooses a deterministic frequency-stratified subset from the
Many, Medium, and Few thirds. Pass `--classes all` or explicit comma-separated
class IDs for a broader diagnostic. `--skip-gradients` omits the one-repeat
branch-distance gradient comparison for a faster functional smoke.

Outputs are `manifest.json`, `summary.json`, `functional_rows.csv`, optional
`gradient_summary.csv`, and `report.md`. Ratios of dropout-only, paired, and
disabled distances to the released independent-mask distance are descriptive:
squared distances contain cross terms and are not an additive causal variance
decomposition.

## `fm-lab-sampling-timesteps`

Generate a Geometry Explorer dataset whose class labels identify sampling timesteps:

```bash
fm-lab-sampling-timesteps \
  --run-dir runs/cifar10_original_unet_diffusion_x_bs32 \
  --workspace outputs/geometry_explorer \
  --num-classes 16 \
  --total-rows 2048 \
  --build-view
```

Use `--checkpoint` to override the run checkpoint, `--solver`, `--nfe`, and
`--schedule` to control integration, and `--variant-id` to choose the registered
dataset id. `--overwrite` replaces an existing variant with the same id.

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
| `--dataset-variant` | Registered dataset variant id, e.g. `mnist/tail_digit1`. |
| `--workspace` | Geometry explorer workspace used to resolve `--dataset-variant`. |
| `--steps` | Override `training.steps`. When early stopping is enabled, this is the maximum step count. |
| `--batch-size` | Override `training.batch_size`. |
| `--n-samples` | Override `sampling.n_samples`. |
| `--n-trajectories` | Override `sampling.n_trajectories`. |
| `--nfe` | Override `sampling.nfe`. |
| `--plot-max-points` | Override `sampling.plot_max_points`; defaults to all generated samples. |
| `--sample-batch-size` | Integrate final generated samples in chunks of this size. |
| `--trajectory-target-max-points` | Override `sampling.trajectory_target_max_points`. |
| `--objective` | Override `objective.name`, e.g. `flow_matching`. |
| `--objective-loss` | Override `objective.loss`, currently `mse`. |
| `--model-output` | Override `objective.model_output`: `source`, `target`, or `velocity`. |
| `--loss-space` | Override `objective.loss_space`: `source`, `target`, or `velocity`. |
| `--prediction-min-denom` | Override `objective.min_denom` for prediction-space conversions. |
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
  model_output: velocity
  loss_space: velocity
  min_denom: 0.001
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
  --config configs/geometry_explorer/datasets/mnist/original/models/image_unet_ot.yaml \
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
The MNIST configs also enable `sampling.trajectory_umap`, which writes
`plots/trajectory_umap3d_<solver>_nfe*.png`,
`plots/trajectory_umap3d_<solver>_nfe*.html`, and
`trajectories/<solver>_nfe*_umap3d.npz`.

To train on a registered MNIST variant from the unified geometry explorer:

```bash
fm-lab-train \
  --config configs/geometry_explorer/datasets/mnist/original/models/image_unet_ot.yaml \
  --dataset-variant mnist/tail_digit1 \
  --workspace outputs/geometry_explorer \
  --device auto
```

Experimental MNIST configs also include `configs/mnist/mnist_direction_only_image_unet_ot.yaml`
and `configs/mnist/mnist_learned_acceleration_kernel_vstar_factorized_polynomial_image_unet_ot.yaml`.
The learned-acceleration MNIST config sets `path.network: image_unet`, so the learned
interpolant coefficient map reads `[x0, x1 - x0]` as image channels instead of using the
default flattened MLP pair net.

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

To test the kernel-v* learned interpolant objective, use the V2 config:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_learned_acceleration_kernel_vstar_3d.yaml \
  --steps 50000 \
  --n-samples 8192 \
  --n-trajectories 128 \
  --nfe 64 \
  --device auto
```

This keeps the same K=1 learned-acceleration path family, but changes the path update.
Instead of using only `v_theta` as the advective velocity, it estimates the induced
optimal field

```text
v*_phi(t, x) = E[u_t | x_t = x]
```

by Gaussian-kernel averaging over path states from an auxiliary minibatch pool. The
YAML-only controls are:

```yaml
objective:
  learned_interpolant:
    mode: kernel_vstar
    estimator_size: 256
    query_size: 64
    bandwidth: median
    bandwidth_scale: 1.0
    min_bandwidth: 1.0e-3
```

Use this config to decide whether the K=1 failure was a proxy-optimization issue before
increasing the polynomial/interpolant capacity. It is intended for low-dimensional toy
runs, not MNIST or CIFAR-style image experiments.

For the full factorized polynomial fallback, use:

```bash
fm-lab-train \
  --config configs/toy/gaussian_to_gaussian_mixture_learned_acceleration_kernel_vstar_factorized_polynomial_3d.yaml \
  --steps 50000 \
  --n-samples 8192 \
  --n-trajectories 128 \
  --nfe 64 \
  --device auto
```

This uses the endpoint-constrained correction:

```text
I(t) = x0 + t(x1 - x0) + t(1 - t) [B0(x0,x1) + t B1(x0,x1) + t^2 B2(x0,x1)]
```

The correction vanishes at both endpoints, but unlike `endpoint_bump`, endpoint
velocities are not constrained to match the linear path. This is intentionally more
flexible for the last low-order learned-interpolant test.

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
`patience_steps`. When early stopping is enabled, the trainer restores the best logged
state before writing `checkpoint.pt` and generating samples. In `metrics.json`,
`trained_steps` is the last step actually run, while `checkpoint_step` is the step used
for the saved checkpoint and plots.

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

## `fm-lab-explorer`

Build and launch the unified registry-backed geometry explorer.

```bash
fm-lab-explorer build-dataset \
  --config configs/geometry_explorer/datasets/mnist/original/dataset.yaml

fm-lab-explorer build-dataset \
  --config configs/geometry_explorer/datasets/mnist/tail_digit1/dataset.yaml

fm-lab-explorer build-dataset \
  --config configs/geometry_explorer/datasets/mnist/tail_digit8/dataset.yaml

fm-lab-explorer build-dataset \
  --config configs/geometry_explorer/datasets/mnist/long_tail_monotone/dataset.yaml

fm-lab-explorer build-view \
  --dataset mnist/tail_digit1 \
  --config configs/geometry_explorer/views/raw_pixels.yaml

fm-lab-explorer build-trajectory \
  --run-dir runs/mnist_image_unet_ot \
  --nfe 64

fm-lab-explorer build-model-diagnostics \
  --dataset mnist/tail_digit1 \
  --run-dir runs/mnist_tail_digit1_unet_ot \
  --estimator fm_jacobian \
  --estimator fm_flipd \
  --t-values 0.8 0.9 \
  --max-samples 512 \
  --device auto

fm-lab-explorer summarize --include-classes

fm-lab-explorer launch
```

Key options:

| Option | Meaning |
|---|---|
| `--workspace` | Geometry explorer workspace. Defaults to `outputs/geometry_explorer`. |
| `build-dataset --config` | Build any dataset instance from config: original, edited, long-tail, grayscale, etc. |
| `build-view --dataset --config` | Build projections and diagnostics for a registered variant. |
| `build-trajectory --run-dir` | Project and register saved trajectory arrays from a completed run. |
| `build-model-diagnostics --dataset --run-dir` | Compute checkpoint-dependent ID diagnostics and merge them into registered projection views. |
| `summarize` | Print global and optional per-class intrinsic-dimension summaries for registered views. |
| `launch` | Start the Streamlit UI. Use `--dry-run` to print the launch command. |

Model diagnostics require a real `<run-dir>/checkpoint.pt`; saved samples or trajectory
arrays are not enough because these estimators call the trained model. Available estimators:

| Estimator | Checkpoint type | Main columns |
|---|---|---|
| `fm_jacobian` | Flow-matching velocity model | `fm_jacobian_*_rank_t*` |
| `fm_flipd` | Flow-matching velocity model with independent Gaussian coupling | `fm_flipd_lid_t*`, divergence, recovered score norm |
| `diffusion_normal_bundle` | Gaussian diffusion score/noise model | `diffusion_normal_bundle_lid_t*` upper bound, observed normal rank |
| `diffusion_flipd` | Gaussian diffusion score/noise model | `diffusion_flipd_lid_t*`, score divergence |

For FLIPD estimators, `--num-trace-samples` controls Hutchinson trace probes; pass `0`
only for exact divergence in tiny ambient dimensions. For diffusion estimators,
`--diffusion-sigmas` can override the sigma values inferred from the checkpoint's
Gaussian diffusion schedule. `fm_flipd` is not valid for minibatch-OT flow matching
checkpoints; use `fm_jacobian` for those runs. In high-dimensional pixel space,
`diffusion_normal_bundle_lid_t*` is an upper bound unless `--num-perturbations` is large
enough to resolve the full normal rank.

To compare the raw-geometry class/global ID estimates without opening CSV files:

```bash
fm-lab-explorer summarize --include-classes

fm-lab-explorer summarize \
  --dataset mnist/tail_digit1 \
  --metric global_mle_lid_k20 \
  --include-classes
```

Main workspace layout:

```text
outputs/geometry_explorer/
  registry.sqlite
  datasets/<family>/<variant>/
  model_runs/<family>/<variant>/<run_id>/
```

The registry stores metadata and paths only. Large explorer tables remain Parquet,
trajectory coordinates remain `.npz`, sample arrays remain `.npy`, and thumbnails remain
atlas image files.

## `fm-lab-sample-checkpoint`

Regenerate samples and trajectories from an existing checkpoint without retraining.
This is useful when a trained MNIST model needs more trajectories for UMAP projection.

```bash
fm-lab-sample-checkpoint \
  --run-dir runs/mnist_image_unet_ot \
  --output-dir runs/mnist_image_unet_ot_umap_resample \
  --n-samples 2048 \
  --n-trajectories 256 \
  --sample-batch-size 256 \
  --nfe 64 \
  --trajectory-umap \
  --device auto
```

Key options:

| Option | Meaning |
|---|---|
| `--run-dir` | Completed training run directory. Required. |
| `--checkpoint` | Checkpoint path; defaults to `<run-dir>/checkpoint.pt`. |
| `--output-dir` | Output directory; defaults to writing into `--run-dir`. |
| `--n-samples` | Override `sampling.n_samples`. |
| `--n-trajectories` | Override `sampling.n_trajectories`. |
| `--nfe` | Override `sampling.nfe`. |
| `--weights` | Select `raw` (default) or `ema` checkpoint weights. |
| `--classifier-free-guidance-scale` | Override classifier-free guidance directly. |
| `--sample-batch-size` | Integrate final generated samples in chunks of this size. |
| `--trajectory-umap` | Enable 3D UMAP trajectory plots for this sampling pass. |
| `--no-trajectory-umap` | Disable 3D UMAP trajectory plots for this sampling pass. |
| `--trajectory-umap-target-points` | Maximum target reference points included in the UMAP fit. |
| `--trajectory-umap-neighbors` | UMAP `n_neighbors`. |
| `--trajectory-umap-min-dist` | UMAP `min_dist`. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |

Use `--output-dir` when changing `--nfe`, `--n-samples`, or `--n-trajectories` if you
want to preserve the original run artifacts.

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
