# Generated Dataset Workflow

This is the standard end-to-end workflow for creating model-generated datasets
that can be compared in the geometry explorer.

The workflow is intentionally fixed across datasets:

1. build and register the original dataset
2. train a checkpoint
3. sample generated datasets from that checkpoint
4. register generated datasets during sampling
5. optionally label generated samples
6. build explorer views
7. launch the explorer UI

The commands assume they are run from the repository root.

## Dataset Variables

Set these per dataset and model run.

```bash
WORKSPACE=outputs/geometry_explorer
DEVICE=mps

FAMILY=<dataset_family>
ORIGINAL_VARIANT="$FAMILY/original"

DATASET_CONFIG=<path_to_dataset_config.yaml>
TRAIN_CONFIG=<path_to_model_training_config.yaml>
VIEW_CONFIG=configs/geometry_explorer/views/raw_pixels.yaml

RUN_DIR=<path_to_training_run_dir>

N_SAMPLES=<original_dataset_size>
NFE=64
TRAIN_BATCH=<training_batch_size>
SAMPLE_BATCH=<sampling_batch_size>
ATLAS_TILE_SIZE=<image_side_length>
```

Typical values:

```bash
# MNIST / Fashion-MNIST
N_SAMPLES=70000
ATLAS_TILE_SIZE=28
TRAIN_BATCH=128
SAMPLE_BATCH=512
```

```bash
# CIFAR-10 with split: all
N_SAMPLES=60000
ATLAS_TILE_SIZE=32
TRAIN_BATCH=32
SAMPLE_BATCH=64
```

If MPS runs out of memory, lower `SAMPLE_BATCH` first. If training itself runs
out of memory, lower `TRAIN_BATCH`.

## Generated Dataset Variables

Use stable names so the run directory and registered variant are easy to match.

```bash
CONTROL_RUN="${RUN_DIR}_control"
PRIOR_RUN="${RUN_DIR}_prior_s080"
DENSITY_RUN="${RUN_DIR}_density_q005"

CONTROL_VARIANT="$FAMILY/generated_diffusion_x_control"
PRIOR_VARIANT="$FAMILY/generated_diffusion_x_prior_s080"
DENSITY_VARIANT="$FAMILY/generated_diffusion_x_density_q005"
```

The suffixes encode the sampler:

- `control`: ordinary unguided sampling
- `prior_s080`: prior source latents scaled by `0.80`
- `density_q005`: density guidance with quantile `0.05`

## Build The Original Dataset

```bash
.conda/fm_lab/bin/fm-lab-explorer build-dataset \
  --config "$DATASET_CONFIG"
```

This registers the original dataset variant, usually `$FAMILY/original`, in the
geometry explorer workspace.

## Train A Checkpoint

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config "$TRAIN_CONFIG" \
  --output-dir "$RUN_DIR" \
  --device "$DEVICE" \
  --batch-size "$TRAIN_BATCH" \
  --sample-batch-size "$SAMPLE_BATCH"
```

For the sampler guidance workflows, use a diffusion checkpoint with clean-X
prediction when available. The sampler-side guidance mechanisms were built for
that path.

## Sample And Register A Control Dataset

```bash
.conda/fm_lab/bin/fm-lab-sample-checkpoint \
  --run-dir "$RUN_DIR" \
  --output-dir "$CONTROL_RUN" \
  --device "$DEVICE" \
  --n-samples "$N_SAMPLES" \
  --n-trajectories 16 \
  --nfe "$NFE" \
  --sample-batch-size "$SAMPLE_BATCH" \
  --plot-max-points 64 \
  --no-trajectory-umap \
  --register-dataset "$CONTROL_VARIANT" \
  --dataset-workspace "$WORKSPACE" \
  --dataset-label control_unguided \
  --dataset-atlas-tile-size "$ATLAS_TILE_SIZE"
```

Registration happens through `--register-dataset`. The generated samples are
saved under `outputs/geometry_explorer/datasets/<family>/<variant>/`.

## Sample And Register A Scaled-Prior Dataset

```bash
.conda/fm_lab/bin/fm-lab-sample-checkpoint \
  --run-dir "$RUN_DIR" \
  --output-dir "$PRIOR_RUN" \
  --device "$DEVICE" \
  --n-samples "$N_SAMPLES" \
  --n-trajectories 16 \
  --nfe "$NFE" \
  --sample-batch-size "$SAMPLE_BATCH" \
  --plot-max-points 64 \
  --no-trajectory-umap \
  --prior-guidance-scale 0.80 \
  --register-dataset "$PRIOR_VARIANT" \
  --dataset-workspace "$WORKSPACE" \
  --dataset-label prior_s080 \
  --dataset-atlas-tile-size "$ATLAS_TILE_SIZE"
```

Lower prior scales push initial latents closer to the high-density region of the
Gaussian source. `0.80` is a useful first setting; stronger settings such as
`0.60` may create smoother but less diverse samples.

## Optional: Sample And Register A Density-Guided Dataset

```bash
.conda/fm_lab/bin/fm-lab-sample-checkpoint \
  --run-dir "$RUN_DIR" \
  --output-dir "$DENSITY_RUN" \
  --device "$DEVICE" \
  --n-samples "$N_SAMPLES" \
  --n-trajectories 16 \
  --nfe "$NFE" \
  --sample-batch-size "$SAMPLE_BATCH" \
  --plot-max-points 64 \
  --no-trajectory-umap \
  --density-guidance-quantile 0.05 \
  --density-guidance-strength 1.0 \
  --register-dataset "$DENSITY_VARIANT" \
  --dataset-workspace "$WORKSPACE" \
  --dataset-label density_q005 \
  --dataset-atlas-tile-size "$ATLAS_TILE_SIZE"
```

Use lower quantiles to target higher-density samples more aggressively.

## Optional: Label Generated Samples

For MNIST generated datasets:

```bash
.conda/fm_lab/bin/fm-lab-explorer label-mnist \
  --dataset "$CONTROL_VARIANT" \
  --device "$DEVICE"

.conda/fm_lab/bin/fm-lab-explorer label-mnist \
  --dataset "$PRIOR_VARIANT" \
  --device "$DEVICE"
```

For Fashion-MNIST generated datasets:

```bash
.conda/fm_lab/bin/fm-lab-explorer label-fashion-mnist \
  --dataset "$CONTROL_VARIANT" \
  --device "$DEVICE"

.conda/fm_lab/bin/fm-lab-explorer label-fashion-mnist \
  --dataset "$PRIOR_VARIANT" \
  --device "$DEVICE"
```

Only use these labelers for matching grayscale datasets. Do not use MNIST-style
label tinting for RGB datasets such as CIFAR-10.

## Build Explorer Views

To rebuild views for every dataset currently registered in the workspace:

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  build-registered-views
```

Preview the registered datasets first without running the rebuild:

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  build-registered-views \
  --dry-run
```

To rebuild only selected registered datasets:

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  build-registered-views \
  --dataset "$CONTROL_VARIANT" \
  --dataset "$PRIOR_VARIANT"
```

The default view config computes UMAP 3D and PCA 3D only. To explicitly add
t-SNE 3D for selected datasets, use the manual t-SNE view config:

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  build-registered-views \
  --view-config configs/geometry_explorer/views/raw_pixels_with_tsne.yaml \
  --dataset "$CONTROL_VARIANT"
```

t-SNE is intentionally not part of the default rebuild path because it is costly
on full 60k/70k image datasets.

For one-off manual control, build the original view first, then one view per
generated variant.

```bash
.conda/fm_lab/bin/fm-lab-explorer build-view \
  --dataset "$ORIGINAL_VARIANT" \
  --config "$VIEW_CONFIG"

.conda/fm_lab/bin/fm-lab-explorer build-view \
  --dataset "$CONTROL_VARIANT" \
  --config "$VIEW_CONFIG"

.conda/fm_lab/bin/fm-lab-explorer build-view \
  --dataset "$PRIOR_VARIANT" \
  --config "$VIEW_CONFIG"
```

For the optional density-guided dataset:

```bash
.conda/fm_lab/bin/fm-lab-explorer build-view \
  --dataset "$DENSITY_VARIANT" \
  --config "$VIEW_CONFIG"
```

If you label a generated dataset after a view was already built, rebuild that
view so the UI payload includes the new labels.

## Launch The Explorer

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  launch
```

To print the Streamlit launch command without starting the server:

```bash
.conda/fm_lab/bin/fm-lab-explorer \
  --workspace "$WORKSPACE" \
  launch \
  --dry-run
```

## Check Dataset Registration

```bash
.conda/fm_lab/bin/python - <<'PY'
from fm_lab.geometry_explorer.registry import GeometryRegistry

workspace = "outputs/geometry_explorer"
variants = [
    "<family>/original",
    "<family>/generated_diffusion_x_control",
    "<family>/generated_diffusion_x_prior_s080",
]

registry = GeometryRegistry(workspace)
for variant_id in variants:
    row = registry.get_dataset_variant(variant_id)
    print(variant_id, dict(row)["row_count"], dict(row)["data_path"])
PY
```

## Example Variable Blocks

MNIST:

```bash
FAMILY=mnist
ORIGINAL_VARIANT=mnist/original
DATASET_CONFIG=configs/geometry_explorer/datasets/mnist/original/dataset.yaml
TRAIN_CONFIG=configs/geometry_explorer/datasets/mnist/original/models/image_unet_diffusion_epsilon.yaml
RUN_DIR=runs/mnist_original_unet_diffusion_x
N_SAMPLES=70000
TRAIN_BATCH=128
SAMPLE_BATCH=512
ATLAS_TILE_SIZE=28
```

Fashion-MNIST:

```bash
FAMILY=fashion_mnist
ORIGINAL_VARIANT=fashion_mnist/original
DATASET_CONFIG=configs/geometry_explorer/datasets/fashion_mnist/original/dataset.yaml
TRAIN_CONFIG=configs/geometry_explorer/datasets/fashion_mnist/original/models/image_unet_diffusion_x.yaml
RUN_DIR=runs/fashion_mnist_original_unet_diffusion_x
N_SAMPLES=70000
TRAIN_BATCH=128
SAMPLE_BATCH=512
ATLAS_TILE_SIZE=28
```

CIFAR-10:

```bash
FAMILY=cifar10
ORIGINAL_VARIANT=cifar10/original
DATASET_CONFIG=configs/geometry_explorer/datasets/cifar10/original/dataset.yaml
TRAIN_CONFIG=configs/geometry_explorer/datasets/cifar10/original/models/image_unet_diffusion_x.yaml
RUN_DIR=runs/cifar10_original_unet_diffusion_x_bs32
N_SAMPLES=60000
TRAIN_BATCH=32
SAMPLE_BATCH=64
ATLAS_TILE_SIZE=32
```
