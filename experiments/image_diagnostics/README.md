# Unified Geometry Explorer

The active workflow is registry-backed and starts from `fm-lab-explorer`.
The old `outputs/dataset_explorer` workflow is no longer the recommended path.

Workspace layout:

```text
outputs/geometry_explorer/
  registry.sqlite
  datasets/<family>/<variant>/
  model_runs/<family>/<variant>/<run_id>/
```

Large artifacts stay outside SQLite: dataset indexes are Parquet, arrays are
`.npy`/`.npz`, and previews are atlas images.

## Build Dataset Variants

```bash
fm-lab-explorer build-dataset \
  --config configs/geometry_explorer/mnist_original.yaml

fm-lab-explorer build-variant \
  --config configs/geometry_explorer/mnist_long_tail_001.yaml
```

## Build Projection Views

```bash
fm-lab-explorer build-view \
  --dataset mnist/original \
  --config configs/geometry_explorer/mnist_raw_geometry_view.yaml

fm-lab-explorer build-view \
  --dataset mnist/long_tail_001 \
  --config configs/geometry_explorer/mnist_raw_geometry_view.yaml
```

## Train On A Variant

```bash
fm-lab-train \
  --config configs/mnist/mnist_image_unet_ot.yaml \
  --dataset-variant mnist/long_tail_001 \
  --workspace outputs/geometry_explorer
```

The training run persists its resolved dataset variant in `config.yaml` and is
registered against the geometry workspace.

## Build A Trajectory View

```bash
fm-lab-explorer build-trajectory \
  --run-dir runs/mnist_image_unet_ot \
  --nfe 64 \
  --max-trajectories 512
```

This projects saved high-dimensional sampling trajectories into a shared 3D
UMAP coordinate system and registers the view under
`outputs/geometry_explorer/model_runs/...`.

## Launch

```bash
fm-lab-explorer launch
```

The Streamlit app provides selectors for dataset family, variant, projection
view, mode, and trajectory run when available. The embedded Three.js viewer
handles both dataset geometry and model trajectories.
