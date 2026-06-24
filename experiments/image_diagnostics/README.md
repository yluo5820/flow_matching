# Dataset UMAP Explorer

This module builds an interactive UMAP explorer for image or vector datasets.
It is independent of any image-generation model.

The processing pipeline is:

```text
dataset -> feature matrix -> UMAP/PCA -> local diagnostics -> explorer
```

MNIST and NumPy arrays provide their feature matrices directly and require no
embedding model. Collections of arbitrary image files can use resized raw
pixels or optionally DINOv2. CLIP is not used.

## Install

```bash
python -m pip install -e ".[image-diagnostics]"
```

This installs UMAP, Streamlit, Plotly, PyArrow, and Pillow. It does not install
or download an embedding model.

Only install the optional learned-image feature dependencies when explicitly
using `features.mode: dinov2`:

```bash
python -m pip install -e ".[image-diagnostics,image-embeddings]"
```

## Full MNIST Raw-Pixel Explorer

The repository already contains MNIST under `data/mnist`, so no download is
needed.

Validate the run:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_raw_umap_full.yaml \
  --dry-run
```

The dry run should report 70,000 selected samples. The config reads the 2D and
3D UMAP coordinates produced by
`compute_mnist_reference_projections.py` from `data/umap_explorer_local/` and
computes PCA during the build. It then estimates intrinsic dimension in the
raw-pixel feature space and writes
`explorer/explorer_data_with_raw_id.parquet`. Build everything with:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_raw_umap_full.yaml
```

ID estimation is enabled by default for this config. To rebuild only the
projection explorer, add `--no-id-estimation`.

Launch it:

```bash
streamlit run experiments/image_diagnostics/explorer_app.py
```

The config uses all 60,000 training digits and 10,000 test digits. One build
flattens each `28x28` image into a 784-dimensional raw-pixel vector and
computes UMAP 2D, PCA 2D, and UMAP 3D. The explorer discovers the output
automatically and places all three views in one projection selector.

The primary visualization follows the same broad architecture and interaction
style as
[GrantCuster/umap-explorer](https://github.com/GrantCuster/umap-explorer):
coordinates are computed ahead of time, Python packs the digit thumbnails into
cached texture atlases, and a full dark canvas displays the projected samples.
Every point is an actual digit thumbnail rather than a marker.

The canvas supports:

- drag-to-pan and cursor-centered wheel zoom
- hover enlargement with label, source index, and diagnostics
- click-to-pin selection
- animated UMAP/PCA switching
- reset-to-fit
- a responsive sidebar and compact mobile layout

Streamlit serves the Python application. The canvas is implemented locally
without React; the 3D renderer uses Three.js.

The reference project is MIT-licensed. This implementation is an independent
Python/Canvas version; it uses the reference's public interaction ideas rather
than bundling its React application source.

## Full Reference Configuration

For a presentation much closer to the original 70,000-point explorer, use:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_reference.yaml

streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_umap_reference/explorer/explorer_data.parquet
```

This configuration reproduces the original notebook and renderer settings:

- all 60,000 training digits followed by all 10,000 test digits
- stable label grouping within each split, matching old `fetch_mldata`
- UMAP defaults with `random_state=42`
- t-SNE defaults from the original notebook with `random_state=0`
- UMAP with `min_dist=0.8` and `random_state=42`
- 28px points, 56px hover previews, and 1,000ms linear transitions
- 2048x2048 atlases with 73x73 slots and 14 atlas images
- the original ten-color label palette

The first build downloads the original project's three published projection
JSON files into `data/umap_explorer_reference/`. They total about 9 MB and are
coordinates only: no CLIP, DINOv2, diffusion model, or embedding model is
downloaded. The digit images come from the repository's local MNIST IDX files,
whose ordering was checked byte-for-byte against the original sprite sheets.

### Compute The Full Projections Locally

To produce all three coordinate files from the local MNIST pixels instead of
using the published coordinates:

```bash
python experiments/image_diagnostics/compute_mnist_reference_projections.py
```

The script writes:

```text
data/umap_explorer_local/
  mnist_embeddings.json
  tsne_mnist_embeddings.json
  md08_umap_mnist_embeddings.json
  mnist_labels.json
  manifest.json
```

It uses the original notebook parameters and ordering. Run the two UMAP jobs
first if you want faster feedback:

```bash
python experiments/image_diagnostics/compute_mnist_reference_projections.py \
  --methods umap umap-min-dist-0.8

python experiments/image_diagnostics/compute_mnist_reference_projections.py \
  --methods tsne
```

The full 70,000-point t-SNE is CPU-intensive and can take substantially longer
than UMAP. Existing method outputs are retained unless `--overwrite` is passed.
A quick pipeline smoke test can use `--max-samples 500`, but those shortened
files are not compatible with the full explorer config.

After all three files exist, build and launch the local-coordinate explorer:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_local.yaml

streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_umap_local/explorer/explorer_data.parquet
```

The full MNIST configs keep the projected digits label-colored but render the
selected preview in the original white-on-black grayscale. Hovering or pinning
a digit updates the sidebar immediately with diagnostics for the active
projection: coordinates, k-neighbor radius and mean distance, local label
agreement, distance to the digit-class centroid, and the nearest projected
sample. The separate table workspace is hidden for these configs.

The original repository used older releases of UMAP and scikit-learn. The local
script records installed package versions and all parameters in `manifest.json`;
the result should have comparable cluster structure but is not expected to
match the historical coordinates exactly.

## 3D UMAP

`mnist_raw_umap_full.yaml` computes its 3D UMAP directly alongside the 2D
views. To reproduce the original-project ordering through separately stored
coordinate files instead, run:

```bash
python experiments/image_diagnostics/compute_mnist_reference_projections.py \
  --methods umap-3d

python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_3d_local.yaml

streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_umap_3d_local/explorer/explorer_data.parquet
```

Drag the scene to orbit, use the wheel or `+`/`-` controls to zoom, and
double-click or use reset to restore the initial camera. Hovering or clicking
a digit shows its original preview and diagnostics measured in the 3D UMAP
space, including the displayed `x`, `y`, and `z` coordinates.

The first 3D launch caches the pinned Three.js `0.159.0` browser runtime under
the explorer output directory. This is approximately 650 KB and is not an
embedding model.

## Full MNIST With DINOv2 Features

The DINOv2 config uses the same 70,000 MNIST rows and ordering as the raw-pixel
config, but replaces each flattened 784-value image with the 768-dimensional
CLS token from `facebook/dinov2-base`.

Install the learned-image dependencies:

```bash
python -m pip install -e ".[image-diagnostics,image-embeddings]"
```

Build both DINOv2 UMAP views in one run:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_dinov2_umap_full.yaml
```

The first build downloads DINOv2 Base, computes one normalized `70000x768`
feature matrix, then computes 2D and 3D UMAP from that shared matrix. The model
download is approximately 346 MB; full feature extraction and both projections
are substantially more expensive than the raw-pixel build. Intrinsic-dimension
estimation in DINOv2 feature space runs afterward and writes
`explorer/explorer_data_with_id.parquet`.

Launch all existing compatible views:

```bash
streamlit run experiments/image_diagnostics/explorer_app.py
```

DINOv2 was trained on natural RGB imagery rather than handwritten digits.
These views are useful as a representation comparison, but DINOv2 is not
expected to be intrinsically better than raw pixels for MNIST.

## Full CIFAR-10 Explorer

CIFAR-10 support uses the official binary archive from the
[University of Toronto dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
The loader verifies the published MD5 checksum before extracting the 50,000
training and 10,000 test images. No torchvision dataset dependency is required.

Download CIFAR-10, build raw-pixel UMAP/PCA in 2D and UMAP in 3D, and estimate
intrinsic dimension:

```bash
.conda/fm_lab/bin/python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/cifar10_raw_umap_full.yaml
```

The 32x32 RGB images are packed directly into sprite atlases, so the explorer
shows the original color image for every point without exporting 60,000
individual PNG files.

To build the matching DINOv2 2D/3D views and DINOv2-space ID estimates:

```bash
.conda/fm_lab/bin/python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/cifar10_dinov2_umap_full.yaml
```

Both outputs use identical train-then-test row ordering, so automatic loading
places their compatible views in one projection selector. Launch all available
datasets with:

```bash
.conda/fm_lab/bin/streamlit run experiments/image_diagnostics/explorer_app.py
```

The raw-pixel and DINOv2 builds are computationally independent. Use
`--no-id-estimation` when iterating on projections without recomputing ID.

## Automatic Explorer Loading

The explorer automatically discovers existing outputs, groups views with the
same sample rows, and merges their precomputed projection columns in memory.
Launch it without specifying files:

```bash
.conda/fm_lab/bin/streamlit run experiments/image_diagnostics/explorer_app.py
```

By default it scans `outputs/dataset_explorer`. Compatible raw-pixel and
DINOv2 2D/3D views appear in one projection selector. If the directory
contains incompatible sample sets, an in-app dataset selector is shown.

Automatic loading validates stable sample keys such as `dataset`, `split`,
and `source_index`. It does not load DINOv2 or recompute embeddings or UMAP.
Projection-space diagnostics are calculated for the merged in-memory table.
Generated combined-output directories are ignored to avoid duplicate views,
and `explorer_data_with_id.parquet` is preferred when available.

To scan a different output root:

```bash
.conda/fm_lab/bin/streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data-dir path/to/dataset_explorer
```

Single-file mode remains available:

```bash
.conda/fm_lab/bin/streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data path/to/explorer_data.parquet
```

The unified Three.js renderer uses a front-facing flat camera for 2D
projections and switches to orbit controls for 3D projections. The same
projection selector, preview, and live diagnostics panel are used in both
modes.

## Intrinsic Dimension Estimation

The regular intrinsic-dimension module estimates how many local degrees of
freedom are visible in a selected representation. It does not use FLIPD,
diffusion scores, UMAP coordinates, or t-SNE coordinates.

Run the DINOv2 example:

```bash
python experiments/image_diagnostics/estimate_intrinsic_dimension.py \
  --config configs/image_diagnostics/id_estimation_example.yaml
```

Inspect the plan without computing neighbors or estimators:

```bash
python experiments/image_diagnostics/estimate_intrinsic_dimension.py \
  --config configs/image_diagnostics/id_estimation_example.yaml \
  --dry-run
```

The raw-pixel MNIST comparison uses PCA preprocessing:

```bash
python experiments/image_diagnostics/estimate_intrinsic_dimension.py \
  --config configs/image_diagnostics/id_estimation_mnist_raw.yaml
```

The feature loader supports aligned `.npy` matrices with optional Parquet
metadata, raw pixels loaded from `image_path`, L2 normalization, and optional
PCA preprocessing. Therefore CLIP, classifier, or other feature matrices can
be analyzed without adding model-specific code.

Local outputs are computed for every requested neighborhood size:

- Covariance eigenvalues describe the local spectrum.
- Participation ratio is a smooth effective-rank estimate.
- PCA threshold dimension counts components required for 80%, 90%, 95%, or
  99% local variance.
- TWO-NN uses the ratio between the first two neighbor distances. Its
  per-sample value is especially noisy.
- kNN MLE LID estimates local distance-growth dimension and is sensitive to
  the selected `k`.
- Local ball scaling fits neighbor-count growth against radius and records
  both slope and fit quality.

Global and grouped outputs add full-group covariance estimates, aggregate
TWO-NN and MLE, and correlation/ball-scaling slopes. Ball-scaling curve CSVs
retain the radius and mass values so apparent dimension plateaus can be
inspected instead of trusting only the fitted slope.

Outputs are written under:

```text
outputs/image_diagnostics/<id_estimation_name>/
  config_used.yaml
  manifest.json
  intrinsic_dimension/
    local_id_<feature_space>.parquet
    local_id_<feature_space>.csv
    group_id_<feature_space>.parquet
    group_id_<feature_space>.csv
    id_curves/
```

When explorer merging is enabled, the original explorer table is preserved
and `explorer_data_with_id.parquet` is created beside it. The merged columns
include the feature-space name, representative MLE LID, participation ratio,
PCA dimension, TWO-NN proxy, and neighborhood distances.

Interpretation limits:

1. ID is representation-dependent. DINOv2-space ID and pixel-space ID answer
   different questions.
2. Estimates depend on scale, neighborhood size, density, noise, and sample
   count. Compare several estimators and several `k` values.
3. UMAP or t-SNE separation is not an intrinsic-dimension estimate.
4. Local estimates and small groups are noisy. Relative patterns are usually
   more defensible than absolute dimensions.
5. Duplicate points create zero distances. The module records unstable
   estimates as `NaN` and continues.
6. PCA preprocessing can stabilize noisy features but imposes an upper bound
   on every subsequent estimate.
7. Optional `scikit-dimension` estimators run only when the package is
   installed; core outputs do not depend on it.

For MNIST, raw pixels can produce cleaner class geometry than DINOv2. Digit
identity is strongly tied to stroke-level pixel structure, while DINOv2 was
trained for natural-image invariances and may suppress that variation. This
does not make pixels universally better: DINOv2, CLIP, or classifier features
are often more meaningful for natural images. No representation should be
treated as the unique true manifold.

## NumPy And Toy Data

Any two-dimensional NumPy array with shape `(samples, features)` can be used:

```yaml
input:
  type: numpy
  data_path: path/to/samples.npy
  labels_path: path/to/labels.npy
  max_samples: 5000

features:
  mode: raw
  name: input_vectors
```

For flattened grayscale images, set an image shape to generate previews:

```yaml
input:
  image_shape: [28, 28]
  value_range: [-1.0, 1.0]
```

For 2D or 3D toy samples, leave `image_shape: null`. The original coordinates
are used directly, which is more meaningful than embedding a rendered plot.

## Arbitrary Image Collections

Image collections use:

```yaml
input:
  type: image_metadata
  experiment_dir: path/to/dataset
  metadata_path: metadata/images.jsonl

features:
  mode: raw
  name: resized_pixels
  image_size: [64, 64]
```

The JSONL or CSV metadata must contain `image_path` or `output_path`. Optional
columns such as `label`, `family`, `tags`, and `status` become explorer filters.

Raw resized pixels need no model download but mostly capture color, texture, and
layout. For semantic natural-image similarity, switch to:

```yaml
features:
  mode: dinov2
  name: dinov2
  repo_id: facebook/dinov2-base
```

## Outputs

```text
outputs/dataset_explorer/mnist_raw_umap_full/
  config_used.yaml
  run_log.txt
  dataset_index.parquet
  assets/thumbnails/
  assets/atlases/
    atlas_<content-hash>_00.png
  features/
    raw_pixels_features.npy
    raw_pixels_metadata.parquet
  projections/
    raw_pixels_umap_2d.csv
    raw_pixels_pca_2d.csv
    raw_pixels_umap_3d.csv
  explorer/
    explorer_data.parquet
    manual_labels.csv
```

## Diagnostics

- `knn_radius_k15`: distance to the furthest available local neighbor.
- `knn_mean_distance_k15`: mean local-neighbor distance.
- `local_eig_1` through `local_eig_5`: leading local covariance eigenvalues.
- `participation_ratio_k15`: local effective-rank proxy.
- `two_nn_lid`: rough pointwise local-dimension estimate.
- `distance_to_label_centroid`: distance to the sample-label centroid.
- `distance_to_family_centroid`: distance to a configured family centroid.
- `outlier_score`: negated Local Outlier Factor; larger is more outlier-like.

UMAP is exploratory rather than proof of manifold structure. Local dimension and
outlier scores depend on the feature representation, metric, and neighborhood
scale.
