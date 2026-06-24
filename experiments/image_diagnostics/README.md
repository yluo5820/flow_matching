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

## MNIST Demo

The repository already contains MNIST under `data/mnist`, so no download is
needed.

Validate the run:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_example.yaml \
  --dry-run
```

Build the raw-pixel UMAP explorer:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_example.yaml
```

Launch it:

```bash
streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_umap/explorer/explorer_data.parquet
```

The example selects 2,000 test digits deterministically, exports small grayscale
thumbnails, treats each normalized `28x28` image as a 784-dimensional feature
vector, computes UMAP/PCA and local diagnostics, and colors points by digit.
Selecting a point displays the digit and its nearest raw-pixel neighbors.

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

Streamlit serves the Python application and retains the diagnostics table and
manual-label controls in a secondary workspace below the canvas. The canvas is
implemented locally without React; the optional 3D renderer uses Three.js.

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

The smaller `mnist_umap_example.yaml` remains the better iteration config. It
computes a fresh UMAP/PCA over 2,000 test digits and includes local diagnostics;
the full reference config favors fidelity and disables those extra diagnostics.

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

The explorer also supports a true three-component UMAP rendered with a
perspective Three.js camera. The existing 2D configs are unchanged.

Build the 2,000-digit example directly:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_umap_3d_example.yaml

streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_umap_3d_example/explorer/explorer_data.parquet
```

For all 70,000 digits, compute the 3D coordinates once and then build the
full explorer:

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

## MNIST With DINOv2 Features

The matching DINOv2 examples use the same deterministic 2,000 MNIST test
digits as the raw-pixel examples, but replace each flattened 784-value image
with the 768-dimensional CLS token from `facebook/dinov2-base`.

Install the learned-image dependencies:

```bash
python -m pip install -e ".[image-diagnostics,image-embeddings]"
```

Build the 2D explorer first:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_dinov2_umap_2d.yaml
```

Then build the 3D explorer:

```bash
python experiments/image_diagnostics/build_explorer.py \
  --config configs/image_diagnostics/mnist_dinov2_umap_3d.yaml
```

Both configs use:

```text
outputs/dataset_explorer/mnist_dinov2_shared/features/
```

as their feature cache. The first build downloads DINOv2 Base and computes a
single `2000x768` normalized feature matrix. The second build validates and
reuses that matrix, then computes only its own UMAP. The first download is
approximately 346 MB.

Launch the outputs:

```bash
streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_dinov2_umap_2d/explorer/explorer_data.parquet

streamlit run experiments/image_diagnostics/explorer_app.py -- \
  --data outputs/dataset_explorer/mnist_dinov2_umap_3d/explorer/explorer_data.parquet
```

DINOv2 was trained on natural RGB imagery rather than handwritten digits.
These views are useful as a representation comparison, but DINOv2 is not
expected to be intrinsically better than raw pixels for MNIST.

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
outputs/dataset_explorer/mnist_umap/
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
    raw_pixels_umap.csv
    raw_pixels_pca.csv
  diagnostics/
    raw_pixels_local_diagnostics.csv
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
