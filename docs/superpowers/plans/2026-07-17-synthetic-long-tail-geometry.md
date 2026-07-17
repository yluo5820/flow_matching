# Synthetic Long-Tail Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the controlled three-object experiment that measures how
class frequency and known manifold dimension interact in class-conditional flow
matching.

**Architecture:** Extend the existing analytic renderer and latent-factor system,
write one indexed master-pool format shared by all factorial conditions, and expose
that format through the existing `TargetDistribution` training path. Evaluation is
split into an independent factor oracle, distributional metrics, local FM geometry,
and deterministic aggregation into a living research report.

**Tech Stack:** Python 3.11, NumPy, SciPy, pandas, PyTorch, Pillow, PyYAML, pytest,
existing `fm_lab` flow-matching and geometry-explorer infrastructure.

## Global Constraints

- Use exactly three asymmetric objects: `stepped_monument`, `crooked_arch`, and
  `three_arm_vane`; do not use the marked cube.
- Render `32x32` RGB uint8 training pools without per-image PNG duplication.
- Use dimension levels `low=1`, `medium=3`, and `high=5` with nested factors:
  depth; xyz translation; xyz translation plus bounded camera azimuth/elevation.
- Use class counts `(5000, 500, 50)` and balanced `(5000, 5000, 5000)` controls.
- Cross three geometry mappings with three frequency mappings and add one balanced
  condition per geometry mapping: 12 conditions per replicate.
- Use three paired replicate bundles: 36 main-matrix models.
- All 500-, 50-, and 5,000-example views are nested prefixes of the same seeded
  master pool for an object-dimension cell.
- Use ordinary empirical sampling, class conditioning, no data augmentation, and
  identical model/training settings after the balanced pilot is frozen.
- Save both equal-update and matched-example-pass checkpoints.
- Treat factor recovery as primary; treat FM Jacobian spectra and FM-FLIPD as
  complementary probes, never as literal flow-map rank.
- Renderer, oracle, metric, and balanced-pilot gates block downstream stages on
  failure.
- Preserve invalid, class-leaked, and off-renderer generated samples in summaries.
- Every persisted artifact records seeds, configuration hash, code revision, and
  source checkpoint; completed artifacts are immutable.
- Do not add natural-image validation, single-class models, sharing interventions,
  or long-tail hyperparameter sweeps in this implementation.

## File Structure

The implementation uses existing package boundaries:

- `fm_lab/geometry_explorer/latent_factors.py`: bounded camera-view factor.
- `fm_lab/geometry_explorer/render_maps.py`: translate that factor into a camera
  frame and accept fixed object colors.
- `fm_lab/geometry_explorer/synthetic_objects.py`: the three object meshes and
  uniform OKLCH-derived materials.
- `fm_lab/geometry_explorer/synthetic_long_tail_design.py`: immutable factorial
  design, manifest types, factor ladder, and master-pool generation.
- `fm_lab/geometry_explorer/synthetic_long_tail_calibration.py`: renderer rank,
  factor visibility, nuisance matching, and object-separability gates.
- `fm_lab/data/synthetic_long_tail.py`: memory-mapped indexed training target.
- `fm_lab/geometry_explorer/synthetic_factor_oracle.py`: oracle model, losses,
  renderer-reference stream, checkpoint, and validation gates.
- `fm_lab/geometry_explorer/synthetic_long_tail_metrics.py`: factor distribution,
  oracle-feature quality, and known-collapse controls.
- `fm_lab/geometry_explorer/synthetic_long_tail_geometry.py`: renderer-tangent,
  flow-Jacobian, FLIPD, and memorization measurements.
- `fm_lab/experiments/synthetic_long_tail_geometry.py`: stage services, condition
  configuration generation, run ledger, and aggregation.
- `fm_lab/experiments/run_synthetic_long_tail_geometry.py`: thin CLI.
- `fm_lab/geometry_explorer/synthetic_long_tail_report.py`: deterministic Markdown
  report rendering.
- `scripts/generate_synthetic_long_tail.py`: focused master-pool generation CLI.
- `configs/synthetic_long_tail_geometry/experiment.yaml`: frozen experiment and
  gate settings.
- `configs/synthetic_long_tail_geometry/base_train.yaml`: shared flow-matching
  training configuration.
- `docs/research/synthetic_long_tail_geometry_report.md`: living report.

---

### Task 1: Add the bounded camera-view factor

**Files:**
- Modify: `fm_lab/geometry_explorer/latent_factors.py:110-165`
- Modify: `fm_lab/geometry_explorer/render_maps.py:197-294`
- Test: `tests/test_synthetic_factor_framework.py`

**Interfaces:**
- Produces: `BoundedLookAtView(elevation_bounds: tuple[float, float])`
- Produces factor values shaped `(2,)` as `(azimuth_radians, sin_elevation)`.
- Produces tangent labels `camera_azimuth` and `camera_elevation`.
- Consumed by: `build_factor_space("high")` in Task 3.

- [ ] **Step 1: Write failing sampling, retraction, and rendering tests**

```python
def test_bounded_look_at_view_samples_area_uniform_band() -> None:
    factor = BoundedLookAtView(elevation_bounds=(-np.pi / 6, np.pi / 6))
    values = sample_values(factor.sample(20_000, seed=7))
    assert values.shape == (20_000, 2)
    assert np.all(values[:, 0] >= -np.pi)
    assert np.all(values[:, 0] < np.pi)
    assert np.all(values[:, 1] >= -0.5)
    assert np.all(values[:, 1] <= 0.5)
    assert abs(float(values[:, 1].mean())) < 0.01
    assert factor.dim == 2
    assert factor.tangent_labels(values[0]) == [
        "camera_azimuth",
        "camera_elevation",
    ]


def test_render_map_applies_bounded_look_at_view() -> None:
    factor = BoundedLookAtView(elevation_bounds=(-np.pi / 6, np.pi / 6))
    render_map = RenderMap(
        factor,
        object_name="offset_monument",
        config=RenderConfig(image_size=32, render_mode="silhouette"),
    )
    front = render_map.render(np.asarray([0.0, 0.0], dtype=np.float32))
    side = render_map.render(np.asarray([np.pi / 2, 0.0], dtype=np.float32))
    high = render_map.render(np.asarray([0.0, 0.5], dtype=np.float32))
    assert np.mean(np.abs(front - side)) > 0.01
    assert np.mean(np.abs(front - high)) > 0.01
```

- [ ] **Step 2: Run the tests and verify the missing class failure**

Run:

```bash
pytest tests/test_synthetic_factor_framework.py \
  -k 'bounded_look_at_view or applies_bounded_look_at_view' -v
```

Expected: collection fails because `BoundedLookAtView` is not defined.

- [ ] **Step 3: Implement the bounded factor**

Add this dataclass beside `LookAtViewSphere`:

```python
@dataclass(frozen=True)
class BoundedLookAtView(LatentFactorSpace):
    elevation_bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    name: str = "bounded_look_at_view"
    dim: int = 2
    factor_names: tuple[str, ...] = ("camera_view",)
    factor_dims: tuple[int, ...] = (2,)

    def __post_init__(self) -> None:
        low, high = (float(value) for value in self.elevation_bounds)
        if not -math.pi / 2.0 < low < high < math.pi / 2.0:
            raise ValueError("elevation_bounds must lie inside (-pi/2, pi/2).")

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        low, high = self.elevation_bounds
        values = np.column_stack(
            [
                rng.uniform(-math.pi, math.pi, int(n)),
                rng.uniform(math.sin(low), math.sin(high), int(n)),
            ]
        ).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        del z
        return np.eye(2, dtype=np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        del z
        return ["camera_azimuth", "camera_elevation"]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> np.ndarray:
        value = np.asarray(z, dtype=np.float64) + eps * np.asarray(tangent_vec)
        value[0] = (value[0] + math.pi) % (2.0 * math.pi) - math.pi
        low, high = self.elevation_bounds
        value[1] = np.clip(value[1], math.sin(low), math.sin(high))
        return value.astype(np.float32)

    def distance(self, z1: Any, z2: Any) -> float:
        first = np.asarray(z1, dtype=np.float64)
        second = np.asarray(z2, dtype=np.float64)
        azimuth = abs(first[0] - second[0])
        azimuth = min(azimuth, 2.0 * math.pi - azimuth)
        return float(np.hypot(azimuth, first[1] - second[1]))

    def coordinates(self, z: Any) -> dict[str, float]:
        azimuth, sin_elevation = np.asarray(z, dtype=np.float64)
        return {
            "camera_azimuth": float(azimuth),
            "camera_elevation": float(math.asin(np.clip(sin_elevation, -1.0, 1.0))),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        coordinates = self.coordinates(z)
        azimuth_bin = _linear_bin(
            coordinates["camera_azimuth"],
            (-math.pi, math.pi),
            bins=num_bins,
        )
        elevation_bin = _linear_bin(
            coordinates["camera_elevation"],
            self.elevation_bounds,
            bins=num_bins,
        )
        label_id = azimuth_bin * num_bins + elevation_bin
        return {
            "label": f"bounded_view_bin_{label_id:04d}",
            "label_id": str(label_id),
            "camera_azimuth_bin": str(azimuth_bin),
            "camera_elevation_bin": str(elevation_bin),
        }
```

In `RenderMap._apply_factor`, convert its values to a unit camera direction:

```python
if isinstance(factor, BoundedLookAtView):
    azimuth, sin_elevation = np.asarray(z, dtype=np.float64)
    elevation = math.asin(float(np.clip(sin_elevation, -1.0, 1.0)))
    cos_elevation = math.cos(elevation)
    controls.camera_direction = np.asarray(
        [
            cos_elevation * math.cos(float(azimuth)),
            cos_elevation * math.sin(float(azimuth)),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    return
```

- [ ] **Step 4: Run the focused and existing factor tests**

Run:

```bash
pytest tests/test_synthetic_factor_framework.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add fm_lab/geometry_explorer/latent_factors.py \
  fm_lab/geometry_explorer/render_maps.py \
  tests/test_synthetic_factor_framework.py
git commit -m "feat: add bounded camera view factor"
```

### Task 2: Implement the three object classes and fixed materials

**Files:**
- Modify: `fm_lab/geometry_explorer/synthetic_objects.py:19-40,103-125,853-965`
- Modify: `fm_lab/geometry_explorer/render_maps.py:334-360`
- Test: `tests/test_synthetic_objects.py`

**Interfaces:**
- Produces object kinds `stepped_monument`, `crooked_arch`, `three_arm_vane`.
- Produces `oklch_to_srgb(lightness, chroma, hue_degrees)`.
- Adds `base_color: tuple[float, float, float] | None` to
  `SyntheticObjectSpec`.
- Consumed by: master-pool rendering in Task 3.

- [ ] **Step 1: Write failing object identity and material tests**

```python
@pytest.mark.parametrize(
    "kind,hue",
    [
        ("stepped_monument", 25.0),
        ("crooked_arch", 145.0),
        ("three_arm_vane", 265.0),
    ],
)
def test_long_tail_objects_are_asymmetric_and_colored(kind: str, hue: float) -> None:
    spec = SyntheticObjectSpec(
        kind=kind,
        marker=False,
        base_color=oklch_to_srgb(0.70, 0.12, hue),
    )
    render = SyntheticRenderConfig(image_size=32, supersample=2)
    image = render_synthetic_object(object_spec=spec, render=render, azimuth_deg=25.0)
    mirrored = np.flip(image, axis=1)
    assert image.shape == (32, 32, 3)
    assert float(np.mean(np.abs(image - mirrored))) > 0.005


def test_long_tail_object_silhouettes_are_pairwise_distinct() -> None:
    render = SyntheticRenderConfig(image_size=32, supersample=2)
    images = [
        render_synthetic_object(
            object_spec=SyntheticObjectSpec(kind=kind, marker=False),
            render=render,
            azimuth_deg=35.0,
            render_mode="silhouette",
        )
        for kind in ("stepped_monument", "crooked_arch", "three_arm_vane")
    ]
    distances = [
        float(np.mean(np.abs(images[left] - images[right])))
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    assert min(distances) > 0.03
```

- [ ] **Step 2: Run the tests and verify unsupported-object failures**

Run:

```bash
pytest tests/test_synthetic_objects.py \
  -k 'long_tail_object or long_tail_objects' -v
```

Expected: failures report unsupported object kinds and missing color conversion.

- [ ] **Step 3: Add the material conversion and shape registry**

Add the new kinds to `SUPPORTED_OBJECT_KINDS`, parse `object.base_color`, and add:

```python
def oklch_to_srgb(
    lightness: float,
    chroma: float,
    hue_degrees: float,
) -> tuple[float, float, float]:
    hue = math.radians(float(hue_degrees))
    lab_a = float(chroma) * math.cos(hue)
    lab_b = float(chroma) * math.sin(hue)
    l_root = float(lightness) + 0.3963377774 * lab_a + 0.2158037573 * lab_b
    m_root = float(lightness) - 0.1055613458 * lab_a - 0.0638541728 * lab_b
    s_root = float(lightness) - 0.0894841775 * lab_a - 1.2914855480 * lab_b
    l_value, m_value, s_value = l_root**3, m_root**3, s_root**3
    linear = np.asarray(
        [
            4.0767416621 * l_value - 3.3077115913 * m_value + 0.2309699292 * s_value,
            -1.2684380046 * l_value + 2.6097574011 * m_value - 0.3413193965 * s_value,
            -0.0041960863 * l_value - 0.7034186147 * m_value + 1.707614701 * s_value,
        ]
    )
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    return tuple(float(value) for value in np.clip(srgb, 0.0, 1.0))


def _uniform_colors(spec: SyntheticObjectSpec) -> tuple[tuple[float, float, float], ...]:
    color = spec.base_color or (0.55, 0.55, 0.55)
    return (color,) * 6


def _combine_boxes(
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    *,
    scale: float,
    colors: tuple[tuple[float, float, float], ...],
    prefix: str,
) -> tuple[_CubeFace, ...]:
    faces: list[_CubeFace] = []
    for index, (center, size) in enumerate(boxes):
        faces.extend(
            _box_faces(
                center=tuple(scale * value for value in center),
                size=tuple(scale * value for value in size),
                colors=colors,
                prefix=f"{prefix}_{index}",
            )
        )
    return tuple(faces)
```

Define the shapes with unequal parts and a persistent negative-space difference:

```python
def _stepped_monument_faces(spec: SyntheticObjectSpec) -> tuple[_CubeFace, ...]:
    return _combine_boxes(
        (
            ((0.0, 0.0, -0.55), (0.90, 0.70, 0.28)),
            ((-0.16, 0.04, -0.08), (0.48, 0.46, 0.82)),
            ((0.20, -0.06, 0.43), (0.62, 0.34, 0.22)),
            ((0.34, 0.02, 0.70), (0.18, 0.22, 0.38)),
        ),
        scale=spec.scale,
        colors=_uniform_colors(spec),
        prefix="stepped_monument",
    )


def _crooked_arch_faces(spec: SyntheticObjectSpec) -> tuple[_CubeFace, ...]:
    return _combine_boxes(
        (
            ((-0.42, 0.0, -0.05), (0.24, 0.40, 1.25)),
            ((0.38, 0.02, -0.22), (0.34, 0.34, 0.92)),
            ((-0.02, 0.0, 0.58), (1.02, 0.34, 0.22)),
            ((0.45, -0.16, 0.43), (0.22, 0.28, 0.34)),
        ),
        scale=spec.scale,
        colors=_uniform_colors(spec),
        prefix="crooked_arch",
    )


def _three_arm_vane_faces(spec: SyntheticObjectSpec) -> tuple[_CubeFace, ...]:
    return _combine_boxes(
        (
            ((0.0, 0.0, 0.0), (0.34, 0.34, 0.34)),
            ((0.48, 0.0, 0.03), (0.78, 0.18, 0.20)),
            ((-0.14, 0.0, 0.50), (0.20, 0.22, 0.82)),
            ((-0.32, 0.26, -0.28), (0.58, 0.18, 0.22)),
            ((0.68, -0.12, 0.18), (0.16, 0.34, 0.42)),
        ),
        scale=spec.scale,
        colors=_uniform_colors(spec),
        prefix="three_arm_vane",
    )
```

Route all three kinds in `_object_faces`.

- [ ] **Step 4: Run object and renderer regressions**

Run:

```bash
pytest tests/test_synthetic_objects.py tests/test_synthetic_factor_framework.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Render a 3-by-5 acceptance grid**

Run a small Python command through the project environment that renders each object at
azimuths `[-120, -60, 0, 60, 120]` and writes
`outputs/synthetic_long_tail_geometry/object_acceptance.png`.

Expected: one image exists with 15 non-empty panels; no two rows share the same
silhouette family.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/geometry_explorer/synthetic_objects.py \
  fm_lab/geometry_explorer/render_maps.py \
  tests/test_synthetic_objects.py
git commit -m "feat: add distinct synthetic long-tail objects"
```

### Task 3: Encode the factorial design and build shared master pools

**Files:**
- Create: `fm_lab/geometry_explorer/synthetic_long_tail_design.py`
- Create: `fm_lab/geometry_explorer/synthetic_long_tail_calibration.py`
- Create: `scripts/generate_synthetic_long_tail.py`
- Create: `configs/synthetic_long_tail_geometry/experiment.yaml`
- Create: `tests/test_synthetic_long_tail_design.py`
- Create: `tests/test_synthetic_long_tail_calibration.py`

**Interfaces:**
- Produces: `ConditionManifest`, `ConditionClass`, `PoolCellManifest` dataclasses.
- Produces: `build_factor_space(level: str) -> LatentFactorSpace`.
- Produces: `canonical_factor_rows(factor, values) -> np.ndarray`.
- Produces: `build_condition_manifests(root, replicate, pool_cells) -> tuple[Path, ...]`.
- Produces: `build_master_pools(config, root, replicate) -> tuple[PoolCellManifest, ...]`.
- Produces: `calibrate_renderer(config, output_dir) -> dict[str, Any]`.
- Pool arrays: `images.npy` uint8 `(5000, 3, 32, 32)` and `factors.npy`
  float32 `(5000, 5)` with columns `tx,ty,tz,azimuth,elevation`.
- Consumed by: `SyntheticLongTailImages` in Task 4.

- [ ] **Step 1: Write failing mapping and nested-pool tests**

```python
def test_design_config(
    *,
    master_count: int,
    counts: tuple[int, int, int],
    image_size: int,
) -> dict[str, Any]:
    return {
        "seed": 17,
        "image_size": image_size,
        "master_count": master_count,
        "counts": list(counts),
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {"supersample": 1, "render_batch_size": 8},
    }


def test_factorial_conditions_cover_every_object_dimension_frequency_cell() -> None:
    conditions = build_condition_specs(replicate=0)
    assert len(conditions) == 12
    imbalanced = [item for item in conditions if item.frequency_mapping != "balanced"]
    observed = {
        (entry.object_id, entry.dimension_id, entry.count)
        for condition in imbalanced
        for entry in condition.classes
    }
    assert observed == {
        (object_id, dimension_id, count)
        for object_id in OBJECT_IDS
        for dimension_id in DIMENSION_IDS
        for count in (5000, 500, 50)
    }


def test_tiny_master_pool_is_uint8_and_condition_views_are_nested(tmp_path: Path) -> None:
    config = test_design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    cells = build_master_pools(config, tmp_path, replicate=0)
    paths = build_condition_manifests(tmp_path, 0, cells, counts=(20, 5, 2))
    assert len(cells) == 9
    assert len(paths) == 12
    image_array = np.load(Path(cells[0].image_path), mmap_mode="r")
    assert image_array.dtype == np.uint8
    assert image_array.shape == (20, 3, 16, 16)
    for path in paths:
        manifest = ConditionManifest.read(path)
        for entry in manifest.classes:
            assert entry.index_start == 0
            assert entry.count in {20, 5, 2}
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

```bash
pytest tests/test_synthetic_long_tail_design.py -v
```

Expected: collection fails because `synthetic_long_tail_design` does not exist.

- [ ] **Step 3: Implement exact design constants and manifest types**

```python
OBJECT_IDS = ("stepped_monument", "crooked_arch", "three_arm_vane")
DIMENSION_IDS = ("high", "medium", "low")
FACTOR_COLUMNS = ("tx", "ty", "tz", "azimuth", "elevation")
GEOMETRY_MAPPINGS = (
    ("high", "medium", "low"),
    ("medium", "low", "high"),
    ("low", "high", "medium"),
)
FREQUENCY_MAPPINGS = (
    (5000, 500, 50),
    (500, 50, 5000),
    (50, 5000, 500),
)


@dataclass(frozen=True)
class ConditionClass:
    class_id: int
    object_id: str
    dimension_id: str
    true_dimension: int
    count: int
    image_path: str
    factor_path: str
    index_start: int = 0


@dataclass(frozen=True)
class ConditionManifest:
    condition_id: str
    replicate: int
    geometry_mapping: str
    frequency_mapping: str
    image_shape: tuple[int, int, int]
    classes: tuple[ConditionClass, ...]
    config_hash: str

    def write(self, path: Path) -> Path:
        write_json(asdict(self), path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> "ConditionManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["image_shape"] = tuple(raw["image_shape"])
        raw["classes"] = tuple(ConditionClass(**item) for item in raw["classes"])
        return cls(**raw)
```

Implement `build_factor_space` with the approved ranges:

```python
def build_factor_space(level: str) -> LatentFactorSpace:
    translation_xyz = BoundedTranslation(
        dim=3,
        bounds=((-0.25, 0.25), (-0.25, 0.25), (-0.75, 0.75)),
        name="translation_xyz",
    )
    if level == "low":
        return CameraDepthTranslationInterval(bounds=(-0.75, 0.75))
    if level == "medium":
        return translation_xyz
    if level == "high":
        return ProductFactorSpace(
            [translation_xyz, BoundedLookAtView()],
            name="translation_xyz_bounded_view",
        )
    raise ValueError(f"Unsupported dimension level: {level}")
```

- [ ] **Step 4: Implement deterministic pool rendering and condition manifests**

For every replicate and object-dimension cell, sample one 5,000-item factor pool,
render it with `RenderMap`, transpose HWC to CHW, quantize once, and save with
`open_memmap`:

```python
def canonical_factor_rows(
    factor: LatentFactorSpace,
    values: Sequence[Any],
) -> np.ndarray:
    rows = np.full((len(values), len(FACTOR_COLUMNS)), np.nan, dtype=np.float32)
    for row_id, value in enumerate(values):
        coordinates = factor.coordinates(value)
        rows[row_id, 0] = coordinates.get("translation_x", np.nan)
        rows[row_id, 1] = coordinates.get("translation_y", np.nan)
        rows[row_id, 2] = coordinates.get("translation_z", np.nan)
        rows[row_id, 3] = coordinates.get("camera_azimuth", np.nan)
        rows[row_id, 4] = coordinates.get("camera_elevation", np.nan)
    return rows


images = np.lib.format.open_memmap(
    cell_dir / "images.npy",
    mode="w+",
    dtype=np.uint8,
    shape=(master_count, 3, image_size, image_size),
)
factors = np.lib.format.open_memmap(
    cell_dir / "factors.npy",
    mode="w+",
    dtype=np.float32,
    shape=(master_count, len(FACTOR_COLUMNS)),
)
for start in range(0, master_count, render_batch_size):
    stop = min(master_count, start + render_batch_size)
    rendered = render_map.render_batch(list(values[start:stop]), batch_size=render_batch_size)
    images[start:stop] = np.rint(
        np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    factors[start:stop] = canonical_factor_rows(factor, values[start:stop])
images.flush()
factors.flush()
```

Use seed `base_seed + replicate * 100_000 + object_index * 1_000 +
dimension_index * 10`. Condition manifests use relative paths and prefix counts;
they do not copy arrays.

- [ ] **Step 5: Add the frozen experiment configuration and dry-run CLI**

Create `experiment.yaml` with:

```yaml
experiment: synthetic_long_tail_geometry
seed: 17072026
output_root: outputs/synthetic_long_tail_geometry
image_size: 32
replicates: 3
master_count: 5000
counts: [5000, 500, 50]
objects:
  - id: stepped_monument
    hue_degrees: 25.0
    scale: 1.0
  - id: crooked_arch
    hue_degrees: 145.0
    scale: 1.0
  - id: three_arm_vane
    hue_degrees: 265.0
    scale: 1.0
material:
  oklch_lightness: 0.70
  oklch_chroma: 0.12
render:
  background: [1.0, 1.0, 1.0]
  camera_distance: 4.0
  elevation_bounds_degrees: [-30.0, 30.0]
  supersample: 3
  render_batch_size: 128
calibration:
  renderer_points_per_cell: 256
  relative_singular_threshold: 0.02
  full_rank_fraction: 0.95
  max_pullback_norm_ratio: 4.0
  max_nuisance_standardized_difference: 0.25
oracle:
  training_samples_per_object: 30000
  validation_samples_per_object: 5000
  batch_size: 256
  steps: 20000
  learning_rate: 0.001
  min_object_accuracy: 0.99
  max_normalized_factor_mae: 0.02
evaluation:
  samples_per_class: 5000
  geometry_queries_per_class: 256
  joint_metric_samples: 1000
  fm_directions: 16
  fm_nfe: 32
  bootstrap_draws: 10000
```

The script accepts `--config`, `--output-root`, `--replicate`, and `--dry-run`.

- [ ] **Step 6: Implement and test renderer calibration gates**

Use independent reference renders for all nine object-dimension cells. Compute
foreground occupancy, luminance, contrast, raw-pixel logistic-regression object
accuracy, finite-difference renderer singular values in normalized factor
coordinates, and median per-factor pullback norms.

```python
@dataclass(frozen=True)
class RendererGateThresholds:
    min_object_accuracy: float = 0.99
    max_nuisance_standardized_difference: float = 0.25
    relative_singular_threshold: float = 0.02
    min_full_rank_fraction: float = 0.95
    max_pullback_norm_ratio: float = 4.0


def renderer_gate(
    *,
    object_accuracy: float,
    max_nuisance_difference: float,
    full_rank_fraction: float,
    pullback_norm_ratio: float,
    thresholds: RendererGateThresholds,
) -> dict[str, Any]:
    checks = {
        "object_separability": object_accuracy >= thresholds.min_object_accuracy,
        "nuisance_matching": (
            max_nuisance_difference <= thresholds.max_nuisance_standardized_difference
        ),
        "renderer_rank": full_rank_fraction >= thresholds.min_full_rank_fraction,
        "factor_visibility": pullback_norm_ratio <= thresholds.max_pullback_norm_ratio,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "object_accuracy": object_accuracy,
        "max_nuisance_standardized_difference": max_nuisance_difference,
        "full_rank_fraction": full_rank_fraction,
        "pullback_norm_ratio": pullback_norm_ratio,
    }
```

Write tests in which each threshold is violated independently and assert the exact
failed check. The calibration service writes `renderer_gate.json`, a class-statistics
CSV, and compressed singular values without mutating master pools.

- [ ] **Step 7: Run design, calibration, and dry-run checks**

Run:

```bash
pytest tests/test_synthetic_long_tail_design.py \
  tests/test_synthetic_long_tail_calibration.py -v
python scripts/generate_synthetic_long_tail.py \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  --replicate 0 --dry-run
```

Expected: tests pass; dry run prints 9 pool cells and 12 condition manifests
without creating image arrays.

- [ ] **Step 8: Commit**

```bash
git add fm_lab/geometry_explorer/synthetic_long_tail_design.py \
  fm_lab/geometry_explorer/synthetic_long_tail_calibration.py \
  scripts/generate_synthetic_long_tail.py \
  configs/synthetic_long_tail_geometry/experiment.yaml \
  tests/test_synthetic_long_tail_design.py \
  tests/test_synthetic_long_tail_calibration.py
git commit -m "feat: add synthetic long-tail factorial pools"
```

### Task 4: Add the memory-mapped synthetic training target

**Files:**
- Create: `fm_lab/data/synthetic_long_tail.py`
- Modify: `fm_lab/data/__init__.py`
- Modify: `fm_lab/experiments/factory.py:64-125`
- Create: `tests/test_synthetic_long_tail_data.py`

**Interfaces:**
- Produces: `SyntheticLongTailImages(condition_manifest, normalize, dequantize)`.
- Implements: `sample`, `sample_with_labels`, `all_samples_with_labels`,
  `log_prob`, and `metadata`.
- Exposes: `dim`, `image_shape`, and `class_counts`.
- Consumed by: existing `build_target`, `train_flow_matching`, and Task 5 configs.

- [ ] **Step 1: Write failing loader and empirical-sampling tests**

```python
def write_tiny_condition(
    root: Path,
    *,
    counts: tuple[int, int, int],
) -> Path:
    config = {
        "seed": 17,
        "image_size": 16,
        "master_count": max(counts),
        "counts": list(counts),
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {"supersample": 1, "render_batch_size": 8},
    }
    cells = build_master_pools(config, root, replicate=0)
    manifests = build_condition_manifests(root, 0, cells, counts=counts)
    return next(path for path in manifests if path.stem == "g0_f0")


def test_synthetic_target_loads_indexed_prefixes(tmp_path: Path) -> None:
    manifest_path = write_tiny_condition(tmp_path, counts=(20, 5, 2))
    target = SyntheticLongTailImages(manifest_path, normalize="minus_one_one")
    assert target.dim == 3 * 16 * 16
    assert target.image_shape == (3, 16, 16)
    assert target.class_counts == (20, 5, 2)
    images, labels, source_ids = target.all_samples_with_labels()
    assert images.shape == (27, 3 * 16 * 16)
    assert torch.bincount(labels, minlength=3).tolist() == [20, 5, 2]
    assert len(np.unique(source_ids)) == 27
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0


def test_synthetic_target_sampling_follows_empirical_frequency(tmp_path: Path) -> None:
    target = SyntheticLongTailImages(
        write_tiny_condition(tmp_path, counts=(200, 20, 2)),
        normalize="zero_one",
    )
    _, labels = target.sample_with_labels(20_000)
    frequencies = torch.bincount(labels, minlength=3).float() / len(labels)
    expected = torch.tensor([200.0, 20.0, 2.0]) / 222.0
    assert torch.max(torch.abs(frequencies - expected)) < 0.015
```

- [ ] **Step 2: Run the tests and verify the missing target failure**

Run:

```bash
pytest tests/test_synthetic_long_tail_data.py -v
```

Expected: collection fails because `SyntheticLongTailImages` does not exist.

- [ ] **Step 3: Implement the indexed target**

Use `np.load(..., mmap_mode="r")` per class. Sample a global integer from
`[0, sum(class_counts))`, map it through cumulative counts, and materialize only
the selected batch:

```python
@dataclass
class SyntheticLongTailImages:
    condition_manifest: str | Path
    normalize: str = "minus_one_one"
    dequantize: bool = False
    name: str = "synthetic_long_tail_geometry"
    dim: int = field(default=0, init=False)
    image_shape: tuple[int, ...] = field(default=(), init=False)
    class_counts: tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        self._manifest_path = Path(self.condition_manifest).expanduser().resolve()
        self._manifest = ConditionManifest.read(self._manifest_path)
        self.image_shape = self._manifest.image_shape
        self.dim = int(np.prod(self.image_shape))
        self.class_counts = tuple(entry.count for entry in self._manifest.classes)
        self._arrays = tuple(
            np.load(self._resolve(entry.image_path), mmap_mode="r")
            for entry in self._manifest.classes
        )
        self._offsets = np.cumsum((0, *self.class_counts), dtype=np.int64)

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("SyntheticLongTailImages.sample requires n >= 1.")
        global_indices = np.random.randint(0, int(self._offsets[-1]), size=int(n))
        labels = np.searchsorted(self._offsets[1:], global_indices, side="right")
        output = np.empty((n, *self.image_shape), dtype=np.uint8)
        for class_id, (entry, array) in enumerate(zip(self._manifest.classes, self._arrays)):
            mask = labels == class_id
            local = global_indices[mask] - self._offsets[class_id] + entry.index_start
            output[mask] = np.asarray(array[local], dtype=np.uint8)
        images = self._normalize(torch.from_numpy(output).reshape(n, -1))
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        if device is not None:
            images = images.to(device)
            label_tensor = label_tensor.to(device)
        return images, label_tensor

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        images, _ = self.sample_with_labels(n, device=device)
        return images

    def all_samples_with_labels(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        image_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        for class_id, (entry, array) in enumerate(zip(self._manifest.classes, self._arrays)):
            indices = np.arange(entry.index_start, entry.index_start + entry.count)
            image_parts.append(np.asarray(array[indices], dtype=np.uint8))
            label_parts.append(np.full(entry.count, class_id, dtype=np.int64))
            source_parts.append((np.int64(class_id) << 48) | indices.astype(np.int64))
        images = self._normalize(
            torch.from_numpy(np.concatenate(image_parts)).reshape(-1, self.dim)
        )
        labels = torch.from_numpy(np.concatenate(label_parts))
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return images, labels, np.concatenate(source_parts)

    def log_prob(self, x: torch.Tensor) -> None:
        del x
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition_id": self._manifest.condition_id,
            "condition_manifest": str(self._manifest_path),
            "dim": self.dim,
            "image_shape": list(self.image_shape),
            "class_counts": list(self.class_counts),
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "config_hash": self._manifest.config_hash,
        }

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._manifest_path.parent / candidate

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        values = images.to(dtype=torch.float32) / 255.0
        if self.dequantize:
            values = torch.clamp(values + torch.rand_like(values) / 256.0, 0.0, 1.0)
        if self.normalize == "zero_one":
            return values
        if self.normalize == "minus_one_one":
            return 2.0 * values - 1.0
        raise ValueError(f"Unsupported image normalization: {self.normalize}")
```

Use identifiers `(class_id << 48) | pool_index` in
`all_samples_with_labels` so source IDs remain unique without copied index files.

- [ ] **Step 4: Register the target in the factory**

```python
if name == "synthetic_long_tail_geometry":
    manifest = data_config.get("condition_manifest")
    if not manifest:
        raise ValueError("data.condition_manifest is required.")
    return SyntheticLongTailImages(
        condition_manifest=manifest,
        normalize=str(data_config.get("normalize", "minus_one_one")),
        dequantize=bool(data_config.get("dequantize", False)),
    )
```

- [ ] **Step 5: Run target, factory, and training-contract tests**

Run:

```bash
pytest tests/test_synthetic_long_tail_data.py \
  tests/test_config_smoke.py \
  tests/test_training_sampling.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/data/synthetic_long_tail.py fm_lab/data/__init__.py \
  fm_lab/experiments/factory.py tests/test_synthetic_long_tail_data.py
git commit -m "feat: load indexed synthetic long-tail targets"
```

### Task 5: Generate pilot and main training configurations

**Files:**
- Create: `configs/synthetic_long_tail_geometry/base_train.yaml`
- Create: `fm_lab/experiments/synthetic_long_tail_geometry.py`
- Create: `tests/test_synthetic_long_tail_training.py`

**Interfaces:**
- Produces: `matched_pass_step(total_steps, dataset_size, batch_size) -> int`.
- Produces: `write_condition_training_configs(...) -> tuple[Path, ...]`.
- Produces one pilot configuration with checkpoints `5000,10000,20000,40000`.
- Produces main configurations with `S_*` and the exact matched-pass checkpoint.
- Consumed by: stage CLI in Task 9.

- [ ] **Step 1: Write failing checkpoint and config tests**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_tiny_factorial_manifests(root: Path) -> tuple[Path, ...]:
    config = {
        "seed": 17,
        "image_size": 16,
        "master_count": 20,
        "counts": [20, 5, 2],
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {"supersample": 1, "render_batch_size": 8},
    }
    cells = build_master_pools(config, root, replicate=0)
    return build_condition_manifests(root, 0, cells, counts=(20, 5, 2))


def test_matched_pass_checkpoint_uses_balanced_example_passes() -> None:
    assert matched_pass_step(20_000, dataset_size=5_550, batch_size=256) == 7_400
    assert matched_pass_step(20_000, dataset_size=15_000, batch_size=256) == 20_000


def test_condition_config_freezes_model_and_changes_only_design_fields(tmp_path: Path) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    paths = write_condition_training_configs(
        base_config_path=PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml",
        condition_manifests=manifests,
        output_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        total_steps=20_000,
        batch_size=256,
        model_seed=17,
    )
    configs = [load_config(path) for path in paths]
    assert len(configs) == 12
    assert {tuple(item["training"]["checkpoint_steps"]) for item in configs} == {
        (7_400, 20_000),
        (20_000,),
    }
    assert len({json.dumps(item["model"], sort_keys=True) for item in configs}) == 1
    assert all(item["training"]["early_stopping"]["enabled"] is False for item in configs)
```

- [ ] **Step 2: Run the focused tests and verify missing functions**

Run:

```bash
pytest tests/test_synthetic_long_tail_training.py -v
```

Expected: collection fails on missing configuration helpers.

- [ ] **Step 3: Add the shared baseline configuration**

```yaml
experiment:
  name: synthetic_long_tail_geometry
  seed: 0
  output_dir: runs/synthetic_long_tail_geometry/manual
data:
  name: synthetic_long_tail_geometry
  condition_manifest: outputs/synthetic_long_tail_geometry/replicate_00/conditions/g0_balanced.json
  normalize: minus_one_one
  dequantize: false
source:
  name: gaussian
  dim: 3072
coupling:
  name: independent
path:
  name: linear
model:
  name: image_unet
  image_shape: [3, 32, 32]
  base_channels: 32
  time_embedding_dim: 128
  activation: silu
  zero_init_head: true
conditioning:
  enabled: true
  num_classes: 3
  embedding_dim: 128
  dropout_probability: 0.15
objective:
  name: flow_matching
  model_output: target
  loss_space: velocity
  min_denom: 0.05
  modifiers: []
training:
  batch_size: 256
  steps: 40000
  checkpoint_steps: [5000, 10000, 20000, 40000]
  lr: 0.0001
  warmup_steps: 500
  time_sampling:
    name: logit_normal
    mean: -0.8
    std: 0.8
  log_every: 250
  early_stopping:
    enabled: false
solvers:
  names: [euler]
  nfes: [64]
  schedule: uniform
sampling:
  n_samples: 15000
  n_trajectories: 18
  nfe: 64
  sample_batch_size: 500
  classes: [0, 1, 2]
  seed: 19072026
  classifier_free_guidance:
    scale: 1.0
```

- [ ] **Step 4: Implement deterministic config generation**

`matched_pass_step` uses the balanced 15,000-example reference exactly:

```python
def matched_pass_step(total_steps: int, dataset_size: int, batch_size: int) -> int:
    del batch_size
    return int(math.floor(int(total_steps) * int(dataset_size) / 15_000))
```

For each condition, deep-copy the base config and change only experiment name,
output directory, seeds, condition manifest, total steps, and explicit checkpoint
steps. Save a `config_hash` over the normalized mapping beside each YAML.

- [ ] **Step 5: Run configuration and component-construction tests**

Run:

```bash
pytest tests/test_synthetic_long_tail_training.py tests/test_config_smoke.py -v
```

Expected: all tests pass; each generated config builds a 3-class target, Gaussian
source of dimension 3,072, and a class-conditioned 3-by-32-by-32 image U-Net. The
end-to-end training smoke is Task 12 after the stage runner exists.

- [ ] **Step 6: Commit**

```bash
git add configs/synthetic_long_tail_geometry/base_train.yaml \
  fm_lab/experiments/synthetic_long_tail_geometry.py \
  tests/test_synthetic_long_tail_training.py
git commit -m "feat: generate synthetic geometry training matrix"
```

### Task 6: Train and validate the independent factor oracle

**Files:**
- Create: `fm_lab/geometry_explorer/synthetic_factor_oracle.py`
- Create: `tests/test_synthetic_factor_oracle.py`

**Interfaces:**
- Produces: `SyntheticFactorOracle(nn.Module)`.
- Produces: `OraclePrediction(class_logits, translation, view, features)`.
- Produces: `train_factor_oracle(config, output_dir, device) -> dict[str, Any]`.
- Produces: `load_factor_oracle(checkpoint, device) -> SyntheticFactorOracle`.
- Produces gate metrics object accuracy, per-factor normalized MAE, and the 99.5th
  percentile re-render residual.
- Consumed by: Task 7 and Task 8 evaluators.

- [ ] **Step 1: Write failing model, circular-loss, and gate tests**

```python
def test_oracle_output_shapes() -> None:
    model = SyntheticFactorOracle(num_classes=3)
    prediction = model(torch.zeros(4, 3, 32, 32))
    assert prediction.class_logits.shape == (4, 3)
    assert prediction.translation.shape == (4, 3)
    assert prediction.view.shape == (4, 3)
    assert prediction.features.shape == (4, 256)


def test_oracle_circular_error_wraps_at_pi() -> None:
    predicted = torch.tensor([[math.sin(-math.pi + 0.01), math.cos(-math.pi + 0.01)]])
    target = torch.tensor([[math.sin(math.pi - 0.01), math.cos(math.pi - 0.01)]])
    assert float(circular_vector_error(predicted, target)) < 0.03


def test_oracle_gate_rejects_one_bad_factor() -> None:
    metrics = oracle_gate_metrics(
        object_accuracy=0.999,
        factor_mae={"tx": 0.01, "ty": 0.01, "tz": 0.01, "azimuth": 0.03, "elevation": 0.01},
        min_accuracy=0.99,
        max_factor_mae=0.02,
    )
    assert metrics["passed"] is False
    assert metrics["failed_factors"] == ["azimuth"]
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

```bash
pytest tests/test_synthetic_factor_oracle.py -v
```

Expected: collection fails because the oracle module does not exist.

- [ ] **Step 3: Implement the compact oracle**

```python
@dataclass(frozen=True)
class OraclePrediction:
    class_logits: torch.Tensor
    translation: torch.Tensor
    view: torch.Tensor
    features: torch.Tensor


class SyntheticFactorOracle(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.class_head = nn.Linear(256, num_classes)
        self.translation_head = nn.Linear(256, 3)
        self.view_head = nn.Linear(256, 3)

    def forward(self, images: torch.Tensor) -> OraclePrediction:
        features = self.encoder(images)
        view = self.view_head(features)
        azimuth = F.normalize(view[:, :2], dim=1)
        return OraclePrediction(
            class_logits=self.class_head(features),
            translation=torch.tanh(self.translation_head(features)),
            view=torch.cat([azimuth, torch.tanh(view[:, 2:3])], dim=1),
            features=features,
        )
```

Apply gates without averaging away a failed coordinate:

```python
def oracle_gate_metrics(
    *,
    object_accuracy: float,
    factor_mae: Mapping[str, float],
    min_accuracy: float,
    max_factor_mae: float,
) -> dict[str, Any]:
    failed = sorted(
        name for name, value in factor_mae.items() if float(value) > max_factor_mae
    )
    return {
        "passed": object_accuracy >= min_accuracy and not failed,
        "object_accuracy": float(object_accuracy),
        "factor_mae": {name: float(value) for name, value in factor_mae.items()},
        "failed_factors": failed,
    }
```

Train on 30,000 independently seeded high-dimensional renders per object, validate
on 5,000 per object, and never read a master training pool. Use cross-entropy plus
unit-weight normalized translation MSE, azimuth-vector MSE, and elevation MSE.

- [ ] **Step 4: Implement checkpoint and validation gates**

Save `model_state_dict`, architecture, renderer config hash, training seed, and
factor normalization. Convert translation predictions back to world ranges before
re-rendering. Define off-renderer threshold as the 99.5th percentile of validation
pixel MAE between held-out images and their prediction re-renders.

- [ ] **Step 5: Run oracle model, loss, checkpoint, and gate tests**

Run:

```bash
pytest tests/test_synthetic_factor_oracle.py -v
```

Expected: all tests pass. A fixture training run writes an oracle checkpoint and a
gate JSON whose schema lists all five factor errors and the failure reason.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/geometry_explorer/synthetic_factor_oracle.py \
  tests/test_synthetic_factor_oracle.py
git commit -m "feat: add independent synthetic factor oracle"
```

### Task 7: Implement primary factor-distribution and quality metrics

**Files:**
- Create: `fm_lab/geometry_explorer/synthetic_long_tail_metrics.py`
- Create: `tests/test_synthetic_long_tail_metrics.py`

**Interfaces:**
- Produces: `normalized_wasserstein`, `central_range_ratio`,
  `multivariate_energy_distance`, and `oracle_feature_fid`.
- Produces: `deterministic_subsample` and `summarize_validity`.
- Produces: `evaluate_generated_distribution(...) -> dict[str, Any]`.
- Produces: `calibrate_metric_controls(...) -> dict[str, Any]`.
- Consumes generated `samples/euler_nfe64.npy`, `generated_labels.npy`, the oracle
  checkpoint, and independent renderer references.
- Writes `evaluation/factor_metrics.json` and class-level CSV.

- [ ] **Step 1: Write failing known-collapse and invalid-mass tests**

```python
def test_factor_metrics_order_full_half_and_collapsed_controls() -> None:
    rng = np.random.default_rng(3)
    reference = rng.uniform(-1.0, 1.0, 5000)
    full = rng.uniform(-1.0, 1.0, 5000)
    half = rng.uniform(-0.5, 0.5, 5000)
    collapsed = np.zeros(5000)
    errors = [normalized_wasserstein(values, reference, value_range=2.0) for values in (full, half, collapsed)]
    assert errors[0] < errors[1] < errors[2]
    ratios = [central_range_ratio(values, reference) for values in (full, half, collapsed)]
    assert ratios[0] > ratios[1] > ratios[2]


def test_distribution_summary_keeps_invalid_mass_visible() -> None:
    summary = summarize_validity(
        predicted_class=np.asarray([0, 0, 1, 0]),
        requested_class=np.asarray([0, 0, 0, 0]),
        render_residual=np.asarray([0.01, 0.50, 0.01, 0.02]),
        residual_threshold=0.05,
    )
    assert summary == {
        "class_leakage_rate": 0.25,
        "off_renderer_rate": 0.25,
        "joint_valid_rate": 0.50,
    }
```

- [ ] **Step 2: Run tests and verify missing metric functions**

Run:

```bash
pytest tests/test_synthetic_long_tail_metrics.py -v
```

Expected: collection fails on missing metric imports.

- [ ] **Step 3: Implement deterministic metrics**

```python
def normalized_wasserstein(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    value_range: float,
) -> float:
    if value_range <= 0:
        raise ValueError("value_range must be positive.")
    return float(wasserstein_distance(generated, reference) / value_range)


def central_range_ratio(generated: np.ndarray, reference: np.ndarray) -> float:
    generated_width = float(np.quantile(generated, 0.95) - np.quantile(generated, 0.05))
    reference_width = float(np.quantile(reference, 0.95) - np.quantile(reference, 0.05))
    return generated_width / max(reference_width, np.finfo(np.float64).eps)


def multivariate_energy_distance(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 1000,
    seed: int = 0,
) -> float:
    generated = deterministic_subsample(generated, max_samples, seed)
    reference = deterministic_subsample(reference, max_samples, seed + 1)
    cross = cdist(generated, reference).mean()
    within_generated = pdist(generated).mean()
    within_reference = pdist(reference).mean()
    return float(2.0 * cross - within_generated - within_reference)


def deterministic_subsample(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= maximum:
        return values
    indices = np.random.default_rng(seed).choice(len(values), size=maximum, replace=False)
    return values[np.sort(indices)]


def summarize_validity(
    *,
    predicted_class: np.ndarray,
    requested_class: np.ndarray,
    render_residual: np.ndarray,
    residual_threshold: float,
) -> dict[str, float]:
    class_valid = np.asarray(predicted_class) == np.asarray(requested_class)
    renderer_valid = np.asarray(render_residual) <= float(residual_threshold)
    return {
        "class_leakage_rate": float(1.0 - class_valid.mean()),
        "off_renderer_rate": float(1.0 - renderer_valid.mean()),
        "joint_valid_rate": float(np.mean(class_valid & renderer_valid)),
    }
```

Compute oracle-feature FID from means and covariances using `scipy.linalg.sqrtm`,
discarding only negligible imaginary numerical residue below `1e-6`.

- [ ] **Step 4: Implement checkpoint evaluation and positive controls**

Evaluate exactly 5,000 generated samples per requested class. Report metrics both on
all requested samples and on the joint-valid subset, always adjacent to leakage,
off-renderer, and joint-valid rates. Generate independent full-, half-, and
collapsed-factor controls using the same renderer and oracle.

- [ ] **Step 5: Run metrics tests**

Run:

```bash
pytest tests/test_synthetic_long_tail_metrics.py -v
```

Expected: all tests pass, including strict control ordering.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/geometry_explorer/synthetic_long_tail_metrics.py \
  tests/test_synthetic_long_tail_metrics.py
git commit -m "feat: measure synthetic factor coverage"
```

### Task 8: Add local geometry and memorization diagnostics

**Files:**
- Modify: `fm_lab/diagnostics/fm_lid.py:127-230`
- Create: `fm_lab/geometry_explorer/synthetic_long_tail_geometry.py`
- Create: `tests/test_synthetic_long_tail_geometry.py`
- Modify: `tests/test_fm_lid.py`

**Interfaces:**
- Produces: `FMJacobianSpectrumEstimator.compute_pushforward_matrix(x, t)`.
- Produces: `fixed_class_velocity(model, class_id) -> nn.Module`.
- Produces: `tangent_projection_scores(pushforward, renderer_tangents, rank)`.
- Produces: `evaluate_local_geometry(...) -> dict[str, Any]`.
- Produces: `evaluate_memorization(...) -> dict[str, Any]`.
- Writes per-query parquet/NPZ files plus class summaries.

- [ ] **Step 1: Write a failing pushforward compatibility test**

```python
def test_pushforward_matrix_preserves_existing_spectrum() -> None:
    estimator = FMJacobianSpectrumEstimator(
        model=ZeroVelocity(),
        ode_solver=ProjectionFlowSolver(tangent_dim=2),
        t_values=[0.5],
        eps=1e-2,
        num_directions=16,
        device="cpu",
        nfe=4,
        generator=torch.Generator().manual_seed(3),
    )
    point = torch.tensor([0.2, -0.1, 0.4])
    estimator.generator.manual_seed(3)
    matrix = estimator.compute_pushforward_matrix(point, t=0.5)
    estimator.generator.manual_seed(3)
    spectrum = estimator.compute_spectrum(point, t=0.5)
    assert matrix.ndim == 2
    assert torch.allclose(torch.linalg.svdvals(matrix), spectrum, atol=1e-6)


def test_tangent_projection_detects_preserved_and_missing_direction() -> None:
    pushforward = torch.diag(torch.tensor([2.0, 1.0, 0.001]))
    tangents = torch.eye(3)
    scores = tangent_projection_scores(pushforward, tangents, rank=2)
    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 0.0]), atol=1e-5)


class ContextRecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_labels: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del t
        self.last_labels = context["class_labels"].detach().cpu()
        return torch.zeros_like(x)


def test_fixed_class_velocity_supplies_conditioning_context() -> None:
    model = ContextRecordingVelocity()
    wrapped = fixed_class_velocity(model, class_id=2)
    wrapped(torch.zeros(4, 3), torch.full((4,), 0.5))
    assert torch.equal(model.last_labels, torch.full((4,), 2, dtype=torch.long))
```

- [ ] **Step 2: Run tests and verify missing-method failures**

Run:

```bash
pytest tests/test_fm_lid.py tests/test_synthetic_long_tail_geometry.py -v
```

Expected: failures report missing `compute_pushforward_matrix` and alignment helper.

- [ ] **Step 3: Refactor the FM estimator without changing its spectrum API**

```python
def compute_pushforward_matrix(self, x: torch.Tensor, t: float) -> torch.Tensor:
    x1 = x.to(self.device)
    if x1.ndim < 1:
        raise ValueError("compute_pushforward_matrix expects one data point.")
    x1_batch = x1.unsqueeze(0)
    xt = self._integrate(x1_batch, t0=1.0, t1=float(t))[0]
    directions = sample_unit_directions(
        tuple(xt.shape),
        self.num_directions,
        device=self.device,
        dtype=xt.dtype,
        normalize=self.normalize_directions,
        generator=self.generator,
    )
    x1_perturbed = self._integrate(
        xt.unsqueeze(0) + self.eps * directions,
        t0=float(t),
        t1=1.0,
    )
    base = x1_batch if self.representation_fn is None else self.representation_fn(x1_batch)
    perturbed = x1_perturbed if self.representation_fn is None else self.representation_fn(x1_perturbed)
    return (perturbed.reshape(self.num_directions, -1) - base.reshape(1, -1)).T / self.eps


def compute_spectrum(self, x: torch.Tensor, t: float) -> torch.Tensor:
    return svdvals(self.compute_pushforward_matrix(x, t))
```

- [ ] **Step 4: Implement factor-specific tangent alignment**

```python
def tangent_projection_scores(
    pushforward: torch.Tensor,
    renderer_tangents: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    left, _, _ = torch.linalg.svd(pushforward, full_matrices=False)
    basis = left[:, : int(rank)]
    normalized = F.normalize(renderer_tangents, dim=1)
    return torch.sum((normalized @ basis) ** 2, dim=1).clamp(0.0, 1.0)
```

Wrap the class-conditional velocity model before constructing either geometry
estimator:

```python
class _FixedClassVelocity(nn.Module):
    def __init__(self, model: nn.Module, class_id: int) -> None:
        super().__init__()
        self.model = model
        self.class_id = int(class_id)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        labels = torch.full(
            (len(x),),
            self.class_id,
            dtype=torch.long,
            device=x.device,
        )
        return self.model(x, t, context={"class_labels": labels})


def fixed_class_velocity(model: nn.Module, class_id: int) -> nn.Module:
    return _FixedClassVelocity(model, class_id)
```

After loading the checkpoint, first apply the existing prediction conversion and
then fix the class:

```python
objective = build_objective(config.get("objective", {}))
velocity = velocity_model_for_objective(model, path, objective)
class_velocity = fixed_class_velocity(velocity, class_id)
```

At each of 256 deterministic held-out queries per class, compute renderer finite-
difference tangents in normalized factor coordinates, the 16-direction learned
pushforward matrix with 32 NFEs, its spectrum, factor projection scores, principal
angles, and FM-FLIPD at the configured times. Save only scalar alignment scores and
spectra, not 3,072-by-16 left-vector matrices.

- [ ] **Step 5: Implement memorization measurements**

For every generated sample, store nearest training and held-out distances in raw
factor space and oracle-feature space. Define exact-copy rate by uint8 equality and
near-duplicate rate by the 0.5th percentile of independent held-out-to-held-out
oracle-feature distances. Keep these separate from geometric deficit.

- [ ] **Step 6: Run geometry and existing FM-LID tests**

Run:

```bash
pytest tests/test_fm_lid.py tests/test_synthetic_long_tail_geometry.py \
  tests/test_geometry_explorer.py -v
```

Expected: all tests pass; existing spectrum estimates are unchanged.

- [ ] **Step 7: Commit**

```bash
git add fm_lab/diagnostics/fm_lid.py \
  fm_lab/geometry_explorer/synthetic_long_tail_geometry.py \
  tests/test_fm_lid.py tests/test_synthetic_long_tail_geometry.py
git commit -m "feat: add factor-aligned FM geometry diagnostics"
```

### Task 9: Add the gated stage runner and immutable run ledger

**Files:**
- Modify: `fm_lab/experiments/synthetic_long_tail_geometry.py`
- Create: `fm_lab/experiments/run_synthetic_long_tail_geometry.py`
- Modify: `pyproject.toml`
- Create: `tests/test_synthetic_long_tail_runner.py`

**Interfaces:**
- CLI subcommands: `plan`, `build-pools`, `calibrate-renderer`, `train-oracle`,
  `pilot`, `smoke`, `matrix`, `evaluate`, `aggregate`, `report`.
- Produces: `SyntheticLongTailRunner`, `RunLedger`, `StageBlockedError`,
  `build_matrix_commands`, and `require_gate`.
- Produces: `run_ledger.json` with stage, condition, replicate, config hash, command,
  output path, status, start/end timestamps, and failure text.
- Invokes existing `python -m fm_lab.experiments.run_train` in a fresh process per
  model to release accelerator memory between conditions.

- [ ] **Step 1: Write failing dry-run, gate, and immutability tests**

```python
def test_matrix_dry_run_lists_exactly_36_training_commands(tmp_path: Path) -> None:
    config_paths = {
        replicate: tuple(
            tmp_path / f"rep{replicate:02d}" / f"condition_{condition:02d}.yaml"
            for condition in range(12)
        )
        for replicate in range(3)
    }
    commands = build_matrix_commands(config_paths, run_root=tmp_path / "runs")
    assert len(commands) == 36
    assert len({command.condition_id for command in commands}) == 12
    assert len({command.replicate for command in commands}) == 3


def test_failed_gate_blocks_training(tmp_path: Path) -> None:
    gate_path = tmp_path / "renderer_gate.json"
    write_json({"passed": False, "reasons": ["renderer rank"]}, gate_path)
    with pytest.raises(StageBlockedError, match="renderer rank"):
        require_gate(gate_path, stage="renderer_calibration")


def test_completed_ledger_entry_is_not_overwritten(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "run_ledger.json")
    ledger.complete("rep00_g0_f0", {"metrics": "first.json"})
    with pytest.raises(FileExistsError, match="rep00_g0_f0"):
        ledger.complete("rep00_g0_f0", {"metrics": "second.json"})
```

- [ ] **Step 2: Run tests and verify missing-runner failures**

Run:

```bash
pytest tests/test_synthetic_long_tail_runner.py -v
```

Expected: collection fails because the stage runner is absent.

- [ ] **Step 3: Implement stage state and subprocess commands**

```python
@dataclass(frozen=True)
class TrainingCommand:
    replicate: int
    condition_id: str
    config_path: Path
    run_dir: Path

    def argv(self, device: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "fm_lab.experiments.run_train",
            "--config",
            str(self.config_path),
            "--output-dir",
            str(self.run_dir),
            "--device",
            device,
        ]


def run_training_command(command: TrainingCommand, *, device: str) -> None:
    if command.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {command.run_dir}")
    subprocess.run(command.argv(device), check=True)


def build_matrix_commands(
    config_paths: Mapping[int, Sequence[Path]],
    *,
    run_root: Path,
) -> tuple[TrainingCommand, ...]:
    commands = []
    for replicate, paths in sorted(config_paths.items()):
        for path in paths:
            condition_id = path.stem
            commands.append(
                TrainingCommand(
                    replicate=int(replicate),
                    condition_id=condition_id,
                    config_path=path,
                    run_dir=run_root / f"rep{replicate:02d}" / condition_id,
                )
            )
    return tuple(commands)
```

The runner checks renderer, metric, oracle, and pilot gate JSON before constructing
main-matrix commands. A failed subprocess writes a failed ledger entry and leaves
the partially created run directory intact for diagnosis.

Checkpoint evaluation invokes the existing
`fm_lab.experiments.run_sample_checkpoint` entry point for every explicit pilot or
matched-pass checkpoint, preserving the training configuration and changing only
checkpoint path, sample count, sample seed, and output directory.

- [ ] **Step 4: Add the CLI and project entry point**

Add:

```toml
fm-lab-synthetic-long-tail = "fm_lab.experiments.run_synthetic_long_tail_geometry:main"
```

Every subcommand supports `--config`; training subcommands also support `--device`
and `--dry-run`. `matrix --resume` skips only ledger entries marked complete with a
matching config hash.

- [ ] **Step 5: Run runner tests and inspect the full dry-run matrix**

Run:

```bash
pytest tests/test_synthetic_long_tail_runner.py -v
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  matrix --dry-run
```

Expected: tests pass; output contains 36 unique run directories and no process is
started.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/experiments/synthetic_long_tail_geometry.py \
  fm_lab/experiments/run_synthetic_long_tail_geometry.py \
  pyproject.toml tests/test_synthetic_long_tail_runner.py
git commit -m "feat: orchestrate gated synthetic geometry runs"
```

### Task 10: Aggregate effects and render the living report

**Files:**
- Create: `fm_lab/geometry_explorer/synthetic_long_tail_report.py`
- Create: `docs/research/synthetic_long_tail_geometry_report.md`
- Create: `tests/test_synthetic_long_tail_report.py`
- Modify: `fm_lab/experiments/synthetic_long_tail_geometry.py`

**Interfaces:**
- Produces: `fit_frequency_dimension_effect(frame) -> EffectEstimate`.
- Produces: `paired_hierarchical_bootstrap(frame, draws, seed) -> np.ndarray`.
- Produces: `aggregate_experiment(root, bootstrap_draws) -> dict[str, Any]`.
- Produces: `render_research_report(summary, ledger, destination) -> Path`.
- Writes immutable `analysis/summary.json`, class-level CSV/parquet, figures, and
  the tracked living Markdown report.

- [ ] **Step 1: Write failing interaction and report tests**

```python
def test_effect_fit_recovers_frequency_dimension_interaction() -> None:
    rows = []
    for replicate in range(3):
        for object_id in ("a", "b", "c"):
            for dimension in (1, 3, 5):
                for count in (50, 500, 5000):
                    log_count = math.log10(count)
                    rows.append(
                        {
                            "replicate": replicate,
                            "object_id": object_id,
                            "dimension": dimension,
                            "count": count,
                            "wasserstein_error": (
                                -0.10 * log_count
                                + 0.02 * dimension
                                - 0.08 * log_count * dimension
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    estimate = fit_frequency_dimension_effect(frame, outcome="wasserstein_error")
    assert estimate.interaction == pytest.approx(-0.08, abs=1e-8)


def test_report_keeps_hypotheses_separate_from_observations(tmp_path: Path) -> None:
    path = render_research_report(
        {"effects": {}, "calibration": {}, "conditions": []},
        {"entries": []},
        tmp_path / "report.md",
    )
    text = path.read_text(encoding="utf-8")
    assert "## Frozen hypotheses" in text
    assert "## Calibration record" in text
    assert "## Observations" in text
    assert "## Interpretation" in text
    assert text.index("## Frozen hypotheses") < text.index("## Observations")
```

- [ ] **Step 2: Run tests and verify missing aggregation functions**

Run:

```bash
pytest tests/test_synthetic_long_tail_report.py -v
```

Expected: collection fails because report functions are absent.

- [ ] **Step 3: Implement the fixed-effects design matrix**

Build columns for intercept, `log10(count)`, calibrated dimension, their
interaction, two object indicators, and two replicate indicators; fit with
`np.linalg.lstsq`:

```python
@dataclass(frozen=True)
class EffectEstimate:
    log_count: float
    dimension: float
    interaction: float
    replicate_interactions: tuple[float, ...]
    interval: tuple[float, float] | None = None


design = np.column_stack(
    [
        np.ones(len(frame)),
        log_count,
        dimension,
        log_count * dimension,
        object_indicators,
        replicate_indicators,
    ]
)
coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
```

The 10,000-draw paired bootstrap first resamples replicate IDs and then resamples
the nine object-dimension blocks within each selected replicate. Record the three
per-replicate interaction estimates, pooled estimate, percentile interval, and
whether all signs agree.

For Wasserstein error, H1 is marked supported only when all three replicate
interaction coefficients are negative and the pooled interval upper endpoint is
below zero. H2 is marked supported only when factor loss co-occurs with tangent
alignment loss or positive FM-FLIPD deficit and neither leakage nor off-renderer
mass explains the effect. H3 is always labeled exploratory. H4 reports the paired
balanced-versus-long-tail context effect without inheriting the H1 decision.

- [ ] **Step 4: Implement deterministic report rendering**

Start the tracked report with the question, literature links, frozen design, and
empty run ledger table. On each stage, replace only generated sections delimited by
`<!-- GENERATED:<name>:START -->` and `<!-- GENERATED:<name>:END -->`. Preserve
handwritten observations and interpretations verbatim.

- [ ] **Step 5: Run report tests and render an empty report**

Run:

```bash
pytest tests/test_synthetic_long_tail_report.py -v
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml report
```

Expected: tests pass; the report contains the frozen hypotheses, 12-condition
design table, calibration section, run ledger, observations, interpretation,
limitations, and next-decision sections.

- [ ] **Step 6: Commit**

```bash
git add fm_lab/geometry_explorer/synthetic_long_tail_report.py \
  fm_lab/experiments/synthetic_long_tail_geometry.py \
  docs/research/synthetic_long_tail_geometry_report.md \
  tests/test_synthetic_long_tail_report.py
git commit -m "feat: report synthetic long-tail geometry effects"
```

### Task 11: Verify the complete pipeline and run calibration stages

**Files:**
- Modify on observed evidence only: `docs/research/synthetic_long_tail_geometry_report.md`
- Artifacts: `outputs/synthetic_long_tail_geometry/calibration/`
- Artifacts: `outputs/synthetic_long_tail_geometry/replicate_00/`

**Interfaces:**
- Consumes all previous tasks.
- Produces passing renderer, metric, and oracle gate JSON files.
- Produces the frozen pilot-ready configuration set.

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
pytest -q
ruff check fm_lab scripts tests
```

Expected: zero failures and zero lint errors.

- [ ] **Step 2: Build replicate 0 master pools**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  build-pools --replicate 0
```

Expected: 9 pool cells, 45,000 total images, 12 manifests, uint8 storage below
160 MiB excluding diagnostics, and matching configuration hashes.

- [ ] **Step 3: Run renderer and metric calibration**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  calibrate-renderer
```

Expected gates:

- object oracle separability exceeds 99%;
- occupancy, luminance, and contrast standardized differences are at most 0.25;
- at least 95% of points have nominal renderer rank at relative threshold 0.02;
- median pullback-norm ratio is at most 4;
- full, half-range, and collapsed controls are strictly ordered.

If a gate fails, stop. Change only object scale, factor range, camera, or material
calibration; rerun Tasks 2--3 tests and create a new calibration artifact directory.

- [ ] **Step 4: Train the full oracle and check its gate**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  train-oracle --device auto
```

Expected: object accuracy at least 0.99, every normalized factor MAE at most 0.02,
and a recorded 99.5th-percentile re-render residual. Stop if the gate fails.

- [ ] **Step 5: Update and commit the calibration record**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml report
git add docs/research/synthetic_long_tail_geometry_report.md
git commit -m "docs: record synthetic geometry calibration"
```

Expected: the report links all calibration artifacts and records failures as well as
passing values.

### Task 12: Run the balanced pilot, smoke condition, and main matrix

**Files:**
- Modify on observed evidence only: `docs/research/synthetic_long_tail_geometry_report.md`
- Artifacts: `runs/synthetic_long_tail_geometry/`
- Artifacts: `outputs/synthetic_long_tail_geometry/analysis/`

**Interfaces:**
- Produces frozen `S_*`, 36 main runs, equal-update and matched-pass evaluations,
  factorial effects, and the first scientific conclusion.

- [ ] **Step 1: Run the balanced G0 pilot**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  pilot --device auto
```

Expected: evaluate checkpoints 5k, 10k, 20k, and 40k in ascending order and select
the smallest checkpoint at which every factor has normalized Wasserstein error at
most 0.05, every 5--95% range ratio lies in `[0.80, 1.20]`, and leakage and
off-renderer rates are at most 0.01. Stop if no checkpoint passes; record the failed
pilot before changing capacity.

- [ ] **Step 2: Generate frozen main configurations and run one plumbing smoke**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  smoke --condition g0_f0 --replicate 0 --device auto
```

Expected: training, checkpoint sampling, oracle evaluation, factor metrics, local
geometry, aggregation, and report rendering all complete for one excluded smoke run.

- [ ] **Step 3: Build replicate 1 and 2 pools**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  build-pools --replicate 1
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  build-pools --replicate 2
```

Expected: each replicate has 9 independent pool cells and 12 manifests with no path
or hash collisions.

- [ ] **Step 4: Run the 36-model main matrix**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  matrix --device auto --resume
```

Expected: the ledger ends with 36 complete entries. A failed entry remains failed and
is retried only with an explicit new run directory; completed entries are skipped.

- [ ] **Step 5: Evaluate both checkpoint regimes**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml \
  evaluate --checkpoint-regimes equal_update,matched_pass --device auto --resume
```

Expected: every class/checkpoint has factor metrics over 5,000 generated samples,
validity rates, memorization metrics, and 256-query geometry summaries.

- [ ] **Step 6: Aggregate, render, and verify the scientific report**

Run:

```bash
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml aggregate
python -m fm_lab.experiments.run_synthetic_long_tail_geometry \
  --config configs/synthetic_long_tail_geometry/experiment.yaml report
pytest -q
git status --short
```

Expected: aggregation writes the interaction, per-replicate effects, 10,000-draw
paired interval, balanced-context contrast, shared/private comparison, and all
falsification checks. Tests pass. Git status shows only the intentionally updated
living report unless implementation fixes were required.

- [ ] **Step 7: Commit the first observation record**

```bash
git add docs/research/synthetic_long_tail_geometry_report.md
git commit -m "docs: record synthetic long-tail geometry results"
```

The commit message does not claim support for a hypothesis. The report states the
observed effects first, applies the preregistered support rules second, and records
the next experiment only after those sections are complete.
