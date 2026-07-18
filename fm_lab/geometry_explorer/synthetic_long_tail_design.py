"""Factorial design and shared master pools for the synthetic long-tail study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.latent_factors import (
    BoundedLookAtView,
    BoundedTranslation,
    CameraDepthTranslationInterval,
    LatentFactorSpace,
    ProductFactorSpace,
    sample_values,
)
from fm_lab.geometry_explorer.render_maps import RenderConfig, RenderMap
from fm_lab.geometry_explorer.synthetic_objects import oklch_to_srgb
from fm_lab.utils.logging import write_json

OBJECT_IDS = ("stepped_monument", "crooked_arch", "three_arm_vane")
DIMENSION_IDS = ("high", "medium", "low")
BOUNDED_AZIMUTH_DIMENSION_ID = "high_bounded_azimuth"
VIEW_DEPTH_DIMENSION_ID = "depth_bounded_view"
BOUNDED_ROTATION_CONDITION_ID = "g0_balanced_bounded_azimuth"
BOUNDED_ROTATION_G2_CONDITION_ID = "g2_balanced_bounded_azimuth"
BOUNDED_ROTATION_MEDIUM_CONDITION_ID = "g0_bounded_azimuth_medium"
BOUNDED_ROTATION_TAIL_CONDITION_ID = "g0_bounded_azimuth_tail"
BOUNDED_ROTATION_CONDITION_IDS = frozenset(
    {
        BOUNDED_ROTATION_CONDITION_ID,
        BOUNDED_ROTATION_G2_CONDITION_ID,
        BOUNDED_ROTATION_MEDIUM_CONDITION_ID,
        BOUNDED_ROTATION_TAIL_CONDITION_ID,
    }
)
FACTOR_IDENTITY_CONDITION_IDS = frozenset(
    f"g{geometry}_balanced_view_depth_3d" for geometry in range(3)
)
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

# The azimuth span is chosen so that its estimated total pixel-space arc length
# matches the existing depth interval's.  The Jacobian norms come from the frozen
# renderer's pullback diagnostic for the 5D stepped-monument cell.
AZIMUTH_PULLBACK_NORM = 119.40936660766602
DEPTH_PULLBACK_NORM = 29.531200408935547
DEPTH_TOTAL_RANGE = 1.5
BOUNDED_AZIMUTH_TOTAL_RANGE = DEPTH_TOTAL_RANGE * DEPTH_PULLBACK_NORM / AZIMUTH_PULLBACK_NORM
BOUNDED_AZIMUTH_HALF_RANGE = BOUNDED_AZIMUTH_TOTAL_RANGE / 2.0


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
    def read(cls, path: str | Path) -> ConditionManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["image_shape"] = tuple(raw["image_shape"])
        raw["classes"] = tuple(ConditionClass(**item) for item in raw["classes"])
        return cls(**raw)


@dataclass(frozen=True)
class PoolCellManifest:
    cell_id: str
    replicate: int
    object_id: str
    dimension_id: str
    true_dimension: int
    count: int
    image_shape: tuple[int, int, int]
    factor_columns: tuple[str, ...]
    image_path: str
    factor_path: str
    seed: int
    config_hash: str


def build_factor_space(level: str) -> LatentFactorSpace:
    """Build an approved latent space, including the bounded-rotation control."""

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
    if level == BOUNDED_AZIMUTH_DIMENSION_ID:
        return ProductFactorSpace(
            [
                translation_xyz,
                BoundedLookAtView(
                    azimuth_bounds=(
                        -BOUNDED_AZIMUTH_HALF_RANGE,
                        BOUNDED_AZIMUTH_HALF_RANGE,
                    )
                ),
            ],
            name="translation_xyz_bounded_azimuth_view",
        )
    if level == VIEW_DEPTH_DIMENSION_ID:
        return ProductFactorSpace(
            [
                CameraDepthTranslationInterval(bounds=(-0.75, 0.75)),
                BoundedLookAtView(
                    azimuth_bounds=(
                        -BOUNDED_AZIMUTH_HALF_RANGE,
                        BOUNDED_AZIMUTH_HALF_RANGE,
                    )
                ),
            ],
            name="depth_bounded_azimuth_view",
        )
    raise ValueError(f"Unsupported dimension level: {level}")


def bounded_rotation_condition_spec(
    replicate: int,
    *,
    count: int = 5000,
) -> ConditionManifest:
    """Return the single approved g0 control with only class-0 azimuth restricted."""

    class_count = int(count)
    if class_count <= 0:
        raise ValueError("Bounded-rotation class count must be positive.")
    return _bounded_rotation_spec(
        condition_id=BOUNDED_ROTATION_CONDITION_ID,
        replicate=replicate,
        geometry_index=0,
        class_counts=(class_count, class_count, class_count),
        frequency_mapping="balanced",
    )


def bounded_rotation_followup_condition_specs(
    replicate: int,
    *,
    counts: tuple[int, int, int] = (5000, 500, 50),
) -> tuple[ConditionManifest, ...]:
    """Return the object replication and one-class frequency slice conditions."""

    head, medium, tail = (int(value) for value in counts)
    if not head > medium > tail > 0:
        raise ValueError("Bounded follow-up counts must be strictly descending.")
    return (
        _bounded_rotation_spec(
            condition_id=BOUNDED_ROTATION_G2_CONDITION_ID,
            replicate=replicate,
            geometry_index=2,
            class_counts=(head, head, head),
            frequency_mapping="balanced",
        ),
        _bounded_rotation_spec(
            condition_id=BOUNDED_ROTATION_MEDIUM_CONDITION_ID,
            replicate=replicate,
            geometry_index=0,
            class_counts=(medium, head, head),
            frequency_mapping="class_0_medium_only",
        ),
        _bounded_rotation_spec(
            condition_id=BOUNDED_ROTATION_TAIL_CONDITION_ID,
            replicate=replicate,
            geometry_index=0,
            class_counts=(tail, head, head),
            frequency_mapping="class_0_tail_only",
        ),
    )


def factor_identity_condition_specs(
    replicate: int,
    *,
    count: int = 5000,
) -> tuple[ConditionManifest, ...]:
    """Replace each rotation's 3D XYZ class with 3D depth-plus-view in turn."""

    class_count = int(count)
    if class_count <= 0:
        raise ValueError("Factor-identity class count must be positive.")
    manifests = []
    for geometry_index, baseline_dimensions in enumerate(GEOMETRY_MAPPINGS):
        dimensions = tuple(
            VIEW_DEPTH_DIMENSION_ID if value == "medium" else value for value in baseline_dimensions
        )
        classes = tuple(
            ConditionClass(
                class_id=class_id,
                object_id=object_id,
                dimension_id=dimension_id,
                true_dimension=int(build_factor_space(dimension_id).dim),
                count=class_count,
                image_path="",
                factor_path="",
            )
            for class_id, (object_id, dimension_id) in enumerate(
                zip(OBJECT_IDS, dimensions, strict=True)
            )
        )
        manifests.append(
            ConditionManifest(
                condition_id=f"g{geometry_index}_balanced_view_depth_3d",
                replicate=int(replicate),
                geometry_mapping=f"geometry_{geometry_index}_view_depth_3d",
                frequency_mapping="balanced",
                image_shape=(3, 32, 32),
                classes=classes,
                config_hash="",
            )
        )
    return tuple(manifests)


def _bounded_rotation_spec(
    *,
    condition_id: str,
    replicate: int,
    geometry_index: int,
    class_counts: tuple[int, int, int],
    frequency_mapping: str,
) -> ConditionManifest:
    dimensions = tuple(
        BOUNDED_AZIMUTH_DIMENSION_ID if item == "high" else item
        for item in GEOMETRY_MAPPINGS[geometry_index]
    )
    classes = tuple(
        ConditionClass(
            class_id=class_id,
            object_id=object_id,
            dimension_id=dimension_id,
            true_dimension=int(build_factor_space(dimension_id).dim),
            count=int(count),
            image_path="",
            factor_path="",
        )
        for class_id, (object_id, dimension_id, count) in enumerate(
            zip(OBJECT_IDS, dimensions, class_counts, strict=True)
        )
    )
    return ConditionManifest(
        condition_id=condition_id,
        replicate=int(replicate),
        geometry_mapping=f"geometry_{geometry_index}_bounded_azimuth",
        frequency_mapping=frequency_mapping,
        image_shape=(3, 32, 32),
        classes=classes,
        config_hash="",
    )


def canonical_factor_rows(
    factor: LatentFactorSpace,
    values: Sequence[Any],
) -> np.ndarray:
    """Project heterogeneous latent states into the fixed five-column schema."""

    rows = np.full((len(values), len(FACTOR_COLUMNS)), np.nan, dtype=np.float32)
    for row_id, value in enumerate(values):
        coordinates = factor.coordinates(value)
        rows[row_id, 0] = coordinates.get("translation_x", np.nan)
        rows[row_id, 1] = coordinates.get("translation_y", np.nan)
        rows[row_id, 2] = coordinates.get("translation_z", np.nan)
        rows[row_id, 3] = coordinates.get("camera_azimuth", np.nan)
        rows[row_id, 4] = coordinates.get("camera_elevation", np.nan)
    return rows


def build_local_geometry_queries(
    config: dict[str, Any],
    *,
    object_id: str,
    dimension_id: str,
    count: int,
    seed: int,
    epsilon: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Render deterministic interior queries and normalized renderer tangents."""

    if object_id not in OBJECT_IDS:
        raise ValueError(f"object_id must be one of: {', '.join(OBJECT_IDS)}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    step = float(epsilon)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    factor = build_factor_space(dimension_id)
    scales = _geometry_normalization_scales(dimension_id)
    if len(scales) != int(factor.dim):
        raise ValueError("geometry normalization scales do not match the factor dimension.")
    candidates = sample_values(factor.sample(max(count * 8, count + 16), seed=seed))
    interior = [
        value
        for value in candidates
        if _supports_centered_factor_steps(factor, value, scales=scales, epsilon=step)
    ]
    if len(interior) < count:
        raise ValueError("Could not sample enough interior local-geometry queries.")

    render_map = _render_map(config, _object_configs(config)[object_id], factor)
    query_rows = []
    tangent_rows = []
    selected = []
    for value in interior:
        columns = []
        for tangent, scale in zip(factor.tangent_basis(value), scales, strict=True):
            plus = factor.retract(value, tangent, step * scale)
            minus = factor.retract(value, tangent, -step * scale)
            plus_image = np.asarray(render_map.render(plus), dtype=np.float32)
            minus_image = np.asarray(render_map.render(minus), dtype=np.float32)
            derivative = (plus_image - minus_image).transpose(2, 0, 1).reshape(-1) / step
            columns.append(derivative.astype(np.float32))
        tangent_row = np.stack(columns, axis=0)
        if np.any(np.linalg.norm(tangent_row, axis=1) <= np.finfo(np.float32).eps):
            continue
        image = np.asarray(render_map.render(value), dtype=np.float32)
        query_rows.append(image.transpose(2, 0, 1).reshape(-1) * 2.0 - 1.0)
        tangent_rows.append(tangent_row)
        selected.append(value)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Could not render enough queries with five visible tangents.")
    queries = np.stack(query_rows, axis=0).astype(np.float32)
    tangents = np.stack(tangent_rows, axis=0)
    names = tuple(str(name) for name in factor.tangent_labels(selected[0]))
    if not np.isfinite(queries).all() or not np.isfinite(tangents).all():
        raise ValueError("Rendered local-geometry queries and tangents must be finite.")
    return queries, tangents, names, canonical_factor_rows(factor, selected)


def _geometry_normalization_scales(dimension_id: str) -> tuple[float, ...]:
    translation = (0.25, 0.25, 0.75)
    if dimension_id == "low":
        return (0.75,)
    if dimension_id == "medium":
        return translation
    if dimension_id == "high":
        return (*translation, float(np.pi), 0.5)
    if dimension_id == BOUNDED_AZIMUTH_DIMENSION_ID:
        return (*translation, BOUNDED_AZIMUTH_HALF_RANGE, 0.5)
    if dimension_id == VIEW_DEPTH_DIMENSION_ID:
        return (0.75, BOUNDED_AZIMUTH_HALF_RANGE, 0.5)
    raise ValueError(f"Unsupported dimension level: {dimension_id}")


def _supports_centered_factor_steps(
    factor: LatentFactorSpace,
    value: Any,
    *,
    scales: Sequence[float],
    epsilon: float,
) -> bool:
    baseline = canonical_factor_rows(factor, [value])[0]
    active = np.flatnonzero(np.isfinite(baseline))
    if len(active) != int(factor.dim):
        return False
    for column, tangent, scale in zip(
        active,
        factor.tangent_basis(value),
        scales,
        strict=True,
    ):
        plus = canonical_factor_rows(
            factor,
            [factor.retract(value, tangent, epsilon * scale)],
        )[0, column]
        minus = canonical_factor_rows(
            factor,
            [factor.retract(value, tangent, -epsilon * scale)],
        )[0, column]
        forward = abs(float(plus - baseline[column]))
        backward = abs(float(baseline[column] - minus))
        if min(forward, backward) <= np.finfo(np.float32).eps:
            return False
        ratio = forward / backward
        if not 0.5 <= ratio <= 2.0:
            return False
    return True


def build_condition_specs(
    replicate: int,
    *,
    counts: tuple[int, int, int] = (5000, 500, 50),
) -> tuple[ConditionManifest, ...]:
    """Return the three balanced and nine imbalanced factorial conditions."""

    balanced_count = int(counts[0])
    frequency_mappings = (
        tuple(int(value) for value in counts),
        (int(counts[1]), int(counts[2]), int(counts[0])),
        (int(counts[2]), int(counts[0]), int(counts[1])),
    )
    manifests = []
    for geometry_index, dimensions in enumerate(GEOMETRY_MAPPINGS):
        manifests.append(
            _condition_spec(
                replicate=replicate,
                geometry_index=geometry_index,
                dimensions=dimensions,
                frequency_index=None,
                class_counts=(balanced_count,) * len(OBJECT_IDS),
            )
        )
        for frequency_index, class_counts in enumerate(frequency_mappings):
            manifests.append(
                _condition_spec(
                    replicate=replicate,
                    geometry_index=geometry_index,
                    dimensions=dimensions,
                    frequency_index=frequency_index,
                    class_counts=class_counts,
                )
            )
    return tuple(manifests)


def build_master_pools(
    config: dict[str, Any],
    root: str | Path,
    replicate: int,
) -> tuple[PoolCellManifest, ...]:
    """Sample and render the nine shared object-by-dimension master pools."""

    master_count = int(config["master_count"])
    image_size = int(config["image_size"])
    if master_count <= 0 or image_size <= 0:
        raise ValueError("master_count and image_size must be positive.")
    base_seed = int(config["seed"])
    config_hash = _config_hash(config)
    render_batch_size = int(config.get("render", {}).get("render_batch_size", 128))
    if render_batch_size <= 0:
        raise ValueError("render.render_batch_size must be positive.")

    object_configs = _object_configs(config)
    replicate_root = Path(root).resolve() / f"replicate_{replicate:02d}"
    pool_root = replicate_root / "pools"
    _refuse_existing(pool_root, kind="Pool")
    replicate_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".pools-", dir=replicate_root))
    cells = []
    try:
        for object_index, object_id in enumerate(OBJECT_IDS):
            object_config = object_configs[object_id]
            for dimension_index, dimension_id in enumerate(DIMENSION_IDS):
                seed = (
                    base_seed
                    + int(replicate) * 100_000
                    + object_index * 1_000
                    + dimension_index * 10
                )
                factor = build_factor_space(dimension_id)
                values = sample_values(factor.sample(master_count, seed=seed))
                cell_dir = staging_root / object_id / dimension_id
                cell_dir.mkdir(parents=True, exist_ok=True)
                staging_image_path = cell_dir / "images.npy"
                staging_factor_path = cell_dir / "factors.npy"
                images = np.lib.format.open_memmap(
                    staging_image_path,
                    mode="w+",
                    dtype=np.uint8,
                    shape=(master_count, 3, image_size, image_size),
                )
                factors = np.lib.format.open_memmap(
                    staging_factor_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(master_count, len(FACTOR_COLUMNS)),
                )
                render_map = _render_map(config, object_config, factor)
                for start in range(0, master_count, render_batch_size):
                    stop = min(master_count, start + render_batch_size)
                    rendered = _as_hwc_batch(
                        render_map.render_batch(
                            values[start:stop],
                            batch_size=render_batch_size,
                        ),
                        image_size=image_size,
                    )
                    images[start:stop] = np.rint(
                        np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    factors[start:stop] = canonical_factor_rows(factor, values[start:stop])
                images.flush()
                factors.flush()
                del images, factors
                final_cell_dir = pool_root / object_id / dimension_id
                cells.append(
                    PoolCellManifest(
                        cell_id=f"{object_id}__{dimension_id}",
                        replicate=int(replicate),
                        object_id=object_id,
                        dimension_id=dimension_id,
                        true_dimension=int(factor.dim),
                        count=master_count,
                        image_shape=(3, image_size, image_size),
                        factor_columns=FACTOR_COLUMNS,
                        image_path=str(final_cell_dir / "images.npy"),
                        factor_path=str(final_cell_dir / "factors.npy"),
                        seed=seed,
                        config_hash=config_hash,
                    )
                )
        staging_root.replace(pool_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(cells)


def build_condition_manifests(
    root: str | Path,
    replicate: int,
    pool_cells: Sequence[PoolCellManifest],
    *,
    counts: tuple[int, int, int] = (5000, 500, 50),
) -> tuple[Path, ...]:
    """Write condition metadata as prefix views into shared arrays."""

    by_cell = {(cell.object_id, cell.dimension_id): cell for cell in pool_cells}
    expected_cells = {
        (object_id, dimension_id) for object_id in OBJECT_IDS for dimension_id in DIMENSION_IDS
    }
    if set(by_cell) != expected_cells:
        raise ValueError("pool_cells must contain exactly the nine object-dimension cells.")
    if any(int(count) <= 0 for count in counts):
        raise ValueError("Condition counts must be positive.")
    if max(counts) > min(cell.count for cell in pool_cells):
        raise ValueError("Condition counts cannot exceed the shared master pool count.")

    replicate_root = Path(root).resolve() / f"replicate_{replicate:02d}"
    manifest_dir = replicate_root / "conditions"
    _refuse_existing(manifest_dir, kind="Condition")
    config_hashes = {cell.config_hash for cell in pool_cells}
    if len(config_hashes) != 1:
        raise ValueError("All pool cells must share one config hash.")
    config_hash = next(iter(config_hashes))
    image_shapes = {cell.image_shape for cell in pool_cells}
    if len(image_shapes) != 1:
        raise ValueError("All pool cells must share one image shape.")
    image_shape = next(iter(image_shapes))

    replicate_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".conditions-", dir=replicate_root))
    filenames = []
    try:
        for spec in build_condition_specs(replicate, counts=counts):
            classes = []
            for entry in spec.classes:
                cell = by_cell[(entry.object_id, entry.dimension_id)]
                classes.append(
                    replace(
                        entry,
                        image_path=os.path.relpath(cell.image_path, manifest_dir),
                        factor_path=os.path.relpath(cell.factor_path, manifest_dir),
                    )
                )
            manifest = replace(
                spec,
                image_shape=image_shape,
                classes=tuple(classes),
                config_hash=config_hash,
            )
            filename = f"{manifest.condition_id}.json"
            manifest.write(staging_dir / filename)
            filenames.append(filename)
        staging_dir.replace(manifest_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return tuple(manifest_dir / filename for filename in filenames)


def build_bounded_rotation_control(
    config: dict[str, Any],
    root: str | Path,
    *,
    replicate: int = 0,
) -> dict[str, Path]:
    """Render the one changed 5D pool and reuse the unchanged g0 pool cells.

    The control uses the original class-0 5D seed.  Consequently XYZ and
    elevation are paired sample-for-sample with the full-azimuth pool, while the
    underlying azimuth uniform variate is mapped into the narrower interval.
    """

    master_count = int(config["master_count"])
    image_size = int(config["image_size"])
    if master_count <= 0 or image_size <= 0:
        raise ValueError("master_count and image_size must be positive.")
    replicate_id = int(replicate)
    if replicate_id < 0:
        raise ValueError("replicate must be non-negative.")
    render_batch_size = int(config.get("render", {}).get("render_batch_size", 128))
    if render_batch_size <= 0:
        raise ValueError("render.render_batch_size must be positive.")

    replicate_root = Path(root).resolve() / f"replicate_{replicate_id:02d}"
    canonical_pool_root = replicate_root / "pools"
    reused_cells = (
        (OBJECT_IDS[1], "medium"),
        (OBJECT_IDS[2], "low"),
    )
    baseline_5d_dir = canonical_pool_root / OBJECT_IDS[0] / "high"
    for cell_dir in (baseline_5d_dir, *(canonical_pool_root / o / d for o, d in reused_cells)):
        for filename in ("images.npy", "factors.npy"):
            if not (cell_dir / filename).is_file():
                raise FileNotFoundError(
                    "Canonical pools must be built before the bounded-rotation control: "
                    f"{cell_dir / filename}"
                )

    control_root = replicate_root / "bounded_rotation_control"
    _refuse_existing(control_root, kind="Bounded-rotation control")
    replicate_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".bounded-rotation-", dir=replicate_root))
    condition_dir = staging_root / "conditions"
    new_cell_dir = staging_root / "pools" / OBJECT_IDS[0] / BOUNDED_AZIMUTH_DIMENSION_ID
    final_condition_dir = control_root / "conditions"
    final_new_cell_dir = control_root / "pools" / OBJECT_IDS[0] / BOUNDED_AZIMUTH_DIMENSION_ID
    published = False
    try:
        condition_dir.mkdir(parents=True)
        new_cell_dir.mkdir(parents=True)
        factor = build_factor_space(BOUNDED_AZIMUTH_DIMENSION_ID)
        seed = int(config["seed"]) + replicate_id * 100_000
        values = sample_values(factor.sample(master_count, seed=seed))
        staging_image_path = new_cell_dir / "images.npy"
        staging_factor_path = new_cell_dir / "factors.npy"
        images = np.lib.format.open_memmap(
            staging_image_path,
            mode="w+",
            dtype=np.uint8,
            shape=(master_count, 3, image_size, image_size),
        )
        factors = np.lib.format.open_memmap(
            staging_factor_path,
            mode="w+",
            dtype=np.float32,
            shape=(master_count, len(FACTOR_COLUMNS)),
        )
        object_config = _object_configs(config)[OBJECT_IDS[0]]
        render_map = _render_map(config, object_config, factor)
        for start in range(0, master_count, render_batch_size):
            stop = min(master_count, start + render_batch_size)
            rendered = _as_hwc_batch(
                render_map.render_batch(values[start:stop], batch_size=render_batch_size),
                image_size=image_size,
            )
            images[start:stop] = np.rint(
                np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            factors[start:stop] = canonical_factor_rows(factor, values[start:stop])
        images.flush()
        factors.flush()
        del images, factors

        baseline_factors = np.load(baseline_5d_dir / "factors.npy", mmap_mode="r")
        control_factors = np.load(staging_factor_path, mmap_mode="r")
        if baseline_factors.shape != control_factors.shape:
            raise ValueError("Baseline and control 5D factor pools must have identical shapes.")
        paired_columns = (0, 1, 2, 4)
        if not np.array_equal(
            np.asarray(baseline_factors[:, paired_columns]),
            np.asarray(control_factors[:, paired_columns]),
        ):
            raise ValueError("Control XYZ/elevation factors are not paired with the baseline.")
        del baseline_factors, control_factors

        for object_id, dimension_id in reused_cells:
            link = staging_root / "pools" / object_id / dimension_id
            link.parent.mkdir(parents=True, exist_ok=True)
            target = canonical_pool_root / object_id / dimension_id
            os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)

        control_spec = {
            "schema_version": 1,
            "condition_id": BOUNDED_ROTATION_CONDITION_ID,
            "replicate": replicate_id,
            "class_id_changed": 0,
            "object_id_changed": OBJECT_IDS[0],
            "baseline_dimension_id": "high",
            "control_dimension_id": BOUNDED_AZIMUTH_DIMENSION_ID,
            "true_dimension_unchanged": 5,
            "azimuth_bounds_radians": [
                -BOUNDED_AZIMUTH_HALF_RANGE,
                BOUNDED_AZIMUTH_HALF_RANGE,
            ],
            "azimuth_total_range_radians": BOUNDED_AZIMUTH_TOTAL_RANGE,
            "azimuth_total_range_degrees": math.degrees(BOUNDED_AZIMUTH_TOTAL_RANGE),
            "elevation_bounds_radians": [-math.pi / 6.0, math.pi / 6.0],
            "matching_rule": {
                "description": (
                    "Match estimated total pixel-space arc length of azimuth to the "
                    "full depth interval."
                ),
                "formula": "depth_range * depth_jacobian_norm / azimuth_jacobian_norm",
                "depth_total_range": DEPTH_TOTAL_RANGE,
                "depth_jacobian_norm": DEPTH_PULLBACK_NORM,
                "azimuth_jacobian_norm": AZIMUTH_PULLBACK_NORM,
            },
            "pairing": {
                "seed": seed,
                "identical_factor_columns": ["tx", "ty", "tz", "elevation"],
                "reused_pool_cells": [
                    {"object_id": object_id, "dimension_id": dimension_id}
                    for object_id, dimension_id in reused_cells
                ],
            },
            "interpretation_limit": (
                "This changes the extent and pixel-space scale of one factor, not the "
                "topological intrinsic dimension, which remains five."
            ),
        }
        control_hash = _config_hash({"base_config": config, "control": control_spec})
        spec = bounded_rotation_condition_spec(replicate_id, count=master_count)
        class_dirs = (
            final_new_cell_dir,
            control_root / "pools" / OBJECT_IDS[1] / "medium",
            control_root / "pools" / OBJECT_IDS[2] / "low",
        )
        classes = tuple(
            replace(
                entry,
                image_path=os.path.relpath(cell_dir / "images.npy", final_condition_dir),
                factor_path=os.path.relpath(cell_dir / "factors.npy", final_condition_dir),
            )
            for entry, cell_dir in zip(spec.classes, class_dirs, strict=True)
        )
        manifest = replace(
            spec,
            image_shape=(3, image_size, image_size),
            classes=classes,
            config_hash=control_hash,
        )
        manifest.write(condition_dir / f"{BOUNDED_ROTATION_CONDITION_ID}.json")
        write_json(control_spec | {"config_hash": control_hash}, staging_root / "control_spec.json")
        staging_root.replace(control_root)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "root": control_root,
        "manifest": final_condition_dir / f"{BOUNDED_ROTATION_CONDITION_ID}.json",
        "control_spec": control_root / "control_spec.json",
    }


def build_bounded_rotation_followups(
    config: dict[str, Any],
    root: str | Path,
    *,
    replicate: int = 0,
) -> dict[str, Any]:
    """Build one new g2 pool and immutable manifests for the targeted follow-ups."""

    master_count = int(config["master_count"])
    image_size = int(config["image_size"])
    counts = tuple(int(value) for value in config["counts"])
    if master_count <= 0 or image_size <= 0 or len(counts) != 3:
        raise ValueError("Follow-ups require positive master_count/image_size and three counts.")
    if counts[0] != master_count or not counts[0] > counts[1] > counts[2] > 0:
        raise ValueError("Follow-up counts must be master_count > medium > tail > 0.")
    replicate_id = int(replicate)
    if replicate_id < 0:
        raise ValueError("replicate must be non-negative.")
    render_batch_size = int(config.get("render", {}).get("render_batch_size", 128))
    if render_batch_size <= 0:
        raise ValueError("render.render_batch_size must be positive.")

    replicate_root = Path(root).resolve() / f"replicate_{replicate_id:02d}"
    canonical_pool_root = replicate_root / "pools"
    original_control_root = replicate_root / "bounded_rotation_control"
    g0_bounded_cell = original_control_root / "pools" / OBJECT_IDS[0] / BOUNDED_AZIMUTH_DIMENSION_ID
    g2_baseline_cell = canonical_pool_root / OBJECT_IDS[1] / "high"
    required_cells = (
        g0_bounded_cell,
        g2_baseline_cell,
        canonical_pool_root / OBJECT_IDS[0] / "low",
        canonical_pool_root / OBJECT_IDS[1] / "medium",
        canonical_pool_root / OBJECT_IDS[2] / "low",
        canonical_pool_root / OBJECT_IDS[2] / "medium",
    )
    for cell_dir in required_cells:
        for filename in ("images.npy", "factors.npy"):
            if not (cell_dir / filename).is_file():
                raise FileNotFoundError(
                    f"Required pool is missing before bounded follow-ups: {cell_dir / filename}"
                )

    followup_root = replicate_root / "bounded_rotation_followups"
    _refuse_existing(followup_root, kind="Bounded-rotation follow-up")
    staging_root = Path(tempfile.mkdtemp(prefix=".bounded-followups-", dir=replicate_root))
    condition_dir = staging_root / "conditions"
    new_cell_dir = staging_root / "pools" / OBJECT_IDS[1] / BOUNDED_AZIMUTH_DIMENSION_ID
    final_condition_dir = followup_root / "conditions"
    published = False
    try:
        condition_dir.mkdir(parents=True)
        new_cell_dir.mkdir(parents=True)
        factor = build_factor_space(BOUNDED_AZIMUTH_DIMENSION_ID)
        seed = int(config["seed"]) + replicate_id * 100_000 + 1_000
        values = sample_values(factor.sample(master_count, seed=seed))
        image_path = new_cell_dir / "images.npy"
        factor_path = new_cell_dir / "factors.npy"
        images = np.lib.format.open_memmap(
            image_path,
            mode="w+",
            dtype=np.uint8,
            shape=(master_count, 3, image_size, image_size),
        )
        factors = np.lib.format.open_memmap(
            factor_path,
            mode="w+",
            dtype=np.float32,
            shape=(master_count, len(FACTOR_COLUMNS)),
        )
        render_map = _render_map(config, _object_configs(config)[OBJECT_IDS[1]], factor)
        for start in range(0, master_count, render_batch_size):
            stop = min(master_count, start + render_batch_size)
            rendered = _as_hwc_batch(
                render_map.render_batch(values[start:stop], batch_size=render_batch_size),
                image_size=image_size,
            )
            images[start:stop] = np.rint(
                np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            factors[start:stop] = canonical_factor_rows(factor, values[start:stop])
        images.flush()
        factors.flush()
        del images, factors

        baseline_factors = np.load(g2_baseline_cell / "factors.npy", mmap_mode="r")
        bounded_factors = np.load(factor_path, mmap_mode="r")
        if baseline_factors.shape != bounded_factors.shape or not np.array_equal(
            np.asarray(baseline_factors[:, (0, 1, 2, 4)]),
            np.asarray(bounded_factors[:, (0, 1, 2, 4)]),
        ):
            raise ValueError("g2 bounded XYZ/elevation factors are not paired with baseline.")
        del baseline_factors, bounded_factors

        linked_cells = {
            (OBJECT_IDS[0], BOUNDED_AZIMUTH_DIMENSION_ID): g0_bounded_cell,
            (OBJECT_IDS[0], "low"): canonical_pool_root / OBJECT_IDS[0] / "low",
            (OBJECT_IDS[1], "medium"): canonical_pool_root / OBJECT_IDS[1] / "medium",
            (OBJECT_IDS[2], "low"): canonical_pool_root / OBJECT_IDS[2] / "low",
            (OBJECT_IDS[2], "medium"): canonical_pool_root / OBJECT_IDS[2] / "medium",
        }
        for (object_id, dimension_id), target in linked_cells.items():
            link = staging_root / "pools" / object_id / dimension_id
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)

        followup_spec = {
            "schema_version": 1,
            "replicate": replicate_id,
            "training_runs": 4,
            "object_replication": {
                "condition_id": BOUNDED_ROTATION_G2_CONDITION_ID,
                "changed_class_id": 1,
                "object_id": OBJECT_IDS[1],
                "baseline_condition_id": "g2_balanced",
            },
            "frequency_slice": {
                "changed_class_id": 0,
                "object_id": OBJECT_IDS[0],
                "unique_counts": [counts[0], counts[1], counts[2]],
                "other_class_counts": [counts[0], counts[0]],
                "tail_sampling_policies": ["empirical", "class_balanced"],
            },
            "azimuth_bounds_radians": [
                -BOUNDED_AZIMUTH_HALF_RANGE,
                BOUNDED_AZIMUTH_HALF_RANGE,
            ],
            "pairing": {
                "g2_seed": seed,
                "identical_factor_columns": ["tx", "ty", "tz", "elevation"],
                "model_initialization_and_sampling_seeds": "shared across runs",
            },
            "decision": (
                "Use fewer condition cells at the existing 2000-step comparison budget; "
                "do not repeat the nine-condition factorial."
            ),
        }
        followup_hash = _config_hash({"base_config": config, "followup": followup_spec})
        manifests: dict[str, Path] = {}
        for spec in bounded_rotation_followup_condition_specs(
            replicate_id,
            counts=counts,
        ):
            classes = tuple(
                replace(
                    entry,
                    image_path=os.path.relpath(
                        followup_root
                        / "pools"
                        / entry.object_id
                        / entry.dimension_id
                        / "images.npy",
                        final_condition_dir,
                    ),
                    factor_path=os.path.relpath(
                        followup_root
                        / "pools"
                        / entry.object_id
                        / entry.dimension_id
                        / "factors.npy",
                        final_condition_dir,
                    ),
                )
                for entry in spec.classes
            )
            manifest = replace(
                spec,
                image_shape=(3, image_size, image_size),
                classes=classes,
                config_hash=followup_hash,
            )
            filename = f"{manifest.condition_id}.json"
            manifest.write(condition_dir / filename)
            manifests[manifest.condition_id] = final_condition_dir / filename
        write_json(
            followup_spec | {"config_hash": followup_hash},
            staging_root / "followup_spec.json",
        )
        staging_root.replace(followup_root)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "root": followup_root,
        "manifests": manifests,
        "followup_spec": followup_root / "followup_spec.json",
    }


def build_factor_identity_control(
    config: dict[str, Any],
    root: str | Path,
    *,
    replicate: int = 0,
) -> dict[str, Any]:
    """Render depth-plus-view pools and build three matched 3D interventions."""

    master_count = int(config["master_count"])
    image_size = int(config["image_size"])
    replicate_id = int(replicate)
    if master_count <= 0 or image_size <= 0:
        raise ValueError("master_count and image_size must be positive.")
    if replicate_id < 0:
        raise ValueError("replicate must be non-negative.")
    render_batch_size = int(config.get("render", {}).get("render_batch_size", 128))
    if render_batch_size <= 0:
        raise ValueError("render.render_batch_size must be positive.")

    replicate_root = Path(root).resolve() / f"replicate_{replicate_id:02d}"
    canonical_pool_root = replicate_root / "pools"
    required_cells = tuple(
        canonical_pool_root / object_id / dimension_id
        for object_id in OBJECT_IDS
        for dimension_id in DIMENSION_IDS
    )
    for cell_dir in required_cells:
        for filename in ("images.npy", "factors.npy"):
            if not (cell_dir / filename).is_file():
                raise FileNotFoundError(
                    f"Canonical pool is missing before factor-identity control: "
                    f"{cell_dir / filename}"
                )

    control_root = replicate_root / "factor_identity_control"
    _refuse_existing(control_root, kind="Factor-identity control")
    staging_root = Path(tempfile.mkdtemp(prefix=".factor-identity-", dir=replicate_root))
    condition_dir = staging_root / "conditions"
    final_condition_dir = control_root / "conditions"
    published = False
    try:
        condition_dir.mkdir(parents=True)
        factor = build_factor_space(VIEW_DEPTH_DIMENSION_ID)
        object_configs = _object_configs(config)
        seeds = {}
        for object_index, object_id in enumerate(OBJECT_IDS):
            seed = int(config["seed"]) + replicate_id * 100_000 + object_index * 1_000 + 10
            seeds[object_id] = seed
            values = sample_values(factor.sample(master_count, seed=seed))
            cell_dir = staging_root / "pools" / object_id / VIEW_DEPTH_DIMENSION_ID
            cell_dir.mkdir(parents=True)
            image_path = cell_dir / "images.npy"
            factor_path = cell_dir / "factors.npy"
            images = np.lib.format.open_memmap(
                image_path,
                mode="w+",
                dtype=np.uint8,
                shape=(master_count, 3, image_size, image_size),
            )
            factors = np.lib.format.open_memmap(
                factor_path,
                mode="w+",
                dtype=np.float32,
                shape=(master_count, len(FACTOR_COLUMNS)),
            )
            render_map = _render_map(config, object_configs[object_id], factor)
            for start in range(0, master_count, render_batch_size):
                stop = min(master_count, start + render_batch_size)
                rendered = _as_hwc_batch(
                    render_map.render_batch(
                        values[start:stop],
                        batch_size=render_batch_size,
                    ),
                    image_size=image_size,
                )
                images[start:stop] = np.rint(
                    np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
                ).astype(np.uint8)
                factors[start:stop] = canonical_factor_rows(factor, values[start:stop])
            images.flush()
            factors.flush()
            del images, factors

        for object_id in OBJECT_IDS:
            for dimension_id in DIMENSION_IDS:
                link = staging_root / "pools" / object_id / dimension_id
                link.parent.mkdir(parents=True, exist_ok=True)
                target = canonical_pool_root / object_id / dimension_id
                os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)

        renderer_check = _factor_identity_renderer_check(config)
        if (
            config.get("renderer_profile") == "calibrated_v2"
            and renderer_check["passed"] is not True
        ):
            raise ValueError("Factor-identity view-depth renderer failed its local rank check.")
        control_spec = {
            "schema_version": 1,
            "replicate": replicate_id,
            "baseline_factor_identity": "translation_xyz",
            "intervention_factor_identity": "depth_bounded_azimuth_elevation",
            "nominal_dimension": 3,
            "class_count": master_count,
            "condition_ids": sorted(FACTOR_IDENTITY_CONDITION_IDS),
            "seeds": seeds,
            "renderer_check": renderer_check,
            "factor_bounds": {
                "depth": [-0.75, 0.75],
                "azimuth": [
                    -BOUNDED_AZIMUTH_HALF_RANGE,
                    BOUNDED_AZIMUTH_HALF_RANGE,
                ],
                "elevation_degrees": [-30.0, 30.0],
            },
            "pairing": (
                "Each intervention changes only the balanced rotation's 3D class; "
                "the 1D and 5D class pools, training seed, and sampling seed are unchanged."
            ),
            "interpretation_limit": (
                "This isolates factor identity at fixed nominal dimension, not equal "
                "pixel-space arc length for every factor and object."
            ),
        }
        control_hash = _config_hash({"base_config": config, "control": control_spec})
        manifests = {}
        for spec in factor_identity_condition_specs(replicate_id, count=master_count):
            classes = tuple(
                replace(
                    entry,
                    image_path=os.path.relpath(
                        control_root
                        / "pools"
                        / entry.object_id
                        / entry.dimension_id
                        / "images.npy",
                        final_condition_dir,
                    ),
                    factor_path=os.path.relpath(
                        control_root
                        / "pools"
                        / entry.object_id
                        / entry.dimension_id
                        / "factors.npy",
                        final_condition_dir,
                    ),
                )
                for entry in spec.classes
            )
            manifest = replace(
                spec,
                image_shape=(3, image_size, image_size),
                classes=classes,
                config_hash=control_hash,
            )
            filename = f"{manifest.condition_id}.json"
            manifest.write(condition_dir / filename)
            manifests[manifest.condition_id] = final_condition_dir / filename
        write_json(
            control_spec | {"config_hash": control_hash},
            staging_root / "control_spec.json",
        )
        staging_root.replace(control_root)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)

    return {
        "root": control_root,
        "manifests": manifests,
        "control_spec": control_root / "control_spec.json",
    }


def _factor_identity_renderer_check(config: dict[str, Any]) -> dict[str, Any]:
    count = min(16, int(config["master_count"]))
    epsilon = float(config.get("calibration", {}).get("finite_difference_epsilon", 0.02))
    threshold = float(config.get("calibration", {}).get("relative_singular_threshold", 0.02))
    required_fraction = float(config.get("calibration", {}).get("full_rank_fraction", 0.95))
    records = []
    for object_index, object_id in enumerate(OBJECT_IDS):
        _, tangents, _, _ = build_local_geometry_queries(
            config,
            object_id=object_id,
            dimension_id=VIEW_DEPTH_DIMENSION_ID,
            count=count,
            seed=int(config["seed"]) + 9_300_003 + object_index * 1_000,
            epsilon=epsilon,
        )
        singular = np.linalg.svd(tangents.transpose(0, 2, 1), compute_uv=False)
        ranks = np.sum(singular > singular[:, :1] * threshold, axis=1)
        condition = np.divide(
            singular[:, 0],
            singular[:, -1],
            out=np.full(len(singular), np.inf, dtype=np.float64),
            where=singular[:, -1] > 0.0,
        )
        records.append(
            {
                "object_id": object_id,
                "query_count": count,
                "full_rank_fraction": float(np.mean(ranks == 3)),
                "median_singular_values": [float(value) for value in np.median(singular, axis=0)],
                "median_condition_number": float(np.median(condition)),
            }
        )
    return {
        "passed": all(item["full_rank_fraction"] >= required_fraction for item in records),
        "dimension": 3,
        "relative_singular_threshold": threshold,
        "required_full_rank_fraction": required_fraction,
        "records": records,
    }


def _condition_spec(
    *,
    replicate: int,
    geometry_index: int,
    dimensions: tuple[str, str, str],
    frequency_index: int | None,
    class_counts: tuple[int, int, int],
) -> ConditionManifest:
    geometry_name = f"geometry_{geometry_index}"
    frequency_name = "balanced" if frequency_index is None else f"frequency_{frequency_index}"
    classes = tuple(
        ConditionClass(
            class_id=class_id,
            object_id=object_id,
            dimension_id=dimension_id,
            true_dimension=int(build_factor_space(dimension_id).dim),
            count=int(count),
            image_path="",
            factor_path="",
        )
        for class_id, (object_id, dimension_id, count) in enumerate(
            zip(OBJECT_IDS, dimensions, class_counts, strict=True)
        )
    )
    condition_id = (
        f"g{geometry_index}_balanced"
        if frequency_index is None
        else f"g{geometry_index}_f{frequency_index}"
    )
    return ConditionManifest(
        condition_id=condition_id,
        replicate=int(replicate),
        geometry_mapping=geometry_name,
        frequency_mapping=frequency_name,
        image_shape=(3, 32, 32),
        classes=classes,
        config_hash="",
    )


def _object_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_objects = config.get("objects", [])
    by_id = {str(item["id"]): dict(item) for item in raw_objects}
    if set(by_id) != set(OBJECT_IDS):
        raise ValueError(f"objects must contain exactly: {', '.join(OBJECT_IDS)}")
    lightness = float(config.get("material", {}).get("oklch_lightness", 0.70))
    chroma = float(config.get("material", {}).get("oklch_chroma", 0.12))
    return {
        object_id: {
            "kind": object_id,
            "scale": float(by_id[object_id].get("scale", 1.0)),
            "marker": False,
            "base_color": oklch_to_srgb(
                float(by_id[object_id].get("oklch_lightness", lightness)),
                float(by_id[object_id].get("oklch_chroma", chroma)),
                float(by_id[object_id]["hue_degrees"]),
            ),
        }
        for object_id in OBJECT_IDS
    }


def _render_map(
    config: dict[str, Any],
    object_config: dict[str, Any],
    factor: LatentFactorSpace,
) -> RenderMap:
    render = config.get("render", {})
    background = tuple(float(value) for value in render.get("background", (1.0, 1.0, 1.0)))
    return RenderMap(
        factor,
        str(object_config["kind"]),
        config=RenderConfig(
            image_size=int(config["image_size"]),
            background=background,
            antialias=int(render.get("supersample", 3)) > 1,
            object_config=object_config,
            camera_config={"radius": float(render.get("camera_distance", 4.0))},
            light_config={
                "position": tuple(
                    float(value) for value in render.get("light_position", (3.0, -4.0, 5.0))
                ),
                "energy": float(render.get("light_energy", 400.0)),
                "ambient": float(render.get("ambient", 0.35)),
                "diffuse": float(render.get("diffuse", 0.70)),
            },
        ),
    )


def _as_hwc_batch(rendered: np.ndarray, *, image_size: int) -> np.ndarray:
    array = np.asarray(rendered, dtype=np.float32)
    if array.ndim == 2:
        return array.reshape(len(array), image_size, image_size, 3)
    if array.ndim == 4 and array.shape[1:] == (image_size, image_size, 3):
        return array
    raise ValueError(f"Unexpected render batch shape: {array.shape}")


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _refuse_existing(path: Path, *, kind: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{kind} destination already exists: {path}")
