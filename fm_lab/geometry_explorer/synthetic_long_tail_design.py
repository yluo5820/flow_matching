"""Factorial design and shared master pools for the synthetic long-tail study."""

from __future__ import annotations

import hashlib
import json
import os
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
    """Build one of the three approved latent spaces."""

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
                frequency_name="balanced",
                class_counts=(balanced_count,) * len(OBJECT_IDS),
            )
        )
        for frequency_index, class_counts in enumerate(frequency_mappings):
            manifests.append(
                _condition_spec(
                    replicate=replicate,
                    geometry_index=geometry_index,
                    dimensions=dimensions,
                    frequency_name=f"frequency_{frequency_index}",
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
    replicate_root = Path(root) / f"replicate_{replicate:03d}"
    cells = []
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
            cell_dir = replicate_root / "pools" / object_id / dimension_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            image_path = (cell_dir / "images.npy").resolve()
            factor_path = (cell_dir / "factors.npy").resolve()
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
                    image_path=str(image_path),
                    factor_path=str(factor_path),
                    seed=seed,
                    config_hash=config_hash,
                )
            )
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
        (object_id, dimension_id)
        for object_id in OBJECT_IDS
        for dimension_id in DIMENSION_IDS
    }
    if set(by_cell) != expected_cells:
        raise ValueError("pool_cells must contain exactly the nine object-dimension cells.")
    if any(int(count) <= 0 for count in counts):
        raise ValueError("Condition counts must be positive.")
    if max(counts) > min(cell.count for cell in pool_cells):
        raise ValueError("Condition counts cannot exceed the shared master pool count.")

    manifest_dir = Path(root) / f"replicate_{replicate:03d}" / "conditions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    config_hashes = {cell.config_hash for cell in pool_cells}
    if len(config_hashes) != 1:
        raise ValueError("All pool cells must share one config hash.")
    config_hash = next(iter(config_hashes))
    image_shapes = {cell.image_shape for cell in pool_cells}
    if len(image_shapes) != 1:
        raise ValueError("All pool cells must share one image shape.")
    image_shape = next(iter(image_shapes))

    paths = []
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
        paths.append(manifest.write(manifest_dir / f"{manifest.condition_id}.json"))
    return tuple(paths)


def _condition_spec(
    *,
    replicate: int,
    geometry_index: int,
    dimensions: tuple[str, str, str],
    frequency_name: str,
    class_counts: tuple[int, int, int],
) -> ConditionManifest:
    geometry_name = f"geometry_{geometry_index}"
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
    condition_id = f"replicate_{replicate:03d}__{geometry_name}__{frequency_name}"
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
                lightness,
                chroma,
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
