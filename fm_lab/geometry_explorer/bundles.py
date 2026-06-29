"""Load registry-backed geometry explorer bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.image_diagnostics.canvas_explorer import (
    prepare_array_sprite_atlases,
    prepare_sprite_atlases,
)
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    atlas_point_payload,
    normalize_coordinate_arrays,
    palette_payload,
    projection_columns,
    projection_diagnostics_payload,
    projection_dimensions,
    projection_point_payload,
)
from fm_lab.image_diagnostics.save_utils import read_parquet
from fm_lab.image_diagnostics.trajectory_explorer import (
    _atlas_rows,
    _palette_with_trajectory_labels,
    _trajectory_labels,
    infer_generated_labels_from_target,
)
from fm_lab.utils.config import load_config


def load_projection_payload(
    view_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    """Load a registered dataset projection view into viewer payload JSON."""

    registry = GeometryRegistry(workspace)
    row = registry.get_projection_view(view_id)
    frame = read_parquet(registry.resolve(row["explorer_data_path"]))
    output_dir = registry.resolve(row["output_dir"])
    bundle = prepare_sprite_atlases(
        frame,
        output_dir=output_dir / "assets" / "atlases",
    )
    projection_names = json.loads(row["projection_names_json"] or "{}")
    projections = projection_columns(
        bundle.frame,
        projection_names=projection_names,
        include_z=True,
    )
    if not projections:
        raise ValueError(f"Projection view {view_id} has no projection columns.")
    return {
        "mode": "dataset",
        "points": projection_point_payload(
            bundle.frame,
            projections,
            output_dimensions=3,
        ),
        "trajectory": [],
        "trajectoryLabels": [],
        "trajectoryPreviews": [],
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "palette": palette_payload(bundle.palette),
        "projections": list(projections),
        "projectionDimensions": projection_dimensions(projections),
        "projectionDiagnostics": projection_diagnostics_payload(bundle.frame, projections),
        "tileSize": bundle.tile_size,
        "atlasSize": _atlas_size(bundle.atlas_paths),
        "atlasColumns": bundle.atlas_columns,
        "options": _default_options(),
        "counts": {
            "points": int(len(bundle.frame)),
            "trajectorySteps": 0,
            "trajectories": 0,
        },
    }


def load_trajectory_payload(
    view_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    """Load a registered trajectory view into viewer payload JSON."""

    registry = GeometryRegistry(workspace)
    row = registry.get_trajectory_view(view_id)
    coordinates = np.load(registry.resolve(row["coordinates_path"]))
    trajectory = np.asarray(coordinates["trajectory"], dtype=np.float32)
    target = np.asarray(coordinates["target"], dtype=np.float32)
    generated = np.asarray(coordinates["generated"], dtype=np.float32)
    target_images = _load_optional_array(registry, row["target_path"])
    generated_images = _load_optional_array(registry, row["generated_path"])
    raw_trajectory = _load_optional_array(registry, row["trajectory_path"])
    target_labels = _load_optional_array(registry, row["labels_path"])
    trajectory_images = raw_trajectory[-1] if raw_trajectory is not None else generated_images
    if target_images is None and generated_images is None:
        raise ValueError(f"Trajectory view {view_id} has no endpoint images.")

    generated_labels = None
    if (
        generated_images is not None
        and len(generated)
        and target_labels is not None
        and len(target)
    ):
        generated_labels = infer_generated_labels_from_target(
            generated=generated,
            target=target,
            target_labels=target_labels,
        )
    trajectory_labels = _trajectory_labels(
        trajectory=trajectory,
        target=target if len(target) else None,
        target_labels=target_labels,
    )
    (
        rows,
        atlas_images,
        endpoint_coordinates,
        n_endpoint_rows,
        n_trajectory_preview_rows,
    ) = _atlas_rows(
        target=target if target_images is not None else None,
        generated=generated if generated_images is not None else None,
        target_images=target_images,
        generated_images=generated_images,
        trajectory_images=trajectory_images,
        target_labels=target_labels,
        generated_labels=generated_labels,
        trajectory_labels=trajectory_labels,
        dataset_name=_dataset_name(registry, row),
    )
    if atlas_images is None:
        raise ValueError(f"Trajectory view {view_id} did not produce atlas images.")
    output_dir = registry.resolve(row["output_dir"])
    frame = pd.DataFrame(rows)
    image_shape, value_range = _image_metadata_for_trajectory(registry, row)
    bundle = prepare_array_sprite_atlases(
        frame,
        atlas_images,
        output_dir=output_dir / "assets" / "atlases",
        image_shape=image_shape,
        image_value_range=value_range,
    )
    normalized_endpoints, normalized_trajectory = normalize_coordinate_arrays(
        endpoint_coordinates,
        trajectory,
        output_dimensions=3,
    )
    palette = _palette_with_trajectory_labels(bundle.palette, trajectory_labels)
    return {
        "mode": "trajectory",
        "points": atlas_point_payload(
            bundle.frame,
            coordinates=normalized_endpoints,
            start=0,
            count=n_endpoint_rows,
        ),
        "trajectoryPreviews": atlas_point_payload(
            bundle.frame,
            coordinates=normalized_trajectory[-1],
            start=n_endpoint_rows,
            count=n_trajectory_preview_rows,
        ),
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "palette": palette_payload(palette),
        "projections": ["Trajectory UMAP"],
        "projectionDimensions": {"Trajectory UMAP": 3},
        "projectionDiagnostics": {"Trajectory UMAP": {}},
        "tileSize": bundle.tile_size,
        "atlasSize": _atlas_size(bundle.atlas_paths),
        "atlasColumns": bundle.atlas_columns,
        "trajectory": np.round(normalized_trajectory, 5).tolist(),
        "trajectoryLabels": trajectory_labels,
        "options": _default_options() | {
            "targetAlpha": 0.28,
            "generatedAlpha": 0.9,
            "lineAlpha": 0.34,
            "drawThumbnailsDefault": False,
        },
        "counts": {
            "points": int(n_endpoint_rows),
            "targets": int(len(target)),
            "generated": int(len(generated)),
            "trajectorySteps": int(trajectory.shape[0]),
            "trajectories": int(trajectory.shape[1]),
        },
    }


def _load_optional_array(registry: GeometryRegistry, value: str | None) -> np.ndarray | None:
    if not value:
        return None
    path = registry.resolve(value)
    if not path.exists():
        return None
    return np.load(path)


def _atlas_size(paths: list[Path]) -> int:
    if not paths:
        return 1
    with Image.open(paths[0]) as image:
        return int(image.width)


def _dataset_name(registry: GeometryRegistry, row: Any) -> str:
    variant_id = row["variant_id"]
    if not variant_id:
        return ""
    try:
        variant = registry.get_dataset_variant(variant_id)
    except KeyError:
        return str(variant_id).split("/", 1)[0]
    return str(variant["family"])


def _image_metadata_for_trajectory(
    registry: GeometryRegistry,
    row: Any,
) -> tuple[list[int] | None, tuple[float, float]]:
    variant_id = row["variant_id"]
    if variant_id:
        try:
            variant = registry.get_dataset_variant(variant_id)
            image_shape = (
                json.loads(variant["image_shape_json"])
                if variant["image_shape_json"]
                else None
            )
            value_range = (
                tuple(float(value) for value in json.loads(variant["value_range_json"]))
                if variant["value_range_json"]
                else (0.0, 1.0)
            )
            return image_shape, value_range
        except KeyError:
            pass
    run_dir = registry.resolve(row["output_dir"]).parent
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        data_config = config.get("data", {})
        if str(data_config.get("name", "")).lower() == "mnist":
            normalize = str(data_config.get("normalize", "zero_one")).lower()
            value_range = (
                (-1.0, 1.0)
                if normalize in {"minus_one_one", "-1_1", "centered"}
                else (0.0, 1.0)
            )
            return [28, 28], value_range
    target_path = registry.resolve(row["target_path"]) if row["target_path"] else None
    return _infer_image_shape(target_path), (0.0, 1.0)


def _infer_image_shape(path: Path | None) -> list[int] | None:
    if path is None or not path.exists():
        return None
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2:
        return None
    side = int(round(math.sqrt(values.shape[1])))
    return [side, side] if side * side == values.shape[1] else None


def _default_options() -> dict[str, Any]:
    return {
        "pointSize": 14,
        "hoverSize": 58,
        "previewMode": "original",
        "drawThumbnailsDefault": True,
        "scalePointSizeWithZoom": True,
    }
