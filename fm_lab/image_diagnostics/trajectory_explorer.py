"""Explorer-style HTML renderer for projected sampling trajectories."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.geometry_explorer.viewer import build_geometry_html
from fm_lab.image_diagnostics.canvas_explorer import prepare_array_sprite_atlases
from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    atlas_point_payload,
    normalize_coordinate_arrays,
    palette_payload,
)
from fm_lab.image_diagnostics.palette import LABEL_PALETTE


def write_trajectory_explorer_html(
    output_path: str | Path,
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    trajectory_images: np.ndarray | None = None,
    target_labels: np.ndarray | None = None,
    generated_labels: np.ndarray | None = None,
    image_shape: list[int] | tuple[int, ...] | None = None,
    image_value_range: list[float] | tuple[float, float] = (0.0, 1.0),
    dataset_name: str = "mnist",
    height: int = 760,
    config: ExplorerConfig | None = None,
) -> dict[str, Any]:
    """Write a self-contained trajectory explorer HTML page."""

    trajectory = _as_projected_trajectory(trajectory)
    target = _as_projected_points(target)
    generated = _as_projected_points(generated)
    trajectory_labels = _trajectory_labels(
        trajectory=trajectory,
        target=target,
        target_labels=target_labels,
    )
    (
        rows,
        atlas_images,
        endpoint_coordinates,
        n_endpoint_rows,
        n_trajectory_preview_rows,
    ) = _atlas_rows(
        target=target,
        generated=generated,
        target_images=target_images,
        generated_images=generated_images,
        trajectory_images=trajectory_images,
        target_labels=target_labels,
        generated_labels=generated_labels,
        trajectory_labels=trajectory_labels,
        dataset_name=dataset_name,
    )
    if atlas_images is None or endpoint_coordinates is None:
        raise ValueError("Trajectory explorer requires target or generated endpoint images.")

    explorer_config = config or ExplorerConfig(height=height)
    output_path = Path(output_path)
    frame = pd.DataFrame(rows)
    bundle = prepare_array_sprite_atlases(
        frame,
        atlas_images,
        output_dir=output_path.parent / "assets" / "trajectory_atlases",
        image_shape=image_shape,
        image_value_range=image_value_range,
        tile_size=explorer_config.atlas_tile_size,
        max_atlas_size=explorer_config.atlas_size,
    )
    palette = _palette_with_trajectory_labels(bundle.palette, trajectory_labels)
    normalized_endpoints, normalized_trajectory = normalize_coordinate_arrays(
        endpoint_coordinates,
        trajectory,
        output_dimensions=3,
    )
    payload = {
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
        "tileSize": bundle.tile_size,
        "atlasSize": _atlas_size(bundle.atlas_paths),
        "atlasColumns": bundle.atlas_columns,
        "trajectory": np.round(normalized_trajectory, 5).tolist(),
        "trajectoryLabels": trajectory_labels,
        "projections": ["Trajectory UMAP"],
        "projectionDimensions": {"Trajectory UMAP": 3},
        "projectionDiagnostics": {"Trajectory UMAP": {}},
        "options": {
            "pointSize": explorer_config.point_size,
            "hoverSize": explorer_config.hover_size,
            "previewMode": explorer_config.preview_mode,
            "targetAlpha": 0.28,
            "generatedAlpha": 0.9,
            "lineAlpha": 0.34,
            "drawThumbnailsDefault": False,
        },
        "counts": {
            "targets": 0 if target is None else int(len(target)),
            "generated": 0 if generated is None else int(len(generated)),
            "trajectorySteps": int(trajectory.shape[0]),
            "trajectories": int(trajectory.shape[1]),
            "trajectoryPreviews": int(n_trajectory_preview_rows),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_geometry_html(
            payload,
            height=height,
            vendor_dir=output_path.parent / "assets" / "vendor",
        ),
        encoding="utf-8",
    )
    return {
        "endpoint_points": int(n_endpoint_rows),
        "target_endpoint_points": int(payload["counts"]["targets"]),
        "generated_endpoint_points": int(payload["counts"]["generated"]),
        "trajectory_preview_points": int(n_trajectory_preview_rows),
        "atlas_points": int(len(frame)),
        "atlas_count": int(len(bundle.atlas_paths)),
    }


def _as_projected_trajectory(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Trajectory explorer requires shape (steps, paths, 3), got {array.shape}.")
    return array


def _atlas_size(paths: list[Path]) -> int:
    if not paths:
        return 1
    with Image.open(paths[0]) as image:
        return int(image.width)


def _as_projected_points(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"Endpoint coordinates require shape (n, 3), got {array.shape}.")
    return array


def _atlas_rows(
    *,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    trajectory_images: np.ndarray | None,
    target_labels: np.ndarray | None,
    generated_labels: np.ndarray | None,
    trajectory_labels: list[str],
    dataset_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None, int, int]:
    rows: list[dict[str, Any]] = []
    image_blocks: list[np.ndarray] = []
    coordinate_blocks: list[np.ndarray] = []

    if target is not None and target_images is not None and len(target) > 0:
        target_images = np.asarray(target_images)
        count = min(len(target), len(target_images))
        labels = _string_labels(target_labels, count=count, fallback="target")
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "target",
                    "label_source": "target label" if target_labels is not None else "none",
                }
            )
        image_blocks.append(target_images[:count])
        coordinate_blocks.append(target[:count])

    if generated is not None and generated_images is not None and len(generated) > 0:
        generated_images = np.asarray(generated_images)
        count = min(len(generated), len(generated_images))
        labels = _string_labels(generated_labels, count=count, fallback="generated")
        label_source = "predicted/inferred" if generated_labels is not None else "none"
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "generated",
                    "label_source": label_source,
                }
            )
        image_blocks.append(generated_images[:count])
        coordinate_blocks.append(generated[:count])

    endpoint_rows = len(rows)
    if trajectory_images is not None:
        trajectory_images = np.asarray(trajectory_images)
        count = min(len(trajectory_images), len(trajectory_labels))
        labels = trajectory_labels[:count]
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "trajectory",
                    "label_source": "nearest target final"
                    if labels[index] != "trajectory"
                    else "none",
                }
            )
        image_blocks.append(trajectory_images[:count])

    if not image_blocks:
        return rows, None, None, 0, 0
    endpoint_coordinates = (
        np.concatenate(coordinate_blocks, axis=0)
        if coordinate_blocks
        else np.empty((0, 3), dtype=np.float32)
    )
    return (
        rows,
        np.concatenate(image_blocks, axis=0),
        endpoint_coordinates,
        endpoint_rows,
        len(rows) - endpoint_rows,
    )


def _string_labels(
    labels: np.ndarray | None,
    *,
    count: int,
    fallback: str,
) -> list[str]:
    if labels is None:
        return [fallback] * count
    labels = np.asarray(labels)[:count]
    return [str(value.item() if isinstance(value, np.generic) else value) for value in labels]


def _trajectory_labels(
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
    target_labels: np.ndarray | None,
) -> list[str]:
    if target is None or target_labels is None or len(target) == 0:
        return ["trajectory"] * trajectory.shape[1]
    labels = _nearest_projected_labels(
        query=trajectory[-1],
        reference=target,
        labels=target_labels,
    )
    return _string_labels(labels, count=trajectory.shape[1], fallback="trajectory")


def infer_generated_labels_from_target(
    *,
    generated: np.ndarray | None,
    target: np.ndarray | None,
    target_labels: np.ndarray | None,
) -> np.ndarray | None:
    """Assign generated endpoints the nearest target label in projected space."""

    if generated is None or target is None or target_labels is None:
        return None
    if len(generated) == 0 or len(target) == 0:
        return None
    return _nearest_projected_labels(
        query=np.asarray(generated, dtype=np.float32),
        reference=np.asarray(target, dtype=np.float32),
        labels=np.asarray(target_labels),
    )


def _nearest_projected_labels(
    *,
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=1, algorithm="auto")
        model.fit(reference)
        indices = model.kneighbors(query, return_distance=False)[:, 0]
    except ImportError:
        indices = _nearest_indices_numpy(query=query, reference=reference)
    return np.asarray(labels)[indices]


def _nearest_indices_numpy(
    *,
    query: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    reference = np.asarray(reference, dtype=np.float32)
    for start in range(0, len(query), chunk_size):
        chunk = np.asarray(query[start : start + chunk_size], dtype=np.float32)
        distances = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.argmin(distances, axis=1))
    return np.concatenate(chunks, axis=0)


def _palette_with_trajectory_labels(
    palette: dict[str, tuple[int, int, int]],
    trajectory_labels: list[str],
) -> dict[str, tuple[int, int, int]]:
    updated = dict(palette)
    for label in trajectory_labels:
        if label not in updated:
            updated[label] = LABEL_PALETTE[len(updated) % len(LABEL_PALETTE)]
    return updated
