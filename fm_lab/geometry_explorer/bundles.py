"""Load registry-backed geometry explorer bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.geometry_explorer.display import metric_label
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.image_diagnostics.canvas_explorer import (
    prepare_array_sprite_atlases,
    prepare_sprite_atlases,
)
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    atlas_point_payload,
    json_scalar,
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

GROUP_ID_PREFERRED_METRICS = (
    "global_mle_lid_k20",
    "global_mle_lid_k10",
    "global_two_nn_lid",
    "global_participation_ratio",
    "global_pca_dim_80",
    "global_pca_dim_90",
    "global_pca_dim_95",
    "global_pca_dim_99",
    "correlation_dimension",
    "ball_scaling_dim",
    "ball_scaling_r2",
    "median_local_mle_lid_k15",
    "mean_local_mle_lid_k15",
    "median_participation_ratio_k15",
    "mean_participation_ratio_k15",
    "median_two_nn_lid_local",
    "mean_two_nn_lid_local",
)
GROUP_ID_META_COLUMNS = {
    "groupby_column",
    "group_value",
    "n_samples",
    "feature_space",
    "id_feature_fingerprint",
    "ball_scaling_num_radii",
}
GROUP_ID_PRIMARY_CANDIDATES = (
    "global_mle_lid_k20",
    "global_mle_lid_k10",
    "mean_fm_jacobian_participation_rank_t0900",
    "mean_fm_jacobian_participation_rank_t0800",
    "mean_fm_flipd_lid_t0900",
    "mean_fm_flipd_lid_t0800",
    "mean_diffusion_normal_bundle_lid_t0900",
    "mean_diffusion_normal_bundle_lid_t0800",
    "mean_diffusion_flipd_lid_t0900",
    "mean_diffusion_flipd_lid_t0800",
    "correlation_dimension",
    "ball_scaling_dim",
    "global_participation_ratio",
    "median_local_mle_lid_k15",
)
MODEL_GROUP_METRIC_PREFIXES = (
    "mean_fm_jacobian_",
    "median_fm_jacobian_",
    "std_fm_jacobian_",
    "mean_fm_flipd_",
    "median_fm_flipd_",
    "std_fm_flipd_",
    "mean_diffusion_normal_bundle_",
    "median_diffusion_normal_bundle_",
    "std_diffusion_normal_bundle_",
    "mean_diffusion_flipd_",
    "median_diffusion_flipd_",
    "std_diffusion_flipd_",
)


def load_projection_payload(
    view_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    """Load a registered dataset projection view into viewer payload JSON."""

    registry = GeometryRegistry(workspace)
    indexed = registry.projection_payload(view_id)
    if indexed is not None:
        payload = _projection_payload_from_index(indexed)
        group_diagnostics = _group_diagnostics_payload(
            registry,
            registry.get_projection_view(view_id),
        )
        payload["groupDiagnostics"] = group_diagnostics
        payload["metricLabels"].update(group_diagnostics.get("metricLabels", {}))
        return payload
    return build_and_register_projection_payload_index(view_id, workspace=workspace)


def load_projection_group_diagnostics(
    view_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    """Load class/global intrinsic-dimension summaries for one projection view."""

    registry = GeometryRegistry(workspace)
    return _group_diagnostics_payload(registry, registry.get_projection_view(view_id))


def build_and_register_projection_payload_index(
    view_id: str,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    """Build and register the SQLite-backed projection payload index."""

    registry = GeometryRegistry(workspace)
    row = registry.get_projection_view(view_id)
    payload, index = _projection_payload_from_files(registry, row)
    registry.register_projection_payload(view_id=view_id, **index)
    return payload


def _projection_payload_from_files(
    registry: GeometryRegistry,
    row: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        raise ValueError(
            f"Projection view {row['view_id']} has no projection columns."
        )
    points = projection_point_payload(
        bundle.frame,
        projections,
        output_dimensions=3,
    )
    projection_dimensions_payload = projection_dimensions(projections)
    projection_diagnostics = projection_diagnostics_payload(bundle.frame, projections)
    palette = palette_payload(bundle.palette)
    metric_labels = _metric_labels(points, projection_diagnostics)
    group_diagnostics = _group_diagnostics_payload(registry, row)
    metric_labels.update(group_diagnostics.get("metricLabels", {}))
    atlas_size = _atlas_size(bundle.atlas_paths)
    payload = {
        "mode": "dataset",
        "points": points,
        "trajectory": [],
        "trajectoryLabels": [],
        "trajectoryPreviews": [],
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "palette": palette,
        "projections": list(projections),
        "projectionDimensions": projection_dimensions_payload,
        "projectionDiagnostics": projection_diagnostics,
        "groupDiagnostics": group_diagnostics,
        "metricLabels": metric_labels,
        "tileSize": bundle.tile_size,
        "atlasSize": atlas_size,
        "atlasColumns": bundle.atlas_columns,
        "options": _default_options(),
        "counts": {
            "points": int(len(bundle.frame)),
            "trajectorySteps": 0,
            "trajectories": 0,
        },
    }
    index = {
        "points": points,
        "atlas_paths": bundle.atlas_paths,
        "palette": palette,
        "projections": list(projections),
        "projection_dimensions": projection_dimensions_payload,
        "projection_diagnostics": projection_diagnostics,
        "metric_labels": metric_labels,
        "tile_size": bundle.tile_size,
        "atlas_size": atlas_size,
        "atlas_columns": bundle.atlas_columns,
    }
    return payload, index


def _projection_payload_from_index(indexed: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "dataset",
        "points": indexed["points"],
        "trajectory": [],
        "trajectoryLabels": [],
        "trajectoryPreviews": [],
        "atlases": [atlas_data_url(path) for path in indexed["atlas_paths"]],
        "palette": indexed["palette"],
        "projections": indexed["projections"],
        "projectionDimensions": indexed["projection_dimensions"],
        "projectionDiagnostics": indexed["projection_diagnostics"],
        "groupDiagnostics": {},
        "metricLabels": indexed.get("metric_labels")
        or _metric_labels(indexed["points"], indexed["projection_diagnostics"]),
        "tileSize": indexed["tile_size"],
        "atlasSize": indexed["atlas_size"],
        "atlasColumns": indexed["atlas_columns"],
        "options": _default_options(),
        "counts": {
            "points": int(indexed["point_count"]),
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
    group_diagnostics = _variant_group_diagnostics_payload(registry, row["variant_id"])
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
        "groupDiagnostics": group_diagnostics,
        "metricLabels": group_diagnostics.get("metricLabels", {}),
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


def _group_diagnostics_payload(
    registry: GeometryRegistry,
    row: Any,
) -> dict[str, Any]:
    frames = []
    paths = []
    for path in _find_group_id_paths(registry, row):
        try:
            frame = _read_group_id_frame(path)
        except Exception:
            continue
        if frame.empty or "groupby_column" not in frame or "group_value" not in frame:
            continue
        if not _group_metric_columns(frame):
            continue
        frames.append(frame)
        paths.append(path)
    if not frames:
        return {}
    frame = _merge_group_id_frames(frames)
    if frame.empty or "groupby_column" not in frame or "group_value" not in frame:
        return {}
    metrics = _group_metric_columns(frame)
    if not metrics:
        return {}
    labels = {column: metric_label(column) for column in metrics}
    model_metrics = _model_group_metric_columns(metrics)
    overall_rows = frame[
        (frame["groupby_column"].astype(str) == "__all__")
        & (frame["group_value"].astype(str) == "__all__")
    ]
    label_rows = frame[frame["groupby_column"].astype(str) == "label"]
    groups = {
        str(series.get("group_value", "")): _group_row_payload(series, metrics)
        for _, series in label_rows.iterrows()
    }
    overall = (
        _group_row_payload(overall_rows.iloc[0], metrics)
        if not overall_rows.empty
        else None
    )
    total_samples = _total_group_samples(overall, groups)
    for values in groups.values():
        samples = values.get("n_samples")
        if isinstance(samples, int | float) and total_samples > 0:
            values["class_share"] = samples / total_samples
    if overall is not None:
        overall["class_share"] = 1.0
    if not groups and overall is None:
        return {}
    return {
        "source": _display_path(paths[0], registry.workspace),
        "sources": [_display_path(path, registry.workspace) for path in paths],
        "primaryMetric": _primary_group_metric(metrics),
        "metrics": metrics,
        "modelMetrics": model_metrics,
        "metricLabels": labels,
        "overall": overall,
        "groups": groups,
    }


def _variant_group_diagnostics_payload(
    registry: GeometryRegistry,
    variant_id: str | None,
) -> dict[str, Any]:
    if not variant_id:
        return {}
    views = registry.projection_views(str(variant_id))
    if not views:
        return {}
    return _group_diagnostics_payload(registry, registry.get_projection_view(views[0].view_id))


def _merge_group_id_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    seen_metrics: set[str] = set()
    join_columns = ["groupby_column", "group_value"]
    for frame in frames:
        metrics = [column for column in _group_metric_columns(frame) if column not in seen_metrics]
        if merged is None:
            columns = [*join_columns]
            for column in ("n_samples", "feature_space"):
                if column in frame:
                    columns.append(column)
            columns.extend(metrics)
            merged = frame[columns].copy()
        elif metrics:
            merged = merged.merge(
                frame[[*join_columns, *metrics]],
                on=join_columns,
                how="outer",
            )
        seen_metrics.update(metrics)
    return merged if merged is not None else pd.DataFrame(columns=join_columns)


def _group_metric_columns(frame: pd.DataFrame) -> list[str]:
    available = [
        str(column)
        for column in frame.columns
        if str(column) not in GROUP_ID_META_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    preferred = [column for column in GROUP_ID_PREFERRED_METRICS if column in available]
    model_metrics = [
        column
        for column in sorted(available)
        if column not in preferred
        and column.startswith(MODEL_GROUP_METRIC_PREFIXES)
    ]
    remaining = sorted(
        column for column in available if column not in preferred and column not in model_metrics
    )
    return [*preferred, *model_metrics, *remaining]


def _model_group_metric_columns(metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if metric.startswith(MODEL_GROUP_METRIC_PREFIXES)]


def _primary_group_metric(metrics: list[str]) -> str:
    for column in GROUP_ID_PRIMARY_CANDIDATES:
        if column in metrics:
            return column
    for prefix in (
        "median_fm_jacobian_participation_rank_",
        "mean_fm_jacobian_participation_rank_",
    ):
        for column in metrics:
            if column.startswith(prefix):
                return column
    return metrics[0]


def _total_group_samples(
    overall: dict[str, Any] | None,
    groups: dict[str, dict[str, Any]],
) -> float:
    if overall is not None:
        samples = overall.get("n_samples")
        if isinstance(samples, int | float):
            return float(samples)
    return float(
        sum(
            float(values["n_samples"])
            for values in groups.values()
            if isinstance(values.get("n_samples"), int | float)
        )
    )


def _find_group_id_paths(registry: GeometryRegistry, row: Any) -> list[Path]:
    output_dir = registry.resolve(row["output_dir"])
    id_root = output_dir / "id_estimation"
    paths: list[Path] = []
    manifests = sorted(
        id_root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = data.get("group_id_path")
        if not value:
            continue
        for candidate in _path_candidates(Path(str(value)), output_dir, manifest.parent):
            if candidate.is_file():
                paths.append(candidate)
    candidates = [
        *id_root.glob("*/intrinsic_dimension/group_id_*.csv"),
        *id_root.glob("*/intrinsic_dimension/group_id_*.parquet"),
    ]
    paths.extend(candidates)
    return _dedupe_group_id_paths(paths)


def _dedupe_group_id_paths(paths: list[Path]) -> list[Path]:
    by_stem: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        key = str(resolved.with_suffix(""))
        current = by_stem.get(key)
        if (
            current is None
            or _group_id_extension_priority(resolved) < _group_id_extension_priority(current)
        ):
            by_stem[key] = resolved
    return sorted(by_stem.values(), key=_group_id_path_sort_key)


def _group_id_path_sort_key(path: Path) -> tuple[int, str]:
    text = str(path)
    is_model = int("model_diagnostics_" in text)
    return (is_model, text)


def _group_id_extension_priority(path: Path) -> int:
    return 0 if path.suffix.lower() == ".parquet" else 1


def _path_candidates(path: Path, output_dir: Path, manifest_dir: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    return [output_dir / path, manifest_dir / path, path]


def _read_group_id_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return read_parquet(path)
    return pd.read_csv(path)


def _group_row_payload(row: pd.Series, metrics: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in ("n_samples", "feature_space", *metrics):
        if column in row:
            payload[column] = json_scalar(row[column])
    return payload


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _load_optional_array(registry: GeometryRegistry, value: str | None) -> np.ndarray | None:
    if not value:
        return None
    path = registry.resolve(value)
    if not path.exists():
        return None
    return np.load(path)


def _metric_labels(
    points: list[dict[str, Any]],
    projection_diagnostics: dict[str, Any],
) -> dict[str, str]:
    keys = {
        key
        for point in points
        for key in (point.get("details") or {})
    }
    keys.update(
        key
        for diagnostics in projection_diagnostics.values()
        for key in diagnostics
    )
    return {key: metric_label(key) for key in sorted(keys)}


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
