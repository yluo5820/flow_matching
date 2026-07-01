"""Shared atlas-backed payload helpers for image projection viewers."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def atlas_data_url(path: Path) -> str:
    """Return an image atlas as a MIME-correct data URL."""

    mime_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported sprite atlas format: {path.suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def palette_payload(palette: dict[str, tuple[int, int, int]]) -> dict[str, str]:
    """Serialize an RGB palette into CSS color strings."""

    return {
        label: f"rgb({color[0]}, {color[1]}, {color[2]})"
        for label, color in palette.items()
    }


def projection_columns(
    frame: pd.DataFrame,
    *,
    projection_names: dict[str, str] | None = None,
    include_z: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Discover named x/y(/z) projection columns in display order."""

    projections: dict[str, tuple[str, ...]] = {}
    columns = set(frame.columns)
    discovered = [
        str(column)[:-2]
        for column in frame.columns
        if str(column).endswith("_x")
    ]
    ordered_keys = [
        *(
            key
            for key in (projection_names or {})
            if f"{key}_x" in columns
        ),
        *(key for key in discovered if key not in (projection_names or {})),
    ]
    for key in ordered_keys:
        x_column = f"{key}_x"
        y_column = f"{key}_y"
        z_column = f"{key}_z"
        if y_column not in columns:
            continue
        display_name = (
            projection_names.get(key, projection_display_name(key))
            if projection_names
            else projection_display_name(key)
        )
        projections[display_name] = (
            (x_column, y_column, z_column)
            if include_z and z_column in columns
            else (x_column, y_column)
        )
    return projections


def projection_dimensions(projections: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Return original projection dimensionality by display name."""

    return {name: len(columns) for name, columns in projections.items()}


def projection_point_payload(
    frame: pd.DataFrame,
    projections: dict[str, tuple[str, ...]],
    *,
    output_dimensions: int | None = None,
) -> list[dict[str, Any]]:
    """Build atlas point records with normalized coordinates for each projection."""

    normalized = {
        name: normalized_coordinates(
            frame,
            columns,
            output_dimensions=output_dimensions,
        )
        for name, columns in projections.items()
    }
    diagnostic_columns = sample_metric_columns(frame)
    points: list[dict[str, Any]] = []
    for position, row in frame.iterrows():
        point = atlas_point_fields(row, position)
        point["coordinates"] = {
            name: [float(value) for value in normalized[name][position]]
            for name in projections
        }
        point["details"] = {
            column: json_scalar(row.get(column))
            for column in diagnostic_columns
        }
        points.append(point)
    return points


def atlas_point_payload(
    frame: pd.DataFrame,
    *,
    coordinates: np.ndarray,
    start: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    """Build atlas point records for already-projected coordinate arrays."""

    total = len(frame) - start if count is None else count
    selected = frame.iloc[start : start + total].reset_index(drop=True)
    values = np.asarray(coordinates)
    if len(values) < len(selected):
        raise ValueError(
            f"Coordinate count {len(values)} is smaller than point count {len(selected)}."
        )
    points: list[dict[str, Any]] = []
    for position, row in selected.iterrows():
        point = atlas_point_fields(row, start + position)
        point["kind"] = str(row.get("kind", ""))
        point["labelSource"] = str(row.get("label_source", ""))
        point["coordinates"] = [float(value) for value in values[position]]
        points.append(point)
    return points


def projection_diagnostics_payload(
    frame: pd.DataFrame,
    projections: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Encode projection-level diagnostic vectors for browser-side lookup."""

    payload: dict[str, dict[str, dict[str, str]]] = {}
    for name, columns in projections.items():
        x_column = columns[0]
        projection_key = x_column[:-2]
        details: dict[str, dict[str, str]] = {}
        prefixes = (
            ("knn_radius_k", "kNN radius"),
            ("label_agreement_k", "Local label agreement"),
        )
        for suffix_prefix, label in prefixes:
            column = next(
                (
                    str(value)
                    for value in frame.columns
                    if str(value).startswith(f"{projection_key}_{suffix_prefix}")
                ),
                None,
            )
            if column:
                k_value = column.rsplit("k", 1)[-1]
                details[f"{label} (k={k_value})"] = float32_payload(frame[column])
        centroid_column = f"{projection_key}_distance_to_label_centroid"
        if centroid_column in frame:
            details["Distance to label centroid"] = float32_payload(
                frame[centroid_column]
            )
        payload[name] = details
    return payload


def sample_metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return compact sample-level diagnostics suitable for hover display."""

    prefixes = (
        "knn_radius_k",
        "knn_mean_distance_k",
        "participation_ratio_k",
        "mle_lid_k",
        "pca_dim_",
        "ball_scaling_dim_k",
        "ball_scaling_r2_k",
        "fm_jacobian_participation_rank_",
        "fm_jacobian_entropy_rank_",
        "fm_jacobian_threshold_rank_",
        "fm_flipd_lid_",
        "fm_flipd_divergence_",
        "fm_flipd_score_norm_",
        "diffusion_normal_bundle_lid_",
        "diffusion_normal_bundle_normal_dim_",
        "diffusion_flipd_lid_",
        "diffusion_flipd_divergence_",
    )
    exact = {
        "two_nn_lid",
        "two_nn_lid_local",
        "outlier_score",
        "distance_to_label_centroid",
    }
    return [
        str(column)
        for column in frame.columns
        if str(column) in exact or str(column).startswith(prefixes)
    ]


def normalized_coordinates(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    output_dimensions: int | None = None,
) -> np.ndarray:
    """Center and scale projection coordinates for browser rendering."""

    values = frame[list(columns)].to_numpy(dtype=np.float64, copy=True)
    values -= np.nanmean(values, axis=0, keepdims=True)
    maximum = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0
    normalized = np.nan_to_num(values / maximum * 20.0)
    if output_dimensions is not None and normalized.shape[1] < output_dimensions:
        padding = np.zeros(
            (len(normalized), output_dimensions - normalized.shape[1]),
            dtype=normalized.dtype,
        )
        normalized = np.column_stack([normalized, padding])
    return normalized


def normalize_coordinate_arrays(
    *arrays: np.ndarray,
    output_dimensions: int = 3,
) -> list[np.ndarray]:
    """Center/scale several coordinate arrays in one shared coordinate system."""

    reshaped = []
    for values in arrays:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            reshaped.append(array.reshape(*array.shape[:-1], output_dimensions))
            continue
        if array.shape[-1] < output_dimensions:
            padding = np.zeros(
                (*array.shape[:-1], output_dimensions - array.shape[-1]),
                dtype=array.dtype,
            )
            array = np.concatenate([array, padding], axis=-1)
        reshaped.append(array[..., :output_dimensions])
    flattened = [
        values.reshape(-1, output_dimensions)
        for values in reshaped
        if values.size > 0
    ]
    if not flattened:
        return [np.asarray(values, dtype=np.float32) for values in reshaped]
    combined = np.concatenate(flattened, axis=0)
    center = np.nanmean(combined, axis=0, keepdims=True)
    maximum = float(np.nanmax(np.abs(combined - center))) if combined.size else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0
    return [
        np.nan_to_num((values - center) / maximum * 20.0).astype(np.float32)
        for values in reshaped
    ]


def atlas_point_fields(row: pd.Series, position: int) -> dict[str, Any]:
    """Return the shared metadata and atlas tile location for one point."""

    return {
        "rowId": int(row.get("row_id", position)),
        "sourceIndex": json_scalar(row.get("source_index", position)),
        "label": str(row.get("label", "")),
        "dataset": str(row.get("dataset", "")),
        "atlas": int(row["atlas_index"]),
        "column": int(row["atlas_column"]),
        "row": int(row["atlas_row"]),
    }


def float32_payload(series: pd.Series) -> dict[str, str]:
    """Encode a numeric vector compactly for embedding in JSON."""

    values = np.asarray(series, dtype="<f4")
    return {
        "encoding": "float32-base64",
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def projection_display_name(key: str) -> str:
    if key == "tsne":
        return "T-SNE"
    if key == "umap":
        return "UMAP"
    if key == "pca":
        return "PCA"
    return key.replace("_", " ").title()


def json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
