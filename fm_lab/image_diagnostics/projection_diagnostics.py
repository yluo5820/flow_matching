"""Lightweight point diagnostics computed in each displayed projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_projection_diagnostics(
    projections: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    k_neighbors: int,
) -> pd.DataFrame:
    """Compute selection-oriented diagnostics for every 2D or 3D projection."""

    if projections["row_id"].tolist() != metadata["row_id"].tolist():
        raise ValueError("Projection and metadata row IDs must be aligned.")
    result = pd.DataFrame({"row_id": projections["row_id"].to_numpy()})
    labels = metadata.get("label", pd.Series([""] * len(metadata))).fillna("").astype(str)
    label_values = labels.to_numpy()
    row_ids = metadata["row_id"].to_numpy()
    for key, coordinate_columns in _projection_columns(projections):
        coordinates = projections[coordinate_columns].to_numpy(dtype=np.float64)
        count = len(coordinates)
        if count < 2:
            result[f"{key}_knn_radius_k{k_neighbors}"] = np.nan
            result[f"{key}_knn_mean_distance_k{k_neighbors}"] = np.nan
            result[f"{key}_label_agreement_k{k_neighbors}"] = np.nan
            result[f"{key}_distance_to_label_centroid"] = 0.0
            result[f"{key}_nearest_row_id"] = np.nan
            result[f"{key}_nearest_label"] = ""
            continue
        effective_k = min(k_neighbors, count - 1)
        distances, indices = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric="euclidean",
        ).fit(coordinates).kneighbors(coordinates)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        result[f"{key}_knn_radius_k{k_neighbors}"] = distances[:, -1]
        result[f"{key}_knn_mean_distance_k{k_neighbors}"] = distances.mean(axis=1)
        result[f"{key}_label_agreement_k{k_neighbors}"] = (
            label_values[indices] == label_values[:, None]
        ).mean(axis=1)
        result[f"{key}_distance_to_label_centroid"] = _label_centroid_distances(
            coordinates,
            label_values,
        )
        result[f"{key}_nearest_row_id"] = row_ids[indices[:, 0]]
        result[f"{key}_nearest_label"] = label_values[indices[:, 0]]
    return result


def _projection_columns(frame: pd.DataFrame) -> list[tuple[str, list[str]]]:
    columns = set(frame.columns)
    return [
        (
            str(column)[:-2],
            [
                f"{str(column)[:-2]}_{axis}"
                for axis in ("x", "y", "z")
                if f"{str(column)[:-2]}_{axis}" in columns
            ],
        )
        for column in frame.columns
        if str(column).endswith("_x") and f"{str(column)[:-2]}_y" in columns
    ]


def _label_centroid_distances(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    distances = np.zeros(len(coordinates), dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        centroid = coordinates[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(coordinates[mask] - centroid, axis=1)
    return distances
