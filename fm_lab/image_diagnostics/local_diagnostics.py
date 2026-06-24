"""Local geometric diagnostics in image-embedding space."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from fm_lab.image_diagnostics.config import LocalDiagnosticsConfig

LOGGER = logging.getLogger("fm_lab.image_diagnostics")


def compute_or_load_local_diagnostics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: LocalDiagnosticsConfig,
    output_dir: Path,
    *,
    feature_name: str,
    save: bool = True,
) -> pd.DataFrame:
    """Load cached diagnostics or compute them in the selected embedding space."""

    path = output_dir / "diagnostics" / f"{feature_name}_local_diagnostics.csv"
    if save and config.skip_existing and path.exists():
        frame = pd.read_csv(path)
        if "row_id" not in frame:
            raise RuntimeError(f"Cached diagnostics have no row_id column: {path}")
        if frame["row_id"].tolist() != metadata["row_id"].tolist():
            raise RuntimeError(
                "Cached diagnostics row IDs do not match the embedding cache. "
                "Recompute diagnostics."
            )
        LOGGER.info("Loaded cached local diagnostics: %s", path)
        return frame

    frame = compute_local_diagnostics(embeddings, metadata, config)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        LOGGER.info("Saved local diagnostics: %s", path)
    return frame


def compute_local_diagnostics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: LocalDiagnosticsConfig,
) -> pd.DataFrame:
    """Compute neighborhood, local spectrum, centroid, LID, and LOF metrics."""

    if len(embeddings) != len(metadata):
        raise ValueError("Feature and metadata row counts must match.")
    result = metadata[
        [
            column
            for column in (
                "row_id",
                "image_path",
                "prompt_id",
                "family",
                "seed",
                "model_repo_id",
                "status",
            )
            if column in metadata
        ]
    ].copy()
    n_samples = len(embeddings)
    radius_column = f"knn_radius_k{config.k_neighbors}"
    mean_column = f"knn_mean_distance_k{config.k_neighbors}"
    pr_column = f"participation_ratio_k{config.k_neighbors}"
    if n_samples < 2:
        result[radius_column] = np.nan
        result[mean_column] = np.nan
        _add_empty_spectrum(result, config)
        result[pr_column] = np.nan
        result["two_nn_lid"] = np.nan
        result["distance_to_label_centroid"] = 0.0
        result["distance_to_prompt_centroid"] = 0.0
        result["distance_to_family_centroid"] = 0.0
        result["outlier_score"] = np.nan
        return result

    effective_k = min(config.k_neighbors, n_samples - 1)
    neighbors = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric=config.metric,
    ).fit(embeddings)
    all_distances, all_indices = neighbors.kneighbors(embeddings)
    distances = all_distances[:, 1:]
    indices = all_indices[:, 1:]

    if config.compute_knn_radius:
        result[radius_column] = distances[:, -1]
        result[mean_column] = distances.mean(axis=1)
    else:
        result[radius_column] = np.nan
        result[mean_column] = np.nan

    full_spectra = _local_covariance_spectra(
        embeddings,
        indices,
        max(config.covariance_eigenvalues, effective_k),
    )
    for index in range(config.covariance_eigenvalues):
        result[f"local_eig_{index + 1}"] = (
            full_spectra[:, index] if config.compute_covariance_spectrum else np.nan
        )
    if config.compute_participation_ratio:
        numerator = full_spectra.sum(axis=1) ** 2
        denominator = (full_spectra**2).sum(axis=1)
        result[pr_column] = np.divide(
            numerator,
            denominator,
            out=np.full(n_samples, np.nan),
            where=denominator > 0,
        )
    else:
        result[pr_column] = np.nan

    result["two_nn_lid"] = (
        _two_nn_lid(distances) if config.compute_two_nn_lid else np.nan
    )
    if config.compute_centroid_distances:
        result["distance_to_label_centroid"] = _group_centroid_distances(
            embeddings,
            metadata.get("label", pd.Series([""] * n_samples)),
            config.metric,
        )
        result["distance_to_prompt_centroid"] = _group_centroid_distances(
            embeddings,
            metadata.get("prompt_id", pd.Series([""] * n_samples)),
            config.metric,
        )
        result["distance_to_family_centroid"] = _group_centroid_distances(
            embeddings,
            metadata.get("family", pd.Series([""] * n_samples)),
            config.metric,
        )
    else:
        result["distance_to_label_centroid"] = np.nan
        result["distance_to_prompt_centroid"] = np.nan
        result["distance_to_family_centroid"] = np.nan

    result["outlier_score"] = (
        _lof_scores(embeddings, effective_k, config.metric)
        if config.compute_outlier_score
        else np.nan
    )
    return result


def _local_covariance_spectra(
    embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    n_values: int,
) -> np.ndarray:
    spectra = np.zeros((len(embeddings), n_values), dtype=np.float64)
    for row_index, local_indices in enumerate(neighbor_indices):
        local = embeddings[local_indices]
        if len(local) < 2:
            continue
        centered = local - local.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = (singular_values**2) / max(1, len(local) - 1)
        count = min(n_values, len(eigenvalues))
        spectra[row_index, :count] = eigenvalues[:count]
    return spectra


def _two_nn_lid(distances: np.ndarray) -> np.ndarray:
    if distances.shape[1] < 2:
        return np.full(len(distances), np.nan)
    first = distances[:, 0]
    second = distances[:, 1]
    valid = (first > 0) & (second > first)
    estimates = np.full(len(distances), np.nan)
    estimates[valid] = 1.0 / np.log(second[valid] / first[valid])
    return estimates


def _group_centroid_distances(
    embeddings: np.ndarray,
    groups: pd.Series,
    metric: str,
) -> np.ndarray:
    values = groups.fillna("").astype(str).to_numpy()
    distances = np.zeros(len(embeddings), dtype=np.float64)
    for group in np.unique(values):
        mask = values == group
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        distances[mask] = pairwise_distances(
            embeddings[mask],
            centroid,
            metric=metric,
        )[:, 0]
    return distances


def _lof_scores(embeddings: np.ndarray, effective_k: int, metric: str) -> np.ndarray:
    if len(embeddings) < 3:
        return np.full(len(embeddings), np.nan)
    neighbors = min(effective_k, len(embeddings) - 1)
    model = LocalOutlierFactor(n_neighbors=neighbors, metric=metric)
    model.fit_predict(embeddings)
    return -model.negative_outlier_factor_


def _add_empty_spectrum(
    frame: pd.DataFrame,
    config: LocalDiagnosticsConfig,
) -> None:
    for index in range(config.covariance_eigenvalues):
        frame[f"local_eig_{index + 1}"] = np.nan
