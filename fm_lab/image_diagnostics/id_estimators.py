"""Classical local and global intrinsic-dimension estimators."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from fm_lab.image_diagnostics.id_config import IDEstimationConfig

LOGGER = logging.getLogger("fm_lab.image_diagnostics")
EPSILON = 1e-12


@dataclass(frozen=True)
class NeighborGraph:
    distances: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class ScalingEstimate:
    dimension: float
    r2: float
    curve: pd.DataFrame


def compute_local_id(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: IDEstimationConfig,
    *,
    feature_space: str,
) -> pd.DataFrame:
    """Compute local ID estimates across all configured neighborhood sizes."""

    if len(features) != len(metadata):
        raise ValueError("Feature and metadata row counts must match.")
    selected_columns = [
        column
        for column in (
            "row_id",
            "image_path",
            "prompt_id",
            "family",
            "seed",
            "label",
            "manual_label",
            "model_repo_id",
            "status",
        )
        if column in metadata
    ]
    result = metadata[selected_columns].copy()
    n_samples = len(features)
    columns: dict[str, np.ndarray] = {
        "feature_space": np.full(n_samples, feature_space, dtype=object)
    }
    k_values = sorted(set(config.local_id.k_values))
    if n_samples < 2:
        _add_empty_local_columns(columns, config, n_samples=n_samples)
        return pd.concat([result, pd.DataFrame(columns)], axis=1)

    graph = compute_neighbor_graph(
        features,
        max_neighbors=min(max(k_values), n_samples - 1),
        metric=config.distance.metric,
    )
    columns["two_nn_lid_local"] = (
        two_nn_local(graph.distances)
        if config.local_id.estimators.two_nn
        else np.full(n_samples, np.nan)
    )
    for requested_k in k_values:
        effective_k = min(requested_k, graph.distances.shape[1])
        distances = graph.distances[:, :effective_k]
        indices = graph.indices[:, :effective_k]
        suffix = f"k{requested_k}"
        columns[f"knn_radius_{suffix}"] = (
            distances[:, -1] if effective_k else np.full(n_samples, np.nan)
        )
        columns[f"knn_mean_distance_{suffix}"] = (
            distances.mean(axis=1) if effective_k else np.full(n_samples, np.nan)
        )
        if effective_k < 2:
            _add_empty_k_columns(
                columns,
                config,
                requested_k,
                n_samples=n_samples,
            )
            continue

        spectra = local_covariance_spectra(
            features,
            indices,
            max_values=max(
                config.local_id.covariance_eigenvalues,
                effective_k - 1,
            ),
        )
        if config.local_id.estimators.covariance_spectrum:
            for index in range(config.local_id.covariance_eigenvalues):
                columns[f"local_eig_{index + 1}_{suffix}"] = spectra[:, index]
        if config.local_id.estimators.participation_ratio:
            columns[f"participation_ratio_{suffix}"] = participation_ratio(spectra)
        if config.local_id.estimators.pca_threshold:
            for threshold in config.pca_thresholds.explained_variance:
                columns[f"pca_dim_{_threshold_label(threshold)}_{suffix}"] = (
                    pca_threshold_dimension(spectra, threshold)
                )
        if config.local_id.estimators.mle_lid:
            columns[f"mle_lid_{suffix}"] = mle_lid_local(distances)
        if config.local_id.estimators.ball_scaling:
            dimensions, r2_values = local_ball_scaling(distances)
            columns[f"ball_scaling_dim_{suffix}"] = dimensions
            columns[f"ball_scaling_r2_{suffix}"] = r2_values
    return pd.concat([result, pd.DataFrame(columns)], axis=1)


def compute_global_id(
    features: np.ndarray,
    config: IDEstimationConfig,
    *,
    feature_space: str,
) -> tuple[dict[str, float | int | str], ScalingEstimate | None]:
    """Compute global ID estimates for one full dataset or group."""

    result: dict[str, float | int | str] = {
        "n_samples": len(features),
        "feature_space": feature_space,
    }
    if len(features) < 2:
        return result, None

    spectrum = global_covariance_spectrum(features)
    estimators = config.global_id.estimators
    if estimators.participation_ratio:
        result["global_participation_ratio"] = float(
            participation_ratio(spectrum[None, :])[0]
        )
    if estimators.pca_threshold:
        for threshold in config.pca_thresholds.explained_variance:
            result[f"global_pca_dim_{_threshold_label(threshold)}"] = float(
                pca_threshold_dimension(spectrum[None, :], threshold)[0]
            )

    max_k = min(
        max(config.global_id.mle_k_values),
        len(features) - 1,
    )
    graph = compute_neighbor_graph(
        features,
        max_neighbors=max(2, max_k),
        metric=config.distance.metric,
    )
    if estimators.two_nn:
        result["global_two_nn_lid"] = two_nn_global(graph.distances)
    if estimators.mle_lid:
        for requested_k in config.global_id.mle_k_values:
            effective_k = min(requested_k, graph.distances.shape[1])
            estimates = mle_lid_local(graph.distances[:, :effective_k])
            result[f"global_mle_lid_k{requested_k}"] = _finite_median(estimates)

    scaling = None
    if estimators.correlation_dimension or estimators.ball_scaling:
        scaling = correlation_scaling(
            features,
            metric=config.distance.metric,
            quantiles=config.global_id.scaling_quantiles,
            max_points=config.global_id.scaling_max_points,
        )
        if estimators.correlation_dimension:
            result["correlation_dimension"] = scaling.dimension
        if estimators.ball_scaling:
            result["ball_scaling_dim"] = scaling.dimension
            result["ball_scaling_r2"] = scaling.r2
            result["ball_scaling_num_radii"] = len(scaling.curve)

    result.update(
        compute_optional_skdim(
            features,
            config.global_id.skdim_estimators,
        )
    )
    return result, scaling


def compute_neighbor_graph(
    features: np.ndarray,
    *,
    max_neighbors: int,
    metric: str,
) -> NeighborGraph:
    """Return neighbors with the query sample itself removed by index."""

    n_samples = len(features)
    if n_samples < 2:
        return NeighborGraph(
            distances=np.empty((n_samples, 0), dtype=np.float64),
            indices=np.empty((n_samples, 0), dtype=int),
        )
    effective_k = min(max_neighbors, n_samples - 1)
    model = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric=metric,
    ).fit(features)
    raw_distances, raw_indices = model.kneighbors(features)
    distances = np.full((n_samples, effective_k), np.nan, dtype=np.float64)
    indices = np.full((n_samples, effective_k), -1, dtype=int)
    for row in range(n_samples):
        keep = raw_indices[row] != row
        kept_distances = raw_distances[row][keep][:effective_k]
        kept_indices = raw_indices[row][keep][:effective_k]
        count = len(kept_distances)
        distances[row, :count] = kept_distances
        indices[row, :count] = kept_indices
    zero_fraction = float(np.mean(distances <= EPSILON))
    if zero_fraction >= 0.01:
        LOGGER.warning(
            "%.1f%% of neighbor distances are zero; duplicate points may destabilize ID.",
            100 * zero_fraction,
        )
    return NeighborGraph(distances=distances, indices=indices)


def local_covariance_spectra(
    features: np.ndarray,
    neighbor_indices: np.ndarray,
    *,
    max_values: int,
) -> np.ndarray:
    """Compute local covariance eigenvalues using the smaller Gram matrix."""

    spectra = np.zeros((len(features), max_values), dtype=np.float64)
    for row, local_indices in enumerate(neighbor_indices):
        valid_indices = local_indices[local_indices >= 0]
        if len(valid_indices) < 2:
            continue
        local = np.asarray(features[valid_indices], dtype=np.float64)
        centered = local - local.mean(axis=0, keepdims=True)
        gram = centered @ centered.T / max(1, len(local) - 1)
        try:
            eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        except np.linalg.LinAlgError:
            LOGGER.warning("Local covariance eigendecomposition failed for row %d.", row)
            spectra[row, :] = np.nan
            continue
        eigenvalues = np.maximum(eigenvalues, 0.0)
        count = min(max_values, len(eigenvalues))
        spectra[row, :count] = eigenvalues[:count]
    return spectra


def global_covariance_spectrum(features: np.ndarray) -> np.ndarray:
    """Return descending covariance eigenvalues for a full feature group."""

    centered = np.asarray(features, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return np.maximum(
        singular_values**2 / max(1, len(centered) - 1),
        0.0,
    )


def participation_ratio(spectra: np.ndarray) -> np.ndarray:
    """Compute smooth effective dimension from covariance eigenvalues."""

    numerator = spectra.sum(axis=1) ** 2
    denominator = (spectra**2).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(spectra), np.nan),
        where=denominator > EPSILON,
    )


def pca_threshold_dimension(
    spectra: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Count components required to reach an explained-variance threshold."""

    totals = spectra.sum(axis=1)
    cumulative = np.cumsum(spectra, axis=1)
    fractions = np.divide(
        cumulative,
        totals[:, None],
        out=np.zeros_like(cumulative),
        where=totals[:, None] > EPSILON,
    )
    reached = fractions >= threshold
    dimensions = np.argmax(reached, axis=1).astype(float) + 1.0
    dimensions[~reached.any(axis=1)] = np.nan
    return dimensions


def two_nn_local(distances: np.ndarray) -> np.ndarray:
    """Compute the noisy per-sample TWO-NN proxy."""

    estimates = np.full(len(distances), np.nan)
    if distances.shape[1] < 2:
        return estimates
    first = distances[:, 0]
    second = distances[:, 1]
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first > EPSILON)
        & (second > first + EPSILON)
    )
    estimates[valid] = 1.0 / np.log(second[valid] / first[valid])
    return estimates


def two_nn_global(distances: np.ndarray) -> float:
    """Compute the aggregate TWO-NN estimate from mean log ratios."""

    local = two_nn_log_ratios(distances)
    finite = local[np.isfinite(local) & (local > EPSILON)]
    if not len(finite):
        LOGGER.warning("TWO-NN could not estimate dimension due to duplicate distances.")
        return float("nan")
    return float(1.0 / finite.mean())


def two_nn_log_ratios(distances: np.ndarray) -> np.ndarray:
    """Return log(r2/r1) with invalid or duplicate rows marked NaN."""

    ratios = np.full(len(distances), np.nan)
    if distances.shape[1] < 2:
        return ratios
    first = distances[:, 0]
    second = distances[:, 1]
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first > EPSILON)
        & (second > first + EPSILON)
    )
    ratios[valid] = np.log(second[valid] / first[valid])
    return ratios


def mle_lid_local(distances: np.ndarray) -> np.ndarray:
    """Compute a stable kNN maximum-likelihood LID estimate per sample."""

    estimates = np.full(len(distances), np.nan)
    for row, values in enumerate(distances):
        finite = values[np.isfinite(values) & (values > EPSILON)]
        if len(finite) < 2:
            continue
        radius = finite[-1]
        inner = finite[finite < radius - EPSILON]
        if not len(inner):
            continue
        denominator = np.log(radius / inner).sum()
        if denominator > EPSILON:
            estimates[row] = len(inner) / denominator
    return estimates


def local_ball_scaling(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit local log neighbor count against log radius."""

    dimensions = np.full(len(distances), np.nan)
    r2_values = np.full(len(distances), np.nan)
    for row, values in enumerate(distances):
        finite = values[np.isfinite(values) & (values > EPSILON)]
        if len(finite) < 2:
            continue
        radii, counts = np.unique(finite, return_counts=True)
        cumulative = np.cumsum(counts)
        if len(radii) < 2:
            continue
        x = np.log(radii)
        y = np.log(cumulative)
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = ((y - predicted) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        dimensions[row] = slope
        r2_values[row] = 1.0 - residual / total if total > EPSILON else np.nan
    return dimensions, r2_values


def correlation_scaling(
    features: np.ndarray,
    *,
    metric: str,
    quantiles: tuple[float, ...],
    max_points: int,
) -> ScalingEstimate:
    """Fit log correlation mass against log radius over configured scales."""

    values = np.asarray(features, dtype=np.float64)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
        values = values[indices]
    distances = pdist(values, metric=metric)
    distances = distances[np.isfinite(distances) & (distances > EPSILON)]
    if len(distances) < 2:
        return ScalingEstimate(float("nan"), float("nan"), pd.DataFrame())
    radii = np.unique(np.quantile(distances, quantiles))
    masses = np.asarray([(distances <= radius).mean() for radius in radii])
    valid = (radii > EPSILON) & (masses > 0) & (masses < 1)
    curve = pd.DataFrame(
        {
            "radius": radii,
            "count": [int((distances <= radius).sum()) for radius in radii],
            "mass": masses,
            "log_radius": np.log(np.maximum(radii, EPSILON)),
            "log_mass": np.log(np.maximum(masses, EPSILON)),
        }
    )
    if valid.sum() < 2:
        return ScalingEstimate(float("nan"), float("nan"), curve)
    x = np.log(radii[valid])
    y = np.log(masses[valid])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = ((y - predicted) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - residual / total if total > EPSILON else float("nan")
    return ScalingEstimate(float(slope), float(r2), curve)


def compute_optional_skdim(
    features: np.ndarray,
    estimator_names: tuple[str, ...],
) -> dict[str, float]:
    """Run optional scikit-dimension estimators without making them required."""

    if not estimator_names:
        return {}
    try:
        module = importlib.import_module("skdim.id")
    except ImportError:
        LOGGER.warning(
            "scikit-dimension is not installed; skipping optional estimators: %s",
            ", ".join(estimator_names),
        )
        return {f"skdim_{name.lower()}_global": float("nan") for name in estimator_names}
    class_names = {
        "twonn": "TwoNN",
        "mle": "MLE",
        "danco": "DANCo",
        "pca": "lPCA",
        "corrint": "CorrInt",
    }
    result: dict[str, float] = {}
    for name in estimator_names:
        key = name.lower()
        try:
            estimator = getattr(module, class_names[key])()
            estimator.fit(features)
            dimension = np.asarray(estimator.dimension_, dtype=float)
            result[f"skdim_{key}_global"] = float(np.nanmean(dimension))
        except Exception as exc:
            LOGGER.warning("scikit-dimension estimator %s failed: %s", name, exc)
            result[f"skdim_{key}_global"] = float("nan")
    return result


def _threshold_label(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _add_empty_local_columns(
    result: dict[str, np.ndarray],
    config: IDEstimationConfig,
    *,
    n_samples: int,
) -> None:
    result["two_nn_lid_local"] = np.full(n_samples, np.nan)
    for k_value in config.local_id.k_values:
        result[f"knn_radius_k{k_value}"] = np.full(n_samples, np.nan)
        result[f"knn_mean_distance_k{k_value}"] = np.full(n_samples, np.nan)
        _add_empty_k_columns(
            result,
            config,
            k_value,
            n_samples=n_samples,
        )


def _add_empty_k_columns(
    result: dict[str, np.ndarray],
    config: IDEstimationConfig,
    k_value: int,
    *,
    n_samples: int,
) -> None:
    if config.local_id.estimators.covariance_spectrum:
        for index in range(config.local_id.covariance_eigenvalues):
            result[f"local_eig_{index + 1}_k{k_value}"] = np.full(n_samples, np.nan)
    if config.local_id.estimators.participation_ratio:
        result[f"participation_ratio_k{k_value}"] = np.full(n_samples, np.nan)
    if config.local_id.estimators.pca_threshold:
        for threshold in config.pca_thresholds.explained_variance:
            result[f"pca_dim_{_threshold_label(threshold)}_k{k_value}"] = np.full(
                n_samples,
                np.nan,
            )
    if config.local_id.estimators.mle_lid:
        result[f"mle_lid_k{k_value}"] = np.full(n_samples, np.nan)
    if config.local_id.estimators.ball_scaling:
        result[f"ball_scaling_dim_k{k_value}"] = np.full(n_samples, np.nan)
        result[f"ball_scaling_r2_k{k_value}"] = np.full(n_samples, np.nan)
