"""Diagnostics comparing latent-space and pixel-space neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances

from fm_lab.geometry_explorer.latent_factors import LatentFactorSpace


@dataclass(frozen=True)
class LatentPixelDiagnosticSummary:
    n_samples: int
    spearman_distance_corr: float
    pearson_distance_corr: float
    knn_overlap_by_k: dict[int, float]
    trustworthiness_by_k: dict[int, float]
    continuity_by_k: dict[int, float]


def analyze_latent_pixel_diagnostics(
    render_map: Any,
    factor_space: LatentFactorSpace,
    zs: list[Any] | tuple[Any, ...],
    max_samples: int = 2000,
    pair_count: int = 100_000,
    ks: tuple[int, ...] = (15,),
    seed: int = 0,
) -> LatentPixelDiagnosticSummary:
    """Compare latent and rendered-pixel distances on a bounded subsample."""

    selected = _select_values(list(zs), max_points=max_samples, seed=seed)
    n_samples = len(selected)
    if n_samples < 2:
        return LatentPixelDiagnosticSummary(
            n_samples=n_samples,
            spearman_distance_corr=float("nan"),
            pearson_distance_corr=float("nan"),
            knn_overlap_by_k={int(k): float("nan") for k in ks},
            trustworthiness_by_k={int(k): float("nan") for k in ks},
            continuity_by_k={int(k): float("nan") for k in ks},
        )
    pixels = np.asarray(render_map.render_batch(selected), dtype=np.float32)
    latent_distances = _latent_distance_matrix(factor_space, selected)
    pixel_distances = pairwise_distances(pixels, metric="euclidean")
    pair_indices = _sample_pairs(n_samples, pair_count=pair_count, seed=seed)
    latent_pairs = latent_distances[pair_indices[:, 0], pair_indices[:, 1]]
    pixel_pairs = pixel_distances[pair_indices[:, 0], pair_indices[:, 1]]
    spearman = _correlation_or_nan(latent_pairs, pixel_pairs, spearmanr)
    pearson = _correlation_or_nan(latent_pairs, pixel_pairs, pearsonr)
    latent_ranks = _rank_matrix(latent_distances)
    pixel_ranks = _rank_matrix(pixel_distances)
    overlap = {}
    trust = {}
    continuity = {}
    for k_value in ks:
        k = int(min(k_value, n_samples - 1))
        overlap[k_value] = _knn_overlap(latent_distances, pixel_distances, k=k)
        trust[k_value] = _trustworthiness(latent_ranks, pixel_distances, k=k)
        continuity[k_value] = _trustworthiness(pixel_ranks, latent_distances, k=k)
    return LatentPixelDiagnosticSummary(
        n_samples=n_samples,
        spearman_distance_corr=spearman,
        pearson_distance_corr=pearson,
        knn_overlap_by_k={int(k): float(v) for k, v in overlap.items()},
        trustworthiness_by_k={int(k): float(v) for k, v in trust.items()},
        continuity_by_k={int(k): float(v) for k, v in continuity.items()},
    )


def _latent_distance_matrix(
    factor_space: LatentFactorSpace,
    values: list[Any],
) -> np.ndarray:
    n_values = len(values)
    distances = np.zeros((n_values, n_values), dtype=np.float64)
    for row in range(n_values):
        for column in range(row + 1, n_values):
            value = factor_space.distance(values[row], values[column])
            distances[row, column] = value
            distances[column, row] = value
    return distances


def _correlation_or_nan(
    left: np.ndarray,
    right: np.ndarray,
    correlation_fn: Any,
) -> float:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        left_values.size < 2
        or right_values.size < 2
        or np.ptp(left_values) <= 0.0
        or np.ptp(right_values) <= 0.0
    ):
        return float("nan")
    return float(correlation_fn(left_values, right_values).statistic)


def _sample_pairs(n_samples: int, *, pair_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_pairs = n_samples * (n_samples - 1) // 2
    if pair_count <= 0 or pair_count >= total_pairs:
        return np.asarray(np.triu_indices(n_samples, k=1)).T
    first = rng.integers(0, n_samples, size=pair_count)
    second = rng.integers(0, n_samples - 1, size=pair_count)
    second = np.where(second >= first, second + 1, second)
    return np.column_stack([first, second]).astype(np.int64)


def _rank_matrix(distances: np.ndarray) -> np.ndarray:
    order = np.argsort(distances, axis=1)
    ranks = np.empty_like(order)
    row_indices = np.arange(distances.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(distances.shape[1])[None, :]
    return ranks


def _knn_overlap(
    latent_distances: np.ndarray,
    pixel_distances: np.ndarray,
    *,
    k: int,
) -> float:
    latent_neighbors = np.argsort(latent_distances, axis=1)[:, 1 : k + 1]
    pixel_neighbors = np.argsort(pixel_distances, axis=1)[:, 1 : k + 1]
    overlaps = [
        len(set(latent_neighbors[row]).intersection(pixel_neighbors[row])) / max(1, k)
        for row in range(latent_distances.shape[0])
    ]
    return float(np.mean(overlaps))


def _trustworthiness(
    reference_ranks: np.ndarray,
    candidate_distances: np.ndarray,
    *,
    k: int,
) -> float:
    n_samples = reference_ranks.shape[0]
    if k <= 0 or n_samples <= k + 1:
        return float("nan")
    candidate_neighbors = np.argsort(candidate_distances, axis=1)[:, 1 : k + 1]
    penalty = 0.0
    for row in range(n_samples):
        for neighbor in candidate_neighbors[row]:
            rank = int(reference_ranks[row, neighbor])
            if rank > k:
                penalty += rank - k
    normalizer = n_samples * k * (2 * n_samples - 3 * k - 1)
    if normalizer <= 0:
        return float("nan")
    return float(1.0 - (2.0 / normalizer) * penalty)


def _select_values(values: list[Any], *, max_points: int, seed: int) -> list[Any]:
    if max_points <= 0 or len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=max_points, replace=False))
    return [values[int(index)] for index in indices]
