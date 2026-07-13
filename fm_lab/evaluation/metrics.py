"""Reference-compatible and extended metrics over Inception features."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors


def fid_score(generated: np.ndarray, real: np.ndarray, *, eps: float = 1e-6) -> float:
    generated, real = _feature_pair(generated, real, minimum=2)
    mu_generated = np.mean(generated, axis=0)
    mu_real = np.mean(real, axis=0)
    sigma_generated = np.atleast_2d(np.cov(generated, rowvar=False))
    sigma_real = np.atleast_2d(np.cov(real, rowvar=False))
    difference = np.atleast_1d(mu_generated - mu_real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covariance_mean = linalg.sqrtm(sigma_generated @ sigma_real)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma_generated.shape[0]) * eps
        covariance_mean = linalg.sqrtm(
            (sigma_generated + offset) @ (sigma_real + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0.0, atol=1e-3):
            maximum = float(np.max(np.abs(covariance_mean.imag)))
            raise ValueError(f"FID covariance product has imaginary component {maximum}.")
        covariance_mean = covariance_mean.real
    value = (
        difference @ difference
        + np.trace(sigma_generated)
        + np.trace(sigma_real)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(float(value), 0.0))


def kid_score(
    generated: np.ndarray,
    real: np.ndarray,
    *,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    seed: int = 0,
) -> float:
    generated, real = _feature_pair(generated, real, minimum=2)
    if num_subsets < 1 or max_subset_size < 2:
        raise ValueError("KID subset settings must be positive and include at least two samples.")
    subset_size = min(len(generated), len(real), max_subset_size)
    rng = np.random.default_rng(seed)
    dimension = generated.shape[1]
    total = 0.0
    for _ in range(num_subsets):
        x = generated[rng.choice(len(generated), subset_size, replace=False)]
        y = real[rng.choice(len(real), subset_size, replace=False)]
        within = (x @ x.T / dimension + 1.0) ** 3 + (y @ y.T / dimension + 1.0) ** 3
        cross = (x @ y.T / dimension + 1.0) ** 3
        total += (within.sum() - np.diag(within).sum()) / (subset_size - 1)
        total -= 2.0 * cross.sum() / subset_size
    return float(total / num_subsets / subset_size)


def generative_recall(
    generated: np.ndarray,
    real: np.ndarray,
    *,
    nearest_k: int = 5,
) -> float:
    generated, real = _feature_pair(generated, real, minimum=2)
    if nearest_k < 1 or nearest_k >= len(real):
        raise ValueError("nearest_k must be in [1, number of real samples).")
    real_neighbors = NearestNeighbors(n_neighbors=nearest_k + 1).fit(real)
    real_distances = real_neighbors.kneighbors(real, return_distance=True)[0][:, nearest_k]
    generated_neighbors = NearestNeighbors(n_neighbors=1).fit(generated)
    nearest_generated = generated_neighbors.kneighbors(real, return_distance=True)[0][:, 0]
    return float(np.mean(nearest_generated <= real_distances))


def inception_score(probabilities: np.ndarray, *, splits: int = 10) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) < splits or splits < 1:
        raise ValueError("Inception probabilities must be 2D with at least one row per split.")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Each Inception probability row must be non-negative and sum to one.")
    scores = []
    for indices in np.array_split(np.arange(len(probabilities)), splits):
        subset = probabilities[indices]
        marginal = subset.mean(axis=0, keepdims=True)
        log_subset = np.log(subset.clip(min=1e-16))
        log_marginal = np.log(marginal.clip(min=1e-16))
        scores.append(float(np.exp(np.sum(subset * (log_subset - log_marginal), axis=1).mean())))
    return summarize(scores)


def classwise_fid(
    generated: np.ndarray,
    generated_labels: np.ndarray,
    real: np.ndarray,
    real_labels: np.ndarray,
) -> dict[int, float]:
    generated, real = _feature_pair(generated, real, minimum=2)
    generated_labels = _labels(generated_labels, len(generated), "generated")
    real_labels = _labels(real_labels, len(real), "real")
    classes = sorted(set(real_labels.tolist()) | set(generated_labels.tolist()))
    result = {}
    for class_id in classes:
        generated_class = generated[generated_labels == class_id]
        real_class = real[real_labels == class_id]
        if len(generated_class) < 2 or len(real_class) < 2:
            raise ValueError(f"Class {class_id} requires at least two real and generated samples.")
        result[int(class_id)] = fid_score(generated_class, real_class)
    return result


def summarize(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Cannot summarize an empty or non-vector metric sequence.")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "all": [float(value) for value in array],
    }


def _feature_pair(
    generated: np.ndarray,
    real: np.ndarray,
    *,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if generated.ndim != 2 or real.ndim != 2 or generated.shape[1] != real.shape[1]:
        raise ValueError("Real and generated features must be 2D with matching dimensions.")
    if len(generated) < minimum or len(real) < minimum:
        raise ValueError("Metrics require at least two real and generated feature vectors.")
    if not np.isfinite(generated).all() or not np.isfinite(real).all():
        raise ValueError("Features must contain only finite values.")
    return generated, real


def _labels(labels: np.ndarray, expected: int, name: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError(f"{name} labels must be a vector aligned with its features.")
    return labels
