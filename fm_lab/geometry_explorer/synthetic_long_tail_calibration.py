"""Independent renderer calibration for the synthetic long-tail study."""

from __future__ import annotations

import csv
import itertools
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    DIMENSION_IDS,
    OBJECT_IDS,
    _object_configs,
    _render_map,
    build_factor_space,
)
from fm_lab.utils.logging import write_json

_GATE_FILENAME = "renderer_gate.json"
_STATISTICS_FILENAME = "renderer_class_statistics.csv"
_SINGULAR_VALUES_FILENAME = "renderer_singular_values.npz"
_STATISTIC_FIELDS = (
    "object_id",
    "dimension_id",
    "true_dimension",
    "samples",
    "foreground_occupancy_mean",
    "foreground_occupancy_std",
    "luminance_mean",
    "luminance_std",
    "contrast_mean",
    "contrast_std",
)


@dataclass(frozen=True)
class RendererGateThresholds:
    min_object_accuracy: float = 0.99
    max_nuisance_standardized_difference: float = 0.25
    relative_singular_threshold: float = 0.02
    min_full_rank_fraction: float = 0.95
    max_pullback_norm_ratio: float = 4.0


def renderer_gate(
    *,
    object_accuracy: float,
    max_nuisance_difference: float,
    full_rank_fraction: float,
    pullback_norm_ratio: float,
    thresholds: RendererGateThresholds,
) -> dict[str, Any]:
    """Apply the four renderer acceptance checks without performing calibration."""

    checks = {
        "object_separability": object_accuracy >= thresholds.min_object_accuracy,
        "nuisance_matching": (
            max_nuisance_difference
            <= thresholds.max_nuisance_standardized_difference
        ),
        "renderer_rank": full_rank_fraction >= thresholds.min_full_rank_fraction,
        "factor_visibility": pullback_norm_ratio <= thresholds.max_pullback_norm_ratio,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "object_accuracy": float(object_accuracy),
        "max_nuisance_standardized_difference": float(max_nuisance_difference),
        "full_rank_fraction": float(full_rank_fraction),
        "pullback_norm_ratio": float(pullback_norm_ratio),
    }


def calibrate_renderer(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render independent reference points, compute diagnostics, and write gate artifacts."""

    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Calibration destination already exists: {destination}"
        )
    calibration = config.get("calibration", {})
    points_per_cell = int(calibration.get("renderer_points_per_cell", 256))
    if points_per_cell <= 0:
        raise ValueError("calibration.renderer_points_per_cell must be positive.")
    epsilon = float(calibration.get("finite_difference_epsilon", 0.01))
    if epsilon <= 0.0:
        raise ValueError("calibration finite_difference_epsilon must be positive.")
    thresholds = RendererGateThresholds(
        min_object_accuracy=float(calibration.get("min_object_accuracy", 0.99)),
        max_nuisance_standardized_difference=float(
            calibration.get("max_nuisance_standardized_difference", 0.25)
        ),
        relative_singular_threshold=float(
            calibration.get("relative_singular_threshold", 0.02)
        ),
        min_full_rank_fraction=float(calibration.get("full_rank_fraction", 0.95)),
        max_pullback_norm_ratio=float(calibration.get("max_pullback_norm_ratio", 4.0)),
    )

    object_configs = _object_configs(config)
    background = np.asarray(
        config.get("render", {}).get("background", (1.0, 1.0, 1.0)),
        dtype=np.float32,
    ).reshape(3)
    base_seed = int(config["seed"]) + 9_000_000
    image_rows: list[np.ndarray] = []
    object_labels: list[int] = []
    point_labels: list[int] = []
    statistics: list[dict[str, Any]] = []
    singular_rows: list[np.ndarray] = []
    singular_dimensions: list[int] = []
    singular_cell_ids: list[str] = []
    singular_point_ids: list[int] = []
    pullback_norms: dict[str, list[float]] = {
        label: [] for label in ("tx", "ty", "tz", "azimuth", "elevation")
    }
    full_rank: list[bool] = []

    for object_index, object_id in enumerate(OBJECT_IDS):
        for dimension_index, dimension_id in enumerate(DIMENSION_IDS):
            factor = build_factor_space(dimension_id)
            render_map = _render_map(config, object_configs[object_id], factor)
            seed = base_seed + object_index * 1_000 + dimension_index * 10
            values = factor.sample(points_per_cell, seed=seed).values
            values = _as_values(values, points_per_cell)
            images = np.asarray([render_map.render(value) for value in values], dtype=np.float32)
            cell_statistics, _ = _cell_statistics(
                images,
                background=background,
            )
            statistics.append(
                {
                    "object_id": object_id,
                    "dimension_id": dimension_id,
                    "true_dimension": int(factor.dim),
                    "samples": points_per_cell,
                    **cell_statistics,
                }
            )
            for point_id, (value, image) in enumerate(zip(values, images, strict=True)):
                image_rows.append(image.reshape(-1))
                object_labels.append(object_index)
                point_labels.append(point_id)
                jacobian = _normalized_renderer_jacobian(
                    render_map,
                    factor,
                    value,
                    dimension_id=dimension_id,
                    epsilon=epsilon,
                )
                singular_values = np.linalg.svd(jacobian, compute_uv=False)
                padded = np.full(5, np.nan, dtype=np.float32)
                padded[: len(singular_values)] = singular_values.astype(np.float32)
                singular_rows.append(padded)
                singular_dimensions.append(int(factor.dim))
                singular_cell_ids.append(f"{object_id}__{dimension_id}")
                singular_point_ids.append(point_id)
                largest = float(singular_values[0]) if len(singular_values) else 0.0
                rank = int(
                    np.count_nonzero(
                        singular_values
                        > largest * thresholds.relative_singular_threshold
                    )
                )
                full_rank.append(largest > 0.0 and rank == int(factor.dim))
                for label, norm in zip(
                    _factor_labels(dimension_id),
                    np.linalg.norm(jacobian, axis=0),
                    strict=True,
                ):
                    pullback_norms[label].append(float(norm))
    object_accuracy = _raw_pixel_object_accuracy(
        np.asarray(image_rows, dtype=np.float32),
        np.asarray(object_labels, dtype=np.int64),
        np.asarray(point_labels, dtype=np.int64),
        seed=int(config["seed"]),
    )
    nuisance_difference = _max_nuisance_standardized_difference(statistics)
    full_rank_fraction = float(np.mean(full_rank))
    median_pullback_norms = {
        label: float(np.median(values))
        for label, values in pullback_norms.items()
        if values
    }
    minimum_norm = min(median_pullback_norms.values(), default=0.0)
    maximum_norm = max(median_pullback_norms.values(), default=0.0)
    pullback_norm_ratio = (
        float(maximum_norm / minimum_norm) if minimum_norm > 0.0 else float("inf")
    )
    result = renderer_gate(
        object_accuracy=object_accuracy,
        max_nuisance_difference=nuisance_difference,
        full_rank_fraction=full_rank_fraction,
        pullback_norm_ratio=pullback_norm_ratio,
        thresholds=thresholds,
    )
    result.update(
        {
            "relative_singular_threshold": thresholds.relative_singular_threshold,
            "median_pullback_norms": median_pullback_norms,
            "renderer_points_per_cell": points_per_cell,
            "artifacts": {
                "renderer_gate": _GATE_FILENAME,
                "class_statistics": _STATISTICS_FILENAME,
                "singular_values": _SINGULAR_VALUES_FILENAME,
            },
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        _write_statistics(statistics, staging_dir / _STATISTICS_FILENAME)
        np.savez_compressed(
            staging_dir / _SINGULAR_VALUES_FILENAME,
            values=np.asarray(singular_rows, dtype=np.float32),
            dimensions=np.asarray(singular_dimensions, dtype=np.int16),
            cell_ids=np.asarray(singular_cell_ids),
            point_indices=np.asarray(singular_point_ids, dtype=np.int32),
        )
        write_json(result, staging_dir / _GATE_FILENAME)
        staging_dir.replace(destination)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return result


def _as_values(values: Any, count: int) -> list[Any]:
    if isinstance(values, np.ndarray):
        return [values[index] for index in range(count)]
    return list(values)


def _cell_statistics(
    images: np.ndarray,
    *,
    background: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    foreground = np.any(np.abs(images - background) > (0.5 / 255.0), axis=-1)
    luminance_images = (
        0.2126 * images[..., 0]
        + 0.7152 * images[..., 1]
        + 0.0722 * images[..., 2]
    )
    occupancy = np.mean(foreground, axis=(1, 2))
    luminance = np.asarray(
        [
            float(np.mean(values[mask])) if np.any(mask) else float(np.mean(values))
            for values, mask in zip(luminance_images, foreground, strict=True)
        ],
        dtype=np.float64,
    )
    contrast = np.asarray(
        [
            float(np.std(values[mask])) if np.any(mask) else float(np.std(values))
            for values, mask in zip(luminance_images, foreground, strict=True)
        ],
        dtype=np.float64,
    )
    per_image = {
        "foreground_occupancy": occupancy,
        "luminance": luminance,
        "contrast": contrast,
    }
    statistics = {
        f"{name}_{suffix}": float(function(values))
        for name, values in per_image.items()
        for suffix, function in (("mean", np.mean), ("std", np.std))
    }
    return statistics, per_image


def _normalized_renderer_jacobian(
    render_map: Any,
    factor: Any,
    value: Any,
    *,
    dimension_id: str,
    epsilon: float,
) -> np.ndarray:
    tangents = factor.tangent_basis(value)
    scales = _normalization_scales(dimension_id)
    columns = []
    for tangent, scale in zip(tangents, scales, strict=True):
        plus = factor.retract(value, tangent, epsilon * scale)
        minus = factor.retract(value, tangent, -epsilon * scale)
        difference = render_map.render_flat(plus) - render_map.render_flat(minus)
        columns.append(difference / (2.0 * epsilon))
    return np.column_stack(columns).astype(np.float32)


def _normalization_scales(dimension_id: str) -> tuple[float, ...]:
    translation = (0.25, 0.25, 0.75)
    if dimension_id == "low":
        return (0.75,)
    if dimension_id == "medium":
        return translation
    if dimension_id == "high":
        return (*translation, float(np.pi), 0.5)
    raise ValueError(f"Unsupported dimension level: {dimension_id}")


def _factor_labels(dimension_id: str) -> tuple[str, ...]:
    if dimension_id == "low":
        return ("tz",)
    if dimension_id == "medium":
        return ("tx", "ty", "tz")
    if dimension_id == "high":
        return ("tx", "ty", "tz", "azimuth", "elevation")
    raise ValueError(f"Unsupported dimension level: {dimension_id}")


def _raw_pixel_object_accuracy(
    images: np.ndarray,
    labels: np.ndarray,
    point_ids: np.ndarray,
    *,
    seed: int,
) -> float:
    if int(np.max(point_ids)) == 0:
        train = np.ones(len(point_ids), dtype=bool)
        test = train
    else:
        train = point_ids % 2 == 0
        test = ~train
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(images[train], labels[train])
    return float(classifier.score(images[test], labels[test]))


def _max_nuisance_standardized_difference(
    statistics: list[dict[str, Any]],
) -> float:
    """Return the worst cross-object SMD, equally averaging dimension strata.

    Each object pair is compared within each matching latent-dimension stratum.
    The three absolute standardized mean differences are then averaged with
    equal weight, preventing dimension prevalence from confounding the object
    comparison or opposite-signed strata from cancelling.
    """

    maximum = 0.0
    by_cell = {
        (str(row["object_id"]), str(row["dimension_id"])): row
        for row in statistics
    }
    for first_object, second_object in itertools.combinations(OBJECT_IDS, 2):
        for metric in ("foreground_occupancy", "luminance", "contrast"):
            dimension_differences = []
            for dimension_id in DIMENSION_IDS:
                left = by_cell[(first_object, dimension_id)]
                right = by_cell[(second_object, dimension_id)]
                pooled = np.sqrt(
                    (
                        float(left[f"{metric}_std"]) ** 2
                        + float(right[f"{metric}_std"]) ** 2
                    )
                    / 2.0
                )
                difference = abs(
                    float(left[f"{metric}_mean"])
                    - float(right[f"{metric}_mean"])
                ) / max(float(pooled), 1.0e-6)
                dimension_differences.append(difference)
            maximum = max(maximum, float(np.mean(dimension_differences)))
    return float(maximum)


def _write_statistics(statistics: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_STATISTIC_FIELDS)
        writer.writeheader()
        writer.writerows(statistics)
