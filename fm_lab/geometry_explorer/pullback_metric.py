"""Finite-difference pullback metric diagnostics for renderer-induced maps."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.latent_factors import LatentFactorSpace


@dataclass(frozen=True)
class PullbackMetricResult:
    z_coordinates: dict[str, float]
    tangent_labels: list[str]
    G: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    estimated_rank: int
    trace: float
    determinant: float
    pseudo_determinant: float
    volume_density: float
    condition_number: float
    tangent_norms: np.ndarray
    tangent_correlation: np.ndarray
    mean_abs_offdiag_tangent_correlation: float
    max_abs_offdiag_tangent_correlation: float
    eps: float
    factor_space_name: str
    object_name: str
    render_mode: str


@dataclass(frozen=True)
class PullbackMetricSummary:
    n_points: int
    tangent_labels: list[str]
    mean_eigenvalues: np.ndarray
    std_eigenvalues: np.ndarray
    mean_rank: float
    rank_histogram: dict[int, int]
    mean_trace: float
    std_trace: float
    mean_condition_number: float
    median_condition_number: float
    mean_volume_density: float
    median_volume_density: float
    mean_metric_diagonal: np.ndarray
    mean_tangent_norms: np.ndarray
    relative_tangent_energy: np.ndarray
    tangent_energy: list[dict[str, Any]]
    mean_tangent_correlation: np.ndarray
    pairwise_tangent_correlations: list[dict[str, Any]]
    top_tangent_collapse_pairs: list[dict[str, Any]]
    low_energy_tangent_directions: list[dict[str, Any]]
    mean_abs_offdiag_tangent_correlation: float
    max_abs_offdiag_tangent_correlation: float
    results_path: str


def estimate_jacobian(
    render_map: Any,
    factor_space: LatentFactorSpace,
    z: Any,
    eps: float = 1.0e-3,
    method: str = "central",
    normalize_by_num_pixels: bool = True,
) -> np.ndarray:
    """Estimate J_F(z), shape (pixel_dimension, latent_dimension)."""

    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")
    columns = []
    basis = factor_space.tangent_basis(z)
    base = None
    if method == "forward":
        base = _render_flat(render_map, z)
    for tangent_vec in basis:
        z_plus = factor_space.retract(z, tangent_vec, eps)
        plus = _render_flat(render_map, z_plus)
        if method == "central":
            z_minus = factor_space.retract(z, tangent_vec, -eps)
            minus = _render_flat(render_map, z_minus)
            column = (plus - minus) / (2.0 * float(eps))
        else:
            assert base is not None
            column = (plus - base) / float(eps)
        if normalize_by_num_pixels:
            column = column / math.sqrt(max(1, int(column.size)))
        columns.append(column.astype(np.float64))
    return np.column_stack(columns) if columns else np.empty((0, 0), dtype=np.float64)


def estimate_pullback_metric(
    render_map: Any,
    factor_space: LatentFactorSpace,
    z: Any,
    eps: float = 1.0e-3,
    rank_tol: float = 1.0e-6,
    normalize_by_num_pixels: bool = True,
) -> PullbackMetricResult:
    """Estimate G(z)=J_F(z)^T J_F(z)."""

    jacobian = estimate_jacobian(
        render_map,
        factor_space,
        z,
        eps=eps,
        method="central",
        normalize_by_num_pixels=normalize_by_num_pixels,
    )
    G = np.asarray(jacobian.T @ jacobian, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    singular_values = np.sqrt(eigenvalues)
    positive = eigenvalues > float(rank_tol)
    estimated_rank = int(np.count_nonzero(positive))
    trace = float(np.trace(G))
    determinant = float(np.linalg.det(G)) if G.size else 0.0
    pseudo_determinant = float(np.prod(eigenvalues[positive])) if estimated_rank else 0.0
    volume_density = float(math.sqrt(max(0.0, pseudo_determinant)))
    condition_number = (
        float(eigenvalues[positive][0] / eigenvalues[positive][-1])
        if estimated_rank >= 2
        else 1.0
        if estimated_rank == 1
        else float("inf")
    )
    tangent_correlation = tangent_correlation_matrix(G)
    mean_abs_corr, max_abs_corr = offdiag_abs_correlation_stats(tangent_correlation)
    tangent_labels = _validated_tangent_labels(factor_space, z, int(G.shape[0]))
    return PullbackMetricResult(
        z_coordinates=dict(factor_space.coordinates(z)),
        tangent_labels=tangent_labels,
        G=G,
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        estimated_rank=estimated_rank,
        trace=trace,
        determinant=determinant,
        pseudo_determinant=pseudo_determinant,
        volume_density=volume_density,
        condition_number=condition_number,
        tangent_norms=np.sqrt(np.maximum(np.diag(G), 0.0)),
        tangent_correlation=tangent_correlation,
        mean_abs_offdiag_tangent_correlation=mean_abs_corr,
        max_abs_offdiag_tangent_correlation=max_abs_corr,
        eps=float(eps),
        factor_space_name=str(getattr(factor_space, "name", "")),
        object_name=str(getattr(render_map, "object_name", "")),
        render_mode=str(getattr(render_map, "render_mode", "")),
    )


def analyze_pullback_metrics(
    render_map: Any,
    factor_space: LatentFactorSpace,
    zs: list[Any] | tuple[Any, ...],
    eps: float = 1.0e-3,
    max_points: int = 256,
    rank_tol: float = 1.0e-6,
    normalize_by_num_pixels: bool = True,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> PullbackMetricSummary:
    """Estimate pullback metrics on a subset of latent states and summarize them."""

    selected = _select_values(list(zs), max_points=max_points, seed=seed)
    results = [
        estimate_pullback_metric(
            render_map,
            factor_space,
            z,
            eps=eps,
            rank_tol=rank_tol,
            normalize_by_num_pixels=normalize_by_num_pixels,
        )
        for z in selected
    ]
    if results:
        eigenvalues = np.stack([result.eigenvalues for result in results])
        ranks = [result.estimated_rank for result in results]
        traces = np.asarray([result.trace for result in results], dtype=np.float64)
        conditions = np.asarray(
            [result.condition_number for result in results],
            dtype=np.float64,
        )
        volumes = np.asarray([result.volume_density for result in results], dtype=np.float64)
        metric_diagonals = np.stack([np.diag(result.G) for result in results])
        tangent_norms = np.stack([result.tangent_norms for result in results])
        tangent_correlations = np.stack(
            [result.tangent_correlation for result in results],
        )
        mean_abs_corrs = np.asarray(
            [result.mean_abs_offdiag_tangent_correlation for result in results],
            dtype=np.float64,
        )
        max_abs_corrs = np.asarray(
            [result.max_abs_offdiag_tangent_correlation for result in results],
            dtype=np.float64,
        )
    else:
        eigenvalues = np.empty((0, factor_space.dim), dtype=np.float64)
        ranks = []
        traces = np.empty(0, dtype=np.float64)
        conditions = np.empty(0, dtype=np.float64)
        volumes = np.empty(0, dtype=np.float64)
        metric_diagonals = np.empty((0, factor_space.dim), dtype=np.float64)
        tangent_norms = np.empty((0, factor_space.dim), dtype=np.float64)
        tangent_correlations = np.empty((0, factor_space.dim, factor_space.dim))
        mean_abs_corrs = np.empty(0, dtype=np.float64)
        max_abs_corrs = np.empty(0, dtype=np.float64)
    path = "" if output_path is None else str(Path(output_path))
    tangent_labels = (
        results[0].tangent_labels
        if results
        else _validated_tangent_labels(factor_space, None, int(factor_space.dim))
    )
    mean_metric_diagonal = (
        np.nanmean(metric_diagonals, axis=0) if len(results) else np.empty(0)
    )
    mean_tangent_norms = np.nanmean(tangent_norms, axis=0) if len(results) else np.empty(0)
    relative_tangent_energy = _relative_energy(mean_metric_diagonal)
    tangent_energy = tangent_energy_table(
        tangent_labels,
        mean_metric_diagonal,
        mean_tangent_norms,
    )
    mean_tangent_correlation = (
        np.nanmean(tangent_correlations, axis=0)
        if len(results)
        else np.empty((0, 0))
    )
    pairwise_correlations = tangent_correlation_pairs(
        tangent_labels,
        mean_tangent_correlation,
    )
    summary = PullbackMetricSummary(
        n_points=len(results),
        tangent_labels=tangent_labels,
        mean_eigenvalues=np.nanmean(eigenvalues, axis=0) if len(results) else np.empty(0),
        std_eigenvalues=np.nanstd(eigenvalues, axis=0) if len(results) else np.empty(0),
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
        rank_histogram={int(rank): int(ranks.count(rank)) for rank in sorted(set(ranks))},
        mean_trace=float(np.nanmean(traces)) if traces.size else 0.0,
        std_trace=float(np.nanstd(traces)) if traces.size else 0.0,
        mean_condition_number=_finite_mean(conditions),
        median_condition_number=_finite_median(conditions),
        mean_volume_density=float(np.nanmean(volumes)) if volumes.size else 0.0,
        median_volume_density=float(np.nanmedian(volumes)) if volumes.size else 0.0,
        mean_metric_diagonal=mean_metric_diagonal,
        mean_tangent_norms=mean_tangent_norms,
        relative_tangent_energy=relative_tangent_energy,
        tangent_energy=tangent_energy,
        mean_tangent_correlation=mean_tangent_correlation,
        pairwise_tangent_correlations=pairwise_correlations,
        top_tangent_collapse_pairs=pairwise_correlations[:20],
        low_energy_tangent_directions=low_energy_tangent_directions(
            tangent_energy,
            top_k=10,
        ),
        mean_abs_offdiag_tangent_correlation=(
            float(np.nanmean(mean_abs_corrs)) if mean_abs_corrs.size else 0.0
        ),
        max_abs_offdiag_tangent_correlation=(
            float(np.nanmax(max_abs_corrs)) if max_abs_corrs.size else 0.0
        ),
        results_path=path,
    )
    if output_path is not None:
        _write_pullback_results(Path(output_path), summary, results)
    return summary


def _render_flat(render_map: Any, z: Any) -> np.ndarray:
    if hasattr(render_map, "render_flat"):
        return np.asarray(render_map.render_flat(z), dtype=np.float64).reshape(-1)
    return np.asarray(render_map.render(z), dtype=np.float64).reshape(-1)


def tangent_correlation_matrix(G: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Normalize a pullback metric into tangent-direction cosine overlaps."""

    matrix = np.asarray(G, dtype=np.float64)
    if matrix.size == 0:
        return matrix.copy()
    norms = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=denominator > eps,
    )
    for index, norm in enumerate(norms):
        correlation[index, index] = 1.0 if norm > eps else 0.0
    return np.clip(correlation, -1.0, 1.0)


def offdiag_abs_correlation_stats(correlation: np.ndarray) -> tuple[float, float]:
    """Return mean and max absolute off-diagonal correlations."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.shape[0] <= 1:
        return (0.0, 0.0)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = np.abs(matrix[mask])
    return (float(np.nanmean(values)), float(np.nanmax(values)))


def tangent_energy_table(
    labels: list[str],
    metric_diagonal: np.ndarray,
    tangent_norms: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return labeled per-coordinate pullback energy."""

    diagonal = np.asarray(metric_diagonal, dtype=np.float64).reshape(-1)
    norms = (
        np.asarray(tangent_norms, dtype=np.float64).reshape(-1)
        if tangent_norms is not None
        else np.sqrt(np.maximum(diagonal, 0.0))
    )
    relative = _relative_energy(diagonal)
    return [
        {
            "index": int(index),
            "label": str(labels[index]) if index < len(labels) else f"tangent_{index}",
            "mean_metric_diagonal": float(diagonal[index]),
            "mean_tangent_norm": float(norms[index]) if index < norms.size else 0.0,
            "relative_energy": float(relative[index]) if index < relative.size else 0.0,
        }
        for index in range(int(diagonal.size))
    ]


def low_energy_tangent_directions(
    tangent_energy: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return tangent directions with the smallest pullback diagonal energy."""

    ordered = sorted(
        tangent_energy,
        key=lambda item: (
            float(item.get("relative_energy", 0.0)),
            float(item.get("mean_metric_diagonal", 0.0)),
        ),
    )
    return ordered if top_k is None else ordered[: int(top_k)]


def tangent_correlation_pairs(
    labels: list[str],
    correlation: np.ndarray,
    *,
    min_abs_correlation: float = 0.0,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return all labeled off-diagonal tangent correlations, sorted by magnitude."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return []
    rows = []
    limit = min(matrix.shape[0], matrix.shape[1])
    for left in range(limit):
        for right in range(left + 1, limit):
            value = float(matrix[left, right])
            abs_value = abs(value)
            if not np.isfinite(value) or abs_value < float(min_abs_correlation):
                continue
            rows.append(
                {
                    "left_index": int(left),
                    "right_index": int(right),
                    "left": str(labels[left]) if left < len(labels) else f"tangent_{left}",
                    "right": (
                        str(labels[right])
                        if right < len(labels)
                        else f"tangent_{right}"
                    ),
                    "correlation": value,
                    "abs_correlation": abs_value,
                }
            )
    rows.sort(key=lambda item: float(item["abs_correlation"]), reverse=True)
    return rows if top_k is None else rows[: int(top_k)]


def _select_values(values: list[Any], *, max_points: int, seed: int) -> list[Any]:
    if max_points <= 0 or len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=max_points, replace=False))
    return [values[int(index)] for index in indices]


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("inf")


def _relative_energy(diagonal: np.ndarray) -> np.ndarray:
    values = np.asarray(diagonal, dtype=np.float64).reshape(-1)
    total = float(np.nansum(np.maximum(values, 0.0)))
    if total <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.maximum(values, 0.0) / total


def _validated_tangent_labels(
    factor_space: LatentFactorSpace,
    z: Any | None,
    dim: int,
) -> list[str]:
    labels = list(factor_space.tangent_labels(z))
    if len(labels) == int(dim):
        return labels
    return [f"tangent_{index}" for index in range(int(dim))]


def _write_pullback_results(
    path: Path,
    summary: PullbackMetricSummary,
    results: list[PullbackMetricResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _json_ready(asdict(summary)),
        "results": [_json_ready(asdict(result)) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value
