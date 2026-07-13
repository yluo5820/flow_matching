"""Product-factor pullback metric block diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.latent_factors import ProductFactorSpace
from fm_lab.geometry_explorer.pullback_metric import (
    estimate_pullback_metric,
    low_energy_tangent_directions,
    offdiag_abs_correlation_stats,
    tangent_correlation_matrix,
    tangent_correlation_pairs,
    tangent_energy_table,
)


@dataclass(frozen=True)
class ProductCouplingResult:
    z_coordinates: dict[str, float]
    factor_names: list[str]
    tangent_labels: list[str]
    block_norms: dict[str, float]
    pairwise_couplings: dict[str, float]
    within_factor_tangent_correlations: dict[str, float]
    max_within_factor_tangent_correlations: dict[str, float]
    cross_factor_tangent_correlations: dict[str, float]
    max_cross_factor_tangent_correlations: dict[str, float]
    product_error: float
    G: np.ndarray
    tangent_correlation: np.ndarray


@dataclass(frozen=True)
class ProductCouplingSummary:
    n_points: int
    tangent_labels: list[str]
    mean_pairwise_couplings: dict[str, float]
    std_pairwise_couplings: dict[str, float]
    mean_within_factor_tangent_correlations: dict[str, float]
    max_within_factor_tangent_correlations: dict[str, float]
    mean_cross_factor_tangent_correlations: dict[str, float]
    max_cross_factor_tangent_correlations: dict[str, float]
    mean_metric_diagonal: np.ndarray
    mean_tangent_norms: np.ndarray
    relative_tangent_energy: np.ndarray
    tangent_energy: list[dict[str, Any]]
    mean_tangent_correlation: np.ndarray
    pairwise_tangent_correlations: list[dict[str, Any]]
    top_tangent_collapse_pairs: list[dict[str, Any]]
    low_energy_tangent_directions: list[dict[str, Any]]
    mean_product_error: float
    median_product_error: float
    results_path: str


def block_decompose_metric(
    G: np.ndarray,
    product_space: ProductFactorSpace,
) -> dict[tuple[str, str], np.ndarray]:
    """Return all metric blocks keyed by factor-name pairs."""

    blocks = {}
    matrix = np.asarray(G, dtype=np.float64)
    for left_name, left_slice in product_space.factor_slices.items():
        for right_name, right_slice in product_space.factor_slices.items():
            blocks[(left_name, right_name)] = matrix[left_slice, right_slice]
    return blocks


def factor_coupling_score(
    G_aa: np.ndarray,
    G_bb: np.ndarray,
    G_ab: np.ndarray,
    eps: float = 1.0e-12,
) -> float:
    """Return normalized off-block Frobenius energy."""

    numerator = float(np.linalg.norm(G_ab, ord="fro"))
    denominator = float(
        np.sqrt(
            np.linalg.norm(G_aa, ord="fro") * np.linalg.norm(G_bb, ord="fro") + eps
        )
    )
    return numerator / denominator


def product_metric_error(
    G: np.ndarray,
    product_space: ProductFactorSpace,
    eps: float = 1.0e-12,
) -> float:
    """Relative off-block metric energy."""

    matrix = np.asarray(G, dtype=np.float64)
    blockdiag = np.zeros_like(matrix)
    for block_slice in product_space.factor_slices.values():
        blockdiag[block_slice, block_slice] = matrix[block_slice, block_slice]
    numerator = np.linalg.norm(matrix - blockdiag, ord="fro")
    denominator = np.linalg.norm(matrix, ord="fro") + eps
    return float(numerator / denominator)


def analyze_product_structure_at_point(
    render_map: Any,
    product_space: ProductFactorSpace,
    z: Any,
    eps: float = 1.0e-3,
    rank_tol: float = 1.0e-6,
) -> ProductCouplingResult:
    """Estimate pullback metric at z and summarize product coupling blocks."""

    metric = estimate_pullback_metric(
        render_map,
        product_space,
        z,
        eps=eps,
        rank_tol=rank_tol,
    )
    blocks = block_decompose_metric(metric.G, product_space)
    tangent_correlation = tangent_correlation_matrix(metric.G)
    correlation_blocks = block_decompose_metric(tangent_correlation, product_space)
    block_norms = {
        f"{left}__{right}": float(np.linalg.norm(block, ord="fro"))
        for (left, right), block in blocks.items()
    }
    pairwise = {}
    cross_correlations = {}
    max_cross_correlations = {}
    names = list(product_space.factor_slices)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairwise[f"{left_name}__{right_name}"] = factor_coupling_score(
                blocks[(left_name, left_name)],
                blocks[(right_name, right_name)],
                blocks[(left_name, right_name)],
            )
            correlation_block = np.abs(correlation_blocks[(left_name, right_name)])
            cross_correlations[f"{left_name}__{right_name}"] = float(
                np.nanmean(correlation_block)
            )
            max_cross_correlations[f"{left_name}__{right_name}"] = float(
                np.nanmax(correlation_block)
            )
    within_correlations = {}
    max_within_correlations = {}
    for name in names:
        mean_value, max_value = offdiag_abs_correlation_stats(
            correlation_blocks[(name, name)]
        )
        within_correlations[name] = mean_value
        max_within_correlations[name] = max_value
    return ProductCouplingResult(
        z_coordinates=metric.z_coordinates,
        factor_names=names,
        tangent_labels=metric.tangent_labels,
        block_norms=block_norms,
        pairwise_couplings=pairwise,
        within_factor_tangent_correlations=within_correlations,
        max_within_factor_tangent_correlations=max_within_correlations,
        cross_factor_tangent_correlations=cross_correlations,
        max_cross_factor_tangent_correlations=max_cross_correlations,
        product_error=product_metric_error(metric.G, product_space),
        G=metric.G,
        tangent_correlation=tangent_correlation,
    )


def analyze_product_structure(
    render_map: Any,
    product_space: ProductFactorSpace,
    zs: list[Any] | tuple[Any, ...],
    eps: float = 1.0e-3,
    max_points: int = 256,
    rank_tol: float = 1.0e-6,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> ProductCouplingSummary:
    """Analyze product metric block coupling on a subset of states."""

    selected = _select_values(list(zs), max_points=max_points, seed=seed)
    results = [
        analyze_product_structure_at_point(
            render_map,
            product_space,
            z,
            eps=eps,
            rank_tol=rank_tol,
        )
        for z in selected
    ]
    coupling_keys = sorted(
        {key for result in results for key in result.pairwise_couplings}
    )
    mean_pairwise = {}
    std_pairwise = {}
    for key in coupling_keys:
        values = np.asarray(
            [result.pairwise_couplings.get(key, np.nan) for result in results],
            dtype=np.float64,
        )
        mean_pairwise[key] = float(np.nanmean(values))
        std_pairwise[key] = float(np.nanstd(values))
    within_keys = sorted(
        {key for result in results for key in result.within_factor_tangent_correlations}
    )
    cross_keys = sorted(
        {key for result in results for key in result.cross_factor_tangent_correlations}
    )
    mean_within, max_within = _aggregate_mean_and_max(
        results,
        within_keys,
        "within_factor_tangent_correlations",
        "max_within_factor_tangent_correlations",
    )
    mean_cross, max_cross = _aggregate_mean_and_max(
        results,
        cross_keys,
        "cross_factor_tangent_correlations",
        "max_cross_factor_tangent_correlations",
    )
    errors = np.asarray([result.product_error for result in results], dtype=np.float64)
    tangent_correlations = (
        np.stack([result.tangent_correlation for result in results])
        if results
        else np.empty((0, product_space.dim, product_space.dim))
    )
    metric_diagonals = (
        np.stack([np.diag(result.G) for result in results])
        if results
        else np.empty((0, product_space.dim), dtype=np.float64)
    )
    tangent_norms = (
        np.sqrt(np.maximum(metric_diagonals, 0.0))
        if results
        else np.empty((0, product_space.dim), dtype=np.float64)
    )
    tangent_labels = (
        results[0].tangent_labels
        if results
        else product_space.tangent_labels(None)
    )
    mean_metric_diagonal = (
        np.nanmean(metric_diagonals, axis=0) if results else np.empty(0)
    )
    mean_tangent_norms = (
        np.nanmean(tangent_norms, axis=0)
        if results
        else np.empty(0)
    )
    total_energy = float(np.nansum(np.maximum(mean_metric_diagonal, 0.0)))
    relative_tangent_energy = (
        np.maximum(mean_metric_diagonal, 0.0) / total_energy
        if total_energy > 1.0e-12
        else np.zeros_like(mean_metric_diagonal)
    )
    tangent_energy = tangent_energy_table(
        tangent_labels,
        mean_metric_diagonal,
        mean_tangent_norms,
    )
    mean_tangent_correlation = (
        np.nanmean(tangent_correlations, axis=0)
        if results
        else np.empty((0, 0))
    )
    pairwise_tangent_correlations = tangent_correlation_pairs(
        tangent_labels,
        mean_tangent_correlation,
    )
    path = "" if output_path is None else str(Path(output_path))
    summary = ProductCouplingSummary(
        n_points=len(results),
        tangent_labels=tangent_labels,
        mean_pairwise_couplings=mean_pairwise,
        std_pairwise_couplings=std_pairwise,
        mean_within_factor_tangent_correlations=mean_within,
        max_within_factor_tangent_correlations=max_within,
        mean_cross_factor_tangent_correlations=mean_cross,
        max_cross_factor_tangent_correlations=max_cross,
        mean_metric_diagonal=mean_metric_diagonal,
        mean_tangent_norms=mean_tangent_norms,
        relative_tangent_energy=relative_tangent_energy,
        tangent_energy=tangent_energy,
        mean_tangent_correlation=mean_tangent_correlation,
        pairwise_tangent_correlations=pairwise_tangent_correlations,
        top_tangent_collapse_pairs=pairwise_tangent_correlations[:20],
        low_energy_tangent_directions=low_energy_tangent_directions(
            tangent_energy,
            top_k=10,
        ),
        mean_product_error=float(np.nanmean(errors)) if errors.size else 0.0,
        median_product_error=float(np.nanmedian(errors)) if errors.size else 0.0,
        results_path=path,
    )
    if output_path is not None:
        _write_product_results(Path(output_path), summary, results)
    return summary


def _aggregate_mean_and_max(
    results: list[ProductCouplingResult],
    keys: list[str],
    mean_attribute: str,
    max_attribute: str,
) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    maxes = {}
    for key in keys:
        mean_values = np.asarray(
            [getattr(result, mean_attribute).get(key, np.nan) for result in results],
            dtype=np.float64,
        )
        max_values = np.asarray(
            [getattr(result, max_attribute).get(key, np.nan) for result in results],
            dtype=np.float64,
        )
        means[key] = float(np.nanmean(mean_values))
        maxes[key] = float(np.nanmax(max_values))
    return means, maxes


def _select_values(values: list[Any], *, max_points: int, seed: int) -> list[Any]:
    if max_points <= 0 or len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=max_points, replace=False))
    return [values[int(index)] for index in indices]


def _write_product_results(
    path: Path,
    summary: ProductCouplingSummary,
    results: list[ProductCouplingResult],
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
