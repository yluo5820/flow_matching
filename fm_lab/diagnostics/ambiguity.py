"""Conditional velocity ambiguity estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GridAmbiguityResult:
    ambiguity: float
    heatmap: torch.Tensor
    counts: torch.Tensor
    x_edges: torch.Tensor
    y_edges: torch.Tensor
    valid_bins: int


def grid_ambiguity(
    xt: torch.Tensor,
    velocities: torch.Tensor,
    bins: int = 50,
    bounds: tuple[float, float, float, float] | None = None,
    min_count: int = 5,
) -> GridAmbiguityResult:
    """Estimate E[Tr Cov(U | X)] by 2D spatial binning."""

    if xt.shape[1] != 2:
        raise ValueError("grid_ambiguity only supports 2D samples.")
    if xt.shape != velocities.shape:
        raise ValueError("xt and velocities must have matching shapes.")

    if bounds is None:
        x_min, y_min = xt.min(dim=0).values.tolist()
        x_max, y_max = xt.max(dim=0).values.tolist()
        padding = 1e-6
        bounds = (x_min - padding, x_max + padding, y_min - padding, y_max + padding)

    x_min, x_max, y_min, y_max = bounds
    x_edges = torch.linspace(x_min, x_max, bins + 1, device=xt.device)
    y_edges = torch.linspace(y_min, y_max, bins + 1, device=xt.device)
    x_ids = torch.bucketize(xt[:, 0].contiguous(), x_edges) - 1
    y_ids = torch.bucketize(xt[:, 1].contiguous(), y_edges) - 1
    valid = (x_ids >= 0) & (x_ids < bins) & (y_ids >= 0) & (y_ids < bins)

    heatmap = torch.full((bins, bins), float("nan"), device=xt.device)
    counts = torch.zeros((bins, bins), dtype=torch.long, device=xt.device)
    total = torch.tensor(0.0, device=xt.device)
    total_count = torch.tensor(0.0, device=xt.device)
    valid_bins = 0

    flat_ids = y_ids[valid] * bins + x_ids[valid]
    for flat_id in torch.unique(flat_ids):
        mask = valid & ((y_ids * bins + x_ids) == flat_id)
        count = int(mask.sum().item())
        y_idx = int((flat_id // bins).item())
        x_idx = int((flat_id % bins).item())
        counts[y_idx, x_idx] = count
        if count < min_count:
            continue
        local_v = velocities[mask]
        trace_cov = _trace_covariance(local_v)
        heatmap[y_idx, x_idx] = trace_cov
        total = total + trace_cov * count
        total_count = total_count + count
        valid_bins += 1

    if valid_bins == 0:
        ambiguity = float("nan")
    else:
        ambiguity = float((total / total_count).detach().cpu())
    return GridAmbiguityResult(
        ambiguity=ambiguity,
        heatmap=heatmap.detach().cpu(),
        counts=counts.detach().cpu(),
        x_edges=x_edges.detach().cpu(),
        y_edges=y_edges.detach().cpu(),
        valid_bins=valid_bins,
    )


def knn_ambiguity(
    xt: torch.Tensor,
    velocities: torch.Tensor,
    k: int = 32,
    max_points: int | None = 4096,
) -> dict[str, Any]:
    """Estimate local velocity covariance using k-nearest neighbors."""

    if xt.shape != velocities.shape:
        raise ValueError("xt and velocities must have matching shapes.")
    if xt.shape[0] < 2:
        raise ValueError("Need at least two samples for kNN ambiguity.")

    if max_points is not None and xt.shape[0] > max_points:
        indices = torch.randperm(xt.shape[0], device=xt.device)[:max_points]
        xt = xt[indices]
        velocities = velocities[indices]

    k_eff = min(k, xt.shape[0])
    distances = torch.cdist(xt, xt)
    neighbor_ids = distances.topk(k_eff, largest=False).indices
    local_velocities = velocities[neighbor_ids]
    local_mean = local_velocities.mean(dim=1, keepdim=True)
    local_trace_cov = (local_velocities - local_mean).square().sum(dim=2).mean(dim=1)
    return {
        "ambiguity": float(local_trace_cov.mean().detach().cpu()),
        "local_trace_cov": local_trace_cov.detach().cpu(),
        "k": k_eff,
        "n": xt.shape[0],
    }


def bayes_regression_gap_knn(
    xt: torch.Tensor,
    velocities: torch.Tensor,
    k: int = 32,
    max_points: int | None = 4096,
) -> dict[str, Any]:
    """Approximate irreducible deterministic FM MSE with a kNN conditional mean."""

    result = knn_ambiguity(xt, velocities, k=k, max_points=max_points)
    return {
        "bayes_gap": result["ambiguity"],
        "local_squared_error": result["local_trace_cov"],
        "k": result["k"],
        "n": result["n"],
    }


def _trace_covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.square().sum(dim=1).mean()
