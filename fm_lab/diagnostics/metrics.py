"""Distributional metrics for toy generated samples."""

from __future__ import annotations

import torch


def squared_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0),
) -> float:
    """Compute multi-kernel squared MMD."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("squared_mmd expects 2D sample tensors.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sample dimensions must match.")

    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()

    total = torch.tensor(0.0, device=x.device)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)
        total = total + torch.exp(-gamma * xx).mean()
        total = total + torch.exp(-gamma * yy).mean()
        total = total - 2.0 * torch.exp(-gamma * xy).mean()
    return float((total / len(sigmas)).detach().cpu())


def sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 128,
    p: int = 2,
) -> float:
    """Estimate sliced Wasserstein distance with random projections."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("sliced_wasserstein expects 2D sample tensors.")
    if x.shape != y.shape:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sample dimensions must match.")

    projections = torch.randn(x.shape[1], n_projections, device=x.device)
    projections = projections / projections.norm(dim=0, keepdim=True).clamp_min(1e-12)
    x_proj = torch.sort(x @ projections, dim=0).values
    y_proj = torch.sort(y @ projections, dim=0).values
    distance_p = (x_proj - y_proj).abs().pow(p).mean()
    return float(distance_p.pow(1.0 / p).detach().cpu())
