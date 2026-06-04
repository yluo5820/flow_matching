"""Curvature and material-acceleration diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def material_acceleration(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute `partial_t v + J_x v * v` exactly for low-dimensional batches."""

    x_req = x.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    v = model(x_req, t_req)

    components = []
    for output_idx in range(v.shape[1]):
        grad_x, grad_t = torch.autograd.grad(
            v[:, output_idx].sum(),
            (x_req, t_req),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        if grad_x is None:
            grad_x = torch.zeros_like(x_req)
        if grad_t is None:
            grad_t = torch.zeros_like(t_req)
        directional = (grad_x * v).sum(dim=1)
        components.append(grad_t + directional)
    return torch.stack(components, dim=1).detach()


def curvature_stats(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> dict[str, Any]:
    """Summarize material acceleration norms."""

    acceleration = material_acceleration(model, x, t)
    norms = acceleration.norm(dim=1)
    return {
        "acceleration_mean": float(norms.mean().cpu()),
        "acceleration_max": float(norms.max().cpu()),
        "acceleration_sq_mean": float(norms.square().mean().cpu()),
        "acceleration": acceleration.cpu(),
    }
