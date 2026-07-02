"""Jacobian and stiffness-proxy diagnostics for learned vector fields."""

from __future__ import annotations

from typing import Any

import torch

from fm_lab.diagnostics._linalg import svdvals


def exact_jacobian(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Jacobian `d v_theta(x,t) / d x` for low-dimensional models."""

    x_req = x.detach().clone().requires_grad_(True)
    t_req = t.detach().clone()
    v = model(x_req, t_req)
    columns = []
    for output_idx in range(v.shape[1]):
        grad = torch.autograd.grad(
            v[:, output_idx].sum(),
            x_req,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(x_req)
        columns.append(grad)
    return torch.stack(columns, dim=1).detach()


def jacobian_stats(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> dict[str, Any]:
    """Return Frobenius norm, spectral norm, and divergence summaries."""

    jacobian = exact_jacobian(model, x, t)
    frobenius = jacobian.square().sum(dim=(1, 2)).sqrt()
    singular_values = svdvals(jacobian)
    spectral = singular_values[:, 0]
    divergence = torch.diagonal(jacobian, dim1=1, dim2=2).sum(dim=1)
    return {
        "frobenius_mean": float(frobenius.mean().cpu()),
        "frobenius_max": float(frobenius.max().cpu()),
        "spectral_mean": float(spectral.mean().cpu()),
        "spectral_max": float(spectral.max().cpu()),
        "divergence_mean": float(divergence.mean().cpu()),
        "divergence_std": float(divergence.std(unbiased=False).cpu()),
        "jacobian": jacobian.cpu(),
    }
