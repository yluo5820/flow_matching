"""Geometry mismatch diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def radial_deviation(x: torch.Tensor, radii: tuple[float, ...]) -> dict[str, Any]:
    """Distance from each point to the nearest configured radius."""

    if not radii:
        raise ValueError("At least one radius is required.")
    norms = x.norm(dim=1)
    radius_tensor = torch.tensor(radii, dtype=x.dtype, device=x.device)
    deviations = (norms[:, None] - radius_tensor[None, :]).abs().min(dim=1).values
    return {
        "radial_deviation_mean": float(deviations.mean().detach().cpu()),
        "radial_deviation_max": float(deviations.max().detach().cpu()),
        "radial_deviation": deviations.detach().cpu(),
    }


def radial_tangent_velocity_2d(x: torch.Tensor, velocity: torch.Tensor) -> dict[str, Any]:
    """Decompose 2D velocity into radial and tangent components."""

    if x.shape[1] != 2 or velocity.shape[1] != 2:
        raise ValueError("radial_tangent_velocity_2d expects 2D tensors.")
    radial_unit = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tangent_unit = torch.stack([-radial_unit[:, 1], radial_unit[:, 0]], dim=1)
    radial = (velocity * radial_unit).sum(dim=1)
    tangent = (velocity * tangent_unit).sum(dim=1)
    return {
        "radial_velocity_abs_mean": float(radial.abs().mean().detach().cpu()),
        "tangent_velocity_abs_mean": float(tangent.abs().mean().detach().cpu()),
        "normal_tangent_ratio": float(
            (radial.abs().mean() / tangent.abs().mean().clamp_min(1e-12)).detach().cpu()
        ),
        "radial_velocity": radial.detach().cpu(),
        "tangent_velocity": tangent.detach().cpu(),
    }
