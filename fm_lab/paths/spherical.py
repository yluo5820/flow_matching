"""Spherical/radial interpolation paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.paths.base import expand_time


@dataclass
class SphericalPath:
    """Interpolate directions with slerp and radii linearly."""

    eps: float = 1e-6
    interpolate_radius: bool = True
    name: str = "spherical"

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        radius0, radius1, direction0, direction1 = self._decompose(x0, x1)
        direction = self._slerp(direction0, direction1, t_expanded)
        radius = self._radius(radius0, radius1, t_expanded)
        return radius * direction

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        radius0, radius1, direction0, direction1 = self._decompose(x0, x1)
        direction, direction_dt = self._slerp_with_derivative(direction0, direction1, t_expanded)
        radius = self._radius(radius0, radius1, t_expanded)
        radius_dt = radius1 - radius0 if self.interpolate_radius else torch.zeros_like(radius0)
        return radius_dt * direction + radius * direction_dt

    def _decompose(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        radius0 = x0.norm(dim=1, keepdim=True).clamp_min(self.eps)
        radius1 = x1.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return radius0, radius1, x0 / radius0, x1 / radius1

    def _radius(
        self,
        radius0: torch.Tensor,
        radius1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.interpolate_radius:
            return (1.0 - t) * radius0 + t * radius1
        return torch.ones_like(radius0)

    def _slerp(
        self,
        direction0: torch.Tensor,
        direction1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        direction, _ = self._slerp_with_derivative(direction0, direction1, t)
        return direction

    def _slerp_with_derivative(
        self,
        direction0: torch.Tensor,
        direction1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dot = (direction0 * direction1).sum(dim=1, keepdim=True)
        dot = dot.clamp(-1.0 + self.eps, 1.0 - self.eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp_min(self.eps)

        weight0 = torch.sin((1.0 - t) * omega) / sin_omega
        weight1 = torch.sin(t * omega) / sin_omega
        direction = weight0 * direction0 + weight1 * direction1

        d_weight0 = -omega * torch.cos((1.0 - t) * omega) / sin_omega
        d_weight1 = omega * torch.cos(t * omega) / sin_omega
        direction_dt = d_weight0 * direction0 + d_weight1 * direction1

        near_parallel = omega < 1e-4
        if near_parallel.any():
            linear_direction = (1.0 - t) * direction0 + t * direction1
            linear_direction_dt = direction1 - direction0
            direction = torch.where(near_parallel, linear_direction, direction)
            direction_dt = torch.where(near_parallel, linear_direction_dt, direction_dt)

        return direction, direction_dt
