"""2D tangent-normal path for circle and annulus experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fm_lab.paths.base import expand_time


@dataclass
class TangentNormalPath:
    """Interpolate 2D polar angle along the tangent direction and radius normally."""

    eps: float = 1e-6
    name: str = "tangent_normal"

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        radius, theta, _, _ = self._polar_path(x0, x1, t_expanded)
        return radius * torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        radius, theta, radius_dt, theta_dt = self._polar_path(x0, x1, t_expanded)
        radial = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        tangent = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)
        return radius_dt * radial + radius * theta_dt * tangent

    def _polar_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x0.shape[1] != 2 or x1.shape[1] != 2:
            raise ValueError("TangentNormalPath currently supports 2D inputs only.")

        radius0 = x0.norm(dim=1, keepdim=True).clamp_min(self.eps)
        radius1 = x1.norm(dim=1, keepdim=True).clamp_min(self.eps)
        theta0 = torch.atan2(x0[:, 1:2], x0[:, 0:1])
        theta1 = torch.atan2(x1[:, 1:2], x1[:, 0:1])
        delta_theta = _wrap_angle(theta1 - theta0)
        radius = (1.0 - t) * radius0 + t * radius1
        theta = theta0 + t * delta_theta
        radius_dt = radius1 - radius0
        return radius, theta, radius_dt, delta_theta


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
