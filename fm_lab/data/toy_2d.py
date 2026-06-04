"""Controlled 2D toy target distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu" if device is None else device)


@dataclass
class TwoMoons:
    noise: float = 0.05
    scale: float = 1.0
    name: str = "two_moons"
    dim: int = 2

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        n_first = n // 2
        n_second = n - n_first

        theta_first = torch.rand(n_first, device=device) * math.pi
        theta_second = torch.rand(n_second, device=device) * math.pi

        first = torch.stack([torch.cos(theta_first), torch.sin(theta_first)], dim=1)
        second = torch.stack(
            [1.0 - torch.cos(theta_second), 1.0 - torch.sin(theta_second) - 0.5],
            dim=1,
        )

        x = torch.cat([first, second], dim=0)
        x = x - torch.tensor([0.5, 0.25], device=device)
        x = x * self.scale
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x[torch.randperm(n, device=device)]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {"name": self.name, "noise": self.noise, "scale": self.scale, "dim": self.dim}


@dataclass
class Checkerboard:
    grid_size: int = 4
    extent: float = 2.0
    noise: float = 0.02
    name: str = "checkerboard"
    dim: int = 2

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i + j) % 2 == 0
        ]
        cell_ids = torch.randint(0, len(cells), (n,), device=device)
        cell_tensor = torch.tensor(cells, dtype=torch.float32, device=device)[cell_ids]
        unit = torch.rand(n, 2, device=device)
        cell_size = 2 * self.extent / self.grid_size
        x = -self.extent + (cell_tensor + unit) * cell_size
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "grid_size": self.grid_size,
            "extent": self.extent,
            "noise": self.noise,
            "dim": self.dim,
        }


@dataclass
class GaussianMixture2D:
    n_modes: int = 8
    radius: float = 2.0
    std: float = 0.08
    name: str = "gaussian_mixture"
    dim: int = 2

    def centers(self, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        angles = torch.linspace(0, 2 * math.pi, self.n_modes + 1, device=device)[:-1]
        return self.radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        centers = self.centers(device)
        mode_ids = torch.randint(0, self.n_modes, (n,), device=device)
        return centers[mode_ids] + self.std * torch.randn(n, 2, device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        centers = self.centers(x.device)
        diff = x[:, None, :] - centers[None, :, :]
        exponent = -0.5 * diff.square().sum(dim=-1) / (self.std**2)
        log_norm = -math.log(2 * math.pi * self.std**2)
        return torch.logsumexp(exponent + log_norm - math.log(self.n_modes), dim=1)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "n_modes": self.n_modes,
            "radius": self.radius,
            "std": self.std,
            "dim": self.dim,
        }


@dataclass
class ConcentricCircles:
    radii: tuple[float, ...] = (0.8, 1.6)
    noise: float = 0.04
    name: str = "concentric_circles"
    dim: int = 2

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        radius_ids = torch.randint(0, len(self.radii), (n,), device=device)
        radii = torch.tensor(self.radii, dtype=torch.float32, device=device)[radius_ids]
        theta = torch.rand(n, device=device) * 2 * math.pi
        x = radii[:, None] * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {"name": self.name, "radii": list(self.radii), "noise": self.noise, "dim": self.dim}


@dataclass
class Annulus:
    inner_radius: float = 0.8
    outer_radius: float = 1.6
    name: str = "annulus"
    dim: int = 2

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        theta = torch.rand(n, device=device) * 2 * math.pi
        radius_sq = torch.rand(n, device=device) * (self.outer_radius**2 - self.inner_radius**2)
        radius = torch.sqrt(radius_sq + self.inner_radius**2)
        return radius[:, None] * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        radius = x.norm(dim=1)
        inside = (radius >= self.inner_radius) & (radius <= self.outer_radius)
        area = math.pi * (self.outer_radius**2 - self.inner_radius**2)
        return torch.where(
            inside,
            torch.full_like(radius, -math.log(area)),
            torch.full_like(radius, -float("inf")),
        )

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "dim": self.dim,
        }
