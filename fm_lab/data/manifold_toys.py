"""Low-dimensional manifold and shell toy targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu" if device is None else device)


@dataclass
class SphericalShell:
    dim: int = 3
    radius: float = 1.0
    noise: float = 0.02
    name: str = "spherical_shell"

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        x = torch.randn(n, self.dim, device=device)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x = self.radius * x
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {"name": self.name, "dim": self.dim, "radius": self.radius, "noise": self.noise}


@dataclass
class SwissRoll:
    noise: float = 0.05
    scale: float = 1.0
    name: str = "swiss_roll"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        t = 1.5 * math.pi * (1.0 + 2.0 * torch.rand(n, device=device))
        height = 2.0 * torch.rand(n, device=device) - 1.0
        x = torch.stack([t * torch.cos(t), height, t * torch.sin(t)], dim=1)
        x = self.scale * x / (4.5 * math.pi)
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {"name": self.name, "dim": self.dim, "noise": self.noise, "scale": self.scale}
