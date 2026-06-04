"""Spherical source distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SphericalShellSource:
    dim: int = 2
    radius: float = 1.0
    noise: float = 0.0
    name: str = "spherical_shell"

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device("cpu" if device is None else device)
        x = torch.randn(n, self.dim, device=device)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x = self.radius * x
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def metadata(self) -> dict:
        return {"name": self.name, "dim": self.dim, "radius": self.radius, "noise": self.noise}
